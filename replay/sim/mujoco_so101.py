"""
MuJoCo replay backend for SO-ARM101.

Pipeline position: registered for `(so101, mujoco)` in replay/__init__.py.
Consumed via `python -m replay`.

Two render modes:
  * `replay_to_viewer(...)` — interactive `mujoco.viewer.launch_passive`.
  * `replay_to_mp4(...)` — offscreen `mujoco.Renderer` → mp4 via imageio.

`run(...)` is the unified entrypoint dispatched by replay/__main__.py;
it reads .qpos.npz + .qpos.meta.json (qpos in radians, output of
`python -m retarget --robot so101`), converts to degrees at the
LeRobot-naming boundary, holds-last on invalid frames so the playback
length matches the source recording.

cam_frame view (mp4 default):
  retarget writes `cam_pos_arm` / `cam_quat_arm_xyzw` / `K_flat` into
  the qpos meta. We synthesise a derived scene XML that includes the
  upstream so101_new_calib.xml + a fixed cam_frame camera at that pose
  with fovy = 2*atan(cy/fy) so the rendered viewpoint matches where the
  recording camera was relative to the (anchored) wrist.

Why separate from the loader: the loader is stateless; this module owns
the render-loop timing, scene generation, and the kinematics-vs-dynamics
decision.
"""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import imageio
import mujoco
import mujoco.viewer
import numpy as np

from replay.sim.mujoco_loader import (
    LoadedScene,
    apply_joint_positions_deg,
    load_so_arm101,
    reset_to_keyframe,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_SCENE = _PROJECT_ROOT / "assets" / "mujoco" / "trs_so101" / "scene.xml"
_SCENE_CACHE_DIR = _PROJECT_ROOT / "replay" / "sim" / "scenes"


# Derived scene injected with cam_frame. We patch a copy of
# so101_new_calib.xml with an absolute meshdir (so it resolves from
# anywhere) and write both the patched copy + the wrapper scene under
# `_SCENE_CACHE_DIR/`. Same pattern as mujoco_dex.py.
_SCENE_TEMPLATE = """<mujoco model="so101_replay_scene">
  <include file="{patched_arm_filename}"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="160" elevation="-20" offwidth="{offwidth}" offheight="{offheight}"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane"/>
    <!-- cam_frame: pose + fovy come from retarget meta (recording camera
         relative to first-valid wrist, lifted into arm frame). -->
    <camera name="cam_frame" pos="{cam_pos}" quat="{cam_quat_wxyz}"
            mode="fixed" fovy="{cam_fovy_deg}"/>
  </worldbody>

  <keyframe>
    <key name="home" qpos="0 0 0 0 0 0" ctrl="0 0 0 0 0 0"/>
  </keyframe>
</mujoco>
"""


def _fovy_from_K(K_flat: list[float]) -> float:
    """Vertical FOV (deg) from a 3×3 intrinsic matrix.  fovy = 2·atan(cy/fy)."""
    import math
    K = np.asarray(K_flat, dtype=np.float64).reshape(3, 3)
    fy = float(K[1, 1])
    cy = float(K[1, 2])
    if fy <= 0:
        raise ValueError(f"invalid K[1,1]={fy}; expected positive focal length")
    return float(np.degrees(2.0 * math.atan(cy / fy)))


def _patch_arm_mjcf() -> Path:
    """Copy so101_new_calib.xml with absolute meshdir into scene cache.
    Idempotent: rewrite is cheap, costs <1ms."""
    src = _PROJECT_ROOT / "assets" / "mujoco" / "trs_so101" / "so101_new_calib.xml"
    if not src.exists():
        raise FileNotFoundError(f"upstream MJCF not found: {src}")
    raw = src.read_text(encoding="utf-8")
    abs_meshdir = (src.parent / "assets").resolve().as_posix()
    if 'meshdir="assets"' not in raw:
        raise RuntimeError(
            "expected `meshdir=\"assets\"` in so101_new_calib.xml; not found"
        )
    raw = raw.replace('meshdir="assets"', f'meshdir="{abs_meshdir}"', 1)
    _SCENE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = _SCENE_CACHE_DIR / "_patched_so101_new_calib.xml"
    out.write_text(raw, encoding="utf-8")
    return out


def _apply_cam_zoom(
    cam_pos: list[float], centroid: list[float] | None, cam_zoom: float,
) -> tuple[list[float], dict]:
    """Push cam back along its viewing axis (cam→centroid) so the rendered
    robot covers fewer pixels. Visual-only — caller keeps the original
    cam_quat / fovy / K so the projection math is unchanged.

    Same semantics in mp4 + rerun: `back_off = (1/cam_zoom - 1) * baseline`.
    cam_zoom = 1.0 → no change; < 1.0 → cam moves away from centroid;
    > 1.0 → cam moves toward centroid.

    Returns (cam_pos_eff, info_dict). info_dict includes baseline + back_off
    for caller logging; empty when zoom skipped.
    """
    if cam_zoom <= 0:
        raise ValueError(f"cam_zoom must be > 0, got {cam_zoom}")
    cam_pos_eff = np.asarray(cam_pos, dtype=np.float64).copy()
    if cam_zoom == 1.0 or centroid is None:
        return cam_pos_eff.tolist(), {}
    ctr = np.asarray(centroid, dtype=np.float64)
    view_axis = ctr - cam_pos_eff
    baseline = float(np.linalg.norm(view_axis))
    if baseline <= 1e-6:
        return cam_pos_eff.tolist(), {}
    view_axis /= baseline
    back_off = (1.0 / cam_zoom - 1.0) * baseline
    cam_pos_eff = cam_pos_eff - view_axis * back_off
    return cam_pos_eff.tolist(), {
        "baseline_depth_m": baseline,
        "back_off_m": back_off,
        "new_depth_m": baseline + back_off,
    }


def _build_cam_scene(
    cam_pos: list[float], cam_quat_xyzw: list[float],
    fovy_deg: float, *, width: int, height: int,
) -> Path:
    """Write a derived scene with cam_frame injected. Returns its path.

    MuJoCo quat order is [w x y z]; scipy/our meta is [x y z w] — convert here.
    """
    patched_arm = _patch_arm_mjcf()
    qx, qy, qz, qw = cam_quat_xyzw
    xml = _SCENE_TEMPLATE.format(
        patched_arm_filename=patched_arm.name,
        offwidth=width, offheight=height,
        cam_pos=" ".join(f"{x:.6f}" for x in cam_pos),
        cam_quat_wxyz=f"{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}",
        cam_fovy_deg=f"{fovy_deg:.4f}",
    )
    out = _SCENE_CACHE_DIR / "_scene_so101.xml"
    out.write_text(xml, encoding="utf-8")
    return out


# Canonical arm joint order — matches SO101Arm.joint_names (robots/so101.py)
# so we can consume a 5-column joint trajectory by name.
ARM_JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


def replay_joint_trajectory(
    scene: LoadedScene,
    joint_angles_deg: np.ndarray,
    gripper_deg: np.ndarray,
    *,
    fps: int = 30,
    speed: float = 1.0,
    no_gui: bool = False,
    reset_keyframe: str | None = "home",
    physics: bool = False,
    loop: bool = False,
) -> None:
    """Drive the SO-ARM101 through a pre-computed joint trajectory.

    Args:
        scene: loaded scene from replay.sim.mujoco_loader.load_so_arm101.
        joint_angles_deg: (T, 5) arm joints in degrees. Column order must
            match ARM_JOINT_NAMES — the caller is responsible for aligning
            the ikpy chain's joint output to this order.
        gripper_deg: (T,) gripper angle in degrees.
        fps: dataset fps (defines nominal step size).
        speed: playback multiplier (1.0 = realtime, 5.0 = 5× faster).
        no_gui: headless mode; skips viewer. Required in CI / unit tests —
            launch_passive opens a GLFW window that blocks.
        reset_keyframe: jump to this keyframe before the loop; None skips.
            Default "home" puts the arm in a neutral pose so the first
            commanded frame has a predictable starting state.
        physics: True → mj_step (dynamics + contact); False → mj_kinematics
            (geometry only). Phase 1.0 default is False — we validate
            geometric feasibility, not servo tracking under load.
        loop: GUI only. True → restart from frame 0 when the trajectory ends.
            False → hold on the final pose until the user closes the window.
            Headless runs exit as soon as the trajectory finishes.
    """
    T = len(joint_angles_deg)
    if joint_angles_deg.shape[1] != len(ARM_JOINT_NAMES):
        raise ValueError(
            f"joint_angles_deg has {joint_angles_deg.shape[1]} cols, "
            f"expected {len(ARM_JOINT_NAMES)} ({ARM_JOINT_NAMES})"
        )
    if len(gripper_deg) != T:
        raise ValueError(
            f"gripper_deg length {len(gripper_deg)} != trajectory length {T}"
        )

    if reset_keyframe is not None:
        reset_to_keyframe(scene, reset_keyframe)

    effective_fps = max(fps * speed, 1e-6)
    dt = 1.0 / effective_fps

    viewer_ctx = (
        nullcontext(None) if no_gui
        else mujoco.viewer.launch_passive(scene.model, scene.data)
    )

    print(
        f"  [sim] Replaying {T} frames at {effective_fps:.1f} Hz  "
        f"({'kinematics' if not physics else 'physics'}, "
        f"{'headless' if no_gui else 'gui'})"
    )

    with viewer_ctx as viewer:
        pass_num = 0
        while True:
            pass_num += 1
            if loop and pass_num > 1:
                print(f"  [sim] Loop pass #{pass_num}")

            user_closed = False
            for t in range(T):
                # Break out promptly if the user closes the GUI window.
                if viewer is not None and not viewer.is_running():
                    user_closed = True
                    break
                t0 = time.perf_counter()

                cmd: dict[str, float] = {
                    name: float(joint_angles_deg[t, i])
                    for i, name in enumerate(ARM_JOINT_NAMES)
                }
                cmd["gripper"] = float(gripper_deg[t])

                # ctrl drives position actuators (mj_step path). In kinematics
                # mode we also set qpos directly to bypass actuator dynamics —
                # kp/dampratio tuning would otherwise lag the preview behind
                # the commanded trajectory.
                apply_joint_positions_deg(scene, cmd, target="ctrl")
                if physics:
                    mujoco.mj_step(scene.model, scene.data)
                else:
                    apply_joint_positions_deg(scene, cmd, target="qpos")
                    mujoco.mj_kinematics(scene.model, scene.data)

                if viewer is not None:
                    viewer.sync()

                if (t + 1) % 60 == 0 or t == T - 1:
                    print(f"    [sim] frame {t + 1}/{T}")

                if not no_gui:
                    elapsed = time.perf_counter() - t0
                    time.sleep(max(dt - elapsed, 0.0))

            if user_closed or no_gui or not loop:
                break

        # GUI + single-play: hold on the final pose so the user can inspect
        # the scene instead of the window snapping shut.
        if viewer is not None and not loop:
            print("  [sim] Replay done. Window stays open — close it to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / 30.0)

    print(f"  [sim] Replay complete.")


# =============================================================================
# .qpos.npz → arm + gripper trajectory loader
# =============================================================================

@dataclass
class _Episode:
    arm_deg: np.ndarray             # (T, 5) degrees, hold-last filled
    gripper_deg: np.ndarray         # (T,)   degrees, hold-last filled
    qpos_valid: np.ndarray          # (T,)   bool — original validity
    n_total: int
    fps: int


def _qpos_to_arm_gripper_deg(
    qpos_rad: np.ndarray, qpos_valid: np.ndarray, joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Slice qpos (radians, joint-name order) → (arm_deg, gripper_deg).

    Hold-last on invalid frames so the playback length matches the source
    recording. Leading invalid frames stay as the first valid pose to
    avoid driving the arm with NaNs.
    """
    name_to_col = {n: i for i, n in enumerate(joint_names)}
    expected_arm = list(ARM_JOINT_NAMES)
    if not all(n in name_to_col for n in expected_arm + ["gripper"]):
        raise ValueError(
            f"qpos joint_names {joint_names!r} missing expected SO-101 "
            f"joints {expected_arm + ['gripper']!r}"
        )
    arm_cols = [name_to_col[n] for n in expected_arm]
    gripper_col = name_to_col["gripper"]

    arm_rad = qpos_rad[:, arm_cols].astype(np.float64)
    gripper_rad = qpos_rad[:, gripper_col].astype(np.float64)

    # Hold-last fill for invalid frames so the trajectory has no NaNs.
    valid_idx = np.flatnonzero(qpos_valid)
    if len(valid_idx) == 0:
        raise ValueError("qpos has no valid frames; nothing to replay")
    first_valid = int(valid_idx[0])
    # Pre-first-valid frames: clone first valid pose.
    arm_rad[:first_valid] = arm_rad[first_valid]
    gripper_rad[:first_valid] = gripper_rad[first_valid]
    # Mid-trajectory invalid frames: forward-fill last valid pose.
    last_arm = arm_rad[first_valid].copy()
    last_grip = float(gripper_rad[first_valid])
    for t in range(first_valid, len(qpos_valid)):
        if qpos_valid[t]:
            last_arm = arm_rad[t].copy()
            last_grip = float(gripper_rad[t])
        else:
            arm_rad[t] = last_arm
            gripper_rad[t] = last_grip
    return np.degrees(arm_rad), np.degrees(gripper_rad)


def _load_episode(qpos_npz_path: Path, qpos_meta_path: Path, hand: str) -> _Episode:
    arr = np.load(qpos_npz_path, allow_pickle=False)
    qpos_key = f"{hand}_qpos"
    valid_key = f"{hand}_qpos_valid"
    if qpos_key not in arr.files or valid_key not in arr.files:
        raise KeyError(
            f"{qpos_npz_path.name} missing {qpos_key!r}/{valid_key!r}. "
            f"Available: {arr.files}"
        )
    qpos = arr[qpos_key]
    valid = arr[valid_key].astype(bool)

    meta = json.loads(qpos_meta_path.read_text(encoding="utf-8"))
    joint_names = meta["joint_names"]
    if qpos.ndim != 2 or qpos.shape[1] != len(joint_names):
        raise ValueError(
            f"qpos shape {qpos.shape} doesn't match joint_names "
            f"({len(joint_names)} cols). Re-run retarget."
        )

    # Fps source priority: qpos timestamps_us (always present, recording-
    # rate accurate) → meta['fps'] (legacy, retarget never wrote it) → 30
    # default. Mirrors `replay/__main__.py`'s auto-derive so the viewer
    # mode plays at the recording rate, not a hardcoded 30 (which slowed
    # iPhone 60 fps captures to half speed).
    fps_resolved: int
    if "timestamps_us" in arr.files and len(arr["timestamps_us"]) >= 2:
        ts = arr["timestamps_us"].astype(np.int64)
        dt_us = float(np.median(np.diff(ts)))
        fps_resolved = int(round(1e6 / max(dt_us, 1.0)))
    else:
        fps_resolved = int(meta.get("fps", 30))

    arm_deg, gripper_deg = _qpos_to_arm_gripper_deg(qpos, valid, joint_names)
    return _Episode(
        arm_deg=arm_deg, gripper_deg=gripper_deg, qpos_valid=valid,
        n_total=len(qpos), fps=fps_resolved,
    )


# =============================================================================
# Render functions
# =============================================================================

def _render_to_mp4(
    scene: LoadedScene, ep: _Episode, out_mp4: Path,
    *, width: int, height: int, fps: int, camera: str,
) -> dict:
    """Offscreen render → mp4. Drives qpos directly (kinematics)."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    reset_to_keyframe(scene, "home")

    writer = imageio.get_writer(
        str(out_mp4), fps=fps, codec="libx264", quality=8,
        macro_block_size=1,
    )
    n_held = 0
    try:
        with mujoco.Renderer(scene.model, height=height, width=width) as r:
            for t in range(ep.n_total):
                if not bool(ep.qpos_valid[t]):
                    n_held += 1
                cmd: dict[str, float] = {
                    name: float(ep.arm_deg[t, i])
                    for i, name in enumerate(ARM_JOINT_NAMES)
                }
                cmd["gripper"] = float(ep.gripper_deg[t])
                apply_joint_positions_deg(scene, cmd, target="qpos")
                # mj_forward (not mj_kinematics) — kinematics alone leaves
                # cam_xpos at (0,0,0), so any worldbody-mounted camera ends
                # up rendering from world origin instead of its declared
                # pose. mj_forward also computes mj_camlight(), which is
                # what propagates the MJCF camera pos/quat into cam_xpos.
                mujoco.mj_forward(scene.model, scene.data)
                try:
                    r.update_scene(scene.data, camera=camera)
                except Exception:
                    # Fall back to default camera if the requested name is
                    # not in scene.xml (TRS scene only ships a free cam).
                    r.update_scene(scene.data)
                writer.append_data(r.render())
    finally:
        writer.close()
    return {
        "n_total": ep.n_total,
        "n_rendered": ep.n_total,
        "n_held": n_held,
        "n_skipped_leading_invalid": 0,
        "output": str(out_mp4),
    }


# =============================================================================
# Backend entrypoint (called by replay/__main__.py)
# =============================================================================

def run(
    *,
    qpos_npz_path: Path,
    qpos_meta_path: Path,
    output: str,                    # "viewer" | "mp4" | "rerun" | "rrd"
    out_mp4: Path | None = None,
    out_rrd: Path | None = None,
    source_mp4: Path | None = None,
    source_npz: Path | None = None,  # not used by SO-101 (kept for CLI parity)
    scene_xml: Path | None = None,
    fps: int = 30,
    width: int = 1024,
    height: int = 768,
    camera: str = "cam_frame",
    loop: bool = True,
    jpeg_quality: int = 85,
    image_plane_dist_m: float = 2.0,
    cam_zoom: float = 1.0,
) -> dict | None:
    """Unified entrypoint dispatched by replay/__main__.py.

    Output modes:
      viewer  — interactive MuJoCo viewer (live window)
      mp4     — offscreen MuJoCo render to file
      rerun   — spawn rerun.io viewer with AR overlay (hand on source RGB)
      rrd     — write rerun .rrd file (open later via `rerun <path>`)
    """
    if output in ("rerun", "rrd"):
        if source_mp4 is None:
            raise ValueError(
                "output='rerun'/'rrd' requires source_mp4 (stage-1 "
                "preview mp4). The CLI in replay/__main__.py derives it."
            )
        from replay.sim.rerun_so101 import replay_to_rerun
        return replay_to_rerun(
            qpos_npz_path=qpos_npz_path,
            qpos_meta_path=qpos_meta_path,
            source_mp4_path=source_mp4,
            out_rrd_path=out_rrd if output == "rrd" else None,
            width=width, height=height,
            jpeg_quality=jpeg_quality,
            image_plane_dist_m=image_plane_dist_m,
            cam_zoom=cam_zoom,
        )

    meta = json.loads(qpos_meta_path.read_text(encoding="utf-8"))
    hand = meta["hand"]
    extras = meta.get("extras", {})

    # Generate a scene with cam_frame injected if retarget supplied the
    # offset + intrinsics; otherwise fall back to the upstream scene
    # (free-cam view only). Caller can override with --scene-xml.
    if scene_xml is None:
        cam_pos = extras.get("cam_pos_arm")
        cam_quat = extras.get("cam_quat_arm_xyzw")
        centroid = extras.get("target_pos_arm_centroid")
        K_flat = meta.get("K_flat")
        if cam_pos is not None and cam_quat is not None and K_flat is not None:
            fovy = _fovy_from_K(K_flat)
            # Apply cam_zoom (visual-only dolly back along view axis). Same
            # semantics as rerun path so a single CLI flag controls both.
            # Skipped silently if meta lacks centroid (older retarget runs).
            cam_pos_eff, zoom_info = _apply_cam_zoom(
                cam_pos, centroid, cam_zoom,
            )
            scene_xml = _build_cam_scene(
                cam_pos_eff, cam_quat, fovy, width=width, height=height,
            )
            log = (f"  [sim] cam_frame: pos={[round(x,3) for x in cam_pos_eff]} "
                   f"fovy={fovy:.1f}deg")
            if zoom_info:
                log += (f"  cam_zoom={cam_zoom:.3f}  "
                        f"depth {zoom_info['baseline_depth_m']:.3f}→"
                        f"{zoom_info['new_depth_m']:.3f}m")
            elif cam_zoom != 1.0 and centroid is None:
                log += f"  WARN cam_zoom={cam_zoom} ignored (meta has no centroid)"
            print(log)
        else:
            scene_xml = _DEFAULT_SCENE
            print("  [sim] no cam_frame in meta — falling back to upstream scene")

    scene = load_so_arm101(scene_xml)
    ep = _load_episode(qpos_npz_path, qpos_meta_path, hand)
    # Caller fps wins over auto-derived (`replay/__main__.py` already
    # auto-derives args.fps from qpos timestamps; if user passed --fps N
    # explicitly that comes through as `fps`). ep.fps is the fallback.
    effective_fps = int(fps) if fps else ep.fps
    print(f"  [sim] {ep.n_total} frames @ {effective_fps} fps "
          f"(valid={int(ep.qpos_valid.sum())})")

    if output == "viewer":
        replay_joint_trajectory(
            scene,
            joint_angles_deg=ep.arm_deg,
            gripper_deg=ep.gripper_deg,
            fps=effective_fps, speed=1.0, no_gui=False,
            reset_keyframe="home", physics=False, loop=loop,
        )
        return None

    if output == "mp4":
        if out_mp4 is None:
            raise ValueError("output='mp4' requires out_mp4 path")
        return _render_to_mp4(
            scene, ep, out_mp4,
            width=width, height=height, fps=effective_fps, camera=camera,
        )

    raise ValueError(f"unknown output mode {output!r}; use 'viewer' or 'mp4'")
