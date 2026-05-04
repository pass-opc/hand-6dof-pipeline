"""
MuJoCo replay backend for dex hands (shadow / leap / allegro / ...).

Pipeline position: registered for `(robot, mujoco)` keys in
replay/__init__.py. Consumed via `python -m replay`.

Two render modes (officially-recommended MuJoCo APIs):
  * `replay_to_viewer(...)` — interactive GUI window via
    `mujoco.viewer.launch_passive`. Default for visual QA. User can
    pan/zoom/orbit, pause via space, etc.
  * `replay_to_mp4(...)` — offscreen `mujoco.Renderer` → mp4 via
    imageio-ffmpeg. For batch verification / archival.

MJCF + freejoint handling (Q4 fix):
  Menagerie's right_hand.xml has rh_forearm directly under world (fixed
  base). Position-retargeting needs a 6-DoF freejoint there. We patch
  the source XML in a STRING and write the patched copy under
  replay/sim/scenes/ (gitignored) — never to menagerie/ (submodule
  must stay clean).

Joint qpos assignment (Q5 fix):
  Use `data.joint("rh_FFJ4").qpos[0] = value` (named accessor) rather
  than integer-index `data.qpos[14] = value`. Self-documenting; mismatch
  raises immediately rather than silently writing into the wrong slot.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import imageio
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCENE_CACHE_DIR = _PROJECT_ROOT / "replay" / "sim" / "scenes"
_DEFAULT_MENAGERIE_ROOT = _PROJECT_ROOT / "assets" / "menagerie"


# =============================================================================
# Robot → menagerie MJCF mapping
# =============================================================================

@dataclass
class RobotMjcfSpec:
    mjcf_subdir: str               # under assets/menagerie/
    mjcf_filename: str
    target_body: str               # body to inject freejoint on
    follow_target_body: str        # body for hand_follow camera target
    joint_prefix: str = ""         # MJCF strips this when matching dex names


_ROBOT_MJCF: dict[str, RobotMjcfSpec] = {
    "shadow": RobotMjcfSpec(
        mjcf_subdir="shadow_hand",
        mjcf_filename="right_hand.xml",
        target_body="rh_forearm",
        follow_target_body="rh_palm",
        joint_prefix="rh_",
    ),
    # leap / allegro / inspire / svh / ability / panda — TODO when
    # menagerie MJCFs verified and joint name conventions confirmed.
}


@dataclass
class ReplayConfig:
    fps: int = 30
    width: int = 1024
    height: int = 768
    camera: str = "hand_follow"
    codec: str = "libx264"


# =============================================================================
# Scene generation — never write to menagerie submodule
# =============================================================================

_SCENE_TEMPLATE = """<mujoco model="dex_replay_scene">
  <include file="{patched_hand_path}"/>
  <statistic extent="0.8" center="0 0 0.3"/>
  <visual>
    <quality shadowsize="4096"/>
    <global offwidth="{offwidth}" offheight="{offheight}"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
  </asset>
  <worldbody>
    <light pos="0.2 -0.3 0.4" dir="-0.4 0.6 -0.5" directional="true"
           diffuse="0.7 0.7 0.7"/>
    <light pos="-0.2 -0.2 0.5" diffuse="0.4 0.4 0.4"/>
    <!-- No floor and no world-axis triad in this scene: cam_frame is
         placed at the recording camera's actual world pose, and any
         visible debug geometry near the world origin (which can land
         right in front of cam_frame when the recording's ARKit world
         origin happens to coincide with the camera) ruins the "look
         like the recording" contract. Add per-batch contextual props
         (table mesh, object mesh) elsewhere if/when needed. -->
    <body name="world_axes_placeholder" pos="0 0 0"/>
    <!-- cam_frame: pose in world frame from qpos.meta.json (computed at
         retarget time from T_world_cam at the first valid frame). For
         iPhone (handheld but on tripod) and ArUco-anchored 335 with a
         static camera, a single static placement matches the recording
         exactly. fovy = 2*atan(cy/fy) from K. -->
    <camera name="cam_frame" pos="{cam_pos}" {cam_orient}
            mode="fixed" fovy="{cam_frame_fovy_deg}"/>
    <!-- hand_follow: side-rear elevated camera tracking the palm body.
         z=0.5 lifts above origin to give an oblique reference view. -->
    <camera name="hand_follow" pos="0 -0.6 0.5"
            mode="targetbody" target="{follow_target}"/>
  </worldbody>
</mujoco>
"""


def _patch_hand_mjcf(spec: RobotMjcfSpec, menagerie_root: Path) -> Path:
    """Read menagerie hand XML, inject freejoint + abs-path meshdir,
    write to replay/sim/scenes/. Returns path of patched copy. Idempotent."""
    src = menagerie_root / spec.mjcf_subdir / spec.mjcf_filename
    if not src.exists():
        raise FileNotFoundError(f"menagerie MJCF not found: {src}")

    raw = src.read_text(encoding="utf-8")

    # Inject freejoint into target body's opener.
    body_open_substr = f'<body name="{spec.target_body}"'
    if body_open_substr not in raw:
        raise RuntimeError(
            f"could not find '<body name=\"{spec.target_body}\"' in {src}; "
            f"menagerie upstream may have changed — abort to avoid silent "
            f"breakage")
    body_open_start = raw.index(body_open_substr)
    body_open_end = raw.index(">", body_open_start) + 1
    body_opener = raw[body_open_start:body_open_end]
    inject = body_opener + '<freejoint name="hand_base"/>'
    raw = raw.replace(body_opener, inject, 1)

    # Hide Shadow forearm visual meshes. dex_retargeting anchors the
    # floating base at rh_forearm — its 25 cm visual mount sits between
    # camera and wrist in cam_frame view, dwarfing the actual hand action.
    # Setting rgba="0 0 0 0" on the two visual geoms keeps the body +
    # collision (required for the kinematic chain + dex's floating base
    # contract) and just hides the mesh. Per dex_retargeting / MuJoCo
    # Menagerie convention: do NOT remove the body. No-op for non-Shadow
    # robots (leap/allegro lack forearm_N meshes), so safe to apply
    # unconditionally — replace returns the input when the substring
    # isn't found.
    for forearm_geom_orig, forearm_geom_hidden in (
        (
            '<geom class="plastic_visual" mesh="forearm_0" material="gray"/>',
            '<geom class="plastic_visual" mesh="forearm_0" material="gray" rgba="0 0 0 0"/>',
        ),
        (
            '<geom class="plastic_visual" mesh="forearm_1"/>',
            '<geom class="plastic_visual" mesh="forearm_1" rgba="0 0 0 0"/>',
        ),
    ):
        raw = raw.replace(forearm_geom_orig, forearm_geom_hidden, 1)

    # Rewrite meshdir to absolute path so the patched copy resolves
    # meshes from anywhere (we put it under replay/sim/scenes/, not
    # next to the original assets/ folder).
    abs_meshdir = (src.parent / "assets").resolve()
    if 'meshdir="assets"' not in raw:
        raise RuntimeError(
            "expected `meshdir=\"assets\"` in menagerie XML; not found")
    raw = raw.replace(
        'meshdir="assets"',
        f'meshdir="{abs_meshdir.as_posix()}"',
        1,
    )

    _SCENE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = _SCENE_CACHE_DIR / f"_patched_{spec.mjcf_filename}"
    out.write_text(raw, encoding="utf-8")
    return out


def _build_scene(
    robot: str, menagerie_root: Path, cfg: ReplayConfig,
    cam_frame_fovy_deg: float,
    cam_pos_world: list[float] | None = None,
    cam_quat_world_mujoco_xyzw: list[float] | None = None,
) -> Path:
    """Generate per-call scene XML. Returns scene file path.

    cam_frame_fovy_deg: vertical FOV (degrees) for the cam_frame camera.
    Computed from the recording's K matrix per episode so the simulated
    viewpoint matches the real recording's image scale.

    cam_pos_world / cam_quat_world_mujoco_xyzw: pose of the recording
    camera in world frame. Both must be given together (came from
    qpos.meta.json's `cam_pos_world` + `cam_quat_world_mujoco_xyzw`,
    written by retarget). Falls back to the legacy "static at world
    origin looking at +Z" placement when missing — that's what older
    qpos.meta.json from before the dex world-frame refactor look like.
    """
    spec = _ROBOT_MJCF[robot]
    patched_hand = _patch_hand_mjcf(spec, menagerie_root)
    if cam_pos_world is not None and cam_quat_world_mujoco_xyzw is not None:
        cam_pos = " ".join(f"{v:.6f}" for v in cam_pos_world)
        # MuJoCo's <camera> takes either xyaxes (3+3 floats) or quat
        # (w,x,y,z) — quat is cleaner here since retarget already gave us
        # the rotation as xyzw quaternion in MuJoCo cam convention.
        qx, qy, qz, qw = cam_quat_world_mujoco_xyzw
        cam_orient = f'quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}"'
    else:
        cam_pos = "0 0 0"
        cam_orient = 'xyaxes="1 0 0  0 -1 0"'
    scene_xml = _SCENE_TEMPLATE.format(
        patched_hand_path=patched_hand.name,    # relative to scene file
        offwidth=cfg.width,
        offheight=cfg.height,
        follow_target=spec.follow_target_body,
        cam_frame_fovy_deg=f"{cam_frame_fovy_deg:.4f}",
        cam_pos=cam_pos,
        cam_orient=cam_orient,
    )
    scene_path = _SCENE_CACHE_DIR / f"_scene_{robot}.xml"
    scene_path.write_text(scene_xml, encoding="utf-8")
    return scene_path


def _fovy_from_K(K_flat: list[float]) -> float:
    """Vertical field of view (degrees) from a flattened 3x3 intrinsic
    matrix. fovy = 2 * atan(cy / fy).

    Independent of render resolution — fovy is a physical property of
    the lens captured at calibration time. Each recording session may
    have a slightly different K (Gemini 335 firmware reports per-stream),
    so we pull from the saved K rather than hard-coding.
    """
    import math
    K = np.asarray(K_flat, dtype=np.float64).reshape(3, 3)
    fy = float(K[1, 1])
    cy = float(K[1, 2])
    if fy <= 0:
        raise ValueError(f"invalid K[1,1]={fy}; expected positive focal length")
    return float(np.degrees(2.0 * math.atan(cy / fy)))


# =============================================================================
# qpos application — by joint name (Q5 fix)
# =============================================================================

@dataclass
class _PoseLayout:
    """Cached column lookups into dex's qpos vector."""
    dummy_trans_cols: list[int]    # x, y, z translation columns
    dummy_rot_cols: list[int]      # x, y, z Euler rotation columns
    finger_joints: list[tuple[str, int]]  # (mjcf joint name, dex qpos col)


def _build_pose_layout(
    model: mujoco.MjModel, dex_joint_names: list[str], joint_prefix: str,
) -> _PoseLayout:
    """For each MJCF named hinge joint, find its dex qpos column.
    Strip joint_prefix (e.g., 'rh_') on the MJCF side."""
    name_to_dex = {n: i for i, n in enumerate(dex_joint_names)}
    finger_joints: list[tuple[str, int]] = []
    for i in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if model.jnt_type[i] == 0:  # mjJNT_FREE
            continue
        bare = n.removeprefix(joint_prefix)
        if bare not in name_to_dex:
            raise KeyError(
                f"MJCF joint {n!r} (stripped {bare!r}) not in dex output. "
                f"Check joint_prefix or dex retarget config.")
        finger_joints.append((n, name_to_dex[bare]))

    trans_cols = [name_to_dex[f"dummy_{a}_translation_joint"] for a in "xyz"]
    rot_cols = [name_to_dex[f"dummy_{a}_rotation_joint"] for a in "xyz"]
    return _PoseLayout(trans_cols, rot_cols, finger_joints)


def _apply_qpos(
    model: mujoco.MjModel, data: mujoco.MjData,
    q_dex: np.ndarray, layout: _PoseLayout,
    prev_quat_xyzw: np.ndarray | None,
) -> np.ndarray:
    """Write dex qpos into MJCF data.qpos. Returns the freejoint quat
    (xyzw) used (for hemisphere continuity tracking)."""
    # Hinge joints — by name accessor (self-documenting).
    for mjcf_name, dex_col in layout.finger_joints:
        data.joint(mjcf_name).qpos[0] = float(q_dex[dex_col])

    # Freejoint translation: pass through dex output. dex_retargeting
    # uses input frame as the URDF base, so since 02 feeds world-frame
    # joints/wrist these values are already MuJoCo world coords.
    tx, ty, tz = (float(q_dex[c]) for c in layout.dummy_trans_cols)

    # Freejoint rotation. dex outputs intrinsic XYZ Euler in dummy_rot_cols;
    # convert to quat with hemisphere continuity to suppress q vs -q sign
    # flips across adjacent Euler decompositions.
    rx, ry, rz = (float(q_dex[c]) for c in layout.dummy_rot_cols)
    quat_xyzw = Rotation.from_euler("XYZ", [rx, ry, rz]).as_quat()
    if prev_quat_xyzw is not None and float(np.dot(prev_quat_xyzw, quat_xyzw)) < 0:
        quat_xyzw = -quat_xyzw

    # MJCF freejoint qpos slice = [tx, ty, tz, qw, qx, qy, qz] (length 7).
    free_qpos = data.joint("hand_base").qpos
    free_qpos[0] = tx
    free_qpos[1] = ty
    free_qpos[2] = tz
    free_qpos[3] = float(quat_xyzw[3])  # w
    free_qpos[4] = float(quat_xyzw[0])  # x
    free_qpos[5] = float(quat_xyzw[1])  # y
    free_qpos[6] = float(quat_xyzw[2])  # z
    return quat_xyzw


# =============================================================================
# Common loader: model + qpos sequence + layout
# =============================================================================

@dataclass
class _Episode:
    model: mujoco.MjModel
    data: mujoco.MjData
    qpos_seq: np.ndarray            # (T, N) float32
    qpos_valid: np.ndarray          # (T,) bool
    layout: _PoseLayout
    n_total: int
    first_valid_qpos: np.ndarray    # (N,) — used to seed leading invalid
                                     # frames so the rendered timeline
                                     # stays 1:1 with the source mp4.


def _load_episode(
    qpos_npz_path: Path, qpos_meta_path: Path,
    menagerie_root: Path, robot: str, hand: str, cfg: ReplayConfig,
) -> _Episode:
    if hand != "right":
        raise NotImplementedError(f"hand={hand!r} not yet supported")
    if robot not in _ROBOT_MJCF:
        raise NotImplementedError(
            f"robot={robot!r} not in _ROBOT_MJCF; add a spec to register")

    spec = _ROBOT_MJCF[robot]

    # Read meta first to extract K → fovy so the cam_frame camera
    # matches the real recording image scale, plus the cam pose written
    # by retarget so cam_frame sits at the recording camera's actual
    # world location (not the legacy "fixed at world origin" placement).
    meta = json.loads(qpos_meta_path.read_text(encoding="utf-8"))
    K_flat = meta.get("K_flat")
    if K_flat is None:
        # Old meta files (schema v<3) had no K. Fall back to MuJoCo
        # default 45° (likely off, will produce zoomed-in view).
        cam_frame_fovy_deg = 45.0
        print(f"  [warn] qpos meta missing K_flat — cam_frame falls back "
              f"to fovy=45°. Re-run retarget to get accurate fovy.")
    else:
        cam_frame_fovy_deg = _fovy_from_K(K_flat)
        print(f"  cam_frame fovy={cam_frame_fovy_deg:.2f}° from K")

    extras = meta.get("extras") or {}
    cam_pos_world = extras.get("cam_pos_world")
    cam_quat_world = extras.get("cam_quat_world_mujoco_xyzw")
    if cam_pos_world is not None and cam_quat_world is not None:
        anchor_t = extras.get("cam_pose_anchor_frame", "?")
        print(f"  cam_frame world pose from meta (anchor frame={anchor_t}): "
              f"pos={tuple(round(v, 3) for v in cam_pos_world)}")
    else:
        print(f"  [warn] qpos meta missing cam_pos_world — cam_frame falls "
              f"back to legacy world-origin placement. Re-run retarget "
              f"under dex world-frame backend.")

    scene_path = _build_scene(
        robot, menagerie_root, cfg, cam_frame_fovy_deg,
        cam_pos_world=cam_pos_world,
        cam_quat_world_mujoco_xyzw=cam_quat_world,
    )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    layout = _build_pose_layout(model, meta["joint_names"], spec.joint_prefix)

    arr = np.load(qpos_npz_path)
    qpos_seq = arr[f"{hand}_qpos"]
    qpos_valid = arr[f"{hand}_qpos_valid"]
    valid_idx = np.flatnonzero(qpos_valid)
    if len(valid_idx) == 0:
        raise ValueError(
            f"{qpos_npz_path.name}: no valid frames — nothing to replay"
        )
    first_valid_qpos = qpos_seq[int(valid_idx[0])].copy()
    return _Episode(
        model, data, qpos_seq, qpos_valid, layout, len(qpos_seq),
        first_valid_qpos=first_valid_qpos,
    )


# =============================================================================
# Drive helpers (advance one frame, hold-last on invalid)
# =============================================================================

def _drive_one_frame(
    ep: _Episode, t: int,
    last_qpos: np.ndarray | None, prev_quat: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """Apply qpos for frame t. Returns (last_qpos, prev_quat, did_render).

    Hold-fill on every invalid frame including LEADING (frames before
    first-valid get the first-valid pose). This keeps the rendered mp4
    aligned 1:1 with the source recording's frame timeline so AR /
    side-by-side comparisons line up. Caller is responsible for seeding
    last_qpos to ep.first_valid_qpos before t=0.
    """
    if ep.qpos_valid[t]:
        last_qpos = ep.qpos_seq[t]
    prev_quat = _apply_qpos(
        ep.model, ep.data, last_qpos, ep.layout, prev_quat,
    )
    mujoco.mj_forward(ep.model, ep.data)
    return last_qpos, prev_quat, True


# =============================================================================
# Public render functions
# =============================================================================

def replay_to_viewer(
    *,
    qpos_npz_path: Path, qpos_meta_path: Path, menagerie_root: Path,
    robot: str, hand: str, cfg: ReplayConfig | None = None, loop: bool = True,
) -> None:
    """Live MuJoCo viewer. Blocks until user closes the window."""
    cfg = cfg or ReplayConfig()
    ep = _load_episode(qpos_npz_path, qpos_meta_path, menagerie_root,
                       robot, hand, cfg)

    period = 1.0 / cfg.fps
    print(f"  viewer: {ep.n_total} frames @ {cfg.fps} fps "
          f"{'(looping)' if loop else ''}")

    with mujoco.viewer.launch_passive(ep.model, ep.data) as viewer:
        last_qpos = None
        prev_quat = None
        t = 0
        next_step = time.perf_counter()
        while viewer.is_running():
            last_qpos, prev_quat, did_render = _drive_one_frame(
                ep, t, last_qpos, prev_quat,
            )
            if did_render:
                viewer.sync()
            t += 1
            if t >= ep.n_total:
                if not loop:
                    break
                t = 0
                last_qpos = None
                prev_quat = None
            # Pace at fps
            next_step += period
            sleep = next_step - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_step = time.perf_counter()


def replay_to_mp4(
    *,
    qpos_npz_path: Path, qpos_meta_path: Path, menagerie_root: Path,
    robot: str, hand: str, output_mp4_path: Path,
    cfg: ReplayConfig | None = None,
) -> dict:
    """Offscreen render → mp4. Returns stats dict.

    Holds last valid qpos across NaN frames so the playback timeline
    lines up with the source recording's frame count.
    """
    cfg = cfg or ReplayConfig()
    ep = _load_episode(qpos_npz_path, qpos_meta_path, menagerie_root,
                       robot, hand, cfg)

    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_mp4_path), fps=cfg.fps, codec=cfg.codec, quality=8,
        macro_block_size=1,
    )
    # Seed with first-valid pose so leading invalid frames render the
    # action's starting pose (held statically) rather than being skipped
    # or showing the URDF home pose. Keeps mp4 timeline 1:1 with source.
    last_qpos = ep.first_valid_qpos
    prev_quat = None
    n_rendered = 0
    n_held = 0

    with mujoco.Renderer(ep.model, height=cfg.height, width=cfg.width) as r:
        for t in range(ep.n_total):
            was_valid = bool(ep.qpos_valid[t])
            last_qpos, prev_quat, _did_render = _drive_one_frame(
                ep, t, last_qpos, prev_quat,
            )
            if not was_valid:
                n_held += 1
            r.update_scene(ep.data, camera=cfg.camera)
            writer.append_data(r.render())
            n_rendered += 1

    writer.close()
    return {
        "n_total": ep.n_total,
        "n_rendered": n_rendered,
        "n_held": n_held,
        "n_skipped_leading_invalid": 0,  # always 0 since hold-fill is on
        "output": str(output_mp4_path),
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
    source_npz: Path | None = None,
    menagerie_root: Path | None = None,
    fps: int = 30,
    width: int = 1024,
    height: int = 768,
    camera: str = "hand_follow",
    loop: bool = True,
    jpeg_quality: int = 85,
    image_plane_dist_m: float = 2.0,
    # Accepted for CLI parity with so101's --cam-zoom; dex backend does not
    # currently apply it (rerun cam pose comes from meta as-is). Drop the
    # no-op when dex grows the same dolly behavior.
    cam_zoom: float = 1.0,
) -> dict | None:
    """Unified entrypoint dispatched by replay/__main__.py.

    Pulls (robot, hand) from the meta JSON so callers don't double-specify.

    Output modes:
      viewer  — interactive MuJoCo viewer (live window)
      mp4     — offscreen MuJoCo render to file (for sharing / replay)
      rerun   — spawn rerun.io viewer with AR overlay (hand on source RGB)
      rrd     — write rerun .rrd file (open later via `rerun <path>`)
    """
    meta = json.loads(qpos_meta_path.read_text(encoding="utf-8"))
    robot = meta["robot"]
    hand = meta["hand"]

    cfg = ReplayConfig(
        fps=fps, width=width, height=height, camera=camera,
    )
    menagerie = menagerie_root or _DEFAULT_MENAGERIE_ROOT

    if output == "viewer":
        replay_to_viewer(
            qpos_npz_path=qpos_npz_path,
            qpos_meta_path=qpos_meta_path,
            menagerie_root=menagerie,
            robot=robot, hand=hand, cfg=cfg, loop=loop,
        )
        return None

    if output == "mp4":
        if out_mp4 is None:
            raise ValueError("output='mp4' requires out_mp4 path")
        return replay_to_mp4(
            qpos_npz_path=qpos_npz_path,
            qpos_meta_path=qpos_meta_path,
            menagerie_root=menagerie,
            robot=robot, hand=hand,
            output_mp4_path=out_mp4, cfg=cfg,
        )

    if output in ("rerun", "rrd"):
        if source_mp4 is None or source_npz is None:
            raise ValueError(
                "output='rerun'/'rrd' requires source_mp4 and source_npz "
                "(stage-1 preview mp4 + stage-2 .processed.npz). The CLI "
                "in replay/__main__.py derives both from convention."
            )
        # Rebuild scene to capture current cam_pos / cam_quat from meta
        # (same _build_scene call mujoco_dex viewer/mp4 paths use). The
        # rerun renderer wants the patched-and-filled scene xml so its
        # MuJoCo FK matches what the baked mp4 mode produces.
        K_flat = meta.get("K_flat")
        cam_frame_fovy_deg = (
            _fovy_from_K(K_flat) if K_flat is not None else 45.0
        )
        extras = meta.get("extras") or {}
        scene_path = _build_scene(
            robot, menagerie, cfg, cam_frame_fovy_deg,
            cam_pos_world=extras.get("cam_pos_world"),
            cam_quat_world_mujoco_xyzw=extras.get("cam_quat_world_mujoco_xyzw"),
        )
        from replay.sim.rerun_dex import replay_to_rerun
        return replay_to_rerun(
            qpos_npz_path=qpos_npz_path,
            qpos_meta_path=qpos_meta_path,
            source_mp4_path=source_mp4,
            source_npz_path=source_npz,
            scene_xml_path=scene_path,
            out_rrd_path=out_rrd if output == "rrd" else None,
            jpeg_quality=jpeg_quality,
            image_plane_dist_m=image_plane_dist_m,
        )

    raise ValueError(
        f"unknown output mode {output!r}; "
        f"use 'viewer' / 'mp4' / 'rerun' / 'rrd'"
    )
