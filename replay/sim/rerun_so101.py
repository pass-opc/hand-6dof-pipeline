"""
rerun.io AR-overlay renderer for SO-Arm 101 (5-DoF arm + gripper).

Mirror of replay/sim/rerun_dex.py but using mujoco_so101's scene + qpos
application. Logged in the SO-101 arm-base frame (= MuJoCo world for
SO-101). Camera pose comes from retarget meta's `cam_pos_arm` and
`cam_quat_arm_xyzw` (computed in retarget/so101.py L390-396 to place the
recording camera relative to the first-valid wrist anchor).

Conventions:
  * SO-101 hand pose: arm-base frame (no floating base). MuJoCo world.
  * Cam pose stored in meta: MuJoCo cam convention (X right, Y up, Z back
    out of screen) — that's `_R_ARM_CAM_OPENCV @ diag(1, -1, -1)` from
    so101.py. We tell rerun this is `RUB` so the projection works without
    re-flipping. (dex stores OpenCV cam pose and uses RDF — different
    convention but same idea.)
  * Background source RGB: same stage as the qpos meta's `source_npz`,
    so an optimize-fed retarget gets 03_optimized/<sid>/<sid>_preview.mp4
    (overlay drawn from optimized keypoints) and a raw-fed retarget gets
    01_tracking/<sid>/<sid>_preview.mp4. Pairing logic lives in
    replay/__main__.py — backend just uses whatever path the CLI passes.
  * 02 npz `T_world_cam` is NOT used here — SO-101's world ≠ ARKit world.
"""
from __future__ import annotations

import json
from pathlib import Path

import av
import cv2
import mujoco
import numpy as np

from replay.sim.mujoco_loader import apply_joint_positions_deg, load_so_arm101
from replay.sim.mujoco_mesh import iter_visible_mesh_geoms
from replay.sim.mujoco_so101 import (
    ARM_JOINT_NAMES,
    _apply_cam_zoom,
    _build_cam_scene,
    _fovy_from_K,
    _qpos_to_arm_gripper_deg,
)


def replay_to_rerun(
    *,
    qpos_npz_path: Path,
    qpos_meta_path: Path,
    source_mp4_path: Path,
    out_rrd_path: Path | None = None,
    width: int = 1024,
    height: int = 768,
    jpeg_quality: int = 85,
    image_plane_dist_m: float = 2.0,
    max_frames: int | None = None,
    cam_zoom: float = 1.0,
) -> dict:
    """Log one SO-101 episode to rerun. Returns stats.

    `cam_zoom`: visual-only multiplier applied at log time. At 1.0 (default)
    the rerun camera pose equals the retarget meta's `cam_pos_arm`. At
    < 1.0 the camera is translated AWAY from the scene centroid along its
    viewing axis by `(1/cam_zoom - 1) * baseline_depth`, so the same
    intrinsic K projects the robot smaller (covers fewer pixels of the AR
    canvas). Source mp4 + qpos + meta are NOT modified — this only
    changes what rerun draws, never what was retargeted.
    """
    import rerun as rr

    if not qpos_npz_path.exists():
        raise FileNotFoundError(qpos_npz_path)
    if not source_mp4_path.exists():
        raise FileNotFoundError(source_mp4_path)
    if cam_zoom <= 0:
        raise ValueError(f"cam_zoom must be > 0, got {cam_zoom}")

    meta = json.loads(qpos_meta_path.read_text(encoding="utf-8"))
    hand = meta["hand"]
    extras = meta.get("extras") or {}
    cam_pos = extras.get("cam_pos_arm")
    cam_quat_xyzw = extras.get("cam_quat_arm_xyzw")
    centroid = extras.get("target_pos_arm_centroid")
    K_flat = meta.get("K_flat")
    if not (cam_pos and cam_quat_xyzw and K_flat):
        raise KeyError(
            f"{qpos_npz_path.name}: meta missing cam_pos_arm / "
            f"cam_quat_arm_xyzw / K_flat. Re-run retarget."
        )
    K = np.asarray(K_flat, dtype=np.float64).reshape(3, 3)
    fovy = _fovy_from_K(K_flat)

    # Visual-only dolly back along view axis. Shared helper with mp4 path
    # so a single CLI flag controls both modes consistently.
    cam_pos_eff, zoom_info = _apply_cam_zoom(cam_pos, centroid, cam_zoom)
    if zoom_info:
        print(
            f"  cam_zoom={cam_zoom:.3f}  baseline_depth="
            f"{zoom_info['baseline_depth_m']:.3f}m  "
            f"back_off={zoom_info['back_off_m']:+.3f}m"
        )
    elif cam_zoom != 1.0 and centroid is None:
        print(
            f"  WARN: --cam-zoom={cam_zoom} requested but meta has no "
            f"target_pos_arm_centroid; falling back to zoom=1.0. "
            f"Re-run retarget to refresh meta."
        )

    # Build scene with cam_frame at arm-base pose, identical to mp4 mode.
    scene_xml = _build_cam_scene(
        cam_pos, cam_quat_xyzw, fovy, width=width, height=height,
    )
    scene = load_so_arm101(scene_xml)
    model, data = scene.model, scene.data

    # Per-episode arm + gripper joint trajectories (degrees, hold-last
    # filled). Mirrors what mp4 / viewer modes consume.
    arr = np.load(qpos_npz_path)
    qpos = arr[f"{hand}_qpos"]
    qpos_valid = arr[f"{hand}_qpos_valid"].astype(bool)
    arm_deg, gripper_deg = _qpos_to_arm_gripper_deg(
        qpos, qpos_valid, meta["joint_names"],
    )

    container = av.open(str(source_mp4_path))
    stream = container.streams.video[0]
    src_w, src_h = stream.width, stream.height
    src_fps = float(stream.average_rate or 30)

    visible_geoms = list(iter_visible_mesh_geoms(model))

    sid = qpos_npz_path.stem.replace(".qpos", "")
    spawn = out_rrd_path is None
    rr.init(f"opc_ar_overlay_so101_{sid}", spawn=spawn)
    if not spawn:
        out_rrd_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(out_rrd_path))

    # Static logs: meshes + camera.
    # vertex_normals come from the MJCF (model.mesh_normal) so rerun does
    # smooth shading instead of per-face faceted; albedo_factor uses the
    # MATERIAL rgba (yellow plastic / black servo), not the uniform 50%
    # grey fallback that geom_rgba carries on this MJCF.
    geom_paths: dict[int, str] = {}
    for g in visible_geoms:
        path = f"world/so101/{g.body_name}/g{g.geom_id}"
        geom_paths[g.geom_id] = path
        rr.log(
            path,
            rr.Mesh3D(
                vertex_positions=g.vertices,
                triangle_indices=g.triangles,
                vertex_normals=g.normals,
                albedo_factor=[float(c) for c in g.rgba],
            ),
            static=True,
        )

    # Cam pose: stored in MuJoCo cam convention (X right, Y up, Z back).
    # Tell rerun via RUB so its 3D-in-2D projection uses the right ray
    # math. cam_pos / cam_quat_xyzw stay constant (cam is "fixed" in arm
    # base frame for SO-101 — so101.py:374 anchors it once at first valid
    # wrist), so log statically.
    qx, qy, qz, qw = cam_quat_xyzw
    rr.log(
        "world/cam",
        rr.Pinhole(
            image_from_camera=K,
            resolution=(src_w, src_h),
            camera_xyz=rr.ViewCoordinates.RUB,
            image_plane_distance=float(image_plane_dist_m),
        ),
        static=True,
    )
    rr.log(
        "world/cam",
        rr.Transform3D(
            translation=cam_pos_eff,
            quaternion=[float(qx), float(qy), float(qz), float(qw)],
        ),
        static=True,
    )

    n_logged = 0
    for frame_i, av_frame in enumerate(container.decode(video=0)):
        if max_frames is not None and n_logged >= max_frames:
            break
        if frame_i >= len(arm_deg):
            break
        rr.set_time("frame", sequence=frame_i)
        rr.set_time("time", duration=frame_i / src_fps)

        # Source RGB (JPEG-compressed).
        rgb = av_frame.to_ndarray(format="rgb24")
        ok, jpg = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            raise RuntimeError(f"jpeg encode failed at frame {frame_i}")
        rr.log(
            "world/cam/image",
            rr.EncodedImage(contents=jpg.tobytes(), media_type="image/jpeg"),
        )

        # Apply joint angles + FK + log per-geom transforms. Hold-last
        # already done in _qpos_to_arm_gripper_deg, so apply on every
        # frame (no NaNs by this point).
        cmd = {n: float(arm_deg[frame_i, i]) for i, n in enumerate(ARM_JOINT_NAMES)}
        cmd["gripper"] = float(gripper_deg[frame_i])
        apply_joint_positions_deg(scene, cmd, target="qpos")
        mujoco.mj_kinematics(model, data)
        for g in visible_geoms:
            R_w = data.geom_xmat[g.geom_id].reshape(3, 3)
            t_w = data.geom_xpos[g.geom_id]
            rr.log(
                geom_paths[g.geom_id],
                rr.Transform3D(translation=t_w, mat3x3=R_w),
            )
        n_logged += 1

    container.close()
    return {
        "n_logged": int(n_logged),
        "n_total": int(stream.frames),
        "mode": "viewer" if spawn else "rrd",
        "sink": "rerun_viewer" if spawn else str(out_rrd_path),
        "image_plane_distance_m": float(image_plane_dist_m),
    }
