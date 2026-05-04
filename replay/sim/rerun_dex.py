"""
rerun.io AR-overlay renderer for dex hands.

Pipeline position: alternative output mode for stage 6, alongside the
MuJoCo viewer / mp4 modes in mujoco_dex.py. Pinhole archetype + URDF mesh
geoms + per-frame Transform3D → rerun automatically composites the
retargeted hand on top of the recording's RGB stream (3D-in-2D
projection — same archetype the official DROID rerun viewer uses).

Why a separate module rather than inlining into mujoco_dex.py:
  * rerun is an optional dep — keep the import behind a function-level
    guard so MuJoCo-only users don't pay for it
  * the FK + mesh extraction logic is identical to mujoco_dex.py and
    reuses MuJoCo as the kinematics engine; we don't reinvent

Why MuJoCo for FK + rerun for rendering (rather than rerun's URDF loader):
  the dex_retargeting Shadow URDF is patched with a freejoint at
  rh_forearm and the visual forearm is hidden via rgba=0; rerun's URDF
  loader doesn't honor those mods. Using MuJoCo's MjModel + mj_forward
  gives us the exact same geom positions / visibility as the MuJoCo
  viewer / mp4 modes, so the rerun view stays consistent with the
  baked mp4 output.

Inputs:
  qpos_npz_path       output of `python -m retarget` (per-frame qpos)
  qpos_meta_path      sidecar JSON with K_flat, joint_names, source_npz
  source_mp4_path     iPhone/335 01-stage preview mp4 (background plate)
  source_npz_path     02-stage .processed.npz (per-frame T_world_cam)
  out_rrd_path        if given, write .rrd; else spawn the rerun viewer
  jpeg_quality        85 = visually lossless QA, 60 = small file
  image_plane_dist_m  depth at which the source image renders in 3D view;
                      default 2.0 m so it's behind the hand (~0.4 m away)
                      and doesn't occlude in the 3D Spatial view. The 2D
                      Spatial view (the actual AR composite) is unaffected.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from replay.sim.mujoco_mesh import iter_visible_mesh_geoms


@dataclass
class _PoseLayout:
    finger: list[tuple[str, int]]
    trans_cols: list[int]
    rot_cols: list[int]


def _build_pose_layout(model, dex_joint_names: list[str]) -> _PoseLayout:
    """dex qpos column ↔ mjcf joint name. Mirrors mujoco_dex's lookup."""
    name_to_dex = {n: i for i, n in enumerate(dex_joint_names)}
    finger = []
    for ji in range(model.njnt):
        if model.jnt_type[ji] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        mj_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, ji)
        # Shadow has rh_ prefix; dex internal names don't.
        dex_name = mj_name[3:] if mj_name.startswith("rh_") else mj_name
        if dex_name in name_to_dex:
            finger.append((mj_name, name_to_dex[dex_name]))
    return _PoseLayout(
        finger=finger,
        trans_cols=[name_to_dex[f"dummy_{a}_translation_joint"] for a in "xyz"],
        rot_cols=[name_to_dex[f"dummy_{a}_rotation_joint"] for a in "xyz"],
    )


def _apply_qpos(model, data, q_dex: np.ndarray, layout: _PoseLayout) -> None:
    """dex 30-vector → MuJoCo qpos (freejoint + hinges). Same logic as
    mujoco_dex._apply_qpos — kept inline to avoid pulling that module's
    rendering deps. Hemisphere continuity is unnecessary here because
    rerun re-derives orientation from the matrix each frame anyway."""
    for mjcf_name, dex_col in layout.finger:
        data.joint(mjcf_name).qpos[0] = float(q_dex[dex_col])
    tx, ty, tz = (float(q_dex[c]) for c in layout.trans_cols)
    rx, ry, rz = (float(q_dex[c]) for c in layout.rot_cols)
    qx, qy, qz, qw = Rotation.from_euler("XYZ", [rx, ry, rz]).as_quat()
    fq = data.joint("hand_base").qpos
    fq[0], fq[1], fq[2] = tx, ty, tz
    fq[3], fq[4], fq[5], fq[6] = qw, qx, qy, qz


def replay_to_rerun(
    *,
    qpos_npz_path: Path,
    qpos_meta_path: Path,
    source_mp4_path: Path,
    source_npz_path: Path,
    scene_xml_path: Path,
    out_rrd_path: Path | None = None,
    jpeg_quality: int = 85,
    image_plane_dist_m: float = 2.0,
    max_frames: int | None = None,
) -> dict:
    """Log one episode to rerun (viewer or .rrd file).

    Returns stats dict (n_logged, output mode, sink path).
    """
    import rerun as rr

    if not qpos_npz_path.exists():
        raise FileNotFoundError(qpos_npz_path)
    if not source_mp4_path.exists():
        raise FileNotFoundError(source_mp4_path)
    if not source_npz_path.exists():
        raise FileNotFoundError(source_npz_path)

    arr = np.load(qpos_npz_path)
    meta = json.loads(qpos_meta_path.read_text(encoding="utf-8"))
    hand = meta["hand"]
    qpos_seq = arr[f"{hand}_qpos"]
    qpos_valid = arr[f"{hand}_qpos_valid"]

    K = np.asarray(meta["K_flat"], dtype=np.float64).reshape(3, 3)
    proc = np.load(source_npz_path)
    T_world_cam_seq = proc["T_world_cam"]

    container = av.open(str(source_mp4_path))
    stream = container.streams.video[0]
    src_w, src_h = stream.width, stream.height
    src_fps = float(stream.average_rate or 30)

    model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
    data = mujoco.MjData(model)
    layout = _build_pose_layout(model, meta["joint_names"])
    visible_geoms = list(iter_visible_mesh_geoms(model))

    sid = qpos_npz_path.stem.replace(".qpos", "")
    spawn = out_rrd_path is None
    rr.init(f"opc_ar_overlay_{sid}", spawn=spawn)
    if not spawn:
        out_rrd_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(out_rrd_path))

    # Static logs — meshes + camera intrinsics. Per-frame Transform3D
    # below repositions them without re-uploading geometry. Mesh details
    # (vertex_normals + material-aware rgba) come from the shared
    # mujoco_mesh helper so dex and so101 stay visually consistent.
    geom_paths: dict[int, str] = {}
    for g in visible_geoms:
        path = f"world/shadow/{g.body_name}/g{g.geom_id}"
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
    rr.log(
        "world/cam",
        rr.Pinhole(
            image_from_camera=K,
            resolution=(src_w, src_h),
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=float(image_plane_dist_m),
        ),
        static=True,
    )

    n_logged = 0
    for frame_i, av_frame in enumerate(container.decode(video=0)):
        if max_frames is not None and n_logged >= max_frames:
            break
        if frame_i >= len(qpos_seq):
            break
        rr.set_time("frame", sequence=frame_i)
        rr.set_time("time", duration=frame_i / src_fps)

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

        T = T_world_cam_seq[frame_i]
        rr.log(
            "world/cam",
            rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]),
        )

        if bool(qpos_valid[frame_i]):
            _apply_qpos(model, data, qpos_seq[frame_i], layout)
            mujoco.mj_forward(model, data)
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
