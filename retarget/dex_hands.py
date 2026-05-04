"""
dex_retargeting offline-position backend (shadow / leap / allegro / ...).

Pipeline position: implements the `RetargetBackend` protocol for any
robot supported by dex_retargeting. Activated by the `(robot, mujoco)`
keys registered in retarget/__init__.py.

Env requirement: `opc-dex` (numpy>=2 + dex_retargeting + pinocchio + torch
CPU). pyorbbecsdk2's numpy<2 ABI is incompatible — the orchestrator must
not run dex retarget in the recording env.

Algorithm (verbatim from dex_retargeting/example/position_retargeting/
hand_robot_viewer.py):
  1. Build retargeter once per (robot, hand) pair.
  2. At first valid frame: warm_start(wrist_pos, wrist_quat, mano=True).
     ONCE per sequence — re-warming each frame defeats NLopt's
     last_qpos continuity (we tried; the optimizer pops between local
     minima between adjacent frames).
  3. Per valid frame: qpos = retargeter.retarget(joint[indices])

Frame: WORLD (gravity-aligned). 02 lifts cam-frame HaMeR to world via
T_world_cam, and we feed `*_world` arrays here. dummy-6 absorbs the
global pose in URDF base frame, which equals world frame because
dex_retargeting uses input frame as the URDF base. This gives 06 a
well-defined "where is the recording camera in MuJoCo world" for placing
cam_frame — derived per-episode from `T_world_cam` (handheld iPhone +
ArUco-anchored 335 both produce one). Smoke verified 50/50 frames
converge identically on world-frame input as on the legacy cam-frame
input (offset ≈ 12 cm = Shadow forearm length, expected).

OpenMP duplicate-DLL fix (Windows): set KMP_DUPLICATE_LIB_OK=TRUE before
torch / dex_retargeting imports.
"""

from __future__ import annotations

import os

# Must precede torch / dex_retargeting imports (Windows OpenMP bug).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig

from retarget import RetargetResult
from retarget.loader import ProcessedSource


# Public mapping — CLI uses this to validate --robot.
ROBOT_NAMES: dict[str, RobotName] = {
    "shadow":  RobotName.shadow,
    "leap":    RobotName.leap,
    "allegro": RobotName.allegro,
    "inspire": RobotName.inspire,
    "svh":     RobotName.svh,
    "ability": RobotName.ability,
    "panda":   RobotName.panda,
}

HAND_TYPES: dict[str, HandType] = {
    "left":  HandType.left,
    "right": HandType.right,
}


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_URDF_DIR = _PROJECT_ROOT / "assets" / "dex-urdf" / "robots" / "hands"


class DexBackend:
    """Stateful wrapper around `dex_retargeting.SeqRetargeting`.

    Build once per (robot, hand) pair and reuse across episodes. Inside
    an episode, warm_start fires ONCE at the first valid frame; per-frame
    retargeting then rides NLopt's last_qpos continuity.
    """

    name = "dex"

    def __init__(
        self,
        robot: str,
        hand: str,
        urdf_dir: Path | None = None,
    ):
        if robot not in ROBOT_NAMES:
            raise ValueError(
                f"Unknown robot {robot!r}. Choose from {sorted(ROBOT_NAMES)}"
            )
        if hand not in HAND_TYPES:
            raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")

        self.robot = robot
        self.hand = hand
        self.urdf_dir = Path(urdf_dir or _DEFAULT_URDF_DIR).resolve()
        if not self.urdf_dir.exists():
            raise FileNotFoundError(f"urdf_dir not found: {self.urdf_dir}")

        cfg_path = get_default_config_path(
            ROBOT_NAMES[robot], RetargetingType.position, HAND_TYPES[hand],
        )
        RetargetingConfig.set_default_urdf_dir(str(self.urdf_dir))
        self._retargeting = RetargetingConfig.load_from_file(cfg_path).build()
        self._human_indices = np.asarray(
            self._retargeting.optimizer.target_link_human_indices, dtype=int,
        )
        self._joint_names = list(self._retargeting.joint_names)

    # ----------- protocol surface -----------

    @classmethod
    def required_keys(cls, hand: str) -> set[str]:
        """Keys that must be in the source npz for this backend to run."""
        return {
            f"{hand}_joints_world",
            f"{hand}_wrist_world",
            f"{hand}_wrist_quat_world",
            f"{hand}_confidence",
            f"{hand}_quality_passed",
            f"{hand}_trim_first",
            f"{hand}_trim_last",
            "T_world_cam",
        }

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def n_joints(self) -> int:
        return len(self._joint_names)

    @property
    def target_human_indices(self) -> np.ndarray:
        return self._human_indices.copy()

    def retarget_episode(
        self,
        source: ProcessedSource,
        hand: str,
        *,
        min_confidence: float = 0.0,
    ) -> RetargetResult:
        """Run dex_retargeting on an episode's trim range.

        Returns qpos/qpos_valid trimmed (length T_trim = trim_last -
        trim_first). The CLI is responsible for padding back to full
        episode length so timestamps_us indexing stays aligned.

        ~1-5ms per frame (NLopt single-thread); 1k-frame episode in ~3s.
        """
        if hand != self.hand:
            raise ValueError(
                f"Backend built for hand={self.hand!r} but called with hand={hand!r}. "
                f"Build a fresh DexBackend per (robot, hand) pair."
            )
        first, last = source.trim_range(hand)
        T_trim = last - first

        mano_joints = source.get(f"{hand}_joints_world")[first:last].astype(np.float64)
        wrist_pos = source.get(f"{hand}_wrist_world")[first:last].astype(np.float64)
        wrist_quat_xyzw = source.get(
            f"{hand}_wrist_quat_world"
        )[first:last].astype(np.float64)
        confidence = source.get(f"{hand}_confidence")[first:last].astype(np.float64)

        # Validity: confidence threshold + finite checks. The optimizer
        # silently produces nonsense on NaN input (writes prev qpos),
        # so we explicitly skip and mark qpos_valid=False.
        valid_mask = (
            (confidence >= min_confidence)
            & np.isfinite(mano_joints).all(axis=(1, 2))
            & np.isfinite(wrist_pos).all(axis=1)
            & np.isfinite(wrist_quat_xyzw).all(axis=1)
        )

        n = self.n_joints
        qpos = np.full((T_trim, n), np.nan, dtype=np.float32)
        qpos_valid = np.zeros(T_trim, dtype=bool)

        valid_idx = np.flatnonzero(valid_mask)
        if len(valid_idx) == 0:
            return RetargetResult(
                qpos=qpos, qpos_valid=qpos_valid,
                joint_names=self._joint_names,
                extras={
                    "target_human_indices": self._human_indices.tolist(),
                    "n_frames_valid_input": 0,
                },
            )

        # Warm-start ONCE with the first valid frame's MANO global_orient.
        # `is_mano_convention=True` tells dex the quat follows MANO axes;
        # dex internally applies OPERATOR2MANO to align with the URDF
        # wrist link's frame.
        first_valid = int(valid_idx[0])
        self._retargeting.reset()
        wrist_quat_wxyz = _xyzw_to_wxyz(wrist_quat_xyzw[first_valid])
        self._retargeting.warm_start(
            wrist_pos=wrist_pos[first_valid],
            wrist_quat=wrist_quat_wxyz,
            hand_type=HAND_TYPES[self.hand],
            is_mano_convention=True,
        )

        # Per-frame retarget. Joints in cam frame; dummy-6 absorbs
        # global pose. NLopt's last_qpos gives temporal smoothness.
        for t in range(T_trim):
            if not bool(valid_mask[t]):
                continue
            ref = mano_joints[t][self._human_indices]
            try:
                q = self._retargeting.retarget(ref_value=ref)
            except Exception as exc:
                print(f"  [warn] retarget failed at frame {first + t}: {exc}")
                continue
            qpos[t] = q.astype(np.float32, copy=False)
            qpos_valid[t] = True

        # Cam-pose-in-world for replay's cam_frame view. Picked at the first
        # frame where T_world_cam is valid within the trim window so 06's
        # static camera matches the recording at the start of the action.
        # Handheld iPhone moves negligibly (mm) over a single recording, so
        # using a single frame is sufficient. ArUco-anchored 335 also stable.
        cam_pose_extras = self._cam_pose_world_extras(source, first, last)

        return RetargetResult(
            qpos=qpos,
            qpos_valid=qpos_valid,
            joint_names=self._joint_names,
            extras={
                "target_human_indices": self._human_indices.tolist(),
                "n_frames_valid_input": int(valid_mask.sum()),
                "min_confidence": float(min_confidence),
                "input_frame": "world",
                **cam_pose_extras,
            },
        )

    def _cam_pose_world_extras(
        self, source: ProcessedSource, first: int, last: int,
    ) -> dict:
        """Compute the recording camera's pose in MuJoCo world for 06.

        Returns dict with `cam_pos_world` + `cam_quat_world_mujoco_xyzw`
        (or empty if T_world_cam unavailable). MuJoCo cameras use the
        OpenGL/MuJoCo convention (X right, Y up, Z out of screen) while
        T_world_cam stores OpenCV (X right, Y down, Z forward), so we
        right-multiply by diag(1, -1, -1) to flip Y and Z. Same trick as
        retarget/so101.py L396 — keep both backends visually consistent.
        """
        from scipy.spatial.transform import Rotation
        T_world_cam = source.T_world_cam
        if T_world_cam is None:
            return {"cam_pose_world_available": False}
        # Pick first valid T_world_cam in trim window. iPhone has no
        # validity mask (all valid). 335 ArUco emits T_world_cam_valid
        # so first valid != [0] when ArUco was lost at start.
        valid_mask = (
            source.get("T_world_cam_valid")
            if source.has("T_world_cam_valid") else None
        )
        anchor_t = first
        if valid_mask is not None:
            valid_in_trim = np.flatnonzero(valid_mask[first:last].astype(bool))
            if len(valid_in_trim) == 0:
                return {"cam_pose_world_available": False}
            anchor_t = first + int(valid_in_trim[0])
        T = T_world_cam[anchor_t]
        R_world_cam_opencv = T[:3, :3]
        t_world_cam = T[:3, 3]
        # OpenCV cam frame → MuJoCo cam frame: flip Y (down→up) and Z
        # (into-scene → out-of-screen). Translation unchanged.
        R_world_cam_mujoco = R_world_cam_opencv @ np.diag([1.0, -1.0, -1.0])
        cam_quat_xyzw = Rotation.from_matrix(R_world_cam_mujoco).as_quat()
        return {
            "cam_pose_world_available": True,
            "cam_pose_anchor_frame": int(anchor_t),
            "cam_pos_world": [float(v) for v in t_world_cam],
            "cam_quat_world_mujoco_xyzw": [float(v) for v in cam_quat_xyzw],
        }


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """scipy xyzw → pytransform3d wxyz (scalar-first)."""
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
