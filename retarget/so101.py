"""
SO-Arm 101 retarget backend — port of lerobot-seeed's RobotKinematics pattern.

Pipeline position: stage 5 backend for `(so101, mujoco|real)`. Consumes
.processed.npz from stage 2 (cam-frame HaMeR keypoints + 6DoF wrist),
emits .qpos.npz + .qpos.meta.json. Registered as 'so101' in
retarget/__init__.py; same backend covers sim + real arms.

Algorithm (mirrors lerobot-seeed `src/lerobot/model/kinematics.py`,
which uses placo + frame_task + soft constraints):

  1. target_pos_arm = R_arm_cam @ (joints_cam[4] + joints_cam[8]) / 2
       — pinch midpoint of human thumb-tip + index-tip, in arm frame.
       Maps to gripper TCP (where the jaw closes around an object), the
       direct analogue of lerobot-seeed phone teleop's leader-EE-pose →
       follower-EE-pose mapping.
  2. target_R_arm = R_arm_cam @ R_wrist_cam   — wrist 6DoF orientation.
  3. mink FrameTask("gripperframe", "site", position_cost=1.0,
       orientation_cost=0.05) — weighted soft constraint, position
       dominant. lerobot-seeed default is 1.0/0.01 (teleop = small
       per-step corrections); offline retargeting bumps the orient
       weight so wrist tilt actually tracks.
  4. solve_ik N iters → integrate → extract joint angles.
  5. Gripper joint: thumb-tip↔index-tip distance → linear map.

Why mink instead of placo:
  placo (Rhoban) is what lerobot-seeed RobotKinematics uses. As of May
  2026 placo still ships no Windows wheels; pip falls through to a
  pinocchio + eigenpy + eiquadprog C++ build chain that doesn't work
  on win-64. mink (kevinzakka, MuJoCo-native) is the closest equivalent
  — same task-space weighted-QP formulation, pip-installable on Windows.

Workspace adapter (engineering choice, NOT in lerobot-seeed):
  Real-time teleop: leader hand IS in robot's workspace (user adjusts).
  Offline HaMeR retargeting: raw human-hand motion (~50 cm reach) >
  SO-101 reach (~35 cm). Pure 1:1 mapping leaves >95% of frames outside
  reach. We therefore center the bbox of valid pinch positions on a
  workspace_anchor and uniformly scale to fit `workspace_arm_reach *
  workspace_fit_factor`. Override `workspace_scale=1.0` to disable.
"""
from __future__ import annotations

import os
# OpenMP duplicate-DLL fix (mujoco + mink + numpy share OpenMP runtimes).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from optimize.filters.quat_one_euro import _ScalarOneEuro
from retarget import RetargetResult
from retarget.loader import ProcessedSource


# Joint output order (matches retarget/so101.py and replay/sim/mujoco_so101.ARM_JOINT_NAMES).
JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# OpenCV cam (+X right, +Y down, +Z forward) → SO-101 base_link
# (+X forward, +Y left, +Z up). Identical to v1's _R_ARM_CAM_OPENCV;
# kept inline so v1 and v2 are independently auditable.
_R_ARM_CAM_OPENCV: np.ndarray = np.array(
    [
        [0.0, 0.0, 1.0],   # cam +Z (forward) → arm +X
        [-1.0, 0.0, 0.0],  # cam +X (right)   → arm -Y
        [0.0, -1.0, 0.0],  # cam +Y (down)    → arm -Z
    ],
    dtype=np.float64,
)

# MANO 21-keypoint indices used for the gripper map.
_MANO_THUMB_TIP_IDX = 4
_MANO_INDEX_TIP_IDX = 8

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MJCF = (
    _PROJECT_ROOT / "assets" / "mujoco" / "trs_so101" / "so101_new_calib.xml"
)


def _gripper_from_mano_joints(
    joints_cam: np.ndarray,              # (T, 21, 3)
    valid_mask: np.ndarray,              # (T,) bool
    *,
    open_distance_m: float,
    close_distance_m: float,
    open_deg: float,
    close_deg: float,
) -> np.ndarray:
    """thumb-tip ↔ index-tip distance → gripper angle (radians).

    Identical to v1; gripper map is pinch geometry, independent of IK
    target choice.
    """
    thumb = joints_cam[:, _MANO_THUMB_TIP_IDX, :]
    index = joints_cam[:, _MANO_INDEX_TIP_IDX, :]
    dist_m = np.linalg.norm(thumb - index, axis=1)
    span = max(open_distance_m - close_distance_m, 1e-6)
    norm = np.clip((dist_m - close_distance_m) / span, 0.0, 1.0)
    deg = open_deg + (1.0 - norm) * (close_deg - open_deg)
    out = np.full(len(joints_cam), np.nan, dtype=np.float64)
    out[valid_mask] = np.radians(deg[valid_mask])
    return out


class So101Backend:
    """Standard-practice SO-101 retarget via mink weighted soft IK.

    One backend per (robot, hand). Builds the MuJoCo model + mink
    Configuration once; reuses across `retarget_episode` calls.
    """

    name = "so101"
    robot = "so101"

    def __init__(
        self,
        robot: str,
        hand: str,
        *,
        mjcf_path: Path | None = None,
        target_site: str = "gripperframe",
        # Position-orientation weight ratio: lerobot-seeed default
        # (placo) is 1.0/0.01. For *teleop* (real-time, small per-step
        # corrections) that ratio is fine. For *offline retargeting*
        # we want orientation to actually track (pour-style tilt is the
        # whole point), so default lift to 0.1 — still position-dominant
        # but orient is weighted enough to reach within 10° on
        # well-conditioned frames. User can override via constructor or
        # CLI.
        # Empirical sweep on pour_coffee_bean ep02 showed:
        #   0.01 (lerobot default) → pos 31 mm / rot 86°  (orient unusable)
        #   0.05                   → pos 37 mm / rot 62°  ← best balance
        #   0.10                   → pos 72 mm / rot 41°
        #   0.20                   → pos 172 mm / rot 13° (pos unusable)
        # SO-101's 5 DoF can't satisfy both pos<30 and rot<30 — pick the
        # knee. Override via constructor for batch-specific tuning.
        position_cost: float = 1.0,
        orientation_cost: float = 0.05,
        # Posture-task cost: weight on "stay close to previous frame's
        # qpos" as a smoothness regularizer. Without this the QP has
        # multiple local minima per frame (5-DoF arm + 6-DoF target is
        # degenerate); adjacent frames flip between minima → 100°+
        # joint-jumps frame-to-frame. cost=0.1 + per-frame target
        # update reduced p95 jitter from 42°→<5° on pour_coffee_bean
        # ep02. Lower = more pos/orient accuracy, more jitter.
        posture_cost: float = 0.1,
        # IK iteration count per frame: solve_ik returns one velocity
        # step per call. For offline static-target retargeting we want
        # full convergence — 30 iterations with dt=0.05 typically lands
        # under 1mm / 1° on reachable frames (mink uses DLS).
        ik_iters: int = 30,
        ik_dt: float = 0.05,
        ik_solver: str = "daqp",
        ik_damping: float = 1e-4,
        # Pinch-midpoint pre-smoothing. HaMeR's per-frame thumb-tip and
        # index-tip detections occasionally jump 100-400 mm/frame from
        # transient detection errors; raw pinch midpoint inherits those
        # jumps. `optimize/configs/default.yaml` only smooths wrist_cam
        # and wrist_quat, NOT joints_cam (which is what the pinch
        # midpoint comes from), so we apply One-Euro per-axis here. Same
        # filter, same defaults as the optimize stage's wrist filter.
        # Setting smooth_pinch=False reverts to raw pinch (compare for
        # debug); cutoff/beta tuneable per batch motion characteristics.
        smooth_pinch: bool = True,
        pinch_min_cutoff_hz: float = 1.0,
        pinch_beta: float = 0.05,
        pinch_d_cutoff_hz: float = 1.0,
        # Workspace adapter (offline HaMeR specific). lerobot-seeed teleop
        # doesn't need this — the phone leader's EE pose is already in
        # the robot's workspace because the user moves the phone at robot
        # scale. HaMeR offline gives raw human-hand motion (~50 cm reach)
        # which dwarfs SO-101's ~35 cm reach. Adapter:
        #   1. compute valid-pinch bbox center in arm frame (= "natural"
        #      action center after R_arm_cam),
        #   2. uniform scale so max_radius fits arm_reach * fit_factor,
        #   3. shift bbox center to workspace_anchor so the arm always
        #      operates in front of and above its base.
        # auto_scale=True derives scale from data; user can pass an
        # explicit `scale` to override (None = auto).
        workspace_anchor: tuple[float, float, float] = (0.20, 0.0, 0.15),
        workspace_arm_reach: float = 0.30,
        workspace_fit_factor: float = 0.85,
        workspace_scale: float | None = None,  # None = auto-fit
        gripper_open_distance_m: float = 0.10,
        gripper_close_distance_m: float = 0.02,
        gripper_open_deg: float = 0.0,
        gripper_close_deg: float = 100.0,
    ):
        # Accept both registry names — this backend serves the SO-101
        # arm regardless of whether the user typed `so101` or `so101_v2`.
        # `self.robot` (class attribute) stays "so101" so meta + replay
        # dispatch (replay/__init__.py uses 'so101' for sim/real backends)
        # find the right replay backend without a parallel replay v2.
        if robot != "so101":
            raise ValueError(f"So101Backend supports robot='so101', got {robot!r}")
        if hand not in ("left", "right"):
            raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")
        self.hand = hand
        self.mjcf_path = Path(mjcf_path or _DEFAULT_MJCF).resolve()
        if not self.mjcf_path.exists():
            raise FileNotFoundError(f"MJCF not found: {self.mjcf_path}")

        self.target_site = target_site
        self.position_cost = float(position_cost)
        self.orientation_cost = float(orientation_cost)
        self.posture_cost = float(posture_cost)
        self.ik_iters = int(ik_iters)
        self.ik_dt = float(ik_dt)
        self.ik_solver = str(ik_solver)
        self.ik_damping = float(ik_damping)

        self.smooth_pinch = bool(smooth_pinch)
        self.pinch_min_cutoff_hz = float(pinch_min_cutoff_hz)
        self.pinch_beta = float(pinch_beta)
        self.pinch_d_cutoff_hz = float(pinch_d_cutoff_hz)

        self.workspace_anchor = np.asarray(workspace_anchor, dtype=np.float64)
        self.workspace_arm_reach = float(workspace_arm_reach)
        self.workspace_fit_factor = float(workspace_fit_factor)
        self.workspace_scale = workspace_scale  # None = auto

        self.gripper_open_distance_m = float(gripper_open_distance_m)
        self.gripper_close_distance_m = float(gripper_close_distance_m)
        self.gripper_open_deg = float(gripper_open_deg)
        self.gripper_close_deg = float(gripper_close_deg)

        # Build MuJoCo model + mink configuration once. Reused across
        # episodes and frames for warm-starting (config.q persists).
        self._model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        # Validate site exists in model — bail loud if MJCF lacks it
        # rather than silently falling back to a wrong target.
        site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, self.target_site
        )
        if site_id < 0:
            raise RuntimeError(
                f"site {self.target_site!r} not in MJCF {self.mjcf_path}. "
                f"Sites: "
                f"{[mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(self._model.nsite)]}"
            )

        # Map MJCF joint name → qpos column. Used after IK to extract
        # arm joints in JOINT_NAMES order.
        name_to_qadr: dict[str, int] = {}
        for ji in range(self._model.njnt):
            jname = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, ji)
            name_to_qadr[jname] = int(self._model.jnt_qposadr[ji])
        missing = [n for n in JOINT_NAMES if n not in name_to_qadr]
        if missing:
            raise RuntimeError(
                f"MJCF {self.mjcf_path.name} missing expected joints: {missing}"
            )
        # qpos column for each named joint, in JOINT_NAMES order.
        self._qadr_for_joint = {n: name_to_qadr[n] for n in JOINT_NAMES}
        self._gripper_qadr = name_to_qadr["gripper"]

    # ----------- protocol surface -----------

    @classmethod
    def required_keys(cls, hand: str) -> set[str]:
        return {
            f"{hand}_wrist_quat_cam",
            f"{hand}_joints_cam",
            f"{hand}_confidence",
            f"{hand}_quality_passed",
            f"{hand}_trim_first",
            f"{hand}_trim_last",
        }

    @property
    def joint_names(self) -> list[str]:
        return list(JOINT_NAMES)

    @property
    def n_joints(self) -> int:
        return 6

    # ----------- main entry -----------

    def retarget_episode(
        self,
        source: ProcessedSource,
        hand: str,
        *,
        min_confidence: float = 0.0,
    ) -> RetargetResult:
        if hand != self.hand:
            raise ValueError(
                f"Backend built for hand={self.hand!r}, called with hand={hand!r}."
            )

        first, last = source.trim_range(hand)
        T = last - first

        wrist_quat_cam = source.get(f"{hand}_wrist_quat_cam")[first:last].astype(np.float64)
        joints_cam = source.get(f"{hand}_joints_cam")[first:last].astype(np.float64)
        confidence = source.get(f"{hand}_confidence")[first:last].astype(np.float64)

        valid_mask = (
            (confidence >= min_confidence)
            & np.isfinite(wrist_quat_cam).all(axis=1)
            & np.isfinite(joints_cam).all(axis=(1, 2))
        )

        qpos_out = np.full((T, 6), np.nan, dtype=np.float32)
        qpos_valid = np.zeros(T, dtype=bool)
        valid_idx = np.flatnonzero(valid_mask)
        if len(valid_idx) == 0:
            return RetargetResult(
                qpos=qpos_out, qpos_valid=qpos_valid,
                joint_names=self.joint_names,
                extras={"n_frames_valid_input": 0, "ik_engine": "mink"},
            )

        # Pinch midpoint = TCP target source. (T,3) in cam frame.
        pinch_cam = (
            joints_cam[:, _MANO_THUMB_TIP_IDX, :]
            + joints_cam[:, _MANO_INDEX_TIP_IDX, :]
        ) / 2.0

        # Per-axis One-Euro on pinch_cam. Suppresses HaMeR transient
        # detection jumps (measured up to 400 mm/frame on
        # pour_coffee_bean ep02) before they propagate into IK target
        # → joint-jitter. dt comes from frame index (1/fps), source
        # timestamps_us would be more accurate but for fixed-rate
        # iPhone Record3D the difference is sub-percent.
        if self.smooth_pinch and len(valid_idx) > 1:
            ts = source.timestamps_us[first:last].astype(np.float64) / 1e6
            filters_xyz = [
                _ScalarOneEuro(
                    self.pinch_min_cutoff_hz,
                    self.pinch_beta,
                    self.pinch_d_cutoff_hz,
                ) for _ in range(3)
            ]
            pinch_cam_smooth = pinch_cam.copy()
            t_prev = None
            for tt in valid_idx:
                t_now = float(ts[tt])
                dt = (t_now - t_prev) if t_prev is not None else 1.0 / 60.0
                if dt <= 0:
                    dt = 1.0 / 60.0
                for ax in range(3):
                    pinch_cam_smooth[tt, ax] = filters_xyz[ax].filter(
                        float(pinch_cam[tt, ax]), dt
                    )
                t_prev = t_now
            pinch_cam = pinch_cam_smooth

        # Cam → arm rotation, then workspace adapter (HaMeR human-hand
        # motion compressed/recentered into SO-101's 30 cm reach).
        pinch_arm_raw = (_R_ARM_CAM_OPENCV @ pinch_cam.T).T
        valid_pinch = pinch_arm_raw[valid_mask]
        bbox_min = valid_pinch.min(axis=0)
        bbox_max = valid_pinch.max(axis=0)
        bbox_center = 0.5 * (bbox_min + bbox_max)
        # Half-diag of bbox = max single-axis reach from center.
        half_diag = np.linalg.norm(bbox_max - bbox_center)
        if self.workspace_scale is None:
            allowed = self.workspace_arm_reach * self.workspace_fit_factor
            scale_auto = (
                min(1.0, allowed / max(half_diag, 1e-6))
                if half_diag > allowed else 1.0
            )
            scale = float(scale_auto)
        else:
            scale = float(self.workspace_scale)
        target_pos_arm = (pinch_arm_raw - bbox_center) * scale + self.workspace_anchor

        R_wrist_cam = np.zeros((T, 3, 3))
        R_wrist_cam[valid_mask] = Rotation.from_quat(
            wrist_quat_cam[valid_mask]
        ).as_matrix()
        R_target_arm = np.einsum("ij,tjk->tik", _R_ARM_CAM_OPENCV, R_wrist_cam)

        # mink setup. Configuration carries the warm-start qpos across
        # frames; FrameTask carries the per-frame target.
        config = mink.Configuration(self._model)
        # Seed at zero pose (URDF "home"). All joints to 0 rad.
        config.update(np.zeros(self._model.nq, dtype=np.float64))
        frame_task = mink.FrameTask(
            frame_name=self.target_site,
            frame_type="site",
            position_cost=self.position_cost,
            orientation_cost=self.orientation_cost,
        )
        # Smoothness regularizer: posture target updated each frame to
        # the previous solution so the QP penalizes deviation from that
        # configuration. Without this, the 5-DoF arm + 6-DoF target
        # over-determined system has multiple per-frame local minima and
        # IK flips between them frame-to-frame (visible as jitter / 100°+
        # joint jumps). Cost magnitude is the trade-off knob: too low =
        # jitter, too high = sluggish tracking.
        posture_task = mink.PostureTask(self._model, cost=self.posture_cost)
        posture_task.set_target_from_configuration(config)
        # Joint limits read straight from MJCF (jnt_range). Saves ikpy's
        # active_links_mask gymnastics.
        config_limit = mink.ConfigurationLimit(self._model)
        tasks = [frame_task, posture_task]
        limits = [config_limit]

        ik_pos_err = np.full(T, np.nan, dtype=np.float64)
        ik_rot_err_deg = np.full(T, np.nan, dtype=np.float64)
        n_failed = 0

        for t in range(T):
            if not bool(valid_mask[t]):
                continue
            target_pose = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3.from_matrix(R_target_arm[t]),
                translation=target_pos_arm[t],
            )
            frame_task.set_target(target_pose)
            # Track previous-frame solution as posture target — gives
            # the QP a "stay where you were unless EE task pulls you"
            # bias. CRUCIAL for smoothness; without this update the
            # posture target stays at the home pose forever and adjacent
            # frames flip between IK minima.
            posture_task.set_target_from_configuration(config)
            try:
                # Iterate so the soft IK converges to the static target.
                # Each call returns joint velocity (rad/s); integrating
                # with the same dt steps q toward target. With dt=0.05
                # and 30 iters, a static target typically converges to
                # sub-mm / sub-degree on well-conditioned frames.
                for _ in range(self.ik_iters):
                    velocity = mink.solve_ik(
                        config, tasks, dt=self.ik_dt, solver=self.ik_solver,
                        damping=self.ik_damping,
                        limits=limits,
                    )
                    config.integrate_inplace(velocity, self.ik_dt)
            except Exception as exc:
                print(f"  [so101_v2] frame {first + t}: IK failed: {exc}")
                valid_mask[t] = False
                n_failed += 1
                continue

            # Extract arm joint angles in JOINT_NAMES order.
            for i, jname in enumerate(JOINT_NAMES[:5]):
                qpos_out[t, i] = float(config.q[self._qadr_for_joint[jname]])

            # Residual position + orientation error.
            mujoco.mj_forward(self._model, config.data)
            site_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_SITE, self.target_site
            )
            site_pos = config.data.site_xpos[site_id]
            site_xmat = config.data.site_xmat[site_id].reshape(3, 3)
            ik_pos_err[t] = float(np.linalg.norm(target_pos_arm[t] - site_pos))
            R_res = site_xmat.T @ R_target_arm[t]
            trace_clipped = float(np.clip((np.trace(R_res) - 1) / 2, -1, 1))
            ik_rot_err_deg[t] = float(np.degrees(np.arccos(trace_clipped)))

        # Gripper from MANO pinch distance (rigid-invariant, no IK).
        gripper_rad = _gripper_from_mano_joints(
            joints_cam, valid_mask,
            open_distance_m=self.gripper_open_distance_m,
            close_distance_m=self.gripper_close_distance_m,
            open_deg=self.gripper_open_deg,
            close_deg=self.gripper_close_deg,
        )
        qpos_out[valid_mask, 5] = gripper_rad[valid_mask].astype(np.float32)
        qpos_valid[:] = valid_mask

        valid_after = np.flatnonzero(valid_mask)
        if len(valid_after) == 0:
            return RetargetResult(
                qpos=qpos_out, qpos_valid=qpos_valid,
                joint_names=self.joint_names,
                extras={"n_frames_valid_input": 0, "ik_engine": "mink"},
            )
        idx0 = int(valid_after[0])

        # Cam pose for sim cam_frame view: place cam where the recording
        # camera was on average relative to the hand. Anchor uses the
        # per-axis MEDIAN of pinch_cam over valid frames (not idx0):
        # idx0 is brittle — HaMeR detection transients on the first
        # frame and IK failures during the approach segment can throw
        # baseline ||cam→centroid|| 2× off (observed: ep01/04 idx0
        # baseline 0.75 m vs median ~0.43 m), making per-episode mp4 /
        # rerun framing wildly inconsistent across a batch. Median is
        # robust to outlier frames where the hand briefly reaches far
        # from camera, and across pour_coffee_bean the 4 episodes land
        # within 8% of each other (0.41–0.44 m).
        pinch_arm_valid = target_pos_arm[valid_mask]
        centroid = pinch_arm_valid.mean(axis=0)
        cam_offset_cam = -np.median(pinch_cam[valid_mask], axis=0)
        t_arm_cam = centroid + (_R_ARM_CAM_OPENCV @ cam_offset_cam)

        # Cam orientation: look from t_arm_cam at the centroid. world +Z
        # up. MuJoCo cam Z = away from scene (out of screen).
        forward = centroid - t_arm_cam
        forward /= max(np.linalg.norm(forward), 1e-9)
        cam_z = -forward
        world_up = np.array([0.0, 0.0, 1.0])
        cam_x = np.cross(world_up, cam_z)
        if np.linalg.norm(cam_x) < 1e-6:
            cam_x = np.array([1.0, 0.0, 0.0])
        cam_x /= np.linalg.norm(cam_x)
        cam_y = np.cross(cam_z, cam_x)
        R_arm_cam_mujoco = np.column_stack([cam_x, cam_y, cam_z])
        cam_quat_xyzw = Rotation.from_matrix(R_arm_cam_mujoco).as_quat()

        # Reach diagnostic — fraction of valid targets within SO-101
        # reach radius (~0.35 m from base). Surfaces "data outside
        # robot's workspace" loud rather than burying it in IK error.
        dist_from_base = np.linalg.norm(pinch_arm_valid, axis=1)
        n_within = int((dist_from_base <= 0.35).sum())
        n_total = int(len(pinch_arm_valid))

        extras = {
            "n_frames_valid_input": int(valid_mask.sum()),
            "n_ik_failed": int(n_failed),
            "ik_engine": "mink",
            "ik_solver": self.ik_solver,
            "position_cost": self.position_cost,
            "orientation_cost": self.orientation_cost,
            "target_site": self.target_site,
            "source_kp": "pinch_midpoint(thumb_tip,index_tip)",
            "ik_pos_err_mean_mm": (
                float(np.nanmean(ik_pos_err) * 1000)
                if np.any(np.isfinite(ik_pos_err)) else None
            ),
            "ik_pos_err_max_mm": (
                float(np.nanmax(ik_pos_err) * 1000)
                if np.any(np.isfinite(ik_pos_err)) else None
            ),
            "ik_rot_err_mean_deg": (
                float(np.nanmean(ik_rot_err_deg))
                if np.any(np.isfinite(ik_rot_err_deg)) else None
            ),
            "ik_rot_err_max_deg": (
                float(np.nanmax(ik_rot_err_deg))
                if np.any(np.isfinite(ik_rot_err_deg)) else None
            ),
            "reach_within_0.35m_frac": n_within / max(n_total, 1),
            "target_pos_arm_centroid": centroid.tolist(),
            "cam_pos_arm": t_arm_cam.tolist(),
            "cam_quat_arm_xyzw": cam_quat_xyzw.tolist(),
            "workspace_anchor": self.workspace_anchor.tolist(),
            "workspace_arm_reach": self.workspace_arm_reach,
            "workspace_fit_factor": self.workspace_fit_factor,
            "workspace_scale_used": scale,
            "workspace_scale_auto": self.workspace_scale is None,
            "workspace_bbox_half_diag_m": float(half_diag),
            "workspace_bbox_center_arm": bbox_center.tolist(),
        }
        return RetargetResult(
            qpos=qpos_out, qpos_valid=qpos_valid,
            joint_names=self.joint_names,
            extras=extras,
        )
