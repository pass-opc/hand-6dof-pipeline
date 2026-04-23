"""
Trajectory interpolation utilities.

Shared by both ArUco and HaMeR pipelines. Used to fill gaps (NaN frames)
in TrackingResult data from detection failures (occlusion, blur, etc.).

Key design:
  - Position: linear interpolation (scipy interp1d)
  - Rotation: Spherical Linear Interpolation / Slerp (scipy Rotation.Slerp)
  - Gripper width: linear interpolation (1D scalar)
  - Boundary: clamp to first/last valid value (no extrapolation)

Source: Adapted from UMI umi/common/interpolation_util.py (MIT License)
Changes: Added NaN-aware filtering; UMI assumes clean input, we don't.
"""

import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


def interp1d_with_clamp(t: np.ndarray, x: np.ndarray) -> si.interp1d:
    """Create 1D interpolator that clamps to boundary values outside range.

    Source: UMI umi/common/interpolation_util.py::get_interp1d (MIT License)

    Args:
        t: timestamps, shape (N,), must be strictly increasing
        x: values, shape (N,) or (N, D)
    """
    return si.interp1d(
        t, x,
        axis=0, bounds_error=False,
        fill_value=(x[0], x[-1]),
    )


class PoseInterpolator:
    """Interpolate 6DoF poses: linear for position, Slerp for rotation.

    Source: Adapted from UMI umi/common/interpolation_util.py::PoseInterpolator (MIT License)
    Changes: Added from_tracking_result() factory that handles NaN gaps.

    Args:
        t: timestamps of valid (non-NaN) frames, shape (N,)
        x: poses [pos3 + axis_angle3], shape (N, 6)
    """

    def __init__(self, t: np.ndarray, x: np.ndarray):
        assert x.shape[1] == 6, f"Expected (N, 6) poses, got {x.shape}"
        assert len(t) == len(x), f"Timestamp/pose count mismatch: {len(t)} vs {len(x)}"
        assert len(t) >= 2, "Need at least 2 valid frames for interpolation"

        self.pos_interp = interp1d_with_clamp(t, x[:, :3])
        # Slerp requires strictly increasing timestamps
        self.rot_interp = st.Slerp(t, st.Rotation.from_rotvec(x[:, 3:]))
        self.t_min = t[0]
        self.t_max = t[-1]

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """Interpolate poses at given timestamps.

        Args:
            t: query timestamps, shape (M,)
        Returns:
            poses [pos3 + axis_angle3], shape (M, 6)
        """
        t = np.clip(t, self.t_min, self.t_max)
        pos = self.pos_interp(t)
        rot = self.rot_interp(t).as_rotvec()
        return np.concatenate([pos, rot], axis=-1)

    @classmethod
    def from_tracking_result(
        cls,
        timestamps: np.ndarray,
        eef_pos: np.ndarray,
        eef_rot: np.ndarray,
    ) -> "PoseInterpolator":
        """Build interpolator from TrackingResult, filtering out NaN frames.

        Args:
            timestamps: all frame timestamps, shape (T,)
            eef_pos: positions with NaN gaps, shape (T, 3)
            eef_rot: rotations with NaN gaps, shape (T, 3)

        Raises:
            ValueError: if fewer than 2 valid frames exist
        """
        valid = ~np.isnan(eef_pos[:, 0])
        n_valid = valid.sum()
        if n_valid < 2:
            raise ValueError(
                f"Need at least 2 valid frames for interpolation, got {n_valid}/{len(timestamps)}"
            )

        t_valid = timestamps[valid]
        poses_valid = np.concatenate([eef_pos[valid], eef_rot[valid]], axis=-1)
        return cls(t_valid, poses_valid)


def interpolate_gripper_width(
    timestamps: np.ndarray,
    gripper_width: np.ndarray,
) -> np.ndarray:
    """Fill NaN gaps in gripper width using linear interpolation.

    Args:
        timestamps: all frame timestamps, shape (T,)
        gripper_width: width values with NaN gaps, shape (T,)

    Returns:
        Filled gripper width, shape (T,). No NaN values.
    """
    valid = ~np.isnan(gripper_width)
    n_valid = valid.sum()
    if n_valid == 0:
        raise ValueError("No valid gripper width values to interpolate from")
    if n_valid == len(gripper_width):
        return gripper_width.copy()  # nothing to fill

    interp = interp1d_with_clamp(timestamps[valid], gripper_width[valid])
    return interp(timestamps)


def max_consecutive_nans(arr: np.ndarray) -> int:
    """Count the longest consecutive NaN run in a 1D or 2D array.

    For 2D input, checks NaN along the first column (assumes if pos.x is NaN,
    the whole frame is NaN).
    """
    if arr.ndim == 2:
        is_nan = np.isnan(arr[:, 0])
    else:
        is_nan = np.isnan(arr)

    if not np.any(is_nan):
        return 0

    # Diff trick: find runs of consecutive True values
    # Pad with False at boundaries so diff catches start/end runs
    padded = np.concatenate([[False], is_nan, [False]])
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return int(np.max(ends - starts))


def mark_bad_frames(
    tracking_result: dict,
    max_pos_jump_m: float = 0.05,
) -> dict:
    """Mark frames with suspicious PnP results as NaN for later interpolation.

    Detects single-frame position jumps: if frame[t] is far from both
    frame[t-1] and frame[t+1], it's likely a PnP error (half-occluded marker,
    motion blur), not real motion. The key insight: if t-1 and t+1 are close
    to each other but both far from t, then t is the outlier.

    Unlike 2*pi axis-angle wrapping (which disappears when converting to
    rotation matrix), PnP jumps affect xyz position and are real errors.

    Only checks position (xyz), not rotation — rotation has the 2*pi
    ambiguity which is harmless and handled at training time via rot6d.

    Args:
        max_pos_jump_m: Position change threshold in meters per frame.
            Default 0.05 = 5cm/frame = 3m/s @ 60fps (faster than any hand).

    Returns:
        New TrackingResult with bad frames set to NaN in eef_pos, eef_rot,
        gripper_width. Also adds "n_marked_bad" count.
    """
    eef_pos = tracking_result["eef_pos"].copy()
    eef_rot = tracking_result["eef_rot"].copy()
    gripper_width = tracking_result["gripper_width"].copy()
    valid = ~np.isnan(eef_pos[:, 0])
    n_marked = 0

    def _mark(t: int) -> None:
        nonlocal n_marked
        eef_pos[t] = np.nan
        eef_rot[t] = np.nan
        gripper_width[t] = np.nan
        valid[t] = False
        n_marked += 1

    # Interior: 3-point spike — t is outlier if both neighbors are close
    # to each other but far from t.
    for t in range(1, len(eef_pos) - 1):
        if not valid[t] or not valid[t - 1]:
            continue

        d_prev = np.linalg.norm(eef_pos[t] - eef_pos[t - 1])
        if d_prev <= max_pos_jump_m:
            continue

        # t jumped from t-1. Is it real motion or outlier?
        t_next = t + 1
        while t_next < len(eef_pos) and not valid[t_next]:
            t_next += 1
        if t_next >= len(eef_pos):
            continue

        # If t-1 → t+n is closer than t-1 → t, then t is the outlier
        d_skip = np.linalg.norm(eef_pos[t_next] - eef_pos[t - 1])
        if d_skip < d_prev:
            _mark(t)

    # Boundary: t=0 has no t-1. Mirror the logic using (t+1, t+2):
    # frame 0 is an outlier if it's far from frame 1 AND frame 1 is
    # close to frame 2 (1 and 2 lie on the true trajectory, 0 is the spike).
    # Same for t=T-1 against (T-2, T-3). Critical because trim boundaries
    # often land on first/last detected frame where HaMeR had weakest signal.
    def _first_two_valid(indices) -> tuple[int, int] | None:
        found = []
        for i in indices:
            if valid[i]:
                found.append(i)
                if len(found) == 2:
                    return found[0], found[1]
        return None

    if valid[0]:
        nbrs = _first_two_valid(range(1, len(eef_pos)))
        if nbrs is not None:
            n1, n2 = nbrs
            d01 = np.linalg.norm(eef_pos[0] - eef_pos[n1])
            d12 = np.linalg.norm(eef_pos[n1] - eef_pos[n2])
            if d01 > max_pos_jump_m and d12 <= max_pos_jump_m:
                _mark(0)

    last = len(eef_pos) - 1
    if valid[last]:
        nbrs = _first_two_valid(range(last - 1, -1, -1))
        if nbrs is not None:
            n1, n2 = nbrs
            d01 = np.linalg.norm(eef_pos[last] - eef_pos[n1])
            d12 = np.linalg.norm(eef_pos[n1] - eef_pos[n2])
            if d01 > max_pos_jump_m and d12 <= max_pos_jump_m:
                _mark(last)

    if n_marked > 0:
        print(f"  Marked {n_marked} position-jump frames as NaN (threshold: {max_pos_jump_m*100:.0f}cm/frame)")

    return {
        **tracking_result,
        "eef_pos": eef_pos,
        "eef_rot": eef_rot,
        "gripper_width": gripper_width,
        "n_marked_bad": n_marked,
    }


def trim_nan_boundaries(tracking_result: dict) -> dict:
    """Trim leading and trailing NaN frames from a TrackingResult.

    Finds the first and last frame where eef_pos is not NaN,
    and slices all arrays to that range.

    Args:
        tracking_result: dict with timestamps, eef_pos, eef_rot, etc.

    Returns:
        New dict with boundary NaN frames removed. If no valid frames
        exist, returns the original dict unchanged.
    """
    is_valid = ~np.isnan(tracking_result["eef_pos"][:, 0])
    if not np.any(is_valid):
        return tracking_result

    valid_indices = np.where(is_valid)[0]
    first, last = valid_indices[0], valid_indices[-1] + 1  # slice end is exclusive
    if first == 0 and last == len(is_valid):
        return tracking_result  # no boundary NaN, nothing to trim

    n_trimmed = (first) + (len(is_valid) - last)
    print(f"  Trimmed {n_trimmed} boundary NaN frames "
          f"(leading: {first}, trailing: {len(is_valid) - last})")

    seg = slice(first, last)
    return {
        **tracking_result,
        "timestamps": tracking_result["timestamps"][seg],
        "eef_pos": tracking_result["eef_pos"][seg],
        "eef_rot": tracking_result["eef_rot"][seg],
        "gripper_width": tracking_result["gripper_width"][seg],
        "confidence": tracking_result["confidence"][seg],
        "trim_slice": (first, last),  # original frame index range, for RGB alignment
    }


def fill_dual_hand_tracking_result(
    tracking_result: dict,
    hand: str,
    trim_boundary_nans: bool = False,
) -> dict:
    """Extract one hand from dual-hand format and fill NaN gaps.

    Converts dual-hand TrackingResult (from 01_hand_track.py) into single-hand
    TrackingResult compatible with downstream code (same keys as ArUco output).

    Args:
        tracking_result: dual-hand format with left_hand/right_hand sub-dicts
        hand: "left" or "right"
        trim_boundary_nans: trim leading/trailing NaN frames

    Returns:
        Single-hand TrackingResult dict (timestamps, eef_pos, eef_rot, etc.)
        with NaN gaps filled via interpolation.
    """
    hand_key = f"{hand}_hand"
    hand_data = tracking_result[hand_key]

    single = {
        "timestamps": tracking_result["timestamps"],
        "eef_pos": hand_data["eef_pos"],
        "eef_rot": hand_data["eef_rot"],
        "gripper_width": hand_data["gripper_width"],
        "confidence": hand_data["confidence"],
        "source": tracking_result.get("source", "unknown"),
        "episode_name": tracking_result.get("episode_name", ""),
    }

    return fill_tracking_result(single, trim_boundary_nans=trim_boundary_nans)


def fill_tracking_result(
    tracking_result: dict,
    trim_boundary_nans: bool = False,
) -> dict:
    """Fill all NaN gaps in a TrackingResult dict via interpolation.

    Modifies nothing in-place; returns a new dict with filled arrays.
    Also reports fill statistics including max consecutive gap length,
    which downstream quality filters can use to reject bad episodes.

    By default, all NaN frames (including boundary) are filled via
    interpolation (interior) or clamping (boundary). Set
    trim_boundary_nans=True to trim leading/trailing NaN frames first.

    Args:
        tracking_result: dict with timestamps, eef_pos, eef_rot, gripper_width, confidence
        trim_boundary_nans: if True, trim leading/trailing NaN frames before interpolation.
            Default False: boundary NaN frames are filled by clamping to first/last valid value.

    Returns:
        New dict with:
          - NaN-free eef_pos, eef_rot, gripper_width
          - "gap_stats" dict: n_missing, max_consecutive_gap, detect_rate
          - Original confidence values preserved (not interpolated)
    """
    if trim_boundary_nans:
        tracking_result = trim_nan_boundaries(tracking_result)

    ts = tracking_result["timestamps"]
    eef_pos = tracking_result["eef_pos"]
    eef_rot = tracking_result["eef_rot"]
    gripper_width = tracking_result["gripper_width"]

    n_total = len(ts)
    n_pos_missing = np.isnan(eef_pos[:, 0]).sum()
    n_width_missing = np.isnan(gripper_width).sum()
    max_gap = max_consecutive_nans(eef_pos)

    # Interpolate pose
    pose_interp = PoseInterpolator.from_tracking_result(ts, eef_pos, eef_rot)
    filled_poses = pose_interp(ts)

    # Interpolate gripper width
    filled_width = interpolate_gripper_width(ts, gripper_width)

    detect_rate = 1.0 - n_pos_missing / n_total
    print(f"  Interpolation: filled {n_pos_missing}/{n_total} pose frames, "
          f"{n_width_missing}/{n_total} width frames, "
          f"max consecutive gap: {max_gap}, detect rate: {detect_rate:.1%}")

    return {
        **tracking_result,
        "eef_pos": filled_poses[:, :3],
        "eef_rot": filled_poses[:, 3:],
        "gripper_width": filled_width,
        "gap_stats": {
            "n_pos_missing": int(n_pos_missing),
            "n_width_missing": int(n_width_missing),
            "max_consecutive_gap": max_gap,
            "detect_rate": float(detect_rate),
        },
    }
