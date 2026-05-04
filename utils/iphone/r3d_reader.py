"""
Record3D .r3d file reader and iPhone intrinsics utilities.

Shared IO layer for all pipeline scripts (01-04) and future HaMeR main line.
Provides streaming frame access to avoid loading entire recordings into memory.

Orientation handling:
  Record3D stores RGB / depth in either the sensor-native canvas (W > H,
  iPhone wide camera is 4:3) or — when iOS screen rotation is locked to
  portrait while the user holds the phone landscape — a portrait canvas
  (W < H) into which the sensor data has been baked CW 90°. The right
  rotation to normalize a recording therefore depends on what physical
  orientation we *want* the output to be in. `resolve_rotation` returns
  the per-recording image rotation under three named policies:

    * 'auto'      — legacy heuristic. W > H → CCW 90° (results in portrait
                    output). Pre-existing test_HaMeR_4_15 batch was processed
                    this way; kept as default to preserve old behaviour.
    * 'landscape' — restore the canonical sensor-native landscape view.
                    W > H → keep, W < H → CCW 90° (undoes Record3D's CW 90°
                    canvas rotation).
    * 'portrait'  — mirror of 'landscape' for portrait outputs.

  All math (K transform, ARKit pose right-multiply, RGB / depth rotate) is
  driven from a single resolved string ('ccw90' or None) so the policy
  branch lives in exactly one place.

.r3d format: ZIP archive containing:
  - metadata (JSON): fps, timestamps, intrinsics, depth resolution
  - rgbd/N.jpg: RGB frames (N = 0-based index)
  - rgbd/N.depth: LZFSE-compressed float32 depth maps (iPhone LiDAR)
"""

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# 1. Orientation Detection
# =============================================================================

# Public modes accepted by resolve_rotation / read_iphone_intrinsics /
# read_poses / iter_r3d_frames. 'auto' preserves legacy behaviour (W > H →
# CCW 90°). 'landscape' / 'portrait' make the *output* orientation explicit
# regardless of how Record3D stored the canvas.
ORIENTATION_MODES = ("auto", "landscape", "portrait")


def resolve_rotation(metadata: dict, mode: str = "auto") -> str | None:
    """Decide what canvas rotation to apply to a single recording.

    Returns either None or 'ccw90'. The caller plumbs that string into
    `_rotate_frame_ccw90`, the K transform, and the pose right-multiply.

    Why a single resolved string instead of branching on `mode` everywhere:
    keeping the math one-armed (ccw90 only) means the K formula and the
    pose adjustment derived for the legacy landscape→portrait case carry
    over verbatim — direction is the same, only the trigger condition
    differs across modes.
    """
    if mode not in ORIENTATION_MODES:
        raise ValueError(
            f"Unknown orientation mode {mode!r}; expected one of {ORIENTATION_MODES}"
        )
    w = metadata.get("w", 0)
    h = metadata.get("h", 0)
    if mode == "auto":
        # Legacy: W > H assumed sensor-native landscape, rotate to portrait.
        return "ccw90" if w > h else None
    if mode == "landscape":
        # Already landscape canvas → keep. Portrait canvas → undo Record3D's
        # CW 90° bake by applying CCW 90°.
        return None if w > h else "ccw90"
    # mode == "portrait"
    return "ccw90" if w > h else None


def needs_rotation(metadata: dict) -> bool:
    """Backward-compat wrapper. True iff legacy 'auto' mode would rotate.

    Older callers used this as a bool check; new code should call
    resolve_rotation() directly to access non-default modes.
    """
    return resolve_rotation(metadata, mode="auto") == "ccw90"


def _rotate_frame_ccw90(img: np.ndarray) -> np.ndarray:
    """Rotate image 90° counter-clockwise."""
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


# =============================================================================
# 2. iPhone Intrinsics
# =============================================================================

def _extract_intrinsics_raw(metadata: dict, frame_idx: int = 0) -> tuple[float, float, float, float]:
    """Extract raw fx, fy, cx, cy from metadata (before any rotation)."""
    per_frame = metadata.get("perFrameIntrinsicCoeffs", [])
    if per_frame and frame_idx < len(per_frame):
        coeffs = per_frame[frame_idx]
        if len(coeffs) == 4:
            # Compact format: [fx, fy, cx, cy]
            return coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        else:
            # Column-major 3x3: [fx, 0, 0, 0, fy, 0, cx, cy, 1]
            return coeffs[0], coeffs[4], coeffs[6], coeffs[7]
    elif "K" in metadata:
        coeffs = metadata["K"]
        return coeffs[0], coeffs[4], coeffs[6], coeffs[7]
    else:
        raise ValueError("No intrinsics found in Record3D metadata")


def read_iphone_intrinsics(
    metadata: dict,
    frame_idx: int = 0,
    *,
    orientation: str = "auto",
) -> np.ndarray:
    """Extract camera intrinsic matrix K from Record3D metadata.

    Adjusts K to match whatever rotation `resolve_rotation(metadata,
    orientation)` decides will be applied to RGB / depth, so projecting a
    cam-frame 3D point with the returned K lands on the same pixel that
    iter_r3d_frames yields for that frame.

    CCW 90° rotation maps pixel (u, v) → (v, W-1-u), so:
      fx_new = fy, fy_new = fx, cx_new = cy, cy_new = W-1-cx

    Returns:
        K: 3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    """
    fx, fy, cx, cy = _extract_intrinsics_raw(metadata, frame_idx)

    if resolve_rotation(metadata, orientation) == "ccw90":
        W_orig = metadata["w"]
        # CCW 90°: (u, v) → (v, W-1-u)
        fx, fy = fy, fx
        cx, cy = cy, W_orig - 1 - cx

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=np.float64)


def scale_intrinsics(K: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    """Scale intrinsic matrix when image is resized.

    Focal lengths and principal point scale proportionally with resolution.
    """
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    K_scaled = K.copy()
    K_scaled[0, :] *= sx  # fx, cx
    K_scaled[1, :] *= sy  # fy, cy
    return K_scaled


# =============================================================================
# 2b. ARKit World→Camera Poses
# =============================================================================

def read_poses(r3d_path: Path, *, orientation: str = "auto") -> np.ndarray:
    """Read per-frame ARKit T_world_cam, aligned with the rotated RGB stream.

    Record3D metadata['poses'] is a list of [qx, qy, qz, qw, tx, ty, tz] per
    frame. Each represents T_world_cam: the sensor-native camera pose in
    ARKit's gravity-aligned world (+Y up, origin = session start pose).

    When iter_r3d_frames applies CCW 90° to RGB (per `orientation`), the
    camera frame attached to the rotated canvas differs from the sensor
    frame ARKit reports. We right-multiply the pose so consumers see
    T_world_cam in the rotated canvas frame:
        T_world_cam_new = T_world_cam_raw @ R_raw_from_new
    where R_raw_from_new = [[0,1,0],[-1,0,0],[0,0,1]] (image CCW 90° ↔
    camera-frame +90° about +Z optical axis). The rotation matrix is the
    same regardless of starting/ending orientation; only whether to apply
    it depends on `orientation`.
    """
    from scipy.spatial.transform import Rotation

    with zipfile.ZipFile(r3d_path, "r") as zf:
        metadata = json.loads(zf.read("metadata"))

    poses = metadata.get("poses")
    if poses is None:
        raise ValueError(f"No 'poses' field in {r3d_path.name}")
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(
            f"Expected poses shape (T,7), got {poses.shape} in {r3d_path.name}"
        )

    # scipy Rotation.from_quat uses [x, y, z, w] — matches Record3D exactly
    R_wc = Rotation.from_quat(poses[:, :4]).as_matrix()  # (T, 3, 3)
    t_wc = poses[:, 4:7]                                 # (T, 3)

    if resolve_rotation(metadata, orientation) == "ccw90":
        R_raw_from_port = np.array(
            [[0.0,  1.0, 0.0],
             [-1.0, 0.0, 0.0],
             [0.0,  0.0, 1.0]],
            dtype=np.float64,
        )
        R_wc = R_wc @ R_raw_from_port

    T = np.zeros((len(poses), 4, 4), dtype=np.float64)
    T[:, :3, :3] = R_wc
    T[:, :3, 3] = t_wc
    T[:, 3, 3] = 1.0
    return T


# =============================================================================
# 3. R3D Metadata Reader
# =============================================================================

def read_r3d_metadata(r3d_path: Path) -> tuple[dict, list[str]]:
    """Read metadata and sorted jpg list from .r3d without loading any frames.

    Returns:
        metadata: Raw metadata dict (intrinsics, fps, timestamps, etc.)
        jpg_names: Sorted list of jpg entry names inside the zip.
    """
    with zipfile.ZipFile(r3d_path, "r") as zf:
        metadata = json.loads(zf.read("metadata"))
        jpg_names = sorted(
            [n for n in zf.namelist() if n.startswith("rgbd/") and n.endswith(".jpg")],
            key=lambda n: int(n.split("/")[1].split(".")[0]),
        )
    return metadata, jpg_names


# =============================================================================
# 3. Streaming Frame Reader
# =============================================================================

def iter_r3d_frames(
    r3d_path: Path,
    read_depth: bool = False,
    sample_every: int = 1,
    frame_indices: set[int] | None = None,
    *,
    orientation: str = "auto",
):
    """Yield (frame_index, rgb, timestamp, depth_or_None) one frame at a time.

    Streaming reader: only one frame lives in memory at any time.
    This is the standard memory-safe pattern (same as UMI's PyAV decode loop).

    Orientation is resolved per-recording via `resolve_rotation(metadata,
    orientation)`. When the resolution is 'ccw90', RGB and depth are
    rotated CCW 90° before being yielded. K (read_iphone_intrinsics) and
    pose (read_poses) called with the same `orientation` will be aligned.

    Args:
        sample_every: Yield every N-th frame (1 = all). Applied before frame_indices.
        frame_indices: If given, only yield frames whose index is in this set.
            Useful for trim alignment (e.g. skip leading/trailing NaN frames).
        orientation: 'auto' | 'landscape' | 'portrait'. See module docstring.

    Yields:
        (frame_index, rgb, timestamp, depth)
        - frame_index: int, 0-based index in the original recording
        - rgb: (H, W, 3) uint8 RGB, in the canvas frame chosen by `orientation`
        - timestamp: float, seconds
        - depth: (dh, dw) float32 meters, or None if read_depth=False
    """
    with zipfile.ZipFile(r3d_path, "r") as zf:
        metadata = json.loads(zf.read("metadata"))
        ts_list = metadata.get("frameTimestamps", [])
        fps = metadata.get("fps", 60)
        dw = metadata.get("dw", 0)
        dh = metadata.get("dh", 0)
        rotation = resolve_rotation(metadata, orientation)
        rotate = rotation == "ccw90"

        jpg_names = sorted(
            [n for n in zf.namelist() if n.startswith("rgbd/") and n.endswith(".jpg")],
            key=lambda n: int(n.split("/")[1].split(".")[0]),
        )

        if rotate and jpg_names:
            print(f"  [r3d_reader] orientation={orientation!r}: applying CCW 90° "
                  f"(canvas {metadata.get('w')}x{metadata.get('h')} → "
                  f"{metadata.get('h')}x{metadata.get('w')})")

        for i, jpg_name in enumerate(jpg_names):
            if sample_every > 1 and i % sample_every != 0:
                continue
            if frame_indices is not None and i not in frame_indices:
                continue

            # Decode single frame
            img_bytes = zf.read(jpg_name)
            img = cv2.imdecode(
                np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Timestamp
            ts = float(ts_list[i]) if (ts_list and i < len(ts_list)) else i / fps

            # Depth (optional)
            depth = None
            if read_depth and dw > 0 and dh > 0:
                frame_idx = int(jpg_name.split("/")[1].split(".")[0])
                import liblzfse
                depth_bytes = liblzfse.decompress(zf.read(f"rgbd/{frame_idx}.depth"))
                depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(dh, dw).copy()

            # Normalize orientation: landscape → portrait
            if rotate:
                rgb = _rotate_frame_ccw90(rgb)
                if depth is not None:
                    depth = _rotate_frame_ccw90(depth)

            yield i, rgb, ts, depth
            # rgb/depth go out of scope on next iteration → memory reclaimed
