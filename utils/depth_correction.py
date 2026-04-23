"""
LiDAR depth correction for monocular hand tracking estimates.

HaMeR estimates hand position from RGB alone (weak perspective assumption).
The z (depth) component is inaccurate. iPhone LiDAR provides ground-truth
depth which can correct the monocular estimate via perspective scaling.

Correction model (pinhole camera):
  z_true = depth_map[py, px]
  scale = z_true / z_estimated
  corrected = [x * scale, y * scale, z_true]

This preserves the camera ray direction (x/z and y/z ratios from the
original estimate) while fixing the absolute depth.
"""

import numpy as np


def correct_depth_perspective(
    pos_cam: np.ndarray,
    depth_map: np.ndarray,
    K: np.ndarray,
    method: str = "wrist_point",
) -> tuple[np.ndarray, dict]:
    """Correct monocular depth using iPhone LiDAR depth map.

    Args:
        pos_cam: (3,) position in camera frame from HaMeR [x, y, z] (meters)
        depth_map: (H, W) float32 depth map from LiDAR (meters)
        K: (3, 3) camera intrinsic matrix
        method: "wrist_point" (single pixel) or "patch_median" (5x5 robust)

    Returns:
        (corrected_pos, stats): corrected (3,) position and diagnostic dict
        stats contains: z_hamer, z_lidar, scale, px, py, valid
    """
    stats = {
        "z_hamer": float(pos_cam[2]),
        "z_lidar": float("nan"),
        "scale": 1.0,
        "px": -1,
        "py": -1,
        "valid": False,
    }

    z_est = pos_cam[2]
    if z_est <= 0 or not np.isfinite(z_est):
        return pos_cam.copy(), stats

    # Project 3D point to 2D pixel coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    px = int(round(fx * pos_cam[0] / z_est + cx))
    py = int(round(fy * pos_cam[1] / z_est + cy))

    stats["px"] = px
    stats["py"] = py

    dh, dw = depth_map.shape[:2]

    if method == "patch_median":
        # 5x5 patch median for robustness to depth noise
        half = 2
        y0 = max(0, py - half)
        y1 = min(dh, py + half + 1)
        x0 = max(0, px - half)
        x1 = min(dw, px + half + 1)
        if y0 >= y1 or x0 >= x1:
            return pos_cam.copy(), stats
        patch = depth_map[y0:y1, x0:x1]
        valid_depths = patch[patch > 0]
        if len(valid_depths) == 0:
            return pos_cam.copy(), stats
        z_lidar = float(np.median(valid_depths))
    else:
        # Single pixel lookup
        if px < 0 or px >= dw or py < 0 or py >= dh:
            return pos_cam.copy(), stats
        z_lidar = float(depth_map[py, px])

    if z_lidar <= 0 or not np.isfinite(z_lidar):
        return pos_cam.copy(), stats

    # Perspective scaling: preserve ray direction, fix depth
    scale = z_lidar / z_est
    corrected = np.array([
        pos_cam[0] * scale,
        pos_cam[1] * scale,
        z_lidar,
    ])

    stats["z_lidar"] = z_lidar
    stats["scale"] = float(scale)
    stats["valid"] = True

    return corrected, stats


def back_project_depth(
    bbox_center: np.ndarray,
    depth_map: np.ndarray,
    K_real: np.ndarray,
    K_real_src_wh: tuple[int, int],
    depth_wh: tuple[int, int],
    patch_half: int = 2,
) -> tuple[np.ndarray | None, dict]:
    """Back-project bbox center to 3D using LiDAR depth and real camera K.

    Direct approach: (u,v) pixel + z_lidar → real 3D position via pinhole model.
    Bypasses HaMeR's virtual-K z estimation entirely.

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    Args:
        bbox_center: (2,) pixel coords (u, v) in RGB image space
        depth_map: (H, W) float32 depth in meters (LiDAR, may be different res)
        K_real: (3,3) real iPhone intrinsic matrix at RGB resolution
        K_real_src_wh: (W, H) of the RGB image K_real corresponds to
        depth_wh: (W, H) of the depth map
        patch_half: half-size of median patch for robust depth lookup

    Returns:
        (pos_3d, stats): (3,) real position or None if failed; diagnostic dict
    """
    stats = {
        "u": float(bbox_center[0]),
        "v": float(bbox_center[1]),
        "z_lidar": float("nan"),
        "valid": False,
        "method": "back_project",
    }

    u_rgb, v_rgb = float(bbox_center[0]), float(bbox_center[1])

    # Scale bbox center from RGB resolution to depth map resolution
    sx = depth_wh[0] / K_real_src_wh[0]
    sy = depth_wh[1] / K_real_src_wh[1]
    u_dep = int(round(u_rgb * sx))
    v_dep = int(round(v_rgb * sy))

    dh, dw = depth_map.shape[:2]

    # Patch median for robust depth lookup
    y0 = max(0, v_dep - patch_half)
    y1 = min(dh, v_dep + patch_half + 1)
    x0 = max(0, u_dep - patch_half)
    x1 = min(dw, u_dep + patch_half + 1)
    if y0 >= y1 or x0 >= x1:
        return None, stats

    patch = depth_map[y0:y1, x0:x1]
    valid_depths = patch[patch > 0]
    if len(valid_depths) == 0:
        return None, stats

    z_lidar = float(np.median(valid_depths))
    if z_lidar <= 0 or not np.isfinite(z_lidar):
        return None, stats

    # Back-project using real K at RGB resolution
    fx, fy = K_real[0, 0], K_real[1, 1]
    cx, cy = K_real[0, 2], K_real[1, 2]

    x = (u_rgb - cx) * z_lidar / fx
    y = (v_rgb - cy) * z_lidar / fy

    pos_3d = np.array([x, y, z_lidar], dtype=np.float64)

    stats["z_lidar"] = z_lidar
    stats["valid"] = True

    return pos_3d, stats


def print_depth_correction_summary(all_stats: list[dict]) -> None:
    """Print terminal summary of depth correction across all frames."""
    valid = [s for s in all_stats if s["valid"]]
    total = len(all_stats)
    n_valid = len(valid)

    print(f"\n--- Depth Correction Summary ---")
    print(f"  Total frames:     {total}")
    print(f"  Corrected frames: {n_valid} ({n_valid/total*100:.1f}%)" if total > 0 else "  No frames")

    if n_valid == 0:
        print("  No valid depth corrections.")
        return

    z_lidars = np.array([s["z_lidar"] for s in valid])
    print(f"  Z LiDAR:          mean={z_lidars.mean():.3f}m, "
          f"range=[{z_lidars.min():.3f}, {z_lidars.max():.3f}]")

    # HaMeR z comparison (if available)
    has_hamer = [s for s in valid if "z_hamer" in s and np.isfinite(s["z_hamer"])]
    if has_hamer:
        z_hamers = np.array([s["z_hamer"] for s in has_hamer])
        z_lidar_matched = np.array([s["z_lidar"] for s in has_hamer])
        z_errors = np.abs(z_hamers - z_lidar_matched)
        scales = z_lidar_matched / z_hamers
        print(f"  Z HaMeR:          mean={z_hamers.mean():.3f}m")
        print(f"  Scale (lidar/hamer): mean={scales.mean():.3f}, std={scales.std():.3f}")
        print(f"  Z error (before): mean={z_errors.mean()*100:.1f}cm, "
              f"max={z_errors.max()*100:.1f}cm")
