"""
Tests for utils/hand_tracker/depth_correction.py

Covers:
  1. Identity correction (z_hamer == z_lidar)
  2. Known scale factor
  3. Invalid depth (NaN, zero, negative)
  4. Pixel out of bounds
  5. patch_median method
  6. Summary printing

All tests are synthetic (no hardware needed).

Run:
    cd hand-6dof-pipeline
    python -m pytest tests/test_depth_correction.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hand_tracker.depth_correction import correct_depth_perspective, print_depth_correction_summary

ATOL = 1e-6


def make_K(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    """Create a synthetic 3x3 intrinsic matrix."""
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=np.float64)


def make_depth_map(h=480, w=640, fill=1.0):
    """Create a uniform depth map."""
    return np.full((h, w), fill, dtype=np.float32)


# ============================================================
# 1. Identity correction
# ============================================================

def test_identity_no_change():
    """When z_hamer == z_lidar, position should be unchanged."""
    K = make_K()
    pos = np.array([0.1, -0.05, 1.0])
    depth = make_depth_map(fill=1.0)

    corrected, stats = correct_depth_perspective(pos, depth, K)

    np.testing.assert_allclose(corrected, pos, atol=ATOL)
    assert abs(stats["scale"] - 1.0) < ATOL
    assert stats["valid"]


# ============================================================
# 2. Known scale factor
# ============================================================

def test_known_scale_double():
    """z_hamer=1.0, z_lidar=2.0 → x,y doubled, z=2.0."""
    K = make_K()
    pos = np.array([0.1, -0.05, 1.0])
    depth = make_depth_map(fill=2.0)

    corrected, stats = correct_depth_perspective(pos, depth, K)

    np.testing.assert_allclose(corrected, [0.2, -0.1, 2.0], atol=ATOL)
    assert abs(stats["scale"] - 2.0) < ATOL
    assert stats["valid"]


def test_known_scale_half():
    """z_hamer=2.0, z_lidar=1.0 → x,y halved, z=1.0."""
    K = make_K()
    pos = np.array([0.2, -0.1, 2.0])
    depth = make_depth_map(fill=1.0)

    corrected, stats = correct_depth_perspective(pos, depth, K)

    np.testing.assert_allclose(corrected, [0.1, -0.05, 1.0], atol=ATOL)
    assert abs(stats["scale"] - 0.5) < ATOL


# ============================================================
# 3. Invalid depth
# ============================================================

def test_zero_depth_map_returns_original():
    """Zero depth in depth map → return original position."""
    K = make_K()
    pos = np.array([0.1, 0.0, 1.0])
    depth = make_depth_map(fill=0.0)

    corrected, stats = correct_depth_perspective(pos, depth, K)

    np.testing.assert_array_equal(corrected, pos)
    assert not stats["valid"]


def test_negative_z_hamer_returns_original():
    """Negative z in input → return original (behind camera)."""
    K = make_K()
    pos = np.array([0.1, 0.0, -1.0])
    depth = make_depth_map(fill=1.0)

    corrected, stats = correct_depth_perspective(pos, depth, K)

    np.testing.assert_array_equal(corrected, pos)
    assert not stats["valid"]


def test_nan_z_hamer_returns_original():
    """NaN z in input → return original."""
    K = make_K()
    pos = np.array([0.1, 0.0, float("nan")])
    depth = make_depth_map(fill=1.0)

    corrected, stats = correct_depth_perspective(pos, depth, K)

    assert not stats["valid"]


# ============================================================
# 4. Pixel out of bounds
# ============================================================

def test_pixel_out_of_bounds():
    """Projected pixel outside depth map → return original."""
    K = make_K()
    # Position that projects far outside the image
    pos = np.array([100.0, 0.0, 1.0])  # px = 500*100 + 320 = way out of bounds
    depth = make_depth_map(fill=1.0)

    corrected, stats = correct_depth_perspective(pos, depth, K)

    np.testing.assert_array_equal(corrected, pos)
    assert not stats["valid"]


# ============================================================
# 5. patch_median method
# ============================================================

def test_patch_median_basic():
    """patch_median should use median of 5x5 neighborhood."""
    K = make_K()
    pos = np.array([0.0, 0.0, 1.0])  # projects to (cx, cy) = (320, 240)
    depth = make_depth_map(fill=2.0)

    corrected, stats = correct_depth_perspective(pos, depth, K, method="patch_median")

    np.testing.assert_allclose(corrected[2], 2.0, atol=ATOL)
    assert stats["valid"]


def test_patch_median_robust_to_outlier():
    """patch_median should ignore a single outlier pixel."""
    K = make_K()
    pos = np.array([0.0, 0.0, 1.0])  # projects to (320, 240)
    depth = make_depth_map(fill=2.0)
    depth[240, 320] = 100.0  # outlier at center pixel

    corrected, stats = correct_depth_perspective(pos, depth, K, method="patch_median")

    # Median of mostly 2.0 with one 100.0 outlier → should still be 2.0
    np.testing.assert_allclose(corrected[2], 2.0, atol=ATOL)


# ============================================================
# 6. Summary printing
# ============================================================

def test_summary_prints(capsys):
    """Summary should print stats to stdout."""
    stats_list = [
        {"z_hamer": 1.0, "z_lidar": 1.5, "scale": 1.5, "valid": True},
        {"z_hamer": 1.0, "z_lidar": 1.2, "scale": 1.2, "valid": True},
        {"z_hamer": 1.0, "z_lidar": float("nan"), "scale": 1.0, "valid": False},
    ]
    print_depth_correction_summary(stats_list)
    captured = capsys.readouterr()
    assert "Corrected frames: 2" in captured.out
    assert "Scale (lidar/hamer):" in captured.out


def test_summary_no_valid(capsys):
    """Summary with no valid corrections should report that."""
    stats_list = [{"valid": False}]
    print_depth_correction_summary(stats_list)
    captured = capsys.readouterr()
    assert "No valid" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
