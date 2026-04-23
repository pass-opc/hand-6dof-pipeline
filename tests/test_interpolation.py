"""
Tests for utils/interpolation.py

Run:
    cd hand-6dof-pipeline
    python tests/test_interpolation.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.interpolation import (
    PoseInterpolator,
    interpolate_gripper_width,
    fill_tracking_result,
    max_consecutive_nans,
    trim_nan_boundaries,
)


def test_pose_interp_no_gaps():
    """With no NaN gaps, output should match input at same timestamps."""
    t = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    poses = np.column_stack([
        np.linspace(0, 1, 5),   # x: 0 to 1
        np.zeros(5),            # y: 0
        np.full(5, 0.5),        # z: 0.5
        np.zeros(5),            # rx: 0
        np.zeros(5),            # ry: 0
        np.linspace(0, 0.5, 5), # rz: 0 to 0.5
    ])

    interp = PoseInterpolator(t, poses)
    result = interp(t)

    np.testing.assert_allclose(result, poses, atol=1e-10)


def test_pose_interp_midpoint():
    """Interpolate at midpoint between two frames."""
    t = np.array([0.0, 1.0])
    poses = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ], dtype=np.float64)

    interp = PoseInterpolator(t, poses)
    result = interp(np.array([0.5]))

    # Position should be exactly midpoint
    np.testing.assert_allclose(result[0, :3], [0.5, 0, 0], atol=1e-10)


def test_pose_interp_clamp():
    """Queries outside range should clamp to boundary values."""
    t = np.array([1.0, 2.0])
    poses = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ], dtype=np.float64)

    interp = PoseInterpolator(t, poses)

    # Before range -> clamp to first
    result_before = interp(np.array([0.0]))
    np.testing.assert_allclose(result_before[0, :3], [0, 0, 0], atol=1e-10)

    # After range -> clamp to last
    result_after = interp(np.array([3.0]))
    np.testing.assert_allclose(result_after[0, :3], [1, 0, 0], atol=1e-10)


def test_pose_interp_from_tracking_result():
    """Build from TrackingResult with NaN gaps."""
    ts = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    eef_pos = np.array([
        [0, 0, 0.5],
        [np.nan, np.nan, np.nan],  # gap
        [0.2, 0, 0.5],
        [np.nan, np.nan, np.nan],  # gap
        [0.4, 0, 0.5],
    ])
    eef_rot = np.array([
        [0, 0, 0],
        [np.nan, np.nan, np.nan],
        [0, 0, 0],
        [np.nan, np.nan, np.nan],
        [0, 0, 0],
    ])

    interp = PoseInterpolator.from_tracking_result(ts, eef_pos, eef_rot)
    result = interp(ts)

    # No NaN in output
    assert not np.any(np.isnan(result)), "Output should have no NaN"
    # Frame 0 and 4 should match original
    np.testing.assert_allclose(result[0, :3], [0, 0, 0.5], atol=1e-10)
    np.testing.assert_allclose(result[4, :3], [0.4, 0, 0.5], atol=1e-10)
    # Frame 1 (gap) should be interpolated: ~[0.1, 0, 0.5]
    np.testing.assert_allclose(result[1, 0], 0.1, atol=0.01)


def test_pose_interp_too_few_valid():
    """Should raise ValueError with fewer than 2 valid frames."""
    ts = np.array([0.0, 0.1, 0.2])
    eef_pos = np.full((3, 3), np.nan)
    eef_rot = np.full((3, 3), np.nan)
    eef_pos[1] = [0, 0, 0.5]  # only 1 valid frame
    eef_rot[1] = [0, 0, 0]

    try:
        PoseInterpolator.from_tracking_result(ts, eef_pos, eef_rot)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "at least 2" in str(e)


def test_gripper_width_fill():
    """Fill NaN gaps in gripper width."""
    ts = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    width = np.array([0.08, np.nan, 0.06, np.nan, 0.04])

    filled = interpolate_gripper_width(ts, width)

    assert not np.any(np.isnan(filled))
    np.testing.assert_allclose(filled[0], 0.08, atol=1e-10)
    np.testing.assert_allclose(filled[2], 0.06, atol=1e-10)
    np.testing.assert_allclose(filled[4], 0.04, atol=1e-10)
    # Frame 1: interpolated between 0.08 and 0.06 -> 0.07
    np.testing.assert_allclose(filled[1], 0.07, atol=1e-10)
    # Frame 3: interpolated between 0.06 and 0.04 -> 0.05
    np.testing.assert_allclose(filled[3], 0.05, atol=1e-10)


def test_gripper_width_no_gaps():
    """No NaN -> output should match input."""
    ts = np.array([0.0, 0.1, 0.2])
    width = np.array([0.08, 0.06, 0.04])
    filled = interpolate_gripper_width(ts, width)
    np.testing.assert_allclose(filled, width, atol=1e-10)


def test_gripper_width_all_nan():
    """All NaN -> should raise ValueError."""
    ts = np.array([0.0, 0.1, 0.2])
    width = np.full(3, np.nan)
    try:
        interpolate_gripper_width(ts, width)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No valid" in str(e)


def test_fill_tracking_result_default_no_trim():
    """Default: fill all gaps including boundary NaN, no trimming."""
    tracking = {
        "timestamps": np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
        "eef_pos": np.array([
            [np.nan, np.nan, np.nan],  # leading NaN
            [0.1, 0, 0.5],
            [np.nan, np.nan, np.nan],  # interior NaN
            [0.3, 0, 0.5],
            [np.nan, np.nan, np.nan],  # trailing NaN
        ]),
        "eef_rot": np.array([
            [np.nan, np.nan, np.nan],
            [0, 0, 0],
            [np.nan, np.nan, np.nan],
            [0, 0, 0],
            [np.nan, np.nan, np.nan],
        ]),
        "gripper_width": np.array([np.nan, 0.08, np.nan, 0.04, np.nan]),
        "confidence": np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
        "source": "aruco",
    }

    filled = fill_tracking_result(tracking)

    # No trimming: length unchanged
    assert len(filled["timestamps"]) == 5

    # No NaN in filled data
    assert not np.any(np.isnan(filled["eef_pos"]))
    assert not np.any(np.isnan(filled["eef_rot"]))
    assert not np.any(np.isnan(filled["gripper_width"]))

    # Boundary NaN filled by clamping to nearest valid value
    np.testing.assert_allclose(filled["eef_pos"][0], [0.1, 0, 0.5], atol=1e-10)  # clamp to first valid
    np.testing.assert_allclose(filled["eef_pos"][4], [0.3, 0, 0.5], atol=1e-10)  # clamp to last valid

    # Interior NaN interpolated
    np.testing.assert_allclose(filled["eef_pos"][2, 0], 0.2, atol=0.01)


def test_fill_tracking_result_with_trim():
    """With trim_boundary_nans=True: boundary NaN trimmed before interpolation."""
    tracking = {
        "timestamps": np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
        "eef_pos": np.array([
            [np.nan, np.nan, np.nan],  # leading NaN -> trimmed
            [0.1, 0, 0.5],
            [np.nan, np.nan, np.nan],  # interior NaN -> interpolated
            [0.3, 0, 0.5],
            [np.nan, np.nan, np.nan],  # trailing NaN -> trimmed
        ]),
        "eef_rot": np.array([
            [np.nan, np.nan, np.nan],
            [0, 0, 0],
            [np.nan, np.nan, np.nan],
            [0, 0, 0],
            [np.nan, np.nan, np.nan],
        ]),
        "gripper_width": np.array([np.nan, 0.08, np.nan, 0.04, np.nan]),
        "confidence": np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
        "source": "aruco",
    }

    filled = fill_tracking_result(tracking, trim_boundary_nans=True)

    # Trimmed leading + trailing NaN: 5 -> 3 frames
    assert len(filled["timestamps"]) == 3

    # No NaN in filled data
    assert not np.any(np.isnan(filled["eef_pos"]))
    assert not np.any(np.isnan(filled["eef_rot"]))
    assert not np.any(np.isnan(filled["gripper_width"]))


def test_max_consecutive_nans():
    """Count longest NaN run."""
    # No NaN
    assert max_consecutive_nans(np.array([1, 2, 3])) == 0
    # Single NaN
    assert max_consecutive_nans(np.array([1, np.nan, 3])) == 1
    # Two separate gaps, longest is 3
    assert max_consecutive_nans(np.array([1, np.nan, 3, np.nan, np.nan, np.nan, 7])) == 3
    # All NaN
    assert max_consecutive_nans(np.full(5, np.nan)) == 5
    # NaN at boundaries
    assert max_consecutive_nans(np.array([np.nan, np.nan, 3, 4, np.nan])) == 2


def test_trim_nan_boundaries():
    """Leading and trailing NaN frames should be trimmed."""
    tracking = {
        "timestamps": np.arange(8, dtype=np.float64) * 0.1,
        "eef_pos": np.zeros((8, 3)),
        "eef_rot": np.zeros((8, 3)),
        "gripper_width": np.full(8, 0.05),
        "confidence": np.ones(8),
        "source": "aruco",
    }
    # Frames 0,1 and 6,7 are NaN (boundary gaps)
    tracking["eef_pos"][:2] = np.nan
    tracking["eef_pos"][6:] = np.nan
    tracking["eef_rot"][:2] = np.nan
    tracking["eef_rot"][6:] = np.nan

    trimmed = trim_nan_boundaries(tracking)

    # Should keep frames 2-5 (4 frames)
    assert len(trimmed["timestamps"]) == 4
    np.testing.assert_allclose(trimmed["timestamps"][0], 0.2, atol=1e-10)
    np.testing.assert_allclose(trimmed["timestamps"][-1], 0.5, atol=1e-10)
    # No NaN at boundaries
    assert not np.isnan(trimmed["eef_pos"][0, 0])
    assert not np.isnan(trimmed["eef_pos"][-1, 0])


def test_trim_no_boundary_nans():
    """No boundary NaN -> no trimming, same length."""
    tracking = {
        "timestamps": np.arange(5, dtype=np.float64) * 0.1,
        "eef_pos": np.zeros((5, 3)),
        "eef_rot": np.zeros((5, 3)),
        "gripper_width": np.full(5, 0.05),
        "confidence": np.ones(5),
    }
    # Interior NaN only
    tracking["eef_pos"][2] = np.nan
    tracking["eef_rot"][2] = np.nan

    trimmed = trim_nan_boundaries(tracking)
    assert len(trimmed["timestamps"]) == 5  # no trimming


def test_fill_tracking_result_gap_stats():
    """gap_stats should report correct max consecutive gap."""
    tracking = {
        "timestamps": np.arange(10, dtype=np.float64) * 0.1,
        "eef_pos": np.zeros((10, 3)),
        "eef_rot": np.zeros((10, 3)),
        "gripper_width": np.full(10, 0.05),
        "confidence": np.ones(10),
        "source": "aruco",
    }
    # Create a 3-frame gap at frames 4,5,6
    tracking["eef_pos"][4:7] = np.nan
    tracking["eef_rot"][4:7] = np.nan
    tracking["gripper_width"][5] = np.nan

    filled = fill_tracking_result(tracking)

    assert "gap_stats" in filled
    assert filled["gap_stats"]["max_consecutive_gap"] == 3
    assert filled["gap_stats"]["n_pos_missing"] == 3
    assert filled["gap_stats"]["n_width_missing"] == 1
    np.testing.assert_allclose(filled["gap_stats"]["detect_rate"], 0.7, atol=1e-10)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
