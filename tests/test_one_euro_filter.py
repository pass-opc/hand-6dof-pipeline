"""
Tests for utils/one_euro_filter.py

Covers:
  1. OneEuroFilter — constant signal, step, noisy sine, reset, high beta, timestamps
  2. VectorOneEuroFilter — per-dimension filtering
  3. PoseOneEuroFilter — position + rotation (slerp-based) filtering

All tests are synthetic (no hardware needed).

Run:
    cd hand-6dof-pipeline
    python -m pytest tests/test_one_euro_filter.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.one_euro_filter import OneEuroFilter, PoseOneEuroFilter, VectorOneEuroFilter

ATOL = 1e-6


# ============================================================
# 1. OneEuroFilter (scalar)
# ============================================================

class TestOneEuroFilter:

    def test_constant_signal_unchanged(self):
        """Constant signal should pass through with no distortion."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0, freq=60.0)
        outputs = []
        for _ in range(50):
            outputs.append(f.filter(5.0))
        # After warmup, output should equal input
        assert abs(outputs[-1] - 5.0) < 0.01

    def test_first_sample_passthrough(self):
        """First sample should be returned unchanged."""
        f = OneEuroFilter(freq=60.0)
        out = f.filter(42.0)
        assert out == 42.0

    def test_step_response_converges(self):
        """Step from 0 to 1: output should converge to 1.0 within N samples."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0, freq=60.0)
        # Feed 0.0 for 30 samples
        for _ in range(30):
            f.filter(0.0)
        # Step to 1.0
        outputs = []
        for _ in range(120):
            outputs.append(f.filter(1.0))

        # Should not jump immediately
        assert outputs[0] < 0.5
        # Should converge close to 1.0 by end
        assert outputs[-1] > 0.95

    def test_noisy_sine_reduced_variance(self):
        """Filtered noisy sine should have lower error than raw noisy input."""
        rng = np.random.default_rng(42)
        f = OneEuroFilter(min_cutoff=5.0, beta=0.0, freq=60.0)

        n = 120
        t_arr = np.linspace(0, 2.0, n)
        clean = np.sin(2 * np.pi * t_arr)
        noisy = clean + rng.normal(0, 0.3, n)

        filtered = np.array([f.filter(float(x)) for x in noisy])

        # Skip warmup (first 10 samples)
        noise_error = np.std(noisy[10:] - clean[10:])
        filter_error = np.std(filtered[10:] - clean[10:])
        assert filter_error < noise_error

    def test_reset_clears_state(self):
        """After reset, next sample should be returned unchanged (first-sample behavior)."""
        f = OneEuroFilter(freq=60.0)
        f.filter(1.0)
        f.filter(2.0)
        f.filter(3.0)
        f.reset()

        out = f.filter(99.0)
        assert out == 99.0

    def test_high_beta_faster_response(self):
        """Higher beta should track fast changes more aggressively."""
        # Low beta: heavy smoothing
        f_low = OneEuroFilter(min_cutoff=1.0, beta=0.0, freq=60.0)
        # High beta: responsive to speed
        f_high = OneEuroFilter(min_cutoff=1.0, beta=1.0, freq=60.0)

        # Feed constant then step
        for _ in range(30):
            f_low.filter(0.0)
            f_high.filter(0.0)

        # After step, high beta should be closer to target
        out_low = f_low.filter(10.0)
        out_high = f_high.filter(10.0)
        assert out_high > out_low, "High beta should respond faster to velocity changes"

    def test_explicit_timestamps(self):
        """Filter should work correctly with explicit timestamps."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        # Use irregular timestamps
        out0 = f.filter(5.0, timestamp=0.0)
        assert out0 == 5.0  # first sample passthrough

        out1 = f.filter(5.0, timestamp=0.05)
        assert abs(out1 - 5.0) < ATOL  # constant signal

        # Different timestamp spacing should still work
        out2 = f.filter(5.0, timestamp=0.2)
        assert abs(out2 - 5.0) < ATOL

    def test_no_freq_no_timestamp_raises(self):
        """If neither freq nor timestamp is provided, should raise ValueError."""
        f = OneEuroFilter()  # no freq
        with pytest.raises(ValueError, match="timestamp"):
            f.filter(1.0)

    def test_zero_time_delta_returns_previous(self):
        """Same timestamp should return previous filtered value."""
        f = OneEuroFilter()
        f.filter(1.0, timestamp=0.0)
        out = f.filter(5.0, timestamp=0.0)
        assert out == 1.0


# ============================================================
# 2. VectorOneEuroFilter
# ============================================================

class TestVectorOneEuroFilter:

    def test_constant_vector_unchanged(self):
        """Constant 3D vector should pass through with no distortion."""
        f = VectorOneEuroFilter(ndim=3, min_cutoff=1.0, beta=0.0, freq=60.0)
        for _ in range(50):
            out = f.filter(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0], atol=0.01)

    def test_first_sample_passthrough(self):
        """First sample should be returned unchanged."""
        f = VectorOneEuroFilter(ndim=3, freq=60.0)
        x = np.array([1.0, 2.0, 3.0])
        out = f.filter(x)
        np.testing.assert_array_equal(out, x)

    def test_per_dimension_filtering(self):
        """Each dimension should be filtered independently."""
        f = VectorOneEuroFilter(ndim=2, min_cutoff=1.0, beta=0.0, freq=60.0)
        # Feed (0, 5) then step dim0 to 10, dim1 stays at 5
        for _ in range(30):
            f.filter(np.array([0.0, 5.0]))
        out = f.filter(np.array([10.0, 5.0]))
        # dim0 should be smoothed (not jump to 10)
        assert out[0] < 8.0
        # dim1 should stay at 5
        assert abs(out[1] - 5.0) < 0.1

    def test_reset(self):
        """Reset should clear all dimension filters."""
        f = VectorOneEuroFilter(ndim=3, freq=60.0)
        f.filter(np.array([1.0, 2.0, 3.0]))
        f.filter(np.array([4.0, 5.0, 6.0]))
        f.reset()

        out = f.filter(np.array([7.0, 8.0, 9.0]))
        np.testing.assert_array_equal(out, [7.0, 8.0, 9.0])

    def test_wrong_shape_raises(self):
        """Input with wrong number of dimensions should raise AssertionError."""
        f = VectorOneEuroFilter(ndim=3, freq=60.0)
        with pytest.raises(AssertionError):
            f.filter(np.array([1.0, 2.0]))  # ndim=2, expected 3

    def test_with_explicit_timestamps(self):
        """VectorOneEuroFilter should accept explicit timestamps."""
        f = VectorOneEuroFilter(ndim=2, min_cutoff=1.0, beta=0.0)
        out0 = f.filter(np.array([1.0, 2.0]), timestamp=0.0)
        np.testing.assert_array_equal(out0, [1.0, 2.0])

        out1 = f.filter(np.array([1.0, 2.0]), timestamp=0.016)
        np.testing.assert_allclose(out1, [1.0, 2.0], atol=ATOL)


# ============================================================
# 3. PoseOneEuroFilter
# ============================================================

class TestPoseOneEuroFilter:

    def test_constant_pose_unchanged(self):
        """Constant pose should pass through unchanged."""
        f = PoseOneEuroFilter(min_cutoff=1.0, beta=0.0, freq=60.0)
        pos_in = np.array([0.1, -0.05, 0.5])
        rot_in = np.array([0.0, 0.0, 0.0])
        for _ in range(30):
            pos_out, rot_out = f.filter(pos_in, rot_in)
        np.testing.assert_allclose(pos_out, pos_in, atol=0.01)
        np.testing.assert_allclose(rot_out, rot_in, atol=0.01)

    def test_first_sample_passthrough(self):
        """First sample should be returned unchanged."""
        f = PoseOneEuroFilter(freq=60.0)
        pos_in = np.array([1.0, 2.0, 3.0])
        rot_in = np.array([0.1, 0.2, 0.3])
        pos_out, rot_out = f.filter(pos_in, rot_in)
        np.testing.assert_array_equal(pos_out, pos_in)
        np.testing.assert_allclose(rot_out, rot_in, atol=ATOL)

    def test_position_smoothed(self):
        """Position step should be smoothed, not passed through instantly."""
        f = PoseOneEuroFilter(min_cutoff=1.0, beta=0.0, freq=60.0)
        rot = np.array([0.0, 0.0, 0.0])

        for _ in range(30):
            f.filter(np.array([0.0, 0.0, 0.5]), rot)

        # Step position
        pos_out, _ = f.filter(np.array([1.0, 0.0, 0.5]), rot)
        assert pos_out[0] < 0.8, "Position step should be smoothed"

    def test_rotation_smoothed(self):
        """Rotation step should be smoothed via slerp."""
        f = PoseOneEuroFilter(min_cutoff=1.0, beta=0.0, freq=60.0)
        pos = np.zeros(3)

        # Feed zero rotation
        for _ in range(30):
            f.filter(pos, np.array([0.0, 0.0, 0.0]))

        # Step to 90 degrees around z-axis
        rot_step = np.array([0.0, 0.0, np.pi / 2])
        _, rot_out = f.filter(pos, rot_step)

        # Should not jump to full rotation immediately
        assert np.linalg.norm(rot_out) < np.pi / 2 * 0.8

    def test_rotation_normalized(self):
        """Filtered rotation should still be a valid rotation."""
        from scipy.spatial.transform import Rotation

        f = PoseOneEuroFilter(min_cutoff=1.0, beta=0.5, freq=60.0)
        rng = np.random.default_rng(123)
        for _ in range(30):
            rot_noisy = np.array([0.1, 0.0, 0.0]) + rng.normal(0, 0.01, 3)
            _, rot_out = f.filter(np.zeros(3), rot_noisy)

        # Should be convertible to a valid rotation matrix (det = 1)
        R = Rotation.from_rotvec(rot_out).as_matrix()
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-6

    def test_reset(self):
        """Reset should clear both position and rotation filters."""
        f = PoseOneEuroFilter(freq=60.0)
        f.filter(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        f.reset()

        pos_out, rot_out = f.filter(
            np.array([5.0, 0.0, 0.0]), np.array([0.1, 0.0, 0.0])
        )
        # First sample after reset should be passthrough
        np.testing.assert_array_equal(pos_out, [5.0, 0.0, 0.0])
        np.testing.assert_allclose(rot_out, [0.1, 0.0, 0.0], atol=ATOL)

    def test_with_explicit_timestamps(self):
        """PoseOneEuroFilter should work with explicit timestamps."""
        f = PoseOneEuroFilter(min_cutoff=1.0, beta=0.0)
        pos = np.array([0.1, 0.2, 0.3])
        rot = np.array([0.0, 0.0, 0.0])

        pos_out, rot_out = f.filter(pos, rot, timestamp=0.0)
        np.testing.assert_array_equal(pos_out, pos)

        pos_out2, rot_out2 = f.filter(pos, rot, timestamp=0.016)
        np.testing.assert_allclose(pos_out2, pos, atol=0.01)

    def test_legacy_call_interface(self):
        """Legacy __call__(t, pos, rot) interface should still work."""
        f = PoseOneEuroFilter(min_cutoff=1.0, beta=0.0)
        pos = np.array([0.1, 0.2, 0.3])
        rot = np.array([0.0, 0.0, 0.0])
        pos_out, rot_out = f(0.0, pos, rot)
        np.testing.assert_array_equal(pos_out, pos)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
