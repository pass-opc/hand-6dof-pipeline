"""
One Euro Filter for temporal smoothing of noisy hand tracking output.

Pipeline context:
  Applied after hand tracking (HaMeR / ArUco) to smooth 6DoF trajectories
  before feeding into retargeting / dataset generation.

Reference:
  Casiez, Roussel, Vogel, "1€ Filter: A Simple Speed-based Low-pass Filter
  for Noisy Input in Interactive Systems", CHI 2012.

Adapts cutoff frequency based on signal speed:
  - Slow motion → low cutoff → heavy smoothing (reduce jitter)
  - Fast motion → high cutoff → minimal lag (preserve responsiveness)

Classes:
  OneEuroFilter       — scalar signal filter
  VectorOneEuroFilter — per-dimension filter for N-d vectors
  PoseOneEuroFilter   — position (xyz) + rotation (axis-angle via quaternion slerp)
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    """Compute exponential smoothing factor alpha from time interval and cutoff freq.

    alpha = tau / (tau + t_e), where tau = 1 / (2*pi*cutoff)
    Rearranged: alpha = r / (r + 1) with r = 2*pi*cutoff*t_e
    """
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


class OneEuroFilter:
    """One Euro Filter for scalar signals.

    Adaptive low-pass filter whose cutoff frequency increases with signal speed,
    reducing jitter during slow motion while preserving responsiveness during
    fast motion.

    Args:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing.
        beta: Speed coefficient. Higher = less lag on fast motion.
        d_cutoff: Cutoff frequency for derivative estimation (Hz).
        freq: Sampling frequency (Hz). Used to compute dt when timestamps
              are not provided. If None, timestamps must be given to filter().
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        freq: float | None = None,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq = freq
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def reset(self) -> None:
        """Reset filter state (call between episodes)."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def filter(self, value: float, timestamp: float | None = None) -> float:
        """Filter one scalar sample.

        Args:
            value: Input signal value.
            timestamp: Time in seconds. If None, uses 1/freq increment.

        Returns:
            Filtered signal value.
        """
        # Determine timestamp
        if timestamp is None:
            if self.freq is None:
                raise ValueError("Either provide timestamp or set freq in constructor")
            if self._t_prev is None:
                timestamp = 0.0
            else:
                timestamp = self._t_prev + 1.0 / self.freq

        if self._x_prev is None:
            # First sample: passthrough, no filtering possible
            self._x_prev = value
            self._dx_prev = 0.0
            self._t_prev = timestamp
            return value

        t_e = timestamp - self._t_prev
        if t_e <= 0:
            return self._x_prev

        # Estimate derivative (speed) with low-pass filter
        dx = (value - self._x_prev) / t_e
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff: faster signal → higher cutoff → less smoothing
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Low-pass filter the signal
        a = _smoothing_factor(t_e, cutoff)
        x_hat = a * value + (1.0 - a) * self._x_prev

        # Update state
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = timestamp

        return x_hat

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        """Legacy vector interface: filter an ndarray at timestamp t.

        Kept for backward compatibility with existing code that passes
        numpy arrays directly. For new code, prefer VectorOneEuroFilter.
        """
        x = np.asarray(x, dtype=np.float64)

        if self._x_prev is None or not isinstance(self._x_prev, np.ndarray):
            # Re-init for array mode
            self._x_prev = None
            self._dx_prev = None

        return self._filter_array(t, x)

    def _filter_array(self, t: float, x: np.ndarray) -> np.ndarray:
        """Internal: filter an ndarray (used by __call__ for backward compat)."""
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x.copy()

        t_e = t - self._t_prev
        if t_e <= 0:
            return self._x_prev.copy()

        dx = (x - self._x_prev) / t_e
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        speed = np.linalg.norm(dx_hat) if dx_hat.ndim > 0 else abs(float(dx_hat))
        cutoff = self.min_cutoff + self.beta * speed

        a = _smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat.copy()
        self._dx_prev = dx_hat.copy()
        self._t_prev = t

        return x_hat


class VectorOneEuroFilter:
    """Applies OneEuroFilter independently to each dimension of an N-d vector.

    Args:
        ndim: Number of dimensions.
        min_cutoff: Minimum cutoff frequency (Hz).
        beta: Speed coefficient.
        d_cutoff: Cutoff for derivative smoothing (Hz).
        freq: Sampling frequency (Hz). If None, timestamps must be provided.
    """

    def __init__(
        self,
        ndim: int,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        freq: float | None = None,
    ):
        self.ndim = ndim
        self._filters = [
            OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff, freq=freq)
            for _ in range(ndim)
        ]

    def filter(self, value: np.ndarray, timestamp: float | None = None) -> np.ndarray:
        """Filter an N-d vector sample.

        Args:
            value: Input vector of shape (ndim,).
            timestamp: Time in seconds. If None, uses freq-based increment.

        Returns:
            Filtered vector of shape (ndim,).
        """
        value = np.asarray(value, dtype=np.float64)
        assert value.shape == (self.ndim,), f"Expected ({self.ndim},), got {value.shape}"
        return np.array([
            f.filter(float(v), timestamp) for f, v in zip(self._filters, value)
        ])

    def reset(self) -> None:
        """Reset all per-dimension filters."""
        for f in self._filters:
            f.reset()


class PoseOneEuroFilter:
    """One-Euro filter for 6DoF pose (position + axis-angle rotation).

    Position is filtered per-component via VectorOneEuroFilter(3).
    Rotation is filtered in quaternion space using slerp-based smoothing,
    then converted back to axis-angle.

    Args:
        min_cutoff: Minimum cutoff frequency (Hz).
        beta: Speed coefficient.
        d_cutoff: Cutoff for derivative smoothing (Hz).
        freq: Sampling frequency (Hz). If None, timestamps must be provided.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        freq: float | None = None,
    ):
        self._pos_filter = VectorOneEuroFilter(
            ndim=3, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff, freq=freq,
        )
        # Rotation filter params stored for slerp-based smoothing
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._quat_prev: np.ndarray | None = None
        self._angular_speed_prev: float = 0.0
        self._t_prev: float | None = None
        self.freq = freq

    def reset(self) -> None:
        """Reset both position and rotation filter states."""
        self._pos_filter.reset()
        self._quat_prev = None
        self._angular_speed_prev = 0.0
        self._t_prev = None

    def filter(
        self,
        pos: np.ndarray,
        rot: np.ndarray,
        timestamp: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter a 6DoF pose sample.

        Args:
            pos: (3,) position xyz.
            rot: (3,) axis-angle rotation vector.
            timestamp: Time in seconds. If None, uses freq-based increment.

        Returns:
            (filtered_pos, filtered_rot) — both as numpy arrays.
        """
        pos_filtered = self._pos_filter.filter(pos, timestamp)
        rot_filtered = self._filter_rotation(rot, timestamp)
        return pos_filtered, rot_filtered

    def _filter_rotation(self, rot: np.ndarray, timestamp: float | None) -> np.ndarray:
        """Filter rotation using slerp-based adaptive smoothing in quaternion space."""
        # Resolve timestamp
        if timestamp is None:
            if self.freq is None:
                raise ValueError("Either provide timestamp or set freq in constructor")
            if self._t_prev is None:
                timestamp = 0.0
            else:
                timestamp = self._t_prev + 1.0 / self.freq

        quat = Rotation.from_rotvec(rot).as_quat()  # [x, y, z, w] scipy convention

        if self._quat_prev is None:
            # First sample: passthrough
            self._quat_prev = quat.copy()
            self._angular_speed_prev = 0.0
            self._t_prev = timestamp
            return rot.copy()

        t_e = timestamp - self._t_prev
        if t_e <= 0:
            return Rotation.from_quat(self._quat_prev).as_rotvec()

        # Ensure quaternion hemisphere consistency (dot product > 0)
        if np.dot(quat, self._quat_prev) < 0:
            quat = -quat

        # Angular speed: angle between consecutive quaternions / dt
        # angle = 2 * arccos(|q1 . q2|)
        dot = np.clip(np.dot(quat, self._quat_prev), -1.0, 1.0)
        angle = 2.0 * math.acos(abs(dot))
        angular_speed = angle / t_e

        # Low-pass the angular speed estimate
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        angular_speed_hat = a_d * angular_speed + (1.0 - a_d) * self._angular_speed_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * angular_speed_hat

        # Compute alpha (slerp interpolation factor)
        alpha = _smoothing_factor(t_e, cutoff)

        # Slerp between previous filtered quat and current input quat
        # alpha=1 → fully current (no smoothing), alpha=0 → fully previous
        key_rots = Rotation.from_quat(np.stack([self._quat_prev, quat]))
        slerp_fn = Slerp([0.0, 1.0], key_rots)
        quat_filtered = slerp_fn(alpha).as_quat()

        # Normalize for safety
        quat_filtered = quat_filtered / np.linalg.norm(quat_filtered)

        # Update state
        self._quat_prev = quat_filtered.copy()
        self._angular_speed_prev = angular_speed_hat
        self._t_prev = timestamp

        return Rotation.from_quat(quat_filtered).as_rotvec()

    def __call__(
        self,
        t: float,
        pos: np.ndarray,
        rot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Legacy interface for backward compatibility."""
        return self.filter(pos, rot, timestamp=t)
