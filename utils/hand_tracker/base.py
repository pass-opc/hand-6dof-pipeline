"""
Abstract interface for bare-hand 6DoF tracking from RGB images.

Provides:
  - HandBox: 2D hand bounding box from a hand detector
  - HandDetectorBase: ABC for hand detection (localization only, no 3D)
  - HandDetection: per-hand, per-frame 3D detection result (6DoF + joints)
  - HandTracker: ABC for full 3D tracking backends (HaMeR, WiLoR, etc.)

Pipeline position: used by scripts/01_hand_track.py (main line)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# =============================================================================
# Hand detection (2D localization)
# =============================================================================

@dataclass
class HandBox:
    """2D hand bounding box from a hand detector.

    This is the interface between detection (MediaPipe, ViTPose, etc.)
    and 3D reconstruction (HaMeR, WiLoR, etc.).

    Attributes:
        bbox: (4,) array [x1, y1, x2, y2] in pixel coordinates
        is_right: True for right hand, False for left
        confidence: detection confidence in [0, 1]
    """
    bbox: np.ndarray       # (4,) [x1, y1, x2, y2]
    is_right: bool
    confidence: float

    def __post_init__(self):
        assert self.bbox.shape == (4,), \
            f"bbox shape must be (4,), got {self.bbox.shape}"


class HandDetectorBase(ABC):
    """Abstract base class for 2D hand detection (localization only).

    Subclasses provide hand bounding boxes from different backends
    (MediaPipe, ViTPose, etc.). The boxes are consumed by a 3D
    reconstruction model (HaMeR, WiLoR) — this decouples detection
    from reconstruction so either can be swapped independently.
    """

    @abstractmethod
    def detect_hands(self, rgb: np.ndarray) -> list[HandBox]:
        """Detect hands in a single RGB frame.

        Args:
            rgb: (H, W, 3) uint8 RGB image

        Returns:
            List of HandBox (bounding boxes + handedness). Max 2.
        """
        ...

    @abstractmethod
    def get_detector_name(self) -> str:
        """Return detector identifier (e.g. 'mediapipe', 'vitpose')."""
        ...


# =============================================================================
# Hand tracking (full 3D)
# =============================================================================

@dataclass
class HandDetection:
    """Single hand detection result from one RGB frame.

    Attributes:
        handedness: "left" or "right"
        wrist_pos: (3,) position in camera frame (meters)
        wrist_rot: (3,) axis-angle rotation (Rodrigues vector)
        joints_3d: (21, 3) MANO joint positions in camera frame (meters)
        confidence: detection confidence in [0, 1]
        bbox: optional (4,) array [x1, y1, x2, y2] in pixels, carried from detector
    """
    handedness: str
    wrist_pos: np.ndarray
    wrist_rot: np.ndarray
    joints_3d: np.ndarray
    confidence: float
    bbox: np.ndarray | None = None

    @property
    def gripper_width(self) -> float:
        """Thumb tip (joint 4) to index tip (joint 8) distance in meters.

        Standard approach for mapping bare-hand aperture to gripper width.
        Used by EasyMimic, Robotic Telekinesis, and others.
        """
        return float(np.linalg.norm(self.joints_3d[4] - self.joints_3d[8]))

    def __post_init__(self):
        assert self.handedness in ("left", "right"), \
            f"handedness must be 'left' or 'right', got '{self.handedness}'"
        assert self.wrist_pos.shape == (3,), \
            f"wrist_pos shape must be (3,), got {self.wrist_pos.shape}"
        assert self.wrist_rot.shape == (3,), \
            f"wrist_rot shape must be (3,), got {self.wrist_rot.shape}"
        assert self.joints_3d.shape == (21, 3), \
            f"joints_3d shape must be (21, 3), got {self.joints_3d.shape}"


class HandTracker(ABC):
    """Abstract base class for bare-hand 6DoF tracking backends.

    Subclasses implement detect() for a specific model (HaMeR, WiLoR, etc.).
    The factory function create_tracker() instantiates the appropriate backend.

    Design follows the same ABC pattern as robots/base.py (RobotArm).
    """

    @abstractmethod
    def detect(self, rgb: np.ndarray) -> list[HandDetection]:
        """Detect hands in a single RGB frame.

        Args:
            rgb: (H, W, 3) uint8 RGB image

        Returns:
            List of HandDetection objects (0, 1, or 2 entries).
            Empty list if no hands detected.
        """
        ...

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return backend identifier string (e.g. 'hamer', 'wilor')."""
        ...

    def set_focal_length_px(self, focal_length_px: float | None) -> None:
        """Override the focal length used for weak-perspective → 3D lift.

        Backends that project in crop-space (HaMeR's cam_crop_to_full) need
        the real intrinsic to land metric-correct depth. Default no-op for
        backends that don't need it — concrete trackers override when relevant.
        """
        pass
