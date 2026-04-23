"""
Factory function for creating HandTracker instances by backend name.

Supports lazy import so heavy dependencies (HaMeR, WiLoR) are only loaded
when the corresponding backend is requested.

Detection (hand localization) and reconstruction (3D pose) are separate:
  - detector: "mediapipe" (default), "vitpose" (future)
  - backend:  "hamer" (default), "wilor" (future)
"""

from .base import HandDetectorBase, HandTracker


def create_detector(detector: str = "mediapipe", **kwargs) -> HandDetectorBase:
    """Create a HandDetectorBase by detector name.

    Args:
        detector: "mediapipe" or "vitpose" (future)
        **kwargs: passed to detector constructor

    Raises:
        ValueError: unknown detector name
        ImportError: detector dependencies not installed
    """
    if detector == "mediapipe":
        from .mediapipe_detector import MediaPipeHandDetector
        return MediaPipeHandDetector(**kwargs)
    elif detector == "vitpose":
        raise ImportError(
            "ViTPose detector not yet implemented. "
            "Requires mmpose + ViTPose — see docs/HAMER_PIPELINE_PLAN.md"
        )
    else:
        raise ValueError(
            f"Unknown hand detector: '{detector}'. "
            f"Available: mediapipe, vitpose (future)"
        )


def create_tracker(
    backend: str = "hamer",
    detector: str = "mediapipe",
    **kwargs,
) -> HandTracker:
    """Create a HandTracker instance by backend name.

    Args:
        backend: "hamer" or "wilor" (future)
        detector: "mediapipe" or "vitpose" (future)
        **kwargs: passed to backend constructor (device, rescale_factor, etc.)

    Raises:
        ValueError: unknown backend name
        ImportError: backend dependencies not installed
    """
    if backend == "hamer":
        from .hamer_backend import HaMeRTracker
        det = create_detector(detector)
        return HaMeRTracker(detector=det, **kwargs)
    elif backend == "wilor":
        raise ImportError(
            "WiLoR backend not yet implemented. "
            "Coming soon — see docs/HaMeR/HaMeR_Ecosystem.md"
        )
    else:
        raise ValueError(
            f"Unknown hand tracker backend: '{backend}'. "
            f"Available: hamer, wilor (future)"
        )
