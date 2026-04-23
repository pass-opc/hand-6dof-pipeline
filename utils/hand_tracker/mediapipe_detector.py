"""
MediaPipe-based hand detector for HandDetectorBase.

Outputs 2D hand bounding boxes + handedness from MediaPipe Hands.
Does NOT produce 3D reconstruction — that is done by the downstream
HaMeR/WiLoR tracker which consumes these boxes.

Dependencies: mediapipe (pip install mediapipe)
"""

import numpy as np

from .base import HandBox, HandDetectorBase


class MediaPipeHandDetector(HandDetectorBase):
    """Hand detector using MediaPipe Hands.

    Detects hand landmarks via MediaPipe, then computes tight bounding
    boxes from the 21 2D landmarks. Works well for close-up hand views
    (e.g. top-down table manipulation) where ViTDet person detection fails.

    Args:
        min_detection_confidence: MediaPipe detection threshold
        max_num_hands: maximum hands to detect (1 or 2)
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.3,
        max_num_hands: int = 2,
    ):
        self._min_conf = min_detection_confidence
        self._max_hands = max_num_hands
        self._detector = None

    def _ensure_loaded(self):
        if self._detector is not None:
            return

        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "MediaPipe is not installed. Install with:\n"
                "  pip install mediapipe"
            )

        # MediaPipe >= 0.10 uses tasks API
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions, RunningMode,
        )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self._get_model_path(),
            ),
            running_mode=RunningMode.IMAGE,
            num_hands=self._max_hands,
            min_hand_detection_confidence=self._min_conf,
            min_hand_presence_confidence=self._min_conf,
        )
        self._detector = HandLandmarker.create_from_options(options)

    def _get_model_path(self) -> str:
        """Get path to MediaPipe hand_landmarker.task model file.

        Downloads from Google if not present in assets/mediapipe/.
        """
        from pathlib import Path
        import urllib.request

        assets_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "assets" / "mediapipe"
        )
        assets_dir.mkdir(parents=True, exist_ok=True)
        model_path = assets_dir / "hand_landmarker.task"

        if not model_path.exists():
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/"
                "hand_landmarker.task"
            )
            print(f"Downloading MediaPipe hand model to {model_path}...")
            urllib.request.urlretrieve(url, str(model_path))
            print("Done.")

        return str(model_path)

    def detect_hands(self, rgb: np.ndarray) -> list[HandBox]:
        """Detect hands → bounding boxes + handedness.

        Args:
            rgb: (H, W, 3) uint8 RGB image

        Returns:
            List of HandBox with tight bbox around hand landmarks.
        """
        import mediapipe as mp

        self._ensure_loaded()

        h, w = rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        boxes = []
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            # Extract 2D pixel coordinates from normalized landmarks
            xs = [lm.x * w for lm in hand_landmarks]
            ys = [lm.y * h for lm in hand_landmarks]

            # Tight bbox from landmark extent
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            # Handedness: MediaPipe labels from camera's perspective
            # (mirrored), so "Right" in MediaPipe = right hand in image
            handedness_label = result.handedness[i][0].category_name
            is_right = (handedness_label == "Right")
            confidence = result.handedness[i][0].score

            boxes.append(HandBox(
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                is_right=is_right,
                confidence=confidence,
            ))

        return boxes

    def get_detector_name(self) -> str:
        return "mediapipe"
