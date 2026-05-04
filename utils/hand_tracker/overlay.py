"""
Hand-tracking visualization helpers for the 335 path.

Pipeline position: used by scripts335/01_track to render per-frame QA
overlays into a preview .mp4 so a human can eyeball whether HaMeR + depth
correction is working without cracking open the .npz.

Draw conventions:
  - bbox: green rectangle (right hand) / cyan (left hand)
  - 21 MANO keypoints: small circles
  - skeleton: lines connecting MANO joint topology
  - wrist axes: short RGB lines (X red, Y green, Z blue) showing 6DoF orientation
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils.hand_tracker.base import HandDetection


# MANO joint topology — 21 keypoints, 5 fingers + wrist.
# Adjacency for skeleton drawing.
_MANO_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
)

# Tip indices for highlighting (wrist + 5 fingertips).
_TIP_IDX = (0, 4, 8, 12, 16, 20)


def _project(p_cam: np.ndarray, K: np.ndarray) -> tuple[int, int] | None:
    """Pinhole projection: (3,) cam-frame metric → (px, py) pixel int.

    Returns None if behind camera (z<=0).
    """
    z = p_cam[2]
    if z <= 0 or not np.isfinite(z):
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return int(round(fx * p_cam[0] / z + cx)), int(round(fy * p_cam[1] / z + cy))


def draw_overlay(
    rgb: np.ndarray,
    detections: list[HandDetection],
    K: np.ndarray,
    *,
    hud_text: str | None = None,
) -> np.ndarray:
    """Render bbox + skeleton + wrist axes + HUD onto a copy of rgb.

    Returns BGR (cv2-friendly) so the caller can hand it directly to
    cv2.VideoWriter without further conversion.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for det in detections:
        is_right = det.handedness == "right"
        col_box = (0, 255, 0) if is_right else (255, 255, 0)  # right=green, left=cyan
        col_pt = (0, 255, 255) if is_right else (255, 200, 0)

        # MediaPipe detector bbox — thin gray for reference. It's the input
        # to HaMeR but is often loose / palm-only; the actual hand region
        # is the joint-cloud bbox below.
        if det.bbox is not None and np.all(np.isfinite(det.bbox)):
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (128, 128, 128), 1)
            cv2.putText(bgr, "MP", (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1, cv2.LINE_AA)

        # skeleton
        pts2d = [_project(det.joints_3d[i], K) for i in range(21)]

        # Joint-cloud bbox — tight around all 21 projected joints. This is
        # the actual hand extent in image space after depth-corrected 3D.
        valid_pts = [p for p in pts2d if p is not None]
        if valid_pts:
            xs = [p[0] for p in valid_pts]
            ys = [p[1] for p in valid_pts]
            x1, y1 = min(xs) - 8, min(ys) - 8
            x2, y2 = max(xs) + 8, max(ys) + 8
            cv2.rectangle(bgr, (x1, y1), (x2, y2), col_box, 2)
            label = f"{det.handedness[0].upper()} {det.confidence:.2f}"
            cv2.putText(bgr, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_box, 1, cv2.LINE_AA)
        for a, b in _MANO_BONES:
            if pts2d[a] is None or pts2d[b] is None:
                continue
            cv2.line(bgr, pts2d[a], pts2d[b], col_box, 1, cv2.LINE_AA)

        # joints
        for i, p in enumerate(pts2d):
            if p is None:
                continue
            r = 4 if i in _TIP_IDX else 2
            cv2.circle(bgr, p, r, col_pt, -1, cv2.LINE_AA)

        # wrist axes — 5cm long in cam frame, projected to pixels.
        # Build rotation matrix from HaMeR's axis-angle prediction; the
        # matrix representation is unique (no ±π wrap that the persisted
        # quaternion exists to avoid).
        wrist = det.joints_3d[0]
        from scipy.spatial.transform import Rotation
        R = Rotation.from_rotvec(det.wrist_rot).as_matrix()
        axis_len_m = 0.05
        for col_axis, axis_dir in (
            ((0, 0, 255), R[:, 0]),  # X red
            ((0, 255, 0), R[:, 1]),  # Y green
            ((255, 0, 0), R[:, 2]),  # Z blue
        ):
            tip = wrist + axis_dir * axis_len_m
            p0, p1 = _project(wrist, K), _project(tip, K)
            if p0 is None or p1 is None:
                continue
            cv2.arrowedLine(bgr, p0, p1, col_axis, 2, cv2.LINE_AA, tipLength=0.25)

    if hud_text:
        cv2.putText(bgr, hud_text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, hud_text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


class PreviewVideoWriter:
    """Lazy PyAV-backed H.264 mp4 writer.

    Why PyAV instead of cv2.VideoWriter:
      cv2.VideoWriter on conda's opencv build silently fails on H.264
      (avc1) — isOpened() returns True but the bundled libopenh264 hits a
      version mismatch and writes only a header. cv2 falls back to mp4v
      which Windows Movies & TV / Photos / HTML5 <video> refuse to play.
      PyAV uses ffmpeg's libx264 directly (same encoder lerobot uses for
      observation.images.rgb), so output plays in every native player.

    Frames in: BGR uint8 (cv2 convention from draw_overlay). Frames out:
      H.264 baseline mp4, yuv420p, fps from constructor. Lazy-opens on
      first frame so size is taken from the data instead of pre-declared.
    """

    def __init__(self, path: Path, fps: int):
        self.path = path
        self.fps = fps
        self._container = None    # av.container.OutputContainer
        self._stream = None       # av.video.stream.VideoStream
        self.frames_written = 0

    def _open(self, w: int, h: int) -> None:
        import av
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(self.path), mode="w")
        # libx264 ships with PyAV's bundled ffmpeg — no extra install needed.
        # yuv420p is mandatory for browser / Quick Look compatibility.
        stream = self._container.add_stream("libx264", rate=self.fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        # CRF 23 is libx264's default visually-transparent QP; matches the
        # quality range cv2's mp4v produced for QA review while staying
        # native-player friendly.
        stream.options = {"crf": "23", "preset": "veryfast"}
        self._stream = stream

    def write(self, bgr: np.ndarray) -> None:
        import av
        if self._container is None:
            h, w = bgr.shape[:2]
            self._open(w, h)
        # PyAV's VideoFrame.from_ndarray expects RGB; convert here so the
        # call site can stay BGR (cv2's convention used everywhere upstream).
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.frames_written += 1

    def close(self) -> None:
        if self._container is None:
            return
        # Flush the encoder — without this the trailing GOP is dropped.
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._container = None
        self._stream = None
