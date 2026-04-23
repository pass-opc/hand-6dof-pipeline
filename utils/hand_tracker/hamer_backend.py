"""
HaMeR backend for HandTracker.

Wraps the HaMeR model (ViT-H + MANO decoder) for 3D hand reconstruction.
Hand detection (bounding boxes) is delegated to a pluggable HandDetectorBase
(MediaPipe, ViTPose, etc.), decoupling detection from reconstruction.

Full pipeline:
  HandDetectorBase.detect_hands(rgb) → list[HandBox]
  HaMeRTracker._reconstruct(rgb, boxes) → list[HandDetection]

Dependencies: hamer, torch (lazy-imported in _ensure_loaded).

Reference:
  Pavlakos et al., "Reconstructing Hands in 3D with Transformers", CVPR 2024
  https://github.com/geopavlakos/hamer
"""

import sys
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .base import HandBox, HandDetectorBase, HandDetection, HandTracker

# HaMeR checkpoints live under assets/hamer/_DATA (not project root _DATA)
_ASSETS_HAMER_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "hamer" / "_DATA"


def _mock_pyrender():
    """Mock pyrender to bypass OpenGL/EGL on Windows.

    HaMeR's renderer.py imports pyrender at module level, but we never
    use rendering for inference. This mock prevents ImportError on
    headless or Windows systems without EGL/OSMesa.
    """
    if 'pyrender' in sys.modules:
        return

    mock = types.ModuleType('pyrender')

    # Mock classes must accept arbitrary args (HaMeR instantiates some, e.g. OffscreenRenderer)
    class _MockBase:
        def __init__(self, *args, **kwargs):
            pass

    for attr in ('Node', 'Mesh', 'Scene', 'Viewer', 'OffscreenRenderer',
                 'PerspectiveCamera', 'DirectionalLight', 'SpotLight',
                 'PointLight', 'MetallicRoughnessMaterial', 'Primitive',
                 'Trimesh', 'IntrinsicsCamera', 'RenderFlags'):
        setattr(mock, attr, type(attr, (_MockBase,), {}))
    mock.RenderFlags.RGBA = 1
    mock.RenderFlags.SHADOWS_DIRECTIONAL = 2
    mock.RenderFlags.ALL = 3
    sys.modules['pyrender'] = mock
    sys.modules['pyrender.constants'] = types.ModuleType('pyrender.constants')
    sys.modules['pyrender.constants'].RenderFlags = mock.RenderFlags


class HaMeRTracker(HandTracker):
    """HaMeR-based hand tracker.

    Uses a pluggable HandDetectorBase for 2D hand detection, then runs
    HaMeR (ViT-H + Transformer decoder + MANO) for 3D reconstruction.

    Args:
        detector: hand detector providing bounding boxes (MediaPipe, ViTPose, etc.)
        device: torch device string ("cuda" or "cpu")
        rescale_factor: bbox padding factor for HaMeR crop (default 2.0, matches demo.py)
        focal_length_px: focal length (pixels) used to convert HaMeR's
            weak-perspective pred_cam to full-image 3D translation. Pass the
            real camera intrinsic (e.g. iPhone ~1500) for metric-correct depth.
            None → HaMeR's demo default 5000; depth will be biased by
            f_real/5000 (~3× too large for iPhone).
    """

    def __init__(
        self,
        detector: HandDetectorBase,
        device: str = "cuda",
        rescale_factor: float = 2.0,
        focal_length_px: float | None = None,
    ):
        self._detector = detector
        self._device = device
        self._rescale_factor = rescale_factor
        self._focal_length_px = focal_length_px
        self._model = None
        self._model_cfg = None

    def set_focal_length_px(self, focal_length_px: float | None) -> None:
        """Override the focal length used for cam_crop_to_full.

        Allows passing a per-capture real focal without rebuilding the tracker.
        """
        self._focal_length_px = focal_length_px

    def _ensure_loaded(self):
        """Lazy-load HaMeR reconstruction model. Fails fast with clear error."""
        if self._model is not None:
            return

        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("PyTorch is required for HaMeR backend.")

        _mock_pyrender()

        try:
            import hamer.configs
            from hamer.models import load_hamer
        except ImportError:
            raise ImportError(
                "HaMeR is not installed. Install with:\n"
                "  pip install git+https://github.com/geopavlakos/hamer.git\n"
                "Or use a separate conda env (see docs/HAMER_PIPELINE_PLAN.md S11)."
            )

        # Override CACHE_DIR to assets/hamer/_DATA (organized project structure)
        cache_dir = str(_ASSETS_HAMER_DIR)
        hamer.configs.CACHE_DIR_HAMER = cache_dir

        # Verify checkpoints exist
        ckpt_path = f'{cache_dir}/hamer_ckpts/checkpoints/hamer.ckpt'
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"HaMeR checkpoint not found at {ckpt_path}.\n"
                "Run: python scripts/_download_hamer.py\n"
                "Then extract: cd assets/hamer/_DATA && tar -xf hamer_demo_data.tar.gz"
            )

        # Load HaMeR reconstruction model
        self._model, self._model_cfg = load_hamer(checkpoint_path=ckpt_path)
        self._model.eval().to(self._device)

    def detect(self, rgb: np.ndarray) -> list[HandDetection]:
        """Detect + reconstruct hands: detector → bbox → HaMeR → 6DoF.

        Args:
            rgb: (H, W, 3) uint8 RGB image

        Returns:
            List of HandDetection (up to 2, one per hand).
        """
        self._ensure_loaded()

        # Step 1: Hand detection (delegated to pluggable detector)
        hand_boxes = self._detector.detect_hands(rgb)
        if not hand_boxes:
            return []

        # Step 2: HaMeR 3D reconstruction from detected boxes
        return self._reconstruct(rgb, hand_boxes)

    def _reconstruct(self, rgb: np.ndarray, hand_boxes: list[HandBox]) -> list[HandDetection]:
        """Run HaMeR 3D reconstruction on detected hand bounding boxes.

        Args:
            rgb: (H, W, 3) uint8 RGB image
            hand_boxes: list of HandBox from detector

        Returns:
            List of HandDetection with 6DoF pose + MANO joints.
        """
        import torch
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.utils.renderer import cam_crop_to_full

        # Prepare arrays for ViTDetDataset
        boxes = np.array([hb.bbox for hb in hand_boxes], dtype=np.float32)
        right = np.array([1.0 if hb.is_right else 0.0 for hb in hand_boxes],
                         dtype=np.float32)
        scores = np.array([hb.confidence for hb in hand_boxes], dtype=np.float32)

        # ViTDetDataset expects BGR
        bgr = rgb[:, :, ::-1].copy()

        # Crop and preprocess for HaMeR
        dataset = ViTDetDataset(
            self._model_cfg,
            img_cv2=bgr,
            boxes=boxes,
            right=right,
            rescale_factor=self._rescale_factor,
            train=False,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=len(dataset), shuffle=False, num_workers=0,
        )
        batch = next(iter(dataloader))

        # Move batch to device
        batch_device = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # HaMeR inference
        with torch.no_grad():
            output = self._model(batch_device)

        # Convert crop-space camera params to full-image translation
        # pred_cam is weak-perspective (s, tx, ty) in crop; cam_crop_to_full
        # converts to (tx, ty, tz) in full image using box geometry
        # cam_crop_to_full needs all tensors on same device; move to CPU
        pred_cam = output['pred_cam'].cpu()
        box_center = batch['box_center'].float()
        box_size = batch['box_size'].float()
        img_size = batch['img_size'].float()
        # focal_length_px decouples HaMeR's depth from its demo-time virtual
        # f=5000. With real iPhone fx (~1500), t_z comes out metric-correct
        # without needing a post-hoc LiDAR scale rescue.
        focal = self._focal_length_px if self._focal_length_px is not None else 5000.0
        pred_cam_t_full = cam_crop_to_full(
            pred_cam, box_center, box_size, img_size, focal_length=focal,
        ).numpy()                                                  # (B, 3)
        pred_keypoints = output['pred_keypoints_3d'].cpu().numpy() # (B, 21, 3)
        pred_mano = output['pred_mano_params']
        global_orient = pred_mano['global_orient'].cpu().numpy()   # (B, 1, 3, 3)

        detections = []
        for i in range(len(hand_boxes)):
            wrist_pos = pred_cam_t_full[i]                     # (3,)
            rot_mat = global_orient[i, 0]                      # (3, 3)
            wrist_rot = Rotation.from_matrix(rot_mat).as_rotvec()
            joints_3d = pred_keypoints[i]                      # (21, 3)

            # Shift joints from hand-root-relative to camera frame
            joints_cam = joints_3d + wrist_pos[np.newaxis, :]

            is_right = hand_boxes[i].is_right
            handedness = "right" if is_right else "left"

            # Left hand was flipped for inference (MANO is right-hand only), flip back
            if not is_right:
                joints_cam[:, 0] = -joints_cam[:, 0]
                wrist_pos[0] = -wrist_pos[0]
                rot_mat_flipped = rot_mat.copy()
                rot_mat_flipped[0, 1] = -rot_mat_flipped[0, 1]
                rot_mat_flipped[0, 2] = -rot_mat_flipped[0, 2]
                rot_mat_flipped[1, 0] = -rot_mat_flipped[1, 0]
                rot_mat_flipped[2, 0] = -rot_mat_flipped[2, 0]
                wrist_rot = Rotation.from_matrix(rot_mat_flipped).as_rotvec()

            detections.append(HandDetection(
                handedness=handedness,
                wrist_pos=wrist_pos.astype(np.float64),
                wrist_rot=wrist_rot.astype(np.float64),
                joints_3d=joints_cam.astype(np.float64),
                confidence=float(scores[i]),
                bbox=hand_boxes[i].bbox.copy(),
            ))

        # Limit to max 2 hands (highest confidence)
        if len(detections) > 2:
            detections.sort(key=lambda d: d.confidence, reverse=True)
            detections = detections[:2]

        return detections

    def get_backend_name(self) -> str:
        return "hamer"
