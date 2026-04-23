"""
Verify HaMeR inference environment end-to-end.

Checks every dependency in the inference chain:
  1. PyTorch + CUDA
  2. pyrender mock
  3. HaMeR package imports
  4. Checkpoint files exist
  5. MANO model files
  6. detectron2 + ViTDet config
  7. Load HaMeR model (GPU)
  8. Load ViTDet detector (auto-downloads checkpoint)
  9. Dummy inference (synthetic image)

Run:
    cd hand-6dof-pipeline
    python scripts/_verify_hamer_env.py
"""
import sys
import time
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ASSETS_HAMER = PROJECT_ROOT / "assets" / "hamer" / "_DATA"

errors = []
warnings = []

def check(name, func):
    """Run a check, print result."""
    try:
        result = func()
        print(f"  [OK] {name}" + (f" — {result}" if result else ""))
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        errors.append((name, str(e)))
        return False

def warn(name, msg):
    print(f"  [WARN] {name}: {msg}")
    warnings.append((name, msg))


# ============================================================
print("\n=== Step 1: PyTorch + CUDA ===")
# ============================================================
def check_torch():
    import torch
    cuda = torch.cuda.is_available()
    if cuda:
        return f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}"
    else:
        warn("CUDA", "No CUDA available — inference will be very slow on CPU")
        return f"torch {torch.__version__}, CPU only"
check("PyTorch", check_torch)


# ============================================================
print("\n=== Step 2: pyrender mock ===")
# ============================================================
def check_pyrender_mock():
    from utils.hand_tracker.hamer_backend import _mock_pyrender
    _mock_pyrender()
    import pyrender
    assert hasattr(pyrender, 'RenderFlags')
    assert hasattr(pyrender.RenderFlags, 'RGBA')
check("pyrender mock", check_pyrender_mock)


# ============================================================
print("\n=== Step 3: HaMeR package imports ===")
# ============================================================
def check_hamer_imports():
    import hamer
    import hamer.configs
    from hamer.models import HAMER, MANO
    from hamer.models import load_hamer, download_models
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    return f"hamer from {Path(hamer.__file__).parent}"
check("HaMeR imports", check_hamer_imports)


# ============================================================
print("\n=== Step 4: Checkpoint files ===")
# ============================================================
def check_hamer_ckpt():
    ckpt = ASSETS_HAMER / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    assert ckpt.exists(), f"Missing: {ckpt}"
    size_mb = ckpt.stat().st_size / 1024 / 1024
    return f"hamer.ckpt ({size_mb:.0f} MB)"
check("hamer.ckpt", check_hamer_ckpt)

def check_vitpose():
    vp = ASSETS_HAMER / "vitpose_ckpts" / "vitpose+_huge" / "wholebody.pth"
    assert vp.exists(), f"Missing: {vp}"
    size_mb = vp.stat().st_size / 1024 / 1024
    return f"wholebody.pth ({size_mb:.0f} MB)"
check("ViTPose checkpoint", check_vitpose)

def check_model_config():
    cfg = ASSETS_HAMER / "hamer_ckpts" / "model_config.yaml"
    assert cfg.exists(), f"Missing: {cfg}"
check("model_config.yaml", check_model_config)

def check_mano_mean():
    f = ASSETS_HAMER / "data" / "mano_mean_params.npz"
    assert f.exists(), f"Missing: {f}"
check("mano_mean_params.npz", check_mano_mean)


# ============================================================
print("\n=== Step 5: MANO model files ===")
# ============================================================
def check_mano_pkl():
    # MANO_RIGHT.pkl is required by smplx/MANO for forward kinematics
    mano_dir = ASSETS_HAMER / "data" / "mano"
    candidates = list(mano_dir.glob("MANO_RIGHT*")) if mano_dir.exists() else []
    if not candidates:
        # Check if smplx has its own MANO data
        try:
            import smplx
            smplx_dir = Path(smplx.__file__).parent
            # smplx might look for MANO in its own directory or in a configurable path
        except ImportError:
            pass
        warn("MANO_RIGHT",
             f"No MANO_RIGHT.pkl found in {mano_dir}. "
             "Download from https://mano.is.tue.mpg.de/ and place in assets/hamer/_DATA/data/mano/")
        return None
    return f"Found: {candidates[0].name}"

check_mano_result = check_mano_pkl()


# ============================================================
print("\n=== Step 6: detectron2 + ViTDet config ===")
# ============================================================
def check_detectron2():
    import detectron2
    from detectron2.config import LazyConfig
    import hamer
    hamer_dir = Path(hamer.__file__).parent
    cfg_path = hamer_dir / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    assert cfg_path.exists(), f"Missing ViTDet config: {cfg_path}"
    cfg = LazyConfig.load(str(cfg_path))
    return f"detectron2 {detectron2.__version__}, config loaded"
check("detectron2 + ViTDet config", check_detectron2)


# ============================================================
print("\n=== Step 7: Load HaMeR model ===")
# ============================================================
def check_load_hamer():
    import torch
    import hamer.configs
    from hamer.models import load_hamer

    # Override cache dir
    hamer.configs.CACHE_DIR_HAMER = str(ASSETS_HAMER)

    ckpt_path = str(ASSETS_HAMER / "hamer_ckpts" / "checkpoints" / "hamer.ckpt")
    t0 = time.time()
    model, model_cfg = load_hamer(checkpoint_path=ckpt_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    dt = time.time() - t0

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    return f"{n_params:.1f}M params, loaded to {device} in {dt:.1f}s"
check("Load HaMeR model", check_load_hamer)


# ============================================================
print("\n=== Step 8: Load ViTDet detector ===")
# ============================================================
def check_load_detector():
    import hamer
    from detectron2.config import LazyConfig
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    hamer_dir = Path(hamer.__file__).parent
    cfg_path = hamer_dir / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    cfg = LazyConfig.load(str(cfg_path))

    t0 = time.time()
    # This will auto-download model_final_f05665.pkl (~700MB) on first run
    predictor = DefaultPredictor_Lazy(cfg)
    dt = time.time() - t0
    return f"ViTDet loaded in {dt:.1f}s"
check("Load ViTDet detector", check_load_detector)


# ============================================================
print("\n=== Step 9: Dummy inference ===")
# ============================================================
def check_dummy_inference():
    import numpy as np
    from utils.hand_tracker.factory import create_tracker

    tracker = create_tracker("hamer")

    # Create a synthetic image (won't detect any hands, but tests the full pipeline)
    dummy_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    t0 = time.time()
    detections = tracker.detect(dummy_rgb)
    dt = time.time() - t0

    return f"Inference OK in {dt:.1f}s, detected {len(detections)} hands (expected 0 on noise)"
check("Dummy inference", check_dummy_inference)


# ============================================================
print("\n" + "=" * 60)
# ============================================================
if errors:
    print(f"FAILED: {len(errors)} error(s)")
    for name, msg in errors:
        print(f"  - {name}: {msg}")
else:
    print("ALL CHECKS PASSED")

if warnings:
    print(f"\nWARNINGS: {len(warnings)}")
    for name, msg in warnings:
        print(f"  - {name}: {msg}")

print()
sys.exit(1 if errors else 0)
