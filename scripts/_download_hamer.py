"""Download HaMeR model checkpoints."""
import sys
import types

# Mock pyrender for Windows
mock = types.ModuleType('pyrender')
class _MockBase:
    def __init__(self, *args, **kwargs): pass
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

from pathlib import Path
import hamer.configs
from hamer.models import download_models

# Override cache dir to assets/hamer/_DATA
cache_dir = str(Path(__file__).resolve().parent.parent / "assets" / "hamer" / "_DATA")
hamer.configs.CACHE_DIR_HAMER = cache_dir

print(f"Cache dir: {cache_dir}")
print("Downloading HaMeR checkpoints...")
download_models(folder=cache_dir)
print("Done.")
