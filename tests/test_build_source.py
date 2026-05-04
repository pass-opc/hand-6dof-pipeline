"""Pytest suite for scripts/03_build_source.py (iPhone-line source dataset, v2).

Covers v2 schema (world-frame, bimanual stacked):
  - make_features keys / shapes / dtypes
  - make_frame_builder bimanual stacking (L = idx 0, R = idx 1)
  - placeholder fallback for invalid hands (no NaN reaches LeRobot)
  - action stacking (next-frame state passthrough from FrameContext)
  - depth uint16_mm / uint8_cm encoding paths
  - confidence + T_world_cam passthrough
  - discover_episodes file enumeration

End-to-end (build_lerobot_dataset on a real .r3d) is exercised by manually
running scripts/03_build_source.py on a smoke episode.

Run: pytest tests/test_build_source.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# 03_build_source.py starts with a digit — import via spec.
_BUILD_SOURCE_PATH = _PROJECT_ROOT / "scripts" / "03_build_source.py"
_spec = importlib.util.spec_from_file_location(
    "iphone_03_build_source", _BUILD_SOURCE_PATH,
)
build_source = importlib.util.module_from_spec(_spec)
sys.modules["iphone_03_build_source"] = build_source
_spec.loader.exec_module(build_source)  # type: ignore[union-attr]

from utils.dataset.core import FrameContext   # noqa: E402


# =============================================================================
# make_features (v2 schema)
# =============================================================================

def test_features_schema_has_all_v2_keys() -> None:
    f = build_source.make_features(
        (480, 640), include_depth=True, depth_encoding="uint16_mm",
    )
    expected = {
        "observation.images.rgb",
        "observation.depth",
        # State (world-frame, bimanual stacked)
        "observation.state.wrist_pose",
        "observation.state.wrist_valid",
        "observation.state.hand_keypoints",
        "observation.state.gripper",
        # Confidence
        "observation.left_confidence",
        "observation.right_confidence",
        # Action
        "action.wrist_pose",
        "action.gripper",
        # Per-frame extrinsics
        "observation.T_world_cam",
    }
    assert set(f.keys()) == expected


def test_features_omits_depth_when_disabled() -> None:
    f = build_source.make_features(
        (480, 640), include_depth=False, depth_encoding="uint16_mm",
    )
    assert "observation.depth" not in f
    assert "observation.images.rgb" in f


def test_features_state_action_shapes_are_bimanual() -> None:
    """v2 state/action use AgiBot-style (2, ...) bimanual leading dim."""
    f = build_source.make_features(
        (480, 640), include_depth=True, depth_encoding="uint16_mm",
    )
    assert f["observation.state.wrist_pose"]["shape"] == (2, 7)
    assert f["observation.state.hand_keypoints"]["shape"] == (2, 21, 3)
    assert f["observation.state.gripper"]["shape"] == (2,)
    assert f["observation.state.wrist_valid"]["shape"] == (2,)
    assert f["action.wrist_pose"]["shape"] == (2, 7)
    assert f["action.gripper"]["shape"] == (2,)


def test_features_v1_legacy_keys_removed() -> None:
    """v2 deliberately drops v1 cam-frame fields. Customers reading 02 npz
    directly still get cam-frame; 03 source dataset is world-frame only."""
    f = build_source.make_features(
        (480, 640), include_depth=True, depth_encoding="uint16_mm",
    )
    for legacy in ("observation.wrist_pose_left", "observation.wrist_pose_right",
                    "observation.mano_joints_left", "observation.mano_joints_right",
                    "observation.left_valid", "observation.right_valid"):
        assert legacy not in f, f"v2 should not emit {legacy}"


def test_features_depth_uint16_mm() -> None:
    f = build_source.make_features(
        (480, 640), include_depth=True, depth_encoding="uint16_mm",
    )
    depth = f["observation.depth"]
    assert depth["dtype"] == "uint16"
    assert depth["shape"] == (480, 640)


def test_features_depth_uint8_cm() -> None:
    f = build_source.make_features(
        (480, 640), include_depth=True, depth_encoding="uint8_cm",
    )
    depth = f["observation.depth"]
    assert depth["dtype"] == "image"
    assert depth["shape"] == (480, 640, 3)


def test_features_depth_encoding_unknown_raises() -> None:
    with pytest.raises(ValueError, match="depth_encoding"):
        build_source.make_features(
            (480, 640), include_depth=True, depth_encoding="bogus",
        )


def test_features_t_world_cam_shape() -> None:
    f = build_source.make_features(
        (480, 640), include_depth=True, depth_encoding="uint16_mm",
    )
    twc = f["observation.T_world_cam"]
    assert twc["dtype"] == "float32"
    assert twc["shape"] == (4, 4)


# =============================================================================
# Frame-builder fixtures (v2)
# =============================================================================

def _ctx(*, t: int = 0,
          rgb_size: tuple[int, int] = (480, 640),
          include_depth: bool = True,
          left_valid: bool = True,
          right_valid: bool = True,
          left_confidence: float | None = None,
          right_confidence: float | None = None,
          T_world_cam: np.ndarray | None = None,
          # v2 world-frame fields
          left_wrist_pose_world: np.ndarray | None = None,
          right_wrist_pose_world: np.ndarray | None = None,
          left_hand_keypoints_world: np.ndarray | None = None,
          right_hand_keypoints_world: np.ndarray | None = None,
          left_gripper: float | None = None,
          right_gripper: float | None = None,
          left_action_wrist_pose_world: np.ndarray | None = None,
          right_action_wrist_pose_world: np.ndarray | None = None,
          left_action_gripper: float | None = None,
          right_action_gripper: float | None = None) -> FrameContext:
    rgb = np.full(rgb_size + (3,), 100, dtype=np.uint8)
    depth = (np.full(rgb_size, 500, dtype=np.uint16)
              if include_depth else None)

    # Default valid hand state — v2 fields populated when valid
    if left_valid and left_wrist_pose_world is None:
        left_wrist_pose_world = np.array([0.1, 0.2, 0.3, 0, 0, 0, 1], dtype=np.float32)
    if left_valid and left_hand_keypoints_world is None:
        left_hand_keypoints_world = np.full((21, 3), 0.5, dtype=np.float32)
    if left_valid and left_gripper is None:
        left_gripper = 0.4
    if right_valid and right_wrist_pose_world is None:
        right_wrist_pose_world = np.array([0.4, 0.5, 0.6, 0, 0, 0, 1], dtype=np.float32)
    if right_valid and right_hand_keypoints_world is None:
        right_hand_keypoints_world = np.full((21, 3), 0.7, dtype=np.float32)
    if right_valid and right_gripper is None:
        right_gripper = 0.6

    if left_confidence is None:
        left_confidence = 0.95 if left_valid else 0.0
    if right_confidence is None:
        right_confidence = 0.95 if right_valid else 0.0
    if T_world_cam is None:
        T_world_cam = np.eye(4, dtype=np.float64)

    return FrameContext(
        sid="test", t=t,
        rgb=rgb, depth=depth, K=np.eye(3),
        T_world_cam=T_world_cam,
        left_valid=left_valid, right_valid=right_valid,
        left_confidence=left_confidence, right_confidence=right_confidence,
        # v1 cam-frame still populated by core.build_per_hand_fields but v2
        # frame_builder doesn't read them; pass None to mirror invalid path.
        left_wrist_pose=None, right_wrist_pose=None,
        left_mano_joints=None, right_mano_joints=None,
        # v2 fields
        left_wrist_pose_world=left_wrist_pose_world,
        right_wrist_pose_world=right_wrist_pose_world,
        left_hand_keypoints_world=left_hand_keypoints_world,
        right_hand_keypoints_world=right_hand_keypoints_world,
        left_gripper=left_gripper,
        right_gripper=right_gripper,
        left_action_wrist_pose_world=left_action_wrist_pose_world,
        right_action_wrist_pose_world=right_action_wrist_pose_world,
        left_action_gripper=left_action_gripper,
        right_action_gripper=right_action_gripper,
    )


# =============================================================================
# frame_builder behavior (v2)
# =============================================================================

def test_frame_builder_bimanual_stack_order_is_left_then_right() -> None:
    """state.wrist_pose[0] = left, [1] = right (matches AgiBot convention)."""
    fb = build_source.make_frame_builder(
        (480, 640), include_depth=True, depth_encoding="uint16_mm",
    )
    out = fb(_ctx(left_valid=True, right_valid=True))
    np.testing.assert_array_equal(
        out["observation.state.wrist_pose"][0],
        np.array([0.1, 0.2, 0.3, 0, 0, 0, 1], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        out["observation.state.wrist_pose"][1],
        np.array([0.4, 0.5, 0.6, 0, 0, 0, 1], dtype=np.float32),
    )


def test_frame_builder_state_validity_flags() -> None:
    fb = build_source.make_frame_builder(
        (480, 640), include_depth=True, depth_encoding="uint16_mm",
    )
    # Both valid
    out = fb(_ctx(left_valid=True, right_valid=True))
    np.testing.assert_array_equal(
        out["observation.state.wrist_valid"], [1.0, 1.0],
    )
    # Left invalid only
    out = fb(_ctx(left_valid=False, right_valid=True))
    np.testing.assert_array_equal(
        out["observation.state.wrist_valid"], [0.0, 1.0],
    )


def test_frame_builder_placeholder_for_invalid_hand() -> None:
    """Invalid wrist gets identity quat placeholder; invalid hand_keypoints
    gets zeros. No NaN reaches LeRobot."""
    fb = build_source.make_frame_builder(
        (480, 640), include_depth=True, depth_encoding="uint16_mm",
    )
    out = fb(_ctx(left_valid=False, right_valid=True))
    expected_placeholder = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    np.testing.assert_array_equal(
        out["observation.state.wrist_pose"][0], expected_placeholder,
    )
    np.testing.assert_array_equal(
        out["observation.state.hand_keypoints"][0],
        np.zeros((21, 3), dtype=np.float32),
    )
    assert out["observation.state.gripper"][0] == 0.0


def test_frame_builder_gripper_passthrough() -> None:
    fb = build_source.make_frame_builder(
        (480, 640), include_depth=True, depth_encoding="uint16_mm",
    )
    out = fb(_ctx(left_gripper=0.25, right_gripper=0.75))
    np.testing.assert_allclose(
        out["observation.state.gripper"], [0.25, 0.75], atol=1e-6,
    )


def test_frame_builder_action_passthrough() -> None:
    """Action fields land in action.wrist_pose / action.gripper if FrameContext
    has them; otherwise placeholder."""
    fb = build_source.make_frame_builder(
        (480, 640), include_depth=True, depth_encoding="uint16_mm",
    )
    a_left = np.array([1, 2, 3, 0, 0, 0, 1], dtype=np.float32)
    a_right = np.array([4, 5, 6, 0, 0, 0, 1], dtype=np.float32)
    out = fb(_ctx(
        left_action_wrist_pose_world=a_left,
        right_action_wrist_pose_world=a_right,
        left_action_gripper=0.3,
        right_action_gripper=0.8,
    ))
    np.testing.assert_array_equal(out["action.wrist_pose"][0], a_left)
    np.testing.assert_array_equal(out["action.wrist_pose"][1], a_right)
    np.testing.assert_allclose(
        out["action.gripper"], [0.3, 0.8], atol=1e-6,
    )


def test_frame_builder_action_placeholder_when_absent() -> None:
    """If FrameContext has no action fields (last frame, or v1 npz), action
    falls back to placeholder, not None."""
    fb = build_source.make_frame_builder(
        (480, 640), include_depth=True, depth_encoding="uint16_mm",
    )
    out = fb(_ctx())  # no action fields supplied
    np.testing.assert_array_equal(
        out["action.wrist_pose"][0],
        np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
    )
    assert out["action.gripper"][0] == 0.0


def test_frame_builder_includes_rgb_and_depth_uint16() -> None:
    fb = build_source.make_frame_builder(
        (480, 640), include_depth=True, depth_encoding="uint16_mm",
    )
    out = fb(_ctx())
    assert out["observation.images.rgb"].shape == (480, 640, 3)
    assert out["observation.images.rgb"].dtype == np.uint8
    assert out["observation.depth"].shape == (480, 640)
    assert out["observation.depth"].dtype == np.uint16


def test_frame_builder_depth_uint8_cm_is_3channel() -> None:
    fb = build_source.make_frame_builder(
        (480, 640), include_depth=True, depth_encoding="uint8_cm",
    )
    out = fb(_ctx())
    depth = out["observation.depth"]
    assert depth.shape == (480, 640, 3)
    assert depth.dtype == np.uint8


def test_frame_builder_no_depth_when_disabled() -> None:
    fb = build_source.make_frame_builder(
        (480, 640), include_depth=False, depth_encoding="uint16_mm",
    )
    out = fb(_ctx(include_depth=False))
    assert "observation.depth" not in out


def test_frame_builder_dtypes_are_float32() -> None:
    """Trainers are picky; everything non-image must be float32."""
    fb = build_source.make_frame_builder(
        (480, 640), include_depth=True, depth_encoding="uint16_mm",
    )
    out = fb(_ctx())
    for key in ("observation.state.wrist_pose",
                "observation.state.wrist_valid",
                "observation.state.hand_keypoints",
                "observation.state.gripper",
                "action.wrist_pose",
                "action.gripper",
                "observation.left_confidence",
                "observation.right_confidence",
                "observation.T_world_cam"):
        assert out[key].dtype == np.float32, f"{key} dtype: {out[key].dtype}"


def test_frame_builder_confidence_passes_through() -> None:
    fb = build_source.make_frame_builder(
        (480, 640), include_depth=True, depth_encoding="uint16_mm",
    )
    out = fb(_ctx(left_confidence=0.0, right_confidence=0.78))
    assert out["observation.left_confidence"][0] == 0.0
    assert abs(out["observation.right_confidence"][0] - 0.78) < 1e-6


def test_frame_builder_t_world_cam_passthrough() -> None:
    T = np.array([
        [0.0, 1.0, 0.0, 1.5],
        [-1.0, 0.0, 0.0, 2.5],
        [0.0, 0.0, 1.0, 3.5],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    fb = build_source.make_frame_builder(
        (480, 640), include_depth=True, depth_encoding="uint16_mm",
    )
    out = fb(_ctx(T_world_cam=T))
    assert out["observation.T_world_cam"].shape == (4, 4)
    np.testing.assert_allclose(
        out["observation.T_world_cam"], T.astype(np.float32),
    )


# =============================================================================
# discover_episodes
# =============================================================================

def test_discover_episodes_finds_all(tmp_path: Path) -> None:
    for sid in ("ep1", "ep2", "ep3"):
        d = tmp_path / sid
        d.mkdir()
        (d / f"{sid}.processed.npz").write_bytes(b"")
    paths = build_source.discover_episodes(tmp_path, None)
    assert len(paths) == 3
    assert {p.parent.name for p in paths} == {"ep1", "ep2", "ep3"}


def test_discover_episodes_filters_by_list(tmp_path: Path) -> None:
    for sid in ("ep1", "ep2", "ep3"):
        d = tmp_path / sid
        d.mkdir()
        (d / f"{sid}.processed.npz").write_bytes(b"")
    paths = build_source.discover_episodes(tmp_path, ["ep2"])
    assert len(paths) == 1
    assert paths[0].parent.name == "ep2"
