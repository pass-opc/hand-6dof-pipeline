"""
Tests for scripts/03_build_dataset.py (packaging layer).

After the 4-layer refactor, 03 is a pure packaging script: no processing
logic lives here. Coverage:
  1. _validate_processed — coord_frame='episode_local' contract
  2. make_features — depth on/off toggles the depth image key
  3. _append_custom_episode_fields — merges extras into LeRobot v3's
     meta/episodes/**/*.parquet, keyed by episode_index

Processing-layer checks (extract/center/quality/states) now live in
tests/test_process.py.

Run:
    cd hand-6dof-pipeline
    python -m pytest tests/test_build_dataset.py -v
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_spec = importlib.util.spec_from_file_location(
    "build_dataset",
    str(Path(__file__).resolve().parent.parent / "scripts" / "03_build_dataset.py"),
)
_bd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bd)

_validate_processed = _bd._validate_processed
make_features = _bd.make_features
_append_custom_episode_fields = _bd._append_custom_episode_fields


# ============================================================
# Helpers
# ============================================================

def _make_processed(
    coord_frame: str = "episode_local",
    has_left: bool = True,
    has_right: bool = True,
) -> dict:
    out = {
        "episode_name": "ep0",
        "source":       "mock",
    }
    if coord_frame is not None:
        out["coord_frame"] = coord_frame
    if has_left:
        out["left_hand"] = {"quality_passed": False, "skip_reason": ""}
    if has_right:
        out["right_hand"] = {"quality_passed": False, "skip_reason": ""}
    return out


# ============================================================
# 1. _validate_processed
# ============================================================

class TestValidateProcessed:

    def test_accepts_episode_local_dual_hand(self):
        p = _make_processed()
        _validate_processed(p, "ep0")  # no raise

    def test_rejects_missing_coord_frame(self):
        p = _make_processed(coord_frame=None)
        with pytest.raises(ValueError, match="coord_frame"):
            _validate_processed(p, "ep0")

    def test_rejects_wrong_coord_frame(self):
        p = _make_processed(coord_frame="camera")
        with pytest.raises(ValueError, match="coord_frame"):
            _validate_processed(p, "ep0")

    def test_rejects_missing_hand(self):
        p = _make_processed(has_left=False)
        with pytest.raises(ValueError, match="dual-hand"):
            _validate_processed(p, "ep0")


# ============================================================
# 2. make_features
# ============================================================

class TestMakeFeatures:

    def test_base_features(self):
        feats = make_features((480, 640), include_depth=False)
        for key in ("observation.images.rgb", "observation.state",
                    "action", "observation.joints_3d",
                    "observation.confidence"):
            assert key in feats
        assert "observation.images.depth" not in feats
        assert feats["observation.state"]["shape"] == (7,)
        assert feats["action"]["shape"] == (7,)
        assert feats["observation.joints_3d"]["shape"] == (63,)

    def test_include_depth_adds_depth_key(self):
        feats = make_features((240, 320), include_depth=True)
        assert "observation.images.depth" in feats
        assert feats["observation.images.depth"]["shape"] == (240, 320, 3)

    def test_rgb_shape_matches_img_size(self):
        feats = make_features((200, 300), include_depth=False)
        assert feats["observation.images.rgb"]["shape"] == (200, 300, 3)


# ============================================================
# 3. _append_custom_episode_fields
# ============================================================

class TestAppendCustomEpisodeFields:

    def _write_episodes_parquet(
        self,
        output_dir: Path,
        records: list[dict],
        chunk: str = "chunk-000",
        file: str = "file-000",
    ) -> Path:
        pq_dir = output_dir / "meta" / "episodes" / chunk
        pq_dir.mkdir(parents=True, exist_ok=True)
        pq = pq_dir / f"{file}.parquet"
        pd.DataFrame(records).to_parquet(pq, index=False)
        return pq

    def test_merges_fields_by_episode_index(self, tmp_path):
        pq = self._write_episodes_parquet(tmp_path, [
            {"episode_index": 0, "length": 100},
            {"episode_index": 1, "length": 50},
        ])
        extras = {
            0: {"episode_name": "ep_a",
                "center_offset_world": [1.0, 2.0, 3.0]},
            1: {"episode_name": "ep_b",
                "center_offset_world": [4.0, 5.0, 6.0]},
        }
        _append_custom_episode_fields(tmp_path, extras)

        df = pd.read_parquet(pq)
        row0 = df[df["episode_index"] == 0].iloc[0]
        row1 = df[df["episode_index"] == 1].iloc[0]
        assert row0["episode_name"] == "ep_a"
        assert list(row0["center_offset_world"]) == [1.0, 2.0, 3.0]
        assert row0["length"] == 100  # original column preserved
        assert row1["episode_name"] == "ep_b"
        assert list(row1["center_offset_world"]) == [4.0, 5.0, 6.0]

    def test_merges_across_multiple_parquet_chunks(self, tmp_path):
        """When v3 splits episodes across chunks, all chunks get the extras."""
        self._write_episodes_parquet(
            tmp_path,
            [{"episode_index": 0, "length": 100}],
            chunk="chunk-000", file="file-000",
        )
        self._write_episodes_parquet(
            tmp_path,
            [{"episode_index": 1, "length": 50}],
            chunk="chunk-000", file="file-001",
        )
        _append_custom_episode_fields(tmp_path, {
            0: {"episode_name": "ep_a"},
            1: {"episode_name": "ep_b"},
        })
        df0 = pd.read_parquet(
            tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        )
        df1 = pd.read_parquet(
            tmp_path / "meta" / "episodes" / "chunk-000" / "file-001.parquet"
        )
        assert df0.iloc[0]["episode_name"] == "ep_a"
        assert df1.iloc[0]["episode_name"] == "ep_b"

    def test_missing_episodes_dir_does_not_raise(self, tmp_path, capsys):
        """When meta/episodes/ is absent, we warn and return gracefully."""
        _append_custom_episode_fields(tmp_path, {0: {"foo": "bar"}})
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_missing_index_in_extras_leaves_row_null(self, tmp_path):
        pq = self._write_episodes_parquet(tmp_path, [
            {"episode_index": 0, "length": 100},
            {"episode_index": 1, "length": 50},
        ])
        _append_custom_episode_fields(
            tmp_path, {0: {"episode_name": "ep_a"}},
        )
        df = pd.read_parquet(pq)
        row1 = df[df["episode_index"] == 1].iloc[0]
        # Row 1 has no matching extras → new column is present but None
        assert row1["episode_name"] is None
        assert row1["length"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
