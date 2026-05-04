"""
Tests for replay/sim/mujoco_so101.py and replay/real/so101.py.

Covers:
  1. Loader: 6-joint name lookup + ranges (kept here so this file is
     the single SO-101 sim test home).
  2. `apply_joint_positions_deg` write semantics + validation.
  3. `replay_joint_trajectory` shape rejection + final-state check.
  4. The new `_qpos_to_arm_gripper_deg` slice-and-hold-fill helper
     (sim + real share this contract; tested once in the sim file).
  5. Sim `run()` smoke: writes a synthetic .qpos.npz + meta then runs
     mp4 mode end-to-end. Asserts the mp4 exists and has a non-zero size.
  6. Real backend: dry-run path on the same synthetic .qpos.npz —
     must NOT touch hardware and must return executed=False. Real-arm
     execution is gated behind a hardware-only marker.

Run:
    cd code/opc_data_pipeline
    python -m pytest tests/test_replay_so101.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Bail the whole module out cleanly if mujoco isn't installed.
pytest.importorskip("mujoco")

_SCENE = _PROJECT_ROOT / "assets" / "mujoco" / "trs_so101" / "scene.xml"


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def scene():
    from replay.sim.mujoco_loader import load_so_arm101
    return load_so_arm101(_SCENE)


def _write_synthetic_qpos(
    tmp_path: Path, n_frames: int = 8, hand: str = "right",
) -> tuple[Path, Path]:
    """Write a small synthetic <sid>/<sid>.qpos.npz + .qpos.meta.json
    matching the schema produced by retarget/__main__.py."""
    sid = "synth_episode"
    out_dir = tmp_path / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{sid}.qpos.npz"
    meta_path = out_dir / f"{sid}.qpos.meta.json"

    # Tiny safe trajectory: shoulder_pan sweep ±10°, gripper opening.
    qpos = np.zeros((n_frames, 6), dtype=np.float32)
    qpos[:, 0] = np.radians(np.linspace(-10.0, 10.0, n_frames))   # shoulder_pan
    qpos[:, 5] = np.radians(np.linspace(0.0, 30.0, n_frames))     # gripper
    qpos_valid = np.ones(n_frames, dtype=bool)

    timestamps_us = (np.arange(n_frames) * (1_000_000 // 30)).astype(np.int64)
    np.savez_compressed(
        npz_path,
        timestamps_us=timestamps_us,
        right_qpos=qpos, right_qpos_valid=qpos_valid,
    )
    meta = {
        "schema_version": 3,
        "session_id": sid,
        "robot": "so101", "env": "mujoco", "hand": hand,
        "backend": "so101", "retargeting_type": "position",
        "joint_names": [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ],
        "n_joints": 6,
        "trim": [0, n_frames],
        "n_frames_total": n_frames,
        "n_frames_in_trim": n_frames,
        "n_frames_retarget_succeeded": n_frames,
        "K_flat": [600, 0, 320, 0, 600, 240, 0, 0, 1],
        "extras": {},
        "fps": 30,
        "run_timestamp_iso": "2026-04-30T00:00:00",
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return npz_path, meta_path


# ============================================================
# 1. Loader
# ============================================================
def test_loader_maps_all_six_joints(scene):
    from replay.sim.mujoco_loader import LEROBOT_TO_MJCF
    assert set(scene.lerobot_to_qpos_idx) == set(LEROBOT_TO_MJCF)
    assert set(scene.lerobot_to_ctrl_idx) == set(LEROBOT_TO_MJCF)
    assert len(set(scene.lerobot_to_qpos_idx.values())) == 6


def test_loader_joint_ranges_match_trs_so101(scene):
    lo, hi = scene.joint_range_rad["shoulder_pan"]
    assert lo == pytest.approx(-1.9199, abs=1e-3)
    assert hi == pytest.approx(1.9199, abs=1e-3)
    lo, hi = scene.joint_range_rad["gripper"]
    assert lo == pytest.approx(-0.1745, abs=1e-3)
    assert hi == pytest.approx(1.7453, abs=1e-3)


def test_loader_missing_file_raises(tmp_path):
    from replay.sim.mujoco_loader import load_so_arm101
    with pytest.raises(FileNotFoundError):
        load_so_arm101(tmp_path / "does_not_exist.xml")


# ============================================================
# 2. apply_joint_positions_deg
# ============================================================
def test_apply_joints_writes_ctrl(scene):
    from replay.sim.mujoco_loader import apply_joint_positions_deg
    apply_joint_positions_deg(scene, {"shoulder_lift": -45.0}, target="ctrl")
    idx = scene.lerobot_to_ctrl_idx["shoulder_lift"]
    assert scene.data.ctrl[idx] == pytest.approx(np.radians(-45.0), rel=1e-6)


def test_apply_joints_writes_qpos(scene):
    from replay.sim.mujoco_loader import apply_joint_positions_deg
    apply_joint_positions_deg(scene, {"elbow_flex": 60.0}, target="qpos")
    idx = scene.lerobot_to_qpos_idx["elbow_flex"]
    assert scene.data.qpos[idx] == pytest.approx(np.radians(60.0), rel=1e-6)


def test_apply_joints_rejects_unknown_name(scene):
    from replay.sim.mujoco_loader import apply_joint_positions_deg
    with pytest.raises(KeyError):
        apply_joint_positions_deg(scene, {"bogus_joint": 0.0})


def test_apply_joints_out_of_range_warns(scene, capsys):
    from replay.sim.mujoco_loader import apply_joint_positions_deg
    apply_joint_positions_deg(scene, {"shoulder_pan": 200.0}, target="qpos")
    out = capsys.readouterr().out
    assert "WARN" in out and "shoulder_pan" in out


# ============================================================
# 3. replay_joint_trajectory
# ============================================================
def test_replay_rejects_bad_column_count(scene):
    from replay.sim.mujoco_so101 import replay_joint_trajectory
    with pytest.raises(ValueError):
        replay_joint_trajectory(scene, np.zeros((3, 4)), np.zeros(3), no_gui=True)


def test_replay_rejects_length_mismatch(scene):
    from replay.sim.mujoco_so101 import replay_joint_trajectory
    with pytest.raises(ValueError):
        replay_joint_trajectory(scene, np.zeros((3, 5)), np.zeros(5), no_gui=True)


def test_replay_headless_drives_qpos(scene):
    from replay.sim.mujoco_so101 import (
        ARM_JOINT_NAMES, replay_joint_trajectory,
    )
    T = 5
    arm = np.zeros((T, 5))
    arm[:, 0] = np.linspace(-10.0, 10.0, T)
    arm[:, 2] = np.linspace(0.0, 20.0, T)
    gripper = np.linspace(0.0, 15.0, T)
    replay_joint_trajectory(
        scene, arm, gripper, fps=30, speed=5.0, no_gui=True, physics=False,
    )
    for i, name in enumerate(ARM_JOINT_NAMES):
        idx = scene.lerobot_to_qpos_idx[name]
        assert scene.data.qpos[idx] == pytest.approx(
            np.radians(arm[-1, i]), abs=1e-5,
        )
    idx = scene.lerobot_to_qpos_idx["gripper"]
    assert scene.data.qpos[idx] == pytest.approx(
        np.radians(gripper[-1]), abs=1e-5,
    )


# ============================================================
# 4. _qpos_to_arm_gripper_deg
# ============================================================
def test_qpos_slice_picks_arm_and_gripper():
    from replay.sim.mujoco_so101 import _qpos_to_arm_gripper_deg
    joint_names = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]
    qpos = np.zeros((4, 6), dtype=np.float32)
    qpos[:, 0] = np.radians(10.0)
    qpos[:, 5] = np.radians(45.0)
    valid = np.ones(4, dtype=bool)
    arm_deg, grip_deg = _qpos_to_arm_gripper_deg(qpos, valid, joint_names)
    assert arm_deg.shape == (4, 5)
    assert grip_deg.shape == (4,)
    assert arm_deg[0, 0] == pytest.approx(10.0, rel=1e-6)
    assert grip_deg[0] == pytest.approx(45.0, rel=1e-6)


def test_qpos_hold_last_fills_invalid_mid_trajectory():
    from replay.sim.mujoco_so101 import _qpos_to_arm_gripper_deg
    joint_names = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]
    qpos = np.zeros((4, 6), dtype=np.float32)
    qpos[0, 0] = np.radians(5.0)
    qpos[2, 0] = np.radians(20.0)   # frame 1 invalid → should hold frame 0
    valid = np.array([True, False, True, True])
    arm_deg, _ = _qpos_to_arm_gripper_deg(qpos, valid, joint_names)
    assert arm_deg[1, 0] == pytest.approx(5.0, rel=1e-6)
    assert arm_deg[2, 0] == pytest.approx(20.0, rel=1e-6)


def test_qpos_leading_invalid_uses_first_valid_pose():
    from replay.sim.mujoco_so101 import _qpos_to_arm_gripper_deg
    joint_names = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]
    qpos = np.zeros((3, 6), dtype=np.float32)
    qpos[2, 0] = np.radians(15.0)   # frames 0/1 NaN, frame 2 valid
    qpos[0:2] = np.nan
    valid = np.array([False, False, True])
    arm_deg, _ = _qpos_to_arm_gripper_deg(qpos, valid, joint_names)
    # Pre-first-valid frames should clone first valid pose.
    assert arm_deg[0, 0] == pytest.approx(15.0, rel=1e-6)
    assert arm_deg[1, 0] == pytest.approx(15.0, rel=1e-6)
    assert arm_deg[2, 0] == pytest.approx(15.0, rel=1e-6)


def test_qpos_no_valid_frames_raises():
    from replay.sim.mujoco_so101 import _qpos_to_arm_gripper_deg
    joint_names = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]
    qpos = np.full((3, 6), np.nan, dtype=np.float32)
    valid = np.array([False, False, False])
    with pytest.raises(ValueError, match="no valid frames"):
        _qpos_to_arm_gripper_deg(qpos, valid, joint_names)


# ============================================================
# 5. sim run() end-to-end (mp4 mode)
# ============================================================
def test_sim_run_mp4_writes_file(tmp_path):
    """Synthetic qpos → mp4 render → file exists with non-zero size."""
    pytest.importorskip("imageio")
    from replay.sim.mujoco_so101 import run
    npz_path, meta_path = _write_synthetic_qpos(tmp_path, n_frames=8)
    out_mp4 = tmp_path / "out.mp4"
    stats = run(
        qpos_npz_path=npz_path, qpos_meta_path=meta_path,
        output="mp4", out_mp4=out_mp4,
        fps=30, width=320, height=240, camera="cam_frame",
    )
    assert out_mp4.exists()
    assert out_mp4.stat().st_size > 0
    assert stats["n_total"] == 8
    assert stats["n_rendered"] == 8


# ============================================================
# 6. Real backend dry-run
# ============================================================
def test_real_run_dry_run_does_not_touch_hardware(tmp_path):
    """real backend with dry_run=True must succeed without lerobot/serial."""
    from replay.real.so101 import run
    npz_path, meta_path = _write_synthetic_qpos(tmp_path, n_frames=6)
    stats = run(
        qpos_npz_path=npz_path, qpos_meta_path=meta_path,
        output="real", port=None, dry_run=True,
    )
    assert stats["executed"] is False
    assert stats["n_total"] == 6


def test_real_run_without_port_raises(tmp_path):
    """No --port AND not dry-run → must error before importing lerobot."""
    from replay.real.so101 import run
    npz_path, meta_path = _write_synthetic_qpos(tmp_path, n_frames=4)
    with pytest.raises(ValueError, match="port"):
        run(
            qpos_npz_path=npz_path, qpos_meta_path=meta_path,
            output="real", port=None, dry_run=False,
        )
