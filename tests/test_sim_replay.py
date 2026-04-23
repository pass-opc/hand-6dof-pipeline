"""
Tests for sim.mujoco_loader, sim.mujoco_replay, and scripts/05_replay_in_sim.

Covers:
  1. mujoco_loader — scene load, LEROBOT→MJCF index map, qpos/ctrl write,
     out-of-range warning path, unknown joint rejection, keyframe reset.
  2. mujoco_replay — shape validation, headless smoke, final-state check.
  3. Script 05 end-to-end — subprocess run with --no-gui, skipped if no
     dataset is present.

Run:
    cd hand-6dof-pipeline
    python -m pytest tests/test_sim_replay.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCENE = _PROJECT_ROOT / "assets" / "mujoco" / "trs_so101" / "scene.xml"
_DATASET_V3 = _PROJECT_ROOT / "output" / "03_dataset_v3"

sys.path.insert(0, str(_PROJECT_ROOT))

# Bail the whole module out cleanly if mujoco isn't installed yet.
pytest.importorskip("mujoco")


@pytest.fixture
def scene():
    from sim.mujoco_loader import load_so_arm101
    return load_so_arm101(_SCENE)


# ============================================================
# mujoco_loader
# ============================================================
def test_loader_maps_all_six_joints(scene):
    from sim.mujoco_loader import LEROBOT_TO_MJCF

    assert set(scene.lerobot_to_qpos_idx) == set(LEROBOT_TO_MJCF)
    assert set(scene.lerobot_to_ctrl_idx) == set(LEROBOT_TO_MJCF)
    # Distinct indices — mapping is injective.
    assert len(set(scene.lerobot_to_qpos_idx.values())) == 6
    assert len(set(scene.lerobot_to_ctrl_idx.values())) == 6


def test_loader_joint_ranges_match_trs_so101(scene):
    # Values come straight from trs_so101/so101_new_calib.xml (radians).
    lo, hi = scene.joint_range_rad["shoulder_pan"]
    assert lo == pytest.approx(-1.9199, abs=1e-3)
    assert hi == pytest.approx(1.9199, abs=1e-3)
    lo, hi = scene.joint_range_rad["shoulder_lift"]
    assert lo == pytest.approx(-1.7453, abs=1e-3)
    assert hi == pytest.approx(1.7453, abs=1e-3)
    lo, hi = scene.joint_range_rad["gripper"]
    assert lo == pytest.approx(-0.1745, abs=1e-3)
    assert hi == pytest.approx(1.7453, abs=1e-3)


def test_loader_missing_file_raises(tmp_path):
    from sim.mujoco_loader import load_so_arm101
    with pytest.raises(FileNotFoundError):
        load_so_arm101(tmp_path / "does_not_exist.xml")


def test_apply_joints_writes_ctrl(scene):
    from sim.mujoco_loader import apply_joint_positions_deg

    cmd = {
        "shoulder_pan":  10.0,
        "shoulder_lift": -45.0,
        "elbow_flex":    30.0,
        "wrist_flex":    0.0,
        "wrist_roll":    0.0,
        "gripper":       5.0,
    }
    apply_joint_positions_deg(scene, cmd, target="ctrl")
    idx = scene.lerobot_to_ctrl_idx["shoulder_pan"]
    assert scene.data.ctrl[idx] == pytest.approx(np.radians(10.0), rel=1e-6)
    idx = scene.lerobot_to_ctrl_idx["shoulder_lift"]
    assert scene.data.ctrl[idx] == pytest.approx(np.radians(-45.0), rel=1e-6)


def test_apply_joints_writes_qpos(scene):
    from sim.mujoco_loader import apply_joint_positions_deg

    cmd = {"elbow_flex": 60.0}
    apply_joint_positions_deg(scene, cmd, target="qpos")
    idx = scene.lerobot_to_qpos_idx["elbow_flex"]
    assert scene.data.qpos[idx] == pytest.approx(np.radians(60.0), rel=1e-6)


def test_apply_joints_rejects_unknown_name(scene):
    from sim.mujoco_loader import apply_joint_positions_deg
    with pytest.raises(KeyError):
        apply_joint_positions_deg(scene, {"bogus_joint": 0.0})


def test_apply_joints_rejects_bad_target(scene):
    from sim.mujoco_loader import apply_joint_positions_deg
    with pytest.raises(ValueError):
        apply_joint_positions_deg(
            scene, {"shoulder_pan": 0.0}, target="invalid",
        )


def test_apply_joints_out_of_range_warns(scene, capsys):
    from sim.mujoco_loader import apply_joint_positions_deg

    # shoulder_pan range is ~±110°; 200° is clearly out. Caller wants it
    # visible, not clamped.
    apply_joint_positions_deg(scene, {"shoulder_pan": 200.0}, target="qpos")
    captured = capsys.readouterr()
    assert "WARN" in captured.out
    assert "shoulder_pan" in captured.out


def test_reset_to_keyframe_home(scene):
    from sim.mujoco_loader import reset_to_keyframe

    scene.data.qpos[:] = 99.0  # garbage
    reset_to_keyframe(scene, "home")
    # home keyframe from trs_so101/scene.xml: qpos="0 0 0 0 0 0"
    expected = np.zeros(6)
    np.testing.assert_allclose(scene.data.qpos[:6], expected, atol=1e-3)


def test_reset_to_keyframe_missing_raises(scene):
    from sim.mujoco_loader import reset_to_keyframe
    with pytest.raises(ValueError):
        reset_to_keyframe(scene, "no_such_keyframe")


# ============================================================
# mujoco_replay
# ============================================================
def test_replay_rejects_bad_column_count(scene):
    from sim.mujoco_replay import replay_joint_trajectory

    bad = np.zeros((3, 4))         # 4 cols instead of 5
    gripper = np.zeros(3)
    with pytest.raises(ValueError):
        replay_joint_trajectory(scene, bad, gripper, no_gui=True)


def test_replay_rejects_length_mismatch(scene):
    from sim.mujoco_replay import replay_joint_trajectory

    arm = np.zeros((3, 5))
    gripper = np.zeros(5)  # deliberately mismatched
    with pytest.raises(ValueError):
        replay_joint_trajectory(scene, arm, gripper, no_gui=True)


def test_replay_headless_drives_qpos(scene):
    """End-to-end: replay 5 frames, verify final qpos matches last command."""
    from sim.mujoco_replay import ARM_JOINT_NAMES, replay_joint_trajectory

    T = 5
    arm = np.zeros((T, 5))
    arm[:, 0] = np.linspace(-10.0, 10.0, T)   # shoulder_pan
    arm[:, 2] = np.linspace(0.0, 20.0, T)     # elbow_flex
    gripper = np.linspace(0.0, 15.0, T)

    replay_joint_trajectory(
        scene, arm, gripper,
        fps=30, speed=5.0, no_gui=True, physics=False,
    )

    # Kinematics mode writes qpos directly; final state must match frame T-1.
    for i, name in enumerate(ARM_JOINT_NAMES):
        idx = scene.lerobot_to_qpos_idx[name]
        assert scene.data.qpos[idx] == pytest.approx(
            np.radians(arm[-1, i]), abs=1e-5,
        ), f"qpos[{name}] mismatch"
    idx = scene.lerobot_to_qpos_idx["gripper"]
    assert scene.data.qpos[idx] == pytest.approx(
        np.radians(gripper[-1]), abs=1e-5,
    )


# ============================================================
# Script 05 end-to-end
# ============================================================
@pytest.mark.skipif(
    not (_DATASET_V3 / "meta" / "info.json").exists(),
    reason=f"HaMeR v3 dataset not present at {_DATASET_V3}",
)
def test_script_05_smoke_no_gui():
    """Subprocess: read episode → IK → drive MuJoCo headless. Slow (~minutes)."""
    script = _PROJECT_ROOT / "scripts" / "05_replay_in_sim.py"
    cmd = [
        sys.executable, str(script),
        "--dataset-root", str(_DATASET_V3),
        "--episode", "2",       # per project memory: preferred sanity episode
        "--scale", "0.5",       # per project memory: HaMeR v3 requires 0.5
        "--speed", "20.0",      # run fast; we only care about correctness
        "--no-gui",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 0, (
        f"script failed (rc={result.returncode})\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )
    assert "Replay complete" in result.stdout
