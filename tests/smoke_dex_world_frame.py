"""
Smoke test: verify dex_retargeting converges with ARKit world-frame input.

Why: 05 dex_hands.py currently feeds cam-frame wrist+joints. We're about
to switch to world-frame so 06 cam_frame can be placed at the recording
camera's actual pose in world. Before refactoring 05/06 this script
validates that dex_retargeting still produces sensible qpos when the
input frame changes from OpenCV cam to ARKit world (gravity-aligned, +Y up).

Pass criteria:
  1. No exceptions over the first 50 valid frames
  2. dummy_xyz_t magnitudes within ~1 m of source wrist_world
  3. dummy_xyz_t variance pattern follows wrist_world variance pattern

Usage (opc-dex env):
    conda activate opc-dex
    python tests/smoke_dex_world_frame.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from retarget.dex_hands import DexBackend, HAND_TYPES  # noqa: E402


_NPZ = (
    _PROJECT_ROOT / "output" / "iphone" / "pour_coffee_bean"
    / "02_processed" / "pour_coffee_bean_episode02"
    / "pour_coffee_bean_episode02.processed.npz"
)


def _xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def main() -> int:
    if not _NPZ.exists():
        print(f"FAIL: missing input npz {_NPZ}")
        return 1

    print(f"Loading: {_NPZ.name}")
    d = np.load(_NPZ)

    # World-frame fields (added by 02 from cam-frame via T_world_cam)
    joints = d["right_joints_world"].astype(np.float64)
    wrist_pos = d["right_wrist_world"].astype(np.float64)
    wrist_quat = d["right_wrist_quat_world"].astype(np.float64)
    confidence = d["right_confidence"].astype(np.float64)
    trim_first = int(d["right_trim_first"])
    trim_last = int(d["right_trim_last"])

    valid = (
        (confidence >= 0)
        & np.isfinite(joints).all(axis=(1, 2))
        & np.isfinite(wrist_pos).all(axis=1)
        & np.isfinite(wrist_quat).all(axis=1)
    )
    valid_in_trim = valid[trim_first:trim_last]
    valid_idx_in_trim = np.flatnonzero(valid_in_trim)
    print(f"trim=[{trim_first}..{trim_last})  valid_in_trim={valid_in_trim.sum()}")

    if len(valid_idx_in_trim) == 0:
        print("FAIL: no valid frames in trim range")
        return 1

    print(f"\nBuilding shadow/right DexBackend...")
    backend = DexBackend(robot="shadow", hand="right")
    print(f"  joint_names ({backend.n_joints}): {backend.joint_names[:8]}...")
    print(f"  target_human_indices: {backend.target_human_indices}")

    # Warm start with first valid world-frame wrist (mirror dex_hands.py L195-202)
    first_v = valid_idx_in_trim[0] + trim_first
    wp0 = wrist_pos[first_v]
    wq0 = wrist_quat[first_v]
    print(f"\nWarm-start at t={first_v}:")
    print(f"  wrist_world pos:  {wp0}")
    print(f"  wrist_world quat (xyzw): {wq0}")

    backend._retargeting.reset()
    backend._retargeting.warm_start(
        wrist_pos=wp0,
        wrist_quat=_xyzw_to_wxyz(wq0),
        hand_type=HAND_TYPES["right"],
        is_mano_convention=True,
    )

    # Retarget first N=50 valid frames, collect dummy_t for sanity
    N = min(50, len(valid_idx_in_trim))
    dummy_t = np.full((N, 3), np.nan)
    dummy_r = np.full((N, 3), np.nan)
    failures = 0
    print(f"\nRetargeting first {N} valid frames...")
    for k in range(N):
        t = valid_idx_in_trim[k] + trim_first
        ref = joints[t][backend.target_human_indices]
        try:
            q = backend._retargeting.retarget(ref_value=ref)
        except Exception as exc:
            print(f"  FAIL at t={t}: {exc}")
            failures += 1
            continue
        dummy_t[k] = q[:3]
        dummy_r[k] = q[3:6]

    if failures > 0:
        print(f"\nFAIL: {failures}/{N} frames raised exceptions")
        return 1

    print(f"\nPASS: {N}/{N} frames retargeted without exception")

    # Compare magnitudes
    print("\n=== sanity comparison ===")
    used_idx = valid_idx_in_trim[:N] + trim_first
    src_wp = wrist_pos[used_idx]
    print(f"src wrist_world (in MuJoCo will become hand pos):")
    print(f"  X: mean={src_wp[:,0].mean():+.3f}  range={np.ptp(src_wp[:,0]):.3f}")
    print(f"  Y: mean={src_wp[:,1].mean():+.3f}  range={np.ptp(src_wp[:,1]):.3f}")
    print(f"  Z: mean={src_wp[:,2].mean():+.3f}  range={np.ptp(src_wp[:,2]):.3f}")
    print(f"dummy_xyz_t (URDF base in retarget output frame):")
    print(f"  X: mean={dummy_t[:,0].mean():+.3f}  range={np.ptp(dummy_t[:,0]):.3f}")
    print(f"  Y: mean={dummy_t[:,1].mean():+.3f}  range={np.ptp(dummy_t[:,1]):.3f}")
    print(f"  Z: mean={dummy_t[:,2].mean():+.3f}  range={np.ptp(dummy_t[:,2]):.3f}")
    print(f"dummy_xyz_r (Euler XYZ rad):")
    print(f"  X: mean={dummy_r[:,0].mean():+.3f}  range={np.ptp(dummy_r[:,0]):.3f}")
    print(f"  Y: mean={dummy_r[:,1].mean():+.3f}  range={np.ptp(dummy_r[:,1]):.3f}")
    print(f"  Z: mean={dummy_r[:,2].mean():+.3f}  range={np.ptp(dummy_r[:,2]):.3f}")

    delta = np.linalg.norm(dummy_t.mean(axis=0) - src_wp.mean(axis=0))
    print(f"\n||dummy_t.mean - wrist_world.mean|| = {delta:.3f} m  "
          f"(forearm offset; ~25 cm expected for Shadow)")

    if delta > 0.6:
        print(f"WARN: offset > 60 cm — may indicate frame mismatch")
        return 2
    if delta < 0.05:
        print(f"WARN: offset < 5 cm — unusual for Shadow forearm offset")
    print("\nSMOKE PASS — safe to proceed with refactor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
