# Third-Party Sources & Licenses

This file tracks all third-party code, assets, and models vendored into this repository. Each entry lists: where it lives, upstream source, license, and what it's used for.

---

## Assets

### SO-ARM101 URDF + MJCF (TheRobotStudio, same-source)

- **Path**: `assets/mujoco/trs_so101/` and `assets/so101_new_calib.urdf`
- **Source**: https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
- **License**: Apache License 2.0 (file `assets/mujoco/trs_so101/LICENSE`)
- **Version**: sparse-checkout from `main` branch
- **Used by**: `sim/mujoco_loader.py`, `scripts/05_replay_in_sim.py`, `scripts/04_replay_on_arm.py`
- **Notes**:
  - URDF (`so101_new_calib.urdf`) and MJCF (`so101_new_calib.xml`) are *same-source*: both generated from the same Onshape CAD via the `onshape-to-robot` plugin. Zero-points, joint axes, and joint names are identical between the two files — so the IK output semantics match the MJCF `qpos` semantics with no per-joint offset.
  - Joint names (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`) match LeRobot convention verbatim; the `LEROBOT_TO_MJCF` dict in `sim/mujoco_loader.py` is therefore an identity map, retained as the single source of truth for future MJCF swaps.
  - Local modification: `scene.xml` has a `<keyframe name="home">` appended (all-zero qpos). Not present upstream.
