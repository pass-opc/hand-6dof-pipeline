"""
Pure replay layer: `.qpos.npz` → MuJoCo viewer / mp4 / real arm.

Pipeline position: stage 6 of either recording line. Reads the qpos
artefact emitted by `retarget/`, plus the sidecar meta to know which
robot + env was retargeted for, and dispatches to a backend.

Boundary (intentionally narrow):
  - INPUT : `.qpos.npz` + `.qpos.meta.json` (from `python -m retarget`).
  - DOES  : per-frame qpos → MuJoCo state, optional offscreen mp4.
            Hold-last on invalid frames so playback length matches the
            source recording.
  - DOES NOT: any retarget logic, IK, smoothing, frame-shape changes.
              If the qpos shape doesn't match the registered backend,
              fail fast with a message that points at retarget.

Adding a new robot:
  1. Implement a backend in replay/sim/<env>_<robot>.py or
     replay/real/<robot>.py exposing
       run(qpos_npz_path, qpos_meta_path, *, output, **kwargs)
  2. Register in `_BACKENDS` below.
"""

from __future__ import annotations


# (robot, env) -> "module:run_callable". Lazy-imported.
_BACKENDS: dict[tuple[str, str], str] = {
    ("shadow",  "mujoco"): "replay.sim.mujoco_dex:run",
    ("leap",    "mujoco"): "replay.sim.mujoco_dex:run",
    ("allegro", "mujoco"): "replay.sim.mujoco_dex:run",
    ("inspire", "mujoco"): "replay.sim.mujoco_dex:run",
    ("svh",     "mujoco"): "replay.sim.mujoco_dex:run",
    ("ability", "mujoco"): "replay.sim.mujoco_dex:run",
    ("panda",   "mujoco"): "replay.sim.mujoco_dex:run",
    # SO-arm 101 — sim and real share the same retarget output; the
    # backend is selected by `(robot, env)`. CLI `--output real`
    # overrides env to "real" at dispatch time.
    ("so101", "mujoco"): "replay.sim.mujoco_so101:run",
    ("so101", "real"):   "replay.real.so101:run",
}


def get_backend(robot: str, env: str = "mujoco"):
    """Resolve `(robot, env)` to a backend `run` callable."""
    key = (robot, env)
    if key not in _BACKENDS:
        avail = sorted(f"{r}/{e}" for r, e in _BACKENDS)
        raise ValueError(
            f"No replay backend registered for robot={robot!r} env={env!r}. "
            f"Available: {avail}"
        )
    spec = _BACKENDS[key]
    mod_name, func_name = spec.split(":")
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, func_name)
