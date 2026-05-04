"""
Source loader for retarget input — schema-validated `.npz` reader.

Pipeline position: thin npz reader for any source compatible with the
shared schema. Accepts `.processed.npz` (raw, output of stage 2) or
`.optimized.npz` (post-optimize). The retarget contract is purely
schema-based: keys + dtypes + NaN-fraction. Filename prefix is NOT used
for dispatch.

Boundary:
  - INPUTS : any `.npz` with the documented schema (top-level
             `timestamps_us` + `K`, per-hand `wrist_cam` /
             `wrist_quat_cam` / `joints_cam` / `confidence` / trim flags
             / quality flag).
  - DOES   : load → validate → expose typed accessor (`ProcessedSource`).
             Backends call `required_keys(hand)` and the loader's
             `validate_for_retarget(hand)` to fail loud at the start of a
             batch instead of mid-way through.
  - DOES NOT: heuristic detection of optimized vs raw — the npz schema
             is the same, only field values differ. Optimization is
             user-driven and transparent to retarget.

Why npz-only and schema-validated:
  Both recording lines emit the same stage-2 schema; both schema-valid
  optimizer outputs do too. Schema validation is the single source of
  truth, so an artifact from any tool that produces compatible arrays
  retargets without code changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Maximum tolerated NaN fraction in `<hand>_wrist_cam` for the chosen
# hand. Raw `.processed.npz` from a quality-failed hand can be all NaN;
# this guard surfaces "you forgot to run optimize" before the dex
# optimizer chokes silently. Tunable via `validate_for_retarget`.
_DEFAULT_MAX_NAN_FRACTION = 0.5


@dataclass
class ProcessedSource:
    """In-memory view of a schema-valid retarget source npz.

    Holds full-length arrays (no trimming applied). Backends read
    `quality_passed` / `trim_first` / `trim_last` for the chosen hand
    and slice as needed.
    """
    sid: str
    path: Path
    timestamps_us: np.ndarray            # (T,) int64
    K: np.ndarray                        # (3, 3) float64
    T_world_cam: np.ndarray | None       # (T, 4, 4) float64 or None
    raw: dict[str, np.ndarray]

    @property
    def n_frames_total(self) -> int:
        return len(self.timestamps_us)

    def has(self, key: str) -> bool:
        return key in self.raw

    def get(self, key: str) -> np.ndarray:
        if key not in self.raw:
            raise KeyError(
                f"Required key {key!r} missing from {self.path.name}. "
                f"Available: {sorted(self.raw)}"
            )
        return self.raw[key]

    def quality_passed(self, hand: str) -> bool:
        return bool(self.get(f"{hand}_quality_passed"))

    def trim_range(self, hand: str) -> tuple[int, int]:
        first = int(self.get(f"{hand}_trim_first"))
        last = int(self.get(f"{hand}_trim_last"))
        return first, last

    def validate_for_retarget(
        self, hand: str, *,
        max_nan_fraction: float = _DEFAULT_MAX_NAN_FRACTION,
    ) -> None:
        """Hard-check this source can carry retarget for `hand`.

        Surfaces 'too noisy / mostly NaN' before backends run. Suggests
        running optimize when the issue is high NaN density (not when
        quality failed at the trim step — that's a different decision
        the caller has to make).
        """
        first, last = self.trim_range(hand)
        if last <= first:
            return  # empty trim — caller handles, not a schema issue
        wrist = self.get(f"{hand}_wrist_cam")[first:last]
        nan_frac = float(np.isnan(wrist).any(axis=1).mean())
        if nan_frac > max_nan_fraction:
            raise ValueError(
                f"{self.path.name} hand={hand}: "
                f"{nan_frac * 100:.1f}% of in-trim wrist frames are NaN "
                f"(threshold {max_nan_fraction * 100:.0f}%). "
                f"This source is too sparse for direct retarget — run "
                f"`python -m optimize` first to fill gaps and produce "
                f"`.optimized.npz`, then point retarget at that."
            )


def load_npz_source(path: Path) -> ProcessedSource:
    """Load and validate any retarget-compatible npz.

    The validation here is the *common* schema — required top-level keys.
    Per-hand and per-backend validation is the caller's job (via
    `ProcessedSource.validate_for_retarget` and `backend.required_keys`).
    """
    if not path.exists():
        raise FileNotFoundError(f"npz not found: {path}")
    # `.processed.npz` and `.optimized.npz` both end at `.npz`, so strip
    # the secondary extension to get the bare sid.
    sid = path.stem
    for ext in (".processed", ".optimized"):
        if sid.endswith(ext):
            sid = sid[: -len(ext)]
            break
    # allow_pickle=True: iPhone-line .processed.npz includes object-typed
    # metadata strings (`source`, `episode_name`) that 335-line doesn't —
    # we don't use them in retarget but np.load can't enumerate `arr.files`
    # mid-getitem with allow_pickle=False. Files are produced by our own
    # pipeline (no untrusted input), so pickle exposure is bounded.
    arr = np.load(path, allow_pickle=True)
    raw: dict[str, np.ndarray] = {}
    for k in arr.files:
        v = arr[k]
        # Skip object-typed metadata — retarget operates on numeric arrays
        # only. Forward-compatible with whatever string tags lines decide
        # to attach.
        if v.dtype == object:
            continue
        raw[k] = v

    if "timestamps_us" not in raw:
        raise KeyError(
            f"{path.name} missing `timestamps_us`. This file is not a "
            f"valid retarget source npz."
        )
    if "K" not in raw:
        raise KeyError(f"{path.name} missing `K` intrinsics.")

    return ProcessedSource(
        sid=sid,
        path=path,
        timestamps_us=raw["timestamps_us"].astype(np.int64),
        K=raw["K"].astype(np.float64),
        T_world_cam=(
            raw["T_world_cam"].astype(np.float64)
            if "T_world_cam" in raw else None
        ),
        raw=raw,
    )


def discover_episodes(
    source_root: Path, episodes: list[str] | None = None,
) -> list[Path]:
    """Walk `<source_root>/<sid>/<sid>.{optimized,processed}.npz`.

    Discovery order:
      1. Prefer `.optimized.npz` if both exist (user explicitly ran
         optimize for this episode → that's what they want retargeted).
      2. Fall back to `.processed.npz`.

    The `<sid>/<sid>.<stage>.npz` layout is the convention shared by 02
    output, 03 output, optimize output. Anything emitting that layout is
    discoverable.
    """
    if not source_root.exists():
        raise FileNotFoundError(f"source_root not found: {source_root}")
    paths: list[Path] = []
    for sub in sorted(source_root.iterdir()):
        if not sub.is_dir():
            continue
        if episodes and sub.name not in episodes:
            continue
        opt = sub / f"{sub.name}.optimized.npz"
        proc = sub / f"{sub.name}.processed.npz"
        if opt.exists():
            paths.append(opt)
        elif proc.exists():
            paths.append(proc)
        else:
            print(f"  WARN: {sub.name} has no .processed/.optimized npz, skipping")
    return paths
