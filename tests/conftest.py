"""Pytest config for the main-line test suite.

Legacy (ArUco support line) tests live under `tests/legacy/` and are excluded
from the default `pytest` run. Invoke them explicitly with
`pytest tests/legacy/` when needed.
"""

collect_ignore_glob = ["legacy/*"]
