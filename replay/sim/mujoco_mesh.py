"""
Extract per-geom mesh data from a loaded MuJoCo model in a form rerun can log.

Pipeline position: shared helper for `replay/sim/rerun_so101.py` and
`replay/sim/rerun_dex.py`. Both used to roll their own loops that read
`geom_rgba` and skipped vertex normals — that produced flat 50%-gray
robots in rerun even though the MJCF defined per-mesh material colours
(yellow plastic + black servos for SO-101) and MuJoCo carries vertex
normals out of the box. This module fixes both, in one place.

What it changes vs. the old per-file loop:
  * rgba prefers `mat_rgba[geom_matid]` so each part shows its real
    material colour. Falls back to `geom_rgba` when no material set
    (geom_matid < 0).
  * yields `vertex_normals` sliced from `model.mesh_normal[v_start:v_end]`,
    which rerun.io's Mesh3D consumes for smooth (Gouraud-style) shading
    instead of the default per-face faceted look.

What it cannot change:
  * rerun.io's Mesh3D currently has no specular / metallic / Phong
    component — even with normals + colours it won't match MuJoCo's
    headlight-shaded look. Closer is closer; not equivalent.
"""
from __future__ import annotations

from typing import Iterator, NamedTuple

import mujoco
import numpy as np


class MeshGeomVisual(NamedTuple):
    """Per-geom data ready for rerun.Mesh3D.

    Vertices and normals are in the geom's local frame; per-frame world
    pose comes from `data.geom_xpos` / `data.geom_xmat` and is logged as
    a child Transform3D under the static mesh entity.
    """
    geom_id: int
    body_name: str
    vertices: np.ndarray        # (V, 3) float, mesh-local
    triangles: np.ndarray       # (F, 3) int
    normals: np.ndarray         # (V, 3) float, mesh-local; matches MJCF mesh_normal
    rgba: np.ndarray            # (4,) float in [0, 1]; from material if defined


def iter_visible_mesh_geoms(model) -> Iterator[MeshGeomVisual]:
    """Yield MeshGeomVisual for every visible mesh geom (group <= 2, alpha > 0).

    Material rgba lookup:
      Each `<geom>` may reference a `<material>` via `material="..."`
      (compiled into `model.geom_matid[gi] >= 0`). When set, the visual
      colour is `model.mat_rgba[matid]`. When unset (matid == -1),
      fall back to `model.geom_rgba[gi]`. SO-101's MJCF binds materials
      to almost every geom, so this is where the real colour info lives.
    """
    for gi in range(model.ngeom):
        if model.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_group[gi] > 2:
            continue

        # Material rgba takes precedence over the geom-level fallback.
        # geom_matid is (ngeom,) int32 in modern mujoco; older builds may
        # expose (ngeom, 1) — handle both.
        matid_raw = model.geom_matid[gi]
        matid = int(matid_raw if np.ndim(matid_raw) == 0 else matid_raw[0])
        if matid >= 0:
            rgba = model.mat_rgba[matid].copy()
        else:
            rgba = model.geom_rgba[gi].copy()
        if float(rgba[3]) <= 0.0:
            continue

        mesh_id = int(model.geom_dataid[gi])
        if mesh_id < 0:
            continue
        v_start = int(model.mesh_vertadr[mesh_id])
        v_end = v_start + int(model.mesh_vertnum[mesh_id])
        f_start = int(model.mesh_faceadr[mesh_id])
        f_end = f_start + int(model.mesh_facenum[mesh_id])
        verts = model.mesh_vert[v_start:v_end].copy()
        faces = model.mesh_face[f_start:f_end].copy()
        # mesh_normal is the same length as mesh_vert (one normal per
        # vertex). Slicing with the same range gives smooth-shading
        # data. If MJCF didn't ship normals (rare) the slice is empty
        # and rerun falls back to per-face shading on its own.
        normals = model.mesh_normal[v_start:v_end].copy()

        body_id = int(model.geom_bodyid[gi])
        body_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or f"b{body_id}"
        )

        yield MeshGeomVisual(
            geom_id=gi, body_name=body_name,
            vertices=verts, triangles=faces, normals=normals, rgba=rgba,
        )
