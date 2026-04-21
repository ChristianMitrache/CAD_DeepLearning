"""Merge placed part meshes into a scene and sample uniform surface points.

`merge_placed` stitches per-part meshes (each with its own 4x4 transform) into a
single trimesh with a parallel array mapping face -> originating part id. That
enables downstream per-part coloring after a single Poisson-disk sample over the
whole scene.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class MergedScene:
    """Merged multi-part scene.

    face_part_ids: (n_faces,) object array — raw part id per face.
    part_ids: unique raw part ids in *placement order* (first occurrence wins).
        Downstream code uses `part_ids.index(raw)` as the canonical `part_idx`,
        so index 0 is the first-placed part, not the alphabetically-first one.
    """
    mesh: trimesh.Trimesh
    face_part_ids: np.ndarray
    part_ids: list[str]


def merge_placed(
    placements: list[tuple[str, np.ndarray]],
    part_mesh_cache: dict[str, trimesh.Trimesh | None],
) -> MergedScene:
    """Apply each transform to its part's mesh and concatenate into one scene.

    Skips placements with missing meshes or non-finite transforms — the latter
    otherwise trip `RuntimeWarning: divide by zero` inside the vertex matmul.
    """
    verts, faces, face_part_ids_out, ordered_part_ids = [], [], [], []
    seen: set[str] = set()
    offset = 0
    for pid, T in placements:
        m = part_mesh_cache.get(pid)
        if m is None or len(m.vertices) == 0 or len(m.faces) == 0:
            continue
        if not np.all(np.isfinite(T)):
            continue
        # errstate: on some platforms matmul emits divide/overflow warnings for
        # large-but-finite inputs from FPU subnormal flags. The outputs are valid.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            placed_v = (m.vertices @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        if not np.all(np.isfinite(placed_v)):
            continue
        verts.append(placed_v)
        faces.append(m.faces + offset)
        face_part_ids_out.extend([pid] * len(m.faces))
        if pid not in seen:
            seen.add(pid)
            ordered_part_ids.append(pid)
        offset += len(placed_v)
    if not verts:
        empty = trimesh.Trimesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )
        return MergedScene(empty, np.array([], dtype=object), [])
    mesh = trimesh.Trimesh(vertices=np.vstack(verts), faces=np.vstack(faces), process=False)
    return MergedScene(mesh, np.array(face_part_ids_out), ordered_part_ids)


def poisson_sample(scene: MergedScene, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Poisson-disk surface sampling (blue noise). Returns (points, part_ids), len <= n.

    Radius is derived from mesh area: n disks of radius r cover ~n*pi*r^2 == area,
    so r = sqrt(area / (pi * n)). A 0.75 packing factor shrinks the radius to land
    closer to n — real rejection-sampled disk packing on a surface tops out well
    below the ideal density and otherwise chronically undershoots.
    """
    mesh = scene.mesh
    area = float(mesh.area)
    if area <= 0 or n <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=scene.face_part_ids.dtype)
    radius = float(np.sqrt(area / (np.pi * n))) * 0.75
    with contextlib.redirect_stderr(io.StringIO()):
        pts, face_idx = trimesh.sample.sample_surface_even(mesh, n, radius=radius)
    return np.asarray(pts, dtype=np.float32), scene.face_part_ids[face_idx]
