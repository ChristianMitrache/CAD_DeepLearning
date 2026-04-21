"""Point-cloud sampling at the pipeline level.

`sample_scene` is the canonical "scene -> SampledPoints" step used during
preprocess. `resample_from_disk` reads a written scene.ply and re-samples
at any N without re-tessellating STEP.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import cast

import numpy as np
import trimesh
from trimesh.visual import ColorVisuals

from cad_dl.geometry.sampling import MergedScene
from cad_dl.pipeline.io import SampledPoints, load_metadata, load_scene_mesh
from cad_dl.pipeline.schema import AssemblyMetadata


def _sample_from_mesh(
    mesh: trimesh.Trimesh,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Poisson-disk sample the mesh. Returns (points, normals, face_idx)."""
    area = float(mesh.area) if len(mesh.faces) > 0 else 0.0
    if area <= 0 or n <= 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, np.zeros((0,), dtype=np.int64)
    radius = float(np.sqrt(area / (np.pi * n))) * 0.75
    with contextlib.redirect_stderr(io.StringIO()):
        pts, face_idx = trimesh.sample.sample_surface_even(mesh, n, radius=radius)
    pts = np.asarray(pts, dtype=np.float32)
    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
    return pts, normals, np.asarray(face_idx, dtype=np.int64)


def sample_scene(scene: MergedScene, n: int) -> SampledPoints:
    """Poisson-disk sample a MergedScene into SampledPoints (points+normals+part_idx)."""
    pts, normals, face_idx = _sample_from_mesh(scene.mesh, n)
    if len(pts) == 0:
        return SampledPoints(
            points=pts, normals=normals, part_idx=np.zeros((0,), dtype=np.uint16)
        )
    lookup = {pid: i for i, pid in enumerate(scene.part_ids)}
    part_idx = np.fromiter(
        (lookup[scene.face_part_ids[int(f)]] for f in face_idx),
        dtype=np.uint16, count=len(face_idx),
    )
    return SampledPoints(points=pts, normals=normals, part_idx=part_idx)


def resample_from_disk(assembly_dir: Path, n: int) -> SampledPoints:
    """Re-sample an already-processed assembly's scene.ply at a new N.

    Recovers per-vertex part_idx by matching vertex RGB back to
    metadata.parts[].color_rgb (the authoritative mapping).
    """
    mesh = load_scene_mesh(Path(assembly_dir))
    meta = load_metadata(Path(assembly_dir))
    pts, normals, face_idx = _sample_from_mesh(mesh, n)
    if len(pts) == 0:
        return SampledPoints(
            points=pts, normals=normals, part_idx=np.zeros((0,), dtype=np.uint16)
        )

    vert_part_idx = _recover_vert_part_idx(mesh, meta)
    face_pidx = vert_part_idx[mesh.faces[:, 0]]
    part_idx = face_pidx[face_idx].astype(np.uint16)
    return SampledPoints(points=pts, normals=normals, part_idx=part_idx)


def _recover_vert_part_idx(mesh: trimesh.Trimesh, meta: AssemblyMetadata) -> np.ndarray:
    """Map each vertex to its part_idx via RGB lookup into metadata.parts."""
    if not isinstance(mesh.visual, ColorVisuals):
        raise TypeError(f"scene.ply expected ColorVisuals, got {type(mesh.visual).__name__}")
    visual = cast(ColorVisuals, mesh.visual)
    vcols = np.asarray(visual.vertex_colors[:, :3], dtype=np.uint8)
    # Pack RGB into a single uint32 for dict lookup.
    keys = (vcols[:, 0].astype(np.uint32) << 16) | (vcols[:, 1].astype(np.uint32) << 8) | vcols[:, 2].astype(np.uint32)
    lookup: dict[int, int] = {}
    for p in meta.parts:
        r, g, b = p.color_rgb
        lookup[(int(r) << 16) | (int(g) << 8) | int(b)] = p.part_idx
    out = np.zeros(len(keys), dtype=np.uint16)
    for k, pidx in lookup.items():
        out[keys == k] = pidx
    return out
