"""Read/write standardized assembly folders.

`write_assembly` is the ONLY supported way to emit a processed assembly.
`load_assembly` + `validate_assembly` are the reciprocal readers; downstream
code should use them rather than touching files directly.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from cad_dl.geometry.sampling import MergedScene
from cad_dl.pipeline.schema import SCHEMA_VERSION, AssemblyMetadata, PartRecord
from cad_dl.viz.colors import color_for_id


@dataclass
class SampledPoints:
    """N points sampled from a MergedScene. `part_idx` indexes metadata.parts."""
    points: np.ndarray       # (N, 3) float32
    normals: np.ndarray      # (N, 3) float32
    part_idx: np.ndarray     # (N,) uint16

    def __post_init__(self) -> None:
        n = len(self.points)
        assert self.points.shape == (n, 3) and self.points.dtype == np.float32, "points: (N,3) float32"
        assert self.normals.shape == (n, 3) and self.normals.dtype == np.float32, "normals: (N,3) float32"
        assert self.part_idx.shape == (n,) and self.part_idx.dtype == np.uint16, "part_idx: (N,) uint16"


# --------------------------------------------------------------------- write

def _build_scene_ply(scene: MergedScene) -> trimesh.Trimesh:
    """Attach per-vertex part_idx + RGB to the merged mesh, ready to export."""
    mesh = scene.mesh
    n_verts = len(mesh.vertices)
    # Derive per-vertex part_idx by picking the first face each vertex appears in.
    # (A vertex is shared by multiple faces of the same part in practice — parts are
    # disjoint in the placements stage — so any face-of-this-vertex maps back to the
    # same part id.)
    part_idx_lookup = {pid: i for i, pid in enumerate(scene.part_ids)}
    face_part_idx = np.array(
        [part_idx_lookup[p] for p in scene.face_part_ids], dtype=np.uint16
    )
    vert_part_idx = np.full(n_verts, 0, dtype=np.uint16)
    # Assign via face -> vertex. Later writes win, but they all agree for shared verts.
    for face, pidx in zip(mesh.faces, face_part_idx, strict=True):
        vert_part_idx[face] = pidx

    colors = np.zeros((n_verts, 4), dtype=np.uint8)
    colors[:, 3] = 255
    for pidx, pid in enumerate(scene.part_ids):
        c = color_for_id(pid)
        rgb = np.array([int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)], dtype=np.uint8)
        mask = vert_part_idx == pidx
        colors[mask, :3] = rgb

    out = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        process=False,
        vertex_colors=colors,
    )
    # Note: part_idx isn't stored as a PLY vertex attribute — trimesh's PLY
    # writer doesn't round-trip custom vertex attrs. Instead, readers recover
    # per-vertex part_idx from RGB by matching back into metadata.parts[].color_rgb.
    # The colors are authoritative; see pipeline/sampling.py::_recover_vert_part_idx.
    return out


def write_scene_ply(path: Path, scene: MergedScene) -> None:
    """Write scene.ply (binary, with per-vertex part_idx + rgb)."""
    mesh = _build_scene_ply(scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    # trimesh's export(file_type=...) returns str|bytes|dict depending on the format;
    # for binary PLY it's bytes. Writing via `.export(file_obj)` is cleaner.
    with path.open("wb") as f:
        mesh.export(f, file_type="ply", encoding="binary")


def write_points_npz(path: Path, sampled: SampledPoints) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=sampled.points,
        normals=sampled.normals,
        part_idx=sampled.part_idx,
    )


def build_metadata(
    assembly_id: str,
    dataset: str,
    scene: MergedScene,
    sampled: SampledPoints,
    source: dict,
) -> AssemblyMetadata:
    """Derive AssemblyMetadata from a scene + its sampled points."""
    mesh = scene.mesh
    if len(mesh.vertices) > 0:
        bbox = [mesh.vertices.min(axis=0).tolist(), mesh.vertices.max(axis=0).tolist()]
    else:
        bbox = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # Face counts per part_idx
    part_idx_lookup = {pid: i for i, pid in enumerate(scene.part_ids)}
    face_counts = np.zeros(len(scene.part_ids), dtype=np.int64)
    for pid in scene.face_part_ids:
        face_counts[part_idx_lookup[pid]] += 1

    # Point counts per part_idx
    point_counts = np.bincount(sampled.part_idx, minlength=len(scene.part_ids))

    parts = []
    for i, pid in enumerate(scene.part_ids):
        c = color_for_id(pid)
        parts.append(PartRecord(
            part_idx=i,
            part_id=pid,
            color_rgb=(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)),
            n_faces=int(face_counts[i]),
            n_points=int(point_counts[i]),
        ))

    return AssemblyMetadata(
        id=assembly_id,
        dataset=dataset,
        n_faces=len(mesh.faces),
        n_points=len(sampled.points),
        bbox=bbox,
        parts=parts,
        source=source,
    )


def write_assembly(
    out_dir: Path,
    *,
    dataset: str,
    assembly_id: str,
    scene: MergedScene,
    sampled: SampledPoints,
    source: dict,
    validate: bool = True,
) -> AssemblyMetadata:
    """Write a standardized assembly folder. THE canonical write entry point.

    Produces scene.ply, points.npz, metadata.json. Optionally re-reads and
    validates before returning, so a silent write that doesn't round-trip fails loud.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(assembly_id, dataset, scene, sampled, source)

    write_scene_ply(out_dir / "scene.ply", scene)
    write_points_npz(out_dir / "points.npz", sampled)
    (out_dir / "metadata.json").write_text(metadata.to_json())

    if validate:
        validate_assembly(out_dir)
    return metadata


# --------------------------------------------------------------------- read

def load_metadata(assembly_dir: Path) -> AssemblyMetadata:
    return AssemblyMetadata.from_json((Path(assembly_dir) / "metadata.json").read_text())


def load_scene_mesh(assembly_dir: Path) -> trimesh.Trimesh:
    """Load scene.ply."""
    mesh = trimesh.load(Path(assembly_dir) / "scene.ply", process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh from scene.ply, got {type(mesh).__name__}")
    return mesh


def load_points(assembly_dir: Path) -> SampledPoints:
    with np.load(Path(assembly_dir) / "points.npz") as z:
        return SampledPoints(
            points=z["points"].astype(np.float32, copy=False),
            normals=z["normals"].astype(np.float32, copy=False),
            part_idx=z["part_idx"].astype(np.uint16, copy=False),
        )


# --------------------------------------------------------------------- validate

def validate_assembly(assembly_dir: Path) -> None:
    """Raise a clear error if the folder deviates from the canonical format.

    Called at the end of `write_assembly`; also runnable standalone via
    `cad-dl validate`.
    """
    d = Path(assembly_dir)
    for required in ("scene.ply", "points.npz", "metadata.json"):
        if not (d / required).exists():
            raise FileNotFoundError(f"{d} missing required file: {required}")

    meta = load_metadata(d)
    if meta.schema_version != SCHEMA_VERSION:
        raise ValueError(f"{d}: schema_version {meta.schema_version} != expected {SCHEMA_VERSION}")

    # Points structural check
    sampled = load_points(d)  # triggers SampledPoints.__post_init__ asserts
    if len(sampled.points) != meta.n_points:
        raise ValueError(f"{d}: metadata.n_points={meta.n_points} != points.npz len={len(sampled.points)}")
    if sampled.part_idx.size > 0:
        max_idx = int(sampled.part_idx.max())
        if max_idx >= len(meta.parts):
            raise ValueError(f"{d}: points.part_idx max {max_idx} exceeds parts count {len(meta.parts)}")

    # parts[i].part_idx must be 0..n-1 contiguous
    for i, p in enumerate(meta.parts):
        if p.part_idx != i:
            raise ValueError(f"{d}: metadata.parts[{i}].part_idx={p.part_idx} (expected {i})")

    # scene.ply minimal check: loads and has faces
    mesh = load_scene_mesh(d)
    if len(mesh.faces) != meta.n_faces:
        raise ValueError(f"{d}: metadata.n_faces={meta.n_faces} != scene.ply faces={len(mesh.faces)}")


# --------------------------------------------------------------------- index

def write_index(processed_dataset_dir: Path, records: list[AssemblyMetadata]) -> Path:
    """Write <processed>/<dataset>/index.parquet summarizing processed assemblies."""
    import pandas as pd
    now = _dt.datetime.now(_dt.UTC).isoformat()
    rows = [
        {
            "id": m.id,
            "dataset": m.dataset,
            "n_parts": len(m.parts),
            "n_points": m.n_points,
            "bbox_diag": m.bbox_diag(),
            "created_at": now,
        }
        for m in records
    ]
    out = Path(processed_dataset_dir) / "index.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def rebuild_index(processed_dataset_dir: Path) -> Path:
    """Scan <dataset>/*/metadata.json and rewrite index.parquet from scratch."""
    d = Path(processed_dataset_dir)
    records = []
    for meta_path in sorted(d.glob("*/metadata.json")):
        try:
            records.append(AssemblyMetadata.from_json(meta_path.read_text()))
        except Exception as e:
            print(f"[warn] skipping {meta_path}: {e}")
    return write_index(d, records)
