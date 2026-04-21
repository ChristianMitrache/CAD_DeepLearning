"""Standardized metadata schema for processed CAD assemblies.

All datasets (AutoMate, ABC, Fusion360, ...) write the SAME on-disk format:
    <processed_root>/<dataset>/<id>/
        scene.ply       binary PLY with per-vertex x,y,z (float32, meters)
                        + red,green,blue (uint8) + part_idx (uint16)
        points.npz      points (N,3) float32, normals (N,3) float32, part_idx (N,) uint16
        metadata.json   AssemblyMetadata (this module)

    <processed_root>/<dataset>/index.parquet
        one row per assembly: id, dataset, n_parts, n_points, bbox_diag, created_at

Bump SCHEMA_VERSION when anything on-disk changes shape; the loader rejects
mismatched versions rather than silently interpreting old data.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class PartRecord:
    """One part within an assembly. `part_idx` is the canonical int id used in
    scene.ply vertex attrs and points.npz `part_idx` array."""
    part_idx: int
    part_id: str
    color_rgb: tuple[int, int, int]  # 0-255
    n_faces: int
    n_points: int


@dataclass
class AssemblyMetadata:
    """Dataset-agnostic metadata. `source` is opaque per-dataset detail."""
    id: str
    dataset: str
    n_faces: int
    n_points: int
    bbox: list[list[float]]  # [[xmin,ymin,zmin],[xmax,ymax,zmax]] in meters
    parts: list[PartRecord]
    source: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> AssemblyMetadata:
        doc = json.loads(s)
        version = doc.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: file has {version!r}, "
                f"code expects {SCHEMA_VERSION}. Re-run preprocess."
            )
        parts = [PartRecord(**p) for p in doc.pop("parts", [])]
        return cls(parts=parts, **doc)

    def bbox_diag(self) -> float:
        """L2 diagonal of the axis-aligned bbox, in meters."""
        (x0, y0, z0), (x1, y1, z1) = self.bbox
        return float(((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) ** 0.5)


INDEX_COLUMNS = ["id", "dataset", "n_parts", "n_points", "bbox_diag", "created_at"]
