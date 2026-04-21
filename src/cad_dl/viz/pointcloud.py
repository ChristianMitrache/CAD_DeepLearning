"""Colored point-cloud PLY export from a processed assembly folder.

Reads `points.npz` + `metadata.json` from an assembly directory and writes a
colored `points.ply` (binary, little-endian). The PLY is a viz sidecar — it's
drag-and-droppable into MeshLab / CloudCompare / Blender and shows up with
per-part coloring. Not used for training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from cad_dl.pipeline.io import load_metadata, load_points


def points_npz_to_ply(assembly_dir: Path, out_path: Path | None = None) -> Path:
    """Convert an assembly's points.npz into a colored points.ply sidecar.

    Colors are taken from metadata.parts[part_idx].color_rgb so the PLY matches
    scene.ply's coloring. Default output path: <assembly_dir>/points.ply.
    """
    assembly_dir = Path(assembly_dir)
    out_path = Path(out_path) if out_path else assembly_dir / "points.ply"

    sampled = load_points(assembly_dir)
    meta = load_metadata(assembly_dir)

    rgb = np.zeros((len(sampled.points), 3), dtype=np.uint8)
    for p in meta.parts:
        rgb[sampled.part_idx == p.part_idx] = p.color_rgb

    pc = trimesh.PointCloud(sampled.points, colors=rgb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pc.export(out_path)
    return out_path


def sample_and_export(
    processed_dataset_dir: Path,
    k: int = 10,
    seed: int = 42,
) -> list[Path]:
    """Pick k random processed assemblies and write a points.ply into each.

    Skips assemblies that already have a points.ply. Returns the list of
    written/already-existing PLY paths so callers can log them.
    """
    root = Path(processed_dataset_dir)
    candidates = [d for d in sorted(root.glob("*/")) if (d / "points.npz").exists()]
    if not candidates:
        return []
    rng = np.random.default_rng(seed)
    picks = rng.choice(
        len(candidates), size=min(k, len(candidates)), replace=False
    )
    out_paths: list[Path] = []
    for i in picks:
        d = candidates[int(i)]
        try:
            out_paths.append(points_npz_to_ply(d))
        except Exception as e:
            print(f"[warn] {d.name}: ply export failed: {e}")
    return out_paths
