"""Unified `cad-dl` CLI entry point.

Subcommands:
  download     fetch + extract the raw dataset
  preprocess   raw -> processed (scene.ply + points.npz + metadata.json + index.parquet)
  validate     schema-check a processed folder
  ls           summarize processed assemblies via index.parquet
  sample       regenerate points.npz at a new N from scene.ply (no raw needed)
  thumb        render a thumbnail from a processed assembly
  gallery      build a static HTML gallery over processed/<dataset>/
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

from cad_dl.geometry.render import vtk_render_meshes, vtk_render_points
from cad_dl.pipeline.dataset import DATASETS, get_dataset, import_all_datasets
from cad_dl.pipeline.io import (
    load_metadata,
    load_points,
    load_scene_mesh,
    rebuild_index,
    validate_assembly,
    write_points_npz,
)
from cad_dl.pipeline.sampling import _recover_vert_part_idx, resample_from_disk
from cad_dl.viz.gallery import build_gallery
from cad_dl.viz.pointcloud import points_npz_to_ply


def _add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. 'automate')")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                   help="Root for raw datasets (expects <dataset>/ subdir)")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"),
                   help="Root for processed output")


def _get_dataset(args):
    raw_root = Path(args.raw_dir) / args.dataset
    return get_dataset(args.dataset, raw_dir=raw_root, processed_dir=args.processed_dir)


def cmd_download(args) -> int:
    ds = _get_dataset(args)
    ds.download(args)
    return 0


def cmd_preprocess(args) -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    ds = _get_dataset(args)
    if args.ids:
        ids = [ln.strip() for ln in Path(args.ids).read_text().splitlines() if ln.strip()]
    elif args.all:
        ids = None
    elif args.sample:
        ids = list(islice(ds.iter_ids(), args.sample))
    else:
        print("One of --ids, --sample N, or --all is required.", file=sys.stderr)
        return 2
    summary = ds.preprocess_all(
        ids=ids, max_points=args.max_points, workers=args.workers,
        sample_ply=args.sample_ply,
    )
    print(f"{args.dataset}: ok={summary['ok']} failed={summary['failed']} "
          f"skipped={summary['skipped']}  log={summary['log']}")
    return 0


def cmd_validate(args) -> int:
    root = Path(args.processed_dir) / args.dataset
    n_ok = n_fail = 0
    for d in sorted(root.glob("*/")):
        if not (d / "metadata.json").exists():
            continue
        try:
            validate_assembly(d)
            n_ok += 1
        except Exception as e:
            print(f"[fail] {d.name}: {e}")
            n_fail += 1
    print(f"validated: ok={n_ok} fail={n_fail}")
    return 0 if n_fail == 0 else 1


def cmd_ls(args) -> int:
    idx = Path(args.processed_dir) / args.dataset / "index.parquet"
    if not idx.exists():
        print(f"No index.parquet at {idx}. Run preprocess (or `cad-dl reindex`).", file=sys.stderr)
        return 1
    df = pd.read_parquet(idx)
    print(df.to_string(index=False))
    print(f"\n{len(df)} assemblies")
    return 0


def cmd_reindex(args) -> int:
    root = Path(args.processed_dir) / args.dataset
    out = rebuild_index(root)
    print(f"wrote {out}")
    return 0


def cmd_sample(args) -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    assembly_dir = Path(args.processed_dir) / args.dataset / args.id
    sampled = resample_from_disk(assembly_dir, n=args.n)
    out = Path(args.out) if args.out else assembly_dir / "points.npz"
    write_points_npz(out, sampled)
    print(f"wrote {out} with {len(sampled.points)} points")
    return 0


def cmd_thumb(args) -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

    assembly_dir = Path(args.processed_dir) / args.dataset / args.id
    meta = load_metadata(assembly_dir)
    if args.mode == "mesh":
        mesh = load_scene_mesh(assembly_dir)
        vert_part_idx = _recover_vert_part_idx(mesh, meta)
        face_pidx = vert_part_idx[mesh.faces[:, 0]]
        meshes = []
        for p in meta.parts:
            face_mask = face_pidx == p.part_idx
            if not face_mask.any():
                continue
            sub = mesh.submesh([np.where(face_mask)[0]], append=True)
            if isinstance(sub, list): 
                continue
            meshes.append((sub, tuple(c / 255 for c in p.color_rgb)))
        vtk_render_meshes(meshes, Path(args.out), size=args.size)
    else:  # points
        sampled = load_points(assembly_dir)
        clouds = []
        for p in meta.parts:
            mask = sampled.part_idx == p.part_idx
            if not mask.any():
                continue
            clouds.append((sampled.points[mask], tuple(c / 255 for c in p.color_rgb)))
        vtk_render_points(clouds, Path(args.out), size=args.size)
    print(f"wrote {args.out}")
    return 0


def cmd_npz2ply(args) -> int:
    """Convert an assembly's points.npz -> colored points.ply viz sidecar."""
    assembly_dir = Path(args.processed_dir) / args.dataset / args.id
    out = points_npz_to_ply(assembly_dir, out_path=args.out)
    print(f"wrote {out}")
    return 0


def cmd_gallery(args) -> int:
    idx_path = Path(args.processed_dir) / args.dataset / "index.parquet"
    if not idx_path.exists():
        print(f"No index.parquet at {idx_path}", file=sys.stderr)
        return 1
    df = pd.read_parquet(idx_path)

    # Records for the gallery: use a thumb dir if given, otherwise assume thumbs
    # live under processed/<dataset>/<id>/thumb.png (not generated by default).
    thumb_dir = Path(args.thumb_dir) if args.thumb_dir else None
    records = df.to_dict(orient="records")

    columns = [
        {"key": "n_parts", "label": "parts", "sortable": True, "filterable": True},
        {"key": "n_points", "label": "points", "sortable": True, "filterable": False},
        {"key": "bbox_diag", "label": "bbox", "sortable": True, "filterable": False},
        {"key": "id", "label": "id", "sortable": True, "filterable": False},
    ]

    build_gallery(
        records=records,
        out_html=Path(args.out),
        thumb_dir=thumb_dir or (Path(args.processed_dir) / args.dataset),
        columns=columns,
        title=f"{args.dataset} gallery",
        badge_keys=["n_parts", "n_points"],
    )
    print(f"wrote {args.out} ({len(records)} entries)")
    return 0


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("CAD_DL_LOG", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(prog="cad-dl", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # `download` dispatches on --dataset: each registered dataset contributes
    # its own flags via add_download_args so the help text is typed per-dataset.
    import_all_datasets()
    p = sub.add_parser("download", help="Fetch + extract a raw dataset")
    download_sub = p.add_subparsers(dest="dataset", required=True)
    for name, cls in sorted(DATASETS.items()):
        ds_parser = download_sub.add_parser(name, help=f"Download the {name} dataset")
        ds_parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                               help="Root for raw datasets (expects <dataset>/ subdir)")
        ds_parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"),
                               help="Root for processed output")
        cls.add_download_args(ds_parser)
        ds_parser.set_defaults(func=cmd_download)

    p = sub.add_parser("preprocess", help="Raw -> standardized processed/")
    _add_dataset_args(p)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--ids", type=Path, help="File with one id per line")
    g.add_argument("--sample", type=int, help="Take first N from iter_ids()")
    g.add_argument("--all", action="store_true", help="Process every id")
    p.add_argument("--max-points", type=int, default=10000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--sample-ply", type=int, default=10,
                   help="Write a viz points.ply into this many random assemblies after preprocess (0 to skip)")
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("validate", help="Schema-check processed/<dataset>/*")
    _add_dataset_args(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("ls", help="Show index.parquet rows")
    _add_dataset_args(p)
    p.set_defaults(func=cmd_ls)

    p = sub.add_parser("reindex", help="Rebuild index.parquet from folder contents")
    _add_dataset_args(p)
    p.set_defaults(func=cmd_reindex)

    p = sub.add_parser("sample", help="Re-sample an assembly's points.npz from scene.ply")
    _add_dataset_args(p)
    p.add_argument("--id", required=True)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--out", type=Path, help="Default: overwrite the assembly's points.npz")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("thumb", help="Render a thumbnail from a processed assembly")
    _add_dataset_args(p)
    p.add_argument("--id", required=True)
    p.add_argument("--mode", choices=["mesh", "points"], default="mesh")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_thumb)

    p = sub.add_parser("npz2ply", help="Convert one assembly's points.npz -> viz points.ply")
    _add_dataset_args(p)
    p.add_argument("--id", required=True)
    p.add_argument("--out", type=Path, help="Default: <assembly_dir>/points.ply")
    p.set_defaults(func=cmd_npz2ply)

    p = sub.add_parser("gallery", help="Build an HTML gallery over processed/")
    _add_dataset_args(p)
    p.add_argument("--thumb-dir", type=Path, help="Override thumb root")
    p.add_argument("--out", type=Path, default=Path("reports/gallery.html"))
    p.set_defaults(func=cmd_gallery)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
