"""Abstract `Dataset` base class + `DATASETS` registry.

To add a new dataset: subclass `Dataset`, implement `add_download_args`,
`download`, `iter_ids`, `load_scene`; decorate with `@register("myname")`.
The shared `preprocess_all` handles parallel execution, validation, and
index.parquet.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import logging
import pkgutil
import traceback
from abc import ABC, abstractmethod
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import ClassVar

from tqdm import tqdm

from cad_dl.geometry.sampling import MergedScene
from cad_dl.pipeline.io import rebuild_index, write_assembly
from cad_dl.pipeline.sampling import sample_scene
from cad_dl.viz.pointcloud import sample_and_export

log = logging.getLogger(__name__)

DATASETS: dict[str, type[Dataset]] = {}


def register(name: str):
    """Class decorator registering a Dataset subclass under `name`."""
    def deco(cls: type[Dataset]) -> type[Dataset]:
        cls.name = name
        DATASETS[name] = cls
        return cls
    return deco


class Dataset(ABC):
    """Per-dataset adapter. Three methods, rest is shared."""
    name: ClassVar[str]

    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir) / self.name

    # --- per-dataset ---
    @classmethod
    def add_download_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register dataset-specific `download` flags on `parser`.

        Default: no extra flags (intentionally not @abstractmethod — subclasses
        are free to skip it; ruff B027 is suppressed file-wide in pyproject).
        Override to add flags like `--parasolid`. The CLI layer parses these
        and passes the resulting `args` Namespace straight into `download(args)`.
        """

    @abstractmethod
    def download(self, args: argparse.Namespace) -> None:
        """Populate self.raw_dir from the canonical source.

        `args` is the parsed argparse Namespace — it includes both the shared
        flags (dataset, raw_dir, processed_dir) and whatever `add_download_args`
        registered. Subclasses read fields by name (e.g. `args.parasolid`).
        """

    @abstractmethod
    def iter_ids(self) -> Iterator[str]:
        """Yield ids for every processable assembly in self.raw_dir."""

    @abstractmethod
    def load_scene(self, assembly_id: str) -> tuple[MergedScene, dict]:
        """Return (MergedScene, source_metadata_dict) for one id.

        `source_metadata_dict` is stored verbatim in metadata.json["source"].
        """

    # --- shared ---
    def preprocess_one(
        self,
        assembly_id: str,
        *,
        max_points: int = 10000,
    ) -> dict:
        """Load, sample, write a single assembly. Returns a status dict."""
        out_dir = self.processed_dir / assembly_id
        if (out_dir / "metadata.json").exists():
            return {"id": assembly_id, "success": True, "skipped": True}
        try:
            scene, source = self.load_scene(assembly_id)
            if len(scene.mesh.faces) == 0:
                return {"id": assembly_id, "success": False, "error": "empty scene", "skipped": False}
            source.setdefault("point_sampling", {
                "method": "poisson_disk", "max_points": max_points, "packing": 0.75,
            })
            sampled = sample_scene(scene, max_points)
            meta = write_assembly(
                out_dir,
                dataset=self.name,
                assembly_id=assembly_id,
                scene=scene,
                sampled=sampled,
                source=source,
            )
            return {
                "id": assembly_id,
                "success": True,
                "skipped": False,
                "n_faces": meta.n_faces,
                "n_points": meta.n_points,
                "n_parts": len(meta.parts),
            }
        except Exception as e:
            return {
                "id": assembly_id,
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=3),
                "skipped": False,
            }

    def preprocess_all(
        self,
        ids: list[str] | None = None,
        *,
        max_points: int = 10000,
        workers: int = 4,
        rebuild_index_after: bool = True,
        sample_ply: int = 10,
    ) -> dict:
        """Parallel preprocess + index.parquet build. Returns a summary dict.

        `sample_ply`: after processing, write a colored `points.ply` sidecar
        into N random assembly folders so users can eyeball the result in
        MeshLab/CloudCompare. Set to 0 to skip.
        """
        if ids is None:
            ids = list(self.iter_ids())
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        log_path = self.processed_dir / "preprocess_log.jsonl"
        n_ok = n_fail = n_skip = 0

        pool_kwargs: dict = {"max_workers": workers}
        with contextlib.suppress(TypeError):
            pool_kwargs["max_tasks_per_child"] = 100  # Python 3.11+

        with log_path.open("w") as log, ProcessPoolExecutor(**pool_kwargs) as pool:
            futures = [
                pool.submit(_worker_preprocess_one, self.__class__, self.raw_dir,
                            self.processed_dir.parent, aid, max_points)
                for aid in ids
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{self.name} preprocess"):
                res = fut.result()
                log.write(json.dumps(res) + "\n")
                if res.get("skipped"):
                    n_skip += 1
                elif res.get("success"):
                    n_ok += 1
                else:
                    n_fail += 1

        if rebuild_index_after:
            rebuild_index(self.processed_dir)

        ply_paths: list = []
        if sample_ply > 0:
            ply_paths = sample_and_export(self.processed_dir, k=sample_ply)
            if ply_paths:
                print(f"Wrote {len(ply_paths)} sample points.ply files; first: {ply_paths[0]}")

        return {
            "ok": n_ok, "failed": n_fail, "skipped": n_skip,
            "log": str(log_path), "ply_samples": [str(p) for p in ply_paths],
        }


def _worker_preprocess_one(cls, raw_dir, processed_root, assembly_id, max_points):
    """Top-level shim so ProcessPoolExecutor can pickle it."""
    inst = cls(raw_dir=raw_dir, processed_dir=processed_root)
    return inst.preprocess_one(assembly_id, max_points=max_points)


def get_dataset(name: str, raw_dir: Path, processed_dir: Path) -> Dataset:
    """Instantiate a registered dataset by name. Triggers imports so decorators run."""
    import_all_datasets()
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; registered: {sorted(DATASETS)}")
    return DATASETS[name](raw_dir=raw_dir, processed_dir=processed_dir)


def import_all_datasets() -> None:
    """Import every cad_dl.datasets.* subpackage so @register runs.

    Imports `cad_dl.datasets` locally to avoid a circular import at module load
    (cad_dl.datasets.* subpackages import from this module).

    A subpackage without a `dataset.py` (e.g. scripts-only) is silently
    skipped. Any OTHER import failure — e.g. a dataset.py whose own imports
    break because the env is missing a dep — is logged loudly, because that
    used to silently empty the registry and produce a confusing
    `unknown dataset` KeyError downstream.
    """
    # Imported inside the function to avoid a circular import at load time:
    # cad_dl.datasets.* subpackages import from this module.
    import cad_dl.datasets as pkg
    for mod in pkgutil.iter_modules(pkg.__path__):
        if not mod.ispkg:
            continue
        target = f"cad_dl.datasets.{mod.name}.dataset"
        try:
            importlib.import_module(target)
        except ModuleNotFoundError as e:
            # Only swallow "no dataset.py in this subpackage"; re-raise if the
            # missing module is a transitive dep (OCC, trimesh, etc.).
            if e.name == target:
                continue
            log.error("failed to import %s: missing module %r. "
                      "Run `make setup` to install dependencies.", target, e.name)
            raise
