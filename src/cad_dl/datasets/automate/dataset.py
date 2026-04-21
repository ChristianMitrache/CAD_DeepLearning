"""AutoMate dataset adapter for the cad_dl pipeline.

Implements the 3 required Dataset methods:
  - download(): fetch + extract from Zenodo record 7776208
  - iter_ids(): read assemblies.parquet
  - load_scene(): JSON -> tessellate STEP parts -> merged scene with mm->m scaling
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm

from cad_dl.datasets.automate.loader import AutoMateAssembly
from cad_dl.geometry.sampling import MergedScene, merge_placed
from cad_dl.geometry.step import MM_TO_M, load_step_shape, shape_to_trimesh
from cad_dl.pipeline.dataset import Dataset, register

log = logging.getLogger(__name__)

ZENODO_API = "https://zenodo.org/api/records/7776208"
DEFAULT_FILES = {
    "README.md", "config_encodings.json",
    "assemblies.parquet", "parts.parquet", "mates.parquet",
    "assemblies.zip", "step.zip",
}
PARASOLID_FILES = {"parasolid.zip"}


@register("automate")
class AutoMate(Dataset):
    """AutoMate (Autodesk, Zenodo 7776208). 255K mechanical assemblies."""

    @classmethod
    def add_download_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--parasolid", action="store_true",
                            help="Also fetch the (large) parasolid.zip")
        parser.add_argument("--skip-extract", action="store_true",
                            help="Download only; don't unzip")
        parser.add_argument("--only", nargs="*",
                            help="Explicit Zenodo file keys (overrides defaults)")

    def download(self, args: argparse.Namespace) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        log.info("Data dir: %s", self.raw_dir)
        log.info("Free disk here: %.1f GB", _disk_free_gb(self.raw_dir))

        want = set(DEFAULT_FILES)
        if args.parasolid:
            want |= PARASOLID_FILES
        if args.only:
            want = set(args.only)

        manifest = _fetch_manifest()
        entries = [e for e in manifest if e["key"] in want]
        missing = want - {e["key"] for e in entries}
        if missing:
            log.error("files not in Zenodo manifest: %s", missing)
            sys.exit(1)

        # If every target file is already present and size-matches manifest,
        # bail early — no MD5 scan, no curl roundtrip.
        if _all_present(entries, self.raw_dir):
            log.info("All %d files already present in %s; skipping download.",
                     len(entries), self.raw_dir)
        else:
            total_bytes = sum(e["size"] for e in entries)
            log.info("Selected files: %s", sorted(want))
            log.info("Total download size: %.2f GB", total_bytes / 1024**3)
            for entry in tqdm(entries, desc="automate download", unit="file"):
                _download_one(entry, self.raw_dir)

        if args.skip_extract:
            return

        zip_extract_map = {
            "step.zip": self.raw_dir / "step",
            "parasolid.zip": self.raw_dir / "parasolid",
            "assemblies.zip": self.raw_dir / "assemblies",
        }
        zips_to_extract = [
            (n, o) for n, o in zip_extract_map.items() if (self.raw_dir / n).exists()
        ]
        for zip_name, out_dir in tqdm(zips_to_extract, desc="extract", unit="zip"):
            marker = self.raw_dir / f".{zip_name}.extracted"
            _extract_zip(self.raw_dir / zip_name, out_dir, marker)

    def iter_ids(self) -> Iterator[str]:
        parquet = self.raw_dir / "assemblies.parquet"
        if not parquet.exists():
            raise FileNotFoundError(f"{parquet} — run `cad-dl download --dataset automate` first")
        df = pd.read_parquet(parquet)
        id_col = "assemblyId" if "assemblyId" in df.columns else df.columns[0]
        for aid in df[id_col].tolist():
            yield str(aid)

    def load_scene(self, assembly_id: str) -> tuple[MergedScene, dict]:
        deflection = 0.5
        assembly = AutoMateAssembly.from_json(assembly_id, self.raw_dir)

        part_mesh_cache: dict[str, trimesh.Trimesh | None] = {}
        placements: list[tuple[str, np.ndarray]] = []
        n_hidden = sum(1 for o in assembly.occurrences if o.hidden)
        part_errors: list[str] = []

        for occ, step_path in assembly.step_occurrences(self.raw_dir):
            part = assembly.parts[occ.part_index]
            if not step_path.exists():
                part_errors.append(f"{part.id}: missing step file")
                continue
            try:
                if part.id not in part_mesh_cache:
                    shape = load_step_shape(step_path)
                    m = shape_to_trimesh(shape, deflection=deflection)
                    if m is not None and len(m.vertices) > 0:
                        m = m.copy()
                        m.apply_scale(MM_TO_M)
                    else:
                        m = None
                    part_mesh_cache[part.id] = m
                if part_mesh_cache[part.id] is None:
                    continue
                placements.append((part.id, occ.transform))
            except Exception as e:
                part_errors.append(f"{part.id}: {type(e).__name__}: {e}")

        scene = merge_placed(placements, part_mesh_cache)
        source = {
            "n_occurrences": len(assembly.occurrences),
            "n_hidden": n_hidden,
            "n_mates": len(assembly.mates),
            "has_all_step": assembly.has_all_step,
            "deflection": deflection,
            "raw_assembly_json": str(
                self.raw_dir / "assemblies" / "assemblies" / f"{assembly_id}.json"
            ),
        }
        if part_errors:
            source["part_errors"] = part_errors
        return scene, source


# --------------------------------------------------------------------- download helpers

def _fetch_manifest() -> list[dict]:
    with urllib.request.urlopen(ZENODO_API, timeout=30) as resp:
        payload = json.load(resp)
    return payload["files"]


def _md5_of(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _curl_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # curl renders its own byte-level progress bar; don't stack tqdm on top.
    cmd = ["curl", "-L", "-C", "-", "--fail", "--retry", "5", "--retry-delay", "5",
           "-o", str(dest), url]
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _expected_md5(entry: dict) -> str:
    checksum = entry["checksum"]
    assert checksum.startswith("md5:"), checksum
    return checksum[4:]


def _all_present(entries: list[dict], data_dir: Path) -> bool:
    """True iff every manifest entry exists on disk with matching byte size.

    Size is a cheap first-pass check; `_download_one` still does the real
    MD5 verification when called, so this only affects whether we *try* to
    download. If a file is truncated, sizes won't match → we proceed normally.
    """
    for entry in entries:
        dest = data_dir / entry["key"]
        if not dest.exists() or dest.stat().st_size != entry["size"]:
            return False
    return True


def _download_one(entry: dict, data_dir: Path) -> Path:
    name = entry["key"]
    size_gb = entry["size"] / 1024**3
    url = entry["links"]["self"]
    dest = data_dir / name
    want_md5 = _expected_md5(entry)

    if dest.exists():
        have = _md5_of(dest)
        if have == want_md5:
            log.info("[skip] %s (%.2f GB) MD5 matches", name, size_gb)
            return dest
        log.warning("[warn] %s MD5 mismatch (have %s, want %s); re-downloading",
                    name, have, want_md5)
        dest.unlink()

    log.info("[get ] %s (%.2f GB)", name, size_gb)
    _curl_download(url, dest)
    have = _md5_of(dest)
    if have != want_md5:
        raise RuntimeError(f"MD5 mismatch for {name}: have {have}, want {want_md5}")
    log.info("[ok  ] %s MD5 verified", name)
    return dest


def _extract_zip(zip_path: Path, out_dir: Path, marker: Path) -> None:
    if marker.exists():
        log.info("[skip] %s already extracted (%s)", zip_path.name, marker)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("[unzip] %s -> %s", zip_path.name, out_dir)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for member in tqdm(names, desc=f"unzip {zip_path.name}", unit="file", leave=False):
            zf.extract(member, out_dir)
    marker.touch()


def _disk_free_gb(path: Path) -> float:
    stats = shutil.disk_usage(path if path.exists() else path.parent)
    return stats.free / 1024**3
