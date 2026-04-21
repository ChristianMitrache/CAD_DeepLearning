# cad-dl

Work-in-progress toward a CAD embedding model aligned with text, useful for
CAD→CAD retrieval and eventually text→CAD search. Right now the repo is
mostly the preprocessing side: pulling raw STEP-based datasets (currently
just AutoMate), tessellating them, sampling point clouds, and writing
everything out to a uniform on-disk format so the modeling work doesn't
have to care which dataset a part came from.

Model architecture notes live in [PLAN.md](PLAN.md).

## Getting started

You need Homebrew. Everything else the setup script installs for you
(micromamba, p7zip, a conda env at `~/mamba/envs/cad-dl/`).

```bash
make setup
# open a new shell so the micromamba init block takes effect, then:
micromamba activate cad-dl
```

`make setup` writes a small init block to your shell rc (`~/.zshrc` or
`~/.bashrc`) so `micromamba activate` Just Works afterward. From then on,
every new shell can `micromamba activate cad-dl` to use the env.

If you don't want to activate at all, point your tools at
`~/mamba/envs/cad-dl/bin/python` directly — the env lives at a stable path.
More detail in [SETUP.md](SETUP.md).

## Running the pipeline

Everything is wrapped behind a `cad-dl` command. First download the raw
AutoMate data (about 14 GB, resumable, skips any files already on disk):

```bash
cad-dl download automate --raw-dir data/raw
```

Then preprocess. A run on the full 255K assemblies takes a while, so I
usually smoke-test a small slice first:

```bash
cad-dl preprocess --dataset automate --sample 1000 --workers 8
cad-dl preprocess --dataset automate --all --workers 8
```

Preprocessing is idempotent per-assembly. If a run dies halfway, re-running
picks up where it left off because any assembly with a `metadata.json`
already written is skipped.

Each assembly ends up as a folder under `data/processed/automate/<id>/`
containing:

- `scene.ply` — merged triangle mesh, per-vertex RGB encoding the part id
- `points.npz` — N sampled points with normals and per-point `part_idx`
- `metadata.json` — schema-versioned record: bbox, parts, face/point counts, source provenance

Plus a top-level `index.parquet` summarizing the whole dataset.

## Other subcommands

```
validate   re-read every assembly folder and check it matches the schema
ls         dump index.parquet
reindex    rebuild index.parquet by scanning the folders
sample     re-sample an assembly's point cloud to a different N
thumb      render one assembly to a PNG (mesh or points)
npz2ply    turn points.npz into a colored PLY you can drag into MeshLab
gallery    build a static filterable HTML page over the whole dataset
```

`cad-dl <subcommand> --help` has the flags. `CAD_DL_LOG=DEBUG` turns on
verbose logging.

## Layout

```
src/cad_dl/
  pipeline/     dataset abstract base, CLI, canonical I/O, schema
  datasets/     per-dataset adapters (automate for now; ABC and Fusion360 planned)
  geometry/     STEP tessellation, merged-scene sampling, VTK rendering
  viz/          gallery, thumbnails, point-cloud export
environment.yml the dependency list (conda-forge + a small pip block)
scripts/        setup.sh and misc helpers
```

## Adding a new dataset

Subclass `Dataset` in a new file under `src/cad_dl/datasets/<name>/dataset.py`,
implement the three required methods (`download`, `iter_ids`, `load_scene`),
and decorate the class with `@register("<name>")`. Optionally override
`add_download_args` to expose dataset-specific CLI flags. The registry
picks it up automatically and `cad-dl --dataset <name> ...` just works.

## Dev

```
make check     ruff + pyright
make format    ruff format + autofix
```

If pyright suddenly can't start with a `libsimdjson` error after a brew
upgrade, `make doctor` reinstalls node to fix the link.
