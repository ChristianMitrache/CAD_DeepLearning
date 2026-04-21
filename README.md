# cad-dl

CAD embedding model aligned with text descriptions — CAD→CAD retrieval and
text→CAD search. See [PLAN.md](PLAN.md) for the full architecture and
training plan.

## Quickstart

Prereq: [Homebrew](https://brew.sh).

```bash
make setup         # installs micromamba, direnv, p7zip, and the cad-dl conda env
direnv allow       # one-time, in a fresh shell

cad-dl download automate --raw-dir data/raw
cad-dl preprocess --dataset automate --all --workers 8
```

Auto-activation via `.envrc` is scoped to this directory only — no global
PATH changes, safe alongside uv/pyenv projects. Details: [SETUP.md](SETUP.md).

## Layout

```
src/cad_dl/
  pipeline/       # Dataset ABC, CLI, canonical I/O (scene.ply + points.npz + metadata.json)
  datasets/       # Per-dataset adapters (currently: automate)
  geometry/       # STEP I/O, tessellation, sampling, VTK rendering
  viz/            # Gallery, thumbnails, point-cloud PLY export
scripts/setup.sh  # One-shot bootstrap (wrapped by `make setup`)
environment.yml   # Sole source of truth for dependencies
```

## CLI

All dataset operations run through `cad-dl`. Summary:

| subcommand  | what it does                                                |
|-------------|-------------------------------------------------------------|
| `download`  | Fetch + extract raw data (idempotent, size-checked skip)    |
| `preprocess`| Raw → `scene.ply` + `points.npz` + `metadata.json` + index  |
| `validate`  | Schema-check every processed assembly                       |
| `ls`        | Print `index.parquet`                                       |
| `reindex`   | Rebuild `index.parquet` from folder contents                |
| `sample`    | Re-sample `points.npz` at a new N from `scene.ply`          |
| `thumb`     | Render one assembly thumbnail (mesh or point-cloud)         |
| `npz2ply`   | Convert `points.npz` → colored `points.ply` (MeshLab-ready) |
| `gallery`   | Build a static HTML gallery over `processed/<dataset>/`     |

`cad-dl <subcommand> --help` for flags. `CAD_DL_LOG=DEBUG` for verbose logs.

## Dev

```bash
make check     # ruff + pyright
make format    # ruff format + --fix
make doctor    # repair node/simdjson if pyright breaks after a brew upgrade
```

## Adding a dataset

1. Create `src/cad_dl/datasets/<name>/dataset.py`.
2. Subclass `Dataset`, implement `download(args)`, `iter_ids()`,
   `load_scene(id)`. Optionally override `add_download_args(parser)`.
3. Decorate the class with `@register("<name>")`.

`cad-dl <subcommand> --dataset <name>` then works out of the box.
