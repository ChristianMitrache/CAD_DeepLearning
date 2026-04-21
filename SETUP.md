# Setup

## Install

Prereq: **Homebrew** ([install](https://brew.sh)). Everything else is handled.

```bash
make setup
```

Idempotent. Installs `micromamba`, `p7zip`, `direnv`, appends the direnv
hook to your shell rc, then creates the `cad-dl` conda env from
`environment.yml`. Safe to re-run.

## Activate (direnv)

`make setup` already installed `direnv` and appended its shell hook to
your rc. Finish activation by opening a new shell and running **once**:

```bash
direnv allow
```

From then on, `cd` into the repo and the env is active (`cad-dl`, `python`,
`pyright`, `ruff` all resolve to `~/mamba/envs/cad-dl/bin/`). `cd` out and
direnv reverts everything. Activation is **scoped to this directory** — no
global PATH or shell changes, so this won't interfere with a uv/pyenv/
system-Python workflow in your other projects.

Config is committed as [`.envrc`](.envrc); machine-local overrides go in
`.envrc.local` (gitignored).

### If you don't want direnv

Point your tools at the interpreter directly:
`~/mamba/envs/cad-dl/bin/python` / `~/mamba/envs/cad-dl/bin/cad-dl`. No
activation needed — the env lives at a stable path.

## Use

```bash
cad-dl download automate --raw-dir data/raw
cad-dl preprocess --dataset automate --all --workers 8
cad-dl validate --dataset automate
cad-dl gallery --dataset automate --out reports/gallery.html
```

`cad-dl <subcommand> --help` for flags. `CAD_DL_LOG=DEBUG` for verbose logs.

## Add a dependency

Edit `environment.yml` (conda-forge package → `dependencies:`; PyPI/git-only
→ `pip:` block), then re-run `make setup`.

## Gotchas

- **Pins that can't move**: `pythonocc-core=7.7.*` (7.8 removes a symbol occwl
  imports), `pydeprecate==0.3.*` (occwl signature mismatch), `python=3.11`.
- **OpenMP**: `KMP_DUPLICATE_LIB_OK=TRUE` is required — torch and pythonocc
  both ship `libomp.dylib` on macOS.
- **Pyright breaks after `brew upgrade`**: `make doctor` reinstalls node if
  its simdjson link drifted.
- **VSCode shows "package not installed"**: Cmd+Shift+P → "Python: Select
  Interpreter" → `~/mamba/envs/cad-dl/bin/python`.
