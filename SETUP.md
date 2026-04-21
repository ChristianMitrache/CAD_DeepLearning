# Setup

## Install

Prereq: [Homebrew](https://brew.sh).

```bash
make setup
```

Idempotent. Installs `micromamba` + `p7zip`, writes a small init block to
your shell rc (`~/.zshrc` or `~/.bashrc`) so `micromamba activate` works,
then creates the `cad-dl` conda env from `environment.yml`. Safe to re-run.

The init block is marked with `# >>> cad-dl micromamba init >>>` / `<<<` so
re-running setup can replace it cleanly instead of appending duplicates.

## Activate

Open a new shell (so the init block loads) then:

```bash
micromamba activate cad-dl
cad-dl --help
```

To deactivate: `micromamba deactivate`.

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
→ `pip:` block), then re-run `make setup`. `micromamba install --file` only
installs what changed.

## Gotchas

- **Pins that can't move**: `pythonocc-core=7.7.*` (7.8 removes a symbol
  occwl imports), `pydeprecate==0.3.*` (occwl signature mismatch),
  `python=3.11`.
- **OpenMP**: if a script imports both `torch` and `OCC` in the same process,
  set `KMP_DUPLICATE_LIB_OK=TRUE` first (they both ship `libomp.dylib` on
  macOS).
- **Off-screen rendering**: VTK-based thumbnailing needs
  `PYVISTA_OFF_SCREEN=true`.
- **`micromamba: Shell not initialized`**: you're in a shell that was open
  before `make setup` ran, or in a shell type (not zsh/bash) we don't
  auto-configure. Open a new zsh/bash shell, or add the init block manually.
- **Pyright breaks after `brew upgrade`**: `make doctor` reinstalls node if
  its simdjson link drifted.
- **VSCode shows "package not installed"**: Cmd+Shift+P → "Python: Select
  Interpreter" → `~/mamba/envs/cad-dl/bin/python`.

## Verifying the fresh-user flow

If you've been working in this repo you already have micromamba, the conda
env, and the rc init block — so `make setup` no-ops and doesn't really
prove a new user's experience. To simulate one without wrecking your own
config:

```bash
# Exercise the micromamba-install code path
rm ~/.local/bin/micromamba

# Launch a shell that ignores your ~/.zshrc, with only brew + system tools on PATH
env -i HOME=$HOME PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin zsh -f

# Inside that shell:
cd /path/to/CAD_DeepLearning
make setup

# Exit the bare shell, open a regular new terminal, and verify:
micromamba activate cad-dl
cad-dl --help
```

To also exercise env creation (adds 5–15 min to the rebuild), remove the
env first:

```bash
~/.local/bin/micromamba env remove -n cad-dl -y --root-prefix ~/mamba
```

`env -i` wipes environment variables; `zsh -f` skips rc files — together
they approximate a shell with no prior cad-dl state, while leaving your
real `~/.zshrc` untouched.
