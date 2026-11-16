# HomeSense

HomeSense FYP1 is a local Python modular monolith for task-based object searches
using the existing YOLO11n checkpoint and the canonical targets `phone`, `keys`,
and `wallet`. Guidance is image-relative left/center/right; it does not establish
safe routes or physical distance.

## Repository layout

```text
src/homesense/       Production modules: session, camera, detector, perception,
                    stability, and guidance
tests/              All automated tests, including lifecycle regression tests
scripts/            Manual webcam, image-inference, and live-detection tools
models/best.pt      Local trained checkpoint (excluded from Git)
docs/               Component contracts and engineering notes
AGENTS.md           Repository engineering guidance
pyproject.toml      Package metadata and authoritative dependency versions
requirements.txt    Compatibility wrapper for editable installation
```

Private `test_images/`, virtual environments, weights, experiment notebooks, and
generated artifacts stay local and are excluded from Git. The package contains
only `homesense`; scripts, tests, models, and local images are not installed as
application packages.

## Development installation

Use Python 3.12 or later; this repository has been verified with Python 3.12.3.
From the repository root, create the environment if it does not already exist:

```sh
python3.12 -m venv venv
venv/bin/python -m pip install -e .
```

The editable installation makes `homesense` importable from `src/homesense/`
without `PYTHONPATH` or source-path modifications. Reinstall after changing package
metadata or dependencies. Dependency versions are declared only in
`pyproject.toml`; `venv/bin/python -m pip install -r requirements.txt` is the
compatibility equivalent of the editable-install command.

If the existing environment already has the declared dependencies and setuptools
64 or later, an offline installation can use:

```sh
venv/bin/python -m pip install --no-build-isolation --no-deps -e .
```

This offline command does not install or validate missing dependencies. On a fresh
environment use the normal installation command above.

## Automated verification

Run from the repository root:

```sh
venv/bin/python -B -m unittest discover -s tests -v
venv/bin/python -B -c "import homesense; print(homesense.__file__)"
```

The existing suite uses fake sources, detectors, and controlled clocks. See
[tests/README.md](tests/README.md) for test organization and future integration tests.

## Manual diagnostic tools

Run these commands from the repository root after editable installation:

```sh
venv/bin/python scripts/webcam_preview.py --help
venv/bin/python scripts/image_inference_smoke.py --help
venv/bin/python scripts/live_detection_preview.py --help

venv/bin/python scripts/webcam_preview.py
venv/bin/python scripts/image_inference_smoke.py path/to/test_image.jpg --checkpoint models/best.pt
venv/bin/python scripts/live_detection_preview.py --target phone --checkpoint models/best.pt
```

The actual checkpoint remains at `models/best.pt`; supply the trained local file
separately on another checkout. Relative checkpoint and image paths remain relative
to the current working directory, as before. The checkpoint is not bundled in the
package, moved, downloaded, or replaced. There is no fallback to unrelated weights.

The image tool without an image argument loads/checks the model, reports that no
image inference occurred, and exits with status 1. Help commands do not open a
camera or load weights. Camera tools need a local camera and a graphical display;
press `q`, close the window, or use Ctrl+C to stop. Images are not recorded or
uploaded by these tools.

## Design and current limitations

[Perception pipeline and component contracts](docs/perception-pipeline.md) documents
session behavior, worker ownership, freshness, stability, configuration, guidance,
and previously reported manual observations.

[Refactor verification and pending work](docs/repository-structure.md) records the
structural baseline and scope. The previously implemented lifecycle cleanup
hardening is retained. Resource coordination between successive searches remains
pending; this restructure does not introduce an Application Coordinator.
