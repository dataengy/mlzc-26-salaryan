# Project agent instructions

## Contributor Guide

- `src/mlzc_26_salaryan/` contains the Python package: FastAPI app, training, EDA, configuration, and shared model utilities.
- `tests/` contains pytest tests. Keep fixtures synthetic and self-contained; do not test against the downloaded salary dataset.
- `scripts/` holds data download and diagnostics. `data/`, `models/`, and `reports/` contain dataset metadata/examples, the tracked model and metrics, and generated EDA artifacts.
- `notebooks/` contains the analysis notebook; `docs/` contains Russian documentation and operational guides.

### Development commands

- `uv sync --locked` installs the pinned environment.
- `make data`, `make eda`, and `make train` download data, regenerate analysis, and train/save model artifacts.
- `make test` runs the offline pytest suite.
- `make serve` starts the API on port 8080; use `/health` and `/predict` to check it.
- `make docker-check` builds the image and smoke-tests both API endpoints.

### Code and tests

Use Python 3.12, four-space indentation, `snake_case` for modules/functions/variables, and `CapWords` for classes. Reuse helpers in `model_utils.py` and `config.py`. Pin runtime dependencies in `pyproject.toml`; update `uv.lock` with `uv lock` when dependencies change. No formatter or linter is configured.

Use pytest with `test_*.py` files and `test_*` functions. Keep tests deterministic, fast, and offline; use temporary paths and fixed random seeds. Run `make test` before submitting.

### Commits and data

Use short imperative commit summaries; recent history includes Conventional Commit prefixes such as `feat(eda):` and `chore(scripts):`. Keep commits focused. Pull requests should describe changes and checks, link related issues/course requirements, and include screenshots or updated report artifacts when visible output changes. Call out data, model, and configuration changes.

Do not commit secrets or local environment files. Download the salary CSV with `make data` and preserve provenance in `data/source.json`. Describe model metrics as dataset-dependent, not as compensation guarantees.

## Agent preferences

- Use Mermaid diagrams or compact pseudographic schemas more often when they make a workflow, architecture, or comparison easier to understand. Keep them small and omit them when they add no clarity.
- Use the Ponytail and Superpowers skills when they fit the task. Ponytail should guide toward the smallest complete change; Superpowers should guide relevant workflows such as brainstorming, debugging, planning, and verification.
