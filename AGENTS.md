# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: Entry script placeholder.
- `chart2bar/`: Accuracy calc and bar plots
  - `calculate_accuracy.py`, `visualization_barplot.py`
  - `data/` input CSVs, `output/` generated images/CSVs
- `chart2radar/`: Plotly radar figures (`plot_radar.py`, `output/`)
- `chart2violin/`: Violin plots and outputs
- `pyproject.toml`, `uv.lock`: Python 3.12 project and locked deps

## Build, Test, and Development Commands
- Environment (preferred): `uv sync` (creates venv and installs deps)
- Run scripts: `uv run python chart2bar/calculate_accuracy.py --input chart2bar/data/OSS\ Benchmarking\ Results\ -\ Eurorad.csv --output chart2bar/output/accuracy_results.csv`
- Run radar plot: `uv run python chart2radar/plot_radar.py`
- Without uv, install minimal deps: `pip install pandas numpy matplotlib plotly kaleido openpyxl`

## Coding Style & Naming Conventions
- Python, PEP 8, 4‑space indentation, limit ~99 chars.
- Use type hints and concise docstrings for new/edited functions.
- Filenames and functions: `snake_case`; constants: `UPPER_SNAKE`; classes: `CamelCase`.
- Keep paths relative to each script’s folder; write outputs to the local `output/` subdir.

## Testing Guidelines
- Framework: `pytest` (add if/when tests are introduced).
- Location: `tests/`; name files `test_*.py`; mirror module names.
- Focus on pure functions (e.g., `calculate_exact_match_accuracy`) with small, explicit fixtures.
- Run: `pytest -q` (optionally `uv run pytest -q`). Aim to cover data‑handling branches.

## Commit & Pull Request Guidelines
- Git history is brief; adopt Conventional Commits going forward: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Commits: small, focused; present‑tense subject line; include context in body if data sources or assumptions changed.
- PRs: clear description, what/why, sample command and resulting artifact path (e.g., `chart2bar/output/*.png`), and before/after visuals when modifying plots. Link issues where relevant.

## Security & Configuration Tips
- Python `>=3.12` (see `.python-version`). Prefer `uv` for reproducible installs (`uv.lock`).
- Do not commit large datasets; keep generated artifacts small when possible. If updating CSVs, document provenance.
