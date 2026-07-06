# Stage 2 Base Architecture Sprint

## Scope
- Promote the default runner config from Stage 1 tiny sanity to the roadmap's Stage 2 base architecture.
- Keep Stage 1 reproducible through explicit CLI overrides and saved experiment artifacts.
- Add a repeatable comparison tool for completed experiment runs.

## Assumptions
- Stage 1 pipeline is verified and recorded in `docs/experiment-log.md`.
- Stage 2 should use the same architecture family, only larger config values.
- Raw run artifacts remain local under `runs/` and are not committed.

## Architectural Decisions
- `main.py` remains a thin experiment runner.
- Default config becomes:
  - `d_model=128`
  - `num_heads=4`
  - `num_layers=4`
  - `ffn_hidden=512`
  - `context_length=16`
- Default run artifacts move to `runs/stage2/<timestamp>/`.
- Comparison reads canonical artifacts: `config.json`, `metrics.csv`, and `summary.json`.

## Tasks
1. Add Stage 2 tracking to `tasks/todo.md`.
2. Make runner defaults stage-aware.
3. Add a small comparison script for run directories.
4. Add focused tests for Stage 2 defaults and comparison summaries.
5. Update experiment tracking docs.
6. Run tests and a short Stage 2 smoke run.

## Risks
- Stage 2 defaults run slower than Stage 1 defaults.
- Comparing short runs against longer runs can be misleading unless steps and parameter count are visible.
- Existing Stage 1 artifacts predate stage-aware defaults, so comparison must tolerate older artifact shapes.

## Verification Strategy
- Run `python -m unittest discover -s tests -v`.
- Run `scripts/compare_runs.py` against existing Stage 1 runs.
- Run a short Stage 2 smoke command with explicit low step count.
- Confirm default run path is `runs/stage2/...`.

## Verification Results
- Unit tests passed: `.venv/bin/python -m unittest discover -s tests -v`.
- Comparison script passed against saved Stage 1 runs.
- Stage 2 smoke run passed with default config: `runs/stage2/20260617-142727`.
- Smoke loss decreased from `2.8431` to `0.9335` over 20 steps on MPS.
- Default run artifacts now write under `runs/stage2/<timestamp>/`.
- Stage 2 quality still needs a longer run before comparing accuracy against Stage 1.
