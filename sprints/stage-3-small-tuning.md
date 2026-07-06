# Stage 3 Small Tuning Sprint

## Scope
- Move the default experiment config to Stage 3.
- Keep the Stage 2 base architecture fixed.
- Increase default arithmetic difficulty to 2-digit addition.
- Compare Stage 3 runs against the verified Stage 2 baseline.

## Assumptions
- Stage 2 is validated for 1-digit addition.
- Stage 3 should tune experiment/data settings before scaling model size.
- `context_length=16` is sufficient for 2-digit addition examples.

## Architectural Decisions
- Do not change model architecture.
- Default model remains:
  - `d_model=128`
  - `num_heads=4`
  - `num_layers=4`
  - `ffn_hidden=512`
- Default experiment becomes:
  - `stage=stage3`
  - `digit_length=2`
  - `steps=500`
  - `eval_interval=100`
- Raw artifacts write to `runs/stage3/<timestamp>/`.

## Tasks
1. Update runner defaults to Stage 3 tuning baseline.
2. Update tests for new defaults.
3. Update roadmap and experiment tracking docs.
4. Run tests and a short Stage 3 smoke run.
5. Provide comparison commands for Stage 2 vs Stage 3.

## Risks
- 2-digit addition is harder; a 500-step run may not reach high exact accuracy.
- The current data generator samples numbers from `0..99`, so the task is "up to 2 digits" rather than fixed-width 2-digit addition.
- Stage 3 should not be judged by one run; compare learning curves and tune one variable at a time.

## Verification Strategy
- Run unit tests.
- Run a short Stage 3 smoke run.
- Compare Stage 2 and Stage 3 saved runs with `scripts/compare_runs.py`.

## Verification Results
- Unit tests passed: `.venv/bin/python -m unittest discover -s tests -v`.
- Stage 3 smoke run passed: `runs/stage3/smoke-default`.
- Smoke loss decreased from `2.7579` to `1.8549` over 20 steps on MPS.
- Comparison script passed against `runs/stage2/20260617-142931`.
- Exact accuracy remains `0.000` at 20 steps; this validates wiring only, not Stage 3 quality.
