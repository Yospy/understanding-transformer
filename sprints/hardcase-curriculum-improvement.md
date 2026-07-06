# Hard-Case Curriculum Improvement Sprint

## Scope
- Improve the final 2-digit model by oversampling known failure cases during training.
- Keep the selected compact architecture fixed.
- Use `final/eval_exhaustive_2digit.json` as the source of hard cases.

## Assumptions
- Synthetic random training data is unlimited.
- The current final model reaches `9,978 / 10,000` on exhaustive 2-digit addition.
- The best next data improvement is to mix known failed cases into normal random batches.

## Architectural Decisions
- Do not change transformer architecture.
- Add data-loader support for hard cases as `(a, b)` addition pairs.
- Expose training controls:
  - `--hard-case-file`
  - `--hard-case-ratio`
- Keep checkpoints and exhaustive eval flow unchanged.

## Tasks
1. Add hard-case JSON parsing.
2. Add hard-case mixed batch generation.
3. Add CLI flags and config persistence.
4. Add focused unit tests.
5. Update experiment docs with run commands.
6. Run tests and a smoke hard-case training command.

## Risks
- Too much hard-case oversampling can overfit a tiny failure set.
- A ratio around `0.25` was tested and is too aggressive for this tiny failure set.
- The final test remains exhaustive eval over all `10,000` pairs, not sampled accuracy.

## Verification Strategy
- Unit tests for hard-case parsing and batch mixing.
- Smoke training run with `--hard-case-file final/eval_exhaustive_2digit.json`.
- Confirm config records hard-case file and ratio.

## Implementation Notes
- `src/data.py` now loads hard cases from exhaustive eval JSON files.
- `make_batch` can mix random examples and hard cases by ratio.
- `main.py` exposes `--hard-case-file` and `--hard-case-ratio`.
- `main.py` exposes `--init-checkpoint` for fine-tuning from the final checkpoint.
- The ratio is exact per batch via `round(batch_size * hard_case_ratio)`.

## Verification Results
- Unit tests passed: `.venv/bin/python -m unittest discover -s tests -v`.
- Compile check passed: `.venv/bin/python -m compileall main.py src scripts tests`.
- Smoke run passed: `runs/hardcase-smoke`.
- Smoke config recorded `hard_case_file=final/eval_exhaustive_2digit.json` and `hard_case_ratio=0.25`.
- Smoke run loaded `22` hard cases and wrote best/final checkpoints.
- Fine-tune smoke run passed: `runs/hardcase-init-smoke`.
- Fine-tune smoke loaded `final/checkpoint_best_val.pt` at checkpoint step `2000`.

## Full Run Result
- Full hard-case run: `runs/stage5/digit2-hardcase-lr5e-4`.
- Best-val checkpoint exhaustive eval: `9,910 / 10,000`, below the packaged final model's `9,978 / 10,000`.
- Final checkpoint exhaustive eval: `9,953 / 10,000`.
- Final checkpoint fixed all original `22` misses, but introduced `47` new misses.
- Decision: do not promote this run. Keep `final/checkpoint_best_val.pt` as the current best model.
- Next attempt, if needed: lower hard-case ratio to `0.05`, lower LR to `0.0001`, and train only `500` steps.
