# Stage 7 Multi-Operation SFT Warmup Sprint

## Scope
- Run a short supervised fine-tuning warmup for `+`, `-`, `*`, and `/`.
- Keep training under roughly five minutes.
- Use the final addition checkpoint as initialization where possible.
- Prepare a minimally capable multi-operation model for later verifier-based RL.

## Assumptions
- The final addition checkpoint uses the old addition vocab.
- Multi-operation training requires the extended tokenizer vocab.
- Existing non-vocab weights can transfer from the addition checkpoint.
- New operator/output tokens (`-`, `*`, `/`, `R`, `E`) must remain randomly initialized before SFT.
- This warmup is not expected to fully solve multiplication or division.

## Architectural Decisions
- Add an explicit checkpoint vocab-expansion flag instead of weakening strict checkpoint loading by default.
- Reuse the existing `main.py` SFT loop.
- Copy shared token rows from the old checkpoint into the new embedding and LM head.
- Keep all non-vocab model weights identical when architecture dimensions match.
- Train all four operations together to avoid destroying addition behavior immediately.
- Support optional operation sampling weights so short continuation runs can focus more on weak operations.
- Add optional operation-aware diverse operand sampling for the final pre-RL SFT push:
  - addition: carry/no-carry mix
  - subtraction: positive and negative results
  - multiplication: small, one-small, round-number, and full-range cases
  - division: exact division, remainder division, and divide-by-zero cases

## Step-by-Step Tasks
- Add `--allow-vocab-expansion` to the SFT runner.
- Extend checkpoint loading to support old-vocab to new-vocab transfer.
- Add focused tests for shared-token row transfer and strict default behavior.
- Run compile and unit tests.
- Run a short multi-op SFT warmup with the final architecture and extended vocab.
- Report command, runtime, artifact path, and observed metrics.

## Risks
- A short SFT run may only teach format and partial behavior, especially for `*` and `/`.
- The model can forget addition if later runs over-weight new operations.
- RL will still need pass@K diagnostics per operation before serious optimization.

## Verification Strategy
- `python -m compileall src scripts tests main.py`
- `.venv/bin/python -m unittest discover -s tests -p 'test*.py'`
- Short SFT command completes within five minutes.
- Check final metrics and sample outputs.

## Verification Results
- `.venv/bin/python -m compileall main.py src scripts tests` passed.
- `.venv/bin/python -m unittest discover -s tests -p 'test*.py'` passed: `27` tests.
- Added explicit `--allow-vocab-expansion` checkpoint loading.
- Ran 100-step warmup:
  - run dir: `runs/stage7-multiop-sft-warmup/20260704-185050`
  - elapsed: `12.82s`
  - final sampled exact accuracy: `0.34`
- Ran stronger 500-step warmup under the five-minute cap:
  - run dir: `runs/stage7-multiop-sft-warmup-500/20260704-185219`
  - elapsed: `37.82s`
  - final sampled exact accuracy: `0.41`
  - best validation loss: `0.9054`
  - final checkpoint: `runs/stage7-multiop-sft-warmup-500/20260704-185219/checkpoint_final.pt`
- 500-step per-operation greedy exhaustive evaluation:
  - `+`: `0.9644` accuracy, `9644 / 10000`
  - `-`: `0.1291` accuracy, `1291 / 10000`
  - `*`: `0.0367` accuracy, `367 / 10000`
  - `/`: `0.4524` accuracy, `4524 / 10000`
- Conclusion: this is a useful format/capability warmup, but multiplication and subtraction need more SFT/curriculum or pass@K validation before heavy GRPO.
- Added `--operation-weights` for weighted synthetic generation, e.g. `'+:1,-:3,*:4,/:2'`.
- `.venv/bin/python -m unittest discover -s tests -p 'test*.py'` passed after generator weighting: `28` tests.
- Weighted-generator smoke run passed:
  - run dir: `runs/stage7-weighted-generator-smoke`
  - checkpoint: `runs/stage7-weighted-generator-smoke/checkpoint_final.pt`

## Final Pre-RL Diversity Push
- Goal: improve all operations before RL, especially multiplication and division.
- Add `--operand-sampling diverse`.
- Continue from `runs/stage7-multiop-sft-weighted-continue/20260704-190749/checkpoint_final.pt`.
- Keep the run bounded and evaluate per operation afterward.
- First run attempt caught an addition carry sampling bug before training started; fix the generator and re-run verification before continuing.
- `.venv/bin/python -m compileall main.py src scripts tests` passed after the generator fix.
- `.venv/bin/python -m unittest discover -s tests -p 'test*.py'` passed: `29` tests.
- Diverse operand SFT run:
  - run dir: `runs/stage7-multiop-sft-diverse-final/20260704-191822`
  - `+`: `0.9949`
  - `-`: `0.9209`
  - `*`: `0.3833`
  - `/`: `0.7606`
  - combined: `0.7649`
- Multiplication-focused full-range correction run:
  - run dir: `runs/stage7-multiop-sft-mul-correction/20260704-193008`
  - elapsed: `213.19s`
  - checkpoint: `runs/stage7-multiop-sft-mul-correction/20260704-193008/checkpoint_final.pt`
  - `+`: `0.9925`
  - `-`: `0.9540`
  - `*`: `0.6090`
  - `/`: `0.8552`
  - combined: `34107 / 40000 = 0.8527`
- Conclusion: all operations except multiplication are above `75%`; multiplication improved substantially but remains the limiting operation before RL.
