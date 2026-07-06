# Stage 5 Final Model Selection Sprint

## Scope
- Select the final project-stage addition transformer.
- Package the best checkpoint and evaluation artifacts into `final/`.
- Document the experiment ladder, final configuration, evaluation result, and known limitations.

## Assumptions
- Stage 4 established that depth plus longer training beats width-heavy scaling for 2-digit addition.
- The final project-stage goal is a strong compact addition transformer, not a formally perfect arithmetic engine.
- The selected model must have a saved checkpoint and exhaustive evaluation artifact.

## Architectural Decisions
- No more model architecture changes for this stage.
- Final selected config:
  - `d_model=128`
  - `num_heads=4`
  - `num_layers=6`
  - `ffn_hidden=512`
  - `context_length=16`
  - `digit_length=2`
  - `learning_rate=0.001`
  - `steps=2000`
- Use `runs/stage4/digit2-depth6-2000-lr1e-3-ckpt2/checkpoint_best_val.pt` as the selected checkpoint.

## Final Artifacts
- `final/checkpoint_best_val.pt`
- `final/checkpoint_final.pt`
- `final/config.json`
- `final/summary.json`
- `final/metrics.csv`
- `final/samples.txt`
- `final/eval_exhaustive_2digit.json`
- `final/eval_exhaustive_2digit_verify.json`
- `final/final_report.md`
- `final/MANIFEST.md`

## Verification Results
- Checkpointed run completed on MPS.
- Final selected checkpoint has `1,188,736` trainable parameters.
- Sampled exact accuracy at final eval: `0.996` over `500` examples.
- Exhaustive 2-digit accuracy: `9,978 / 10,000 = 0.9978`.
- Known incorrect exhaustive cases: `22 / 10,000`.

## Current Decision
- Stage 5 final package is complete.
- Do not promote the 4.7M model; it was slower and less accurate.
- Future work can target exactness, better curriculum, fixed-width formatting, or larger final models, but those are outside this project-stage final package.
