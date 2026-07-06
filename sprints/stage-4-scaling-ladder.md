# Stage 4 Scaling Ladder Sprint

## Scope
- Scale deliberately from the Stage 3 2-digit baseline.
- Keep the task fixed at 2-digit addition while varying one major factor per run.
- Compare training budget and model capacity before moving toward the final ~10M parameter target.

## Assumptions
- Stage 2 solved 1-digit addition with the 793,728 parameter model.
- Stage 3 baseline used the same model on 2-digit addition and reached only `0.080` final exact accuracy after 500 steps.
- 2-digit addition should remain the target until we understand whether the bottleneck is training budget or capacity.

## Architectural Decisions
- Keep the same architecture family: decoder-only transformer, RoPE, RMSNorm, GELU FFN, AdamW, next-token loss.
- First Stage 4 run scales training budget only:
  - same 793,728 parameter model
  - 2-digit data
  - 1,500 training steps
- Next runs vary model size while keeping 2-digit data and comparable step budgets.

## Candidate Ladder
| Run | Purpose | Config | Approx Params |
|---|---|---|---:|
| `runs/stage4/digit2-budget-1500` | Test training budget bottleneck | `128d, 4h, 4l, 512ffn, 1500 steps` | 793,728 |
| `runs/stage4/digit2-width192-1000` | Test width scaling | `192d, 6h, 4l, 768ffn, 1000 steps` | 1,780,416 |
| `runs/stage4/digit2-depth6-1000` | Test depth scaling | `128d, 4h, 6l, 512ffn, 1000 steps` | 1,188,736 |
| `runs/stage4/digit2-256x6-1000` | Larger capacity checkpoint | `256d, 8h, 6l, 1024ffn, 1000 steps` | 4,736,768 |

## Tasks
1. Record Stage 3 baseline result.
2. Run budget-only scaling.
3. Run width scaling.
4. Run depth scaling.
5. Compare runs with `scripts/compare_runs.py`.
6. Promote the best result into the experiment log.
7. Decide whether to continue toward ~10M parameters or tune training first.

## Risks
- Exact accuracy can be noisy with only 25 eval examples; use validation loss and best accuracy together.
- Larger models may overfit or need lower learning rate.
- If budget-only scaling improves sharply, capacity scaling may be premature.

## Verification Strategy
- Every run must write `config.json`, `metrics.csv`, `summary.json`, and `samples.txt`.
- Compare against `runs/stage3/digit2-baseline`.
- Do not move to 3-digit data until 2-digit accuracy is materially improved.

## Results So Far
| Run | Final Val Loss | Best Val Loss | Final Exact Accuracy | Best Exact Accuracy | Read |
|---|---:|---:|---:|---:|---|
| `runs/stage3/digit2-baseline` | 1.2428 | 1.2428 | 0.080 | 0.120 | Baseline is weak for 2-digit addition. |
| `runs/stage4/digit2-budget-1500` | 0.9711 | 0.9589 | 0.680 | 0.840 | More training fixes much of the gap. |
| `runs/stage4/digit2-width192-1000` | 1.2743 | 1.2541 | 0.120 | 0.120 | Width-only scaling is not the right next lever. |
| `runs/stage4/digit2-depth6-1000` | 0.9743 | 0.9326 | 0.840 | 0.840 | Current best direction; depth helps more than width. |
| `runs/stage4/digit2-depth6-1500` | 0.8777 | 0.8715 | 0.990 | 1.000 | Current best run; depth plus longer training works. |
| `runs/stage4/digit2-depth6-2000-lr1e-3` | 0.8604 | 0.8604 | 0.988 | 0.990 | Strongest final candidate so far; lower LR improves validation loss and fixes saved samples. |
| `runs/stage4/digit2-256x6-1500-lr1e-3` | 0.9002 | 0.8876 | 0.930 | 0.944 | Larger model is slower and worse than the 1.19M depth6-2000 candidate. |
| `runs/stage4/digit2-depth6-2000-lr1e-3-ckpt2` | 0.8585 | 0.8585 | 0.996 | 0.996 | Checkpointed final candidate; exhaustive eval reached 0.9978 accuracy. |

## Current Decision
- Do not move to 3-digit addition yet.
- Do not jump directly to width-heavy scaling.
- Depth plus longer training is validated as the best current lever.
- The strongest candidate is `runs/stage4/digit2-depth6-2000-lr1e-3-ckpt2`.
- The 4.7M model did not improve results, so do not scale further yet.
- Exhaustive 2-digit evaluation is complete: 9,978 / 10,000 correct.
- This is strong enough to enter Stage 5 if the goal is a compact addition transformer, but not a proof of exact arithmetic.

## Checkpoint/Eval Gap
- Current historical runs did not save model weights.
- Add `checkpoint_best_val.pt` and `checkpoint_final.pt` before the next long run.
- Add a reloadable evaluator that reports exact correctness over all `0..99 + 0..99` cases for 2-digit addition.

## Checkpoint/Eval Tooling Verification
- `main.py` now writes `checkpoint_best_val.pt` and `checkpoint_final.pt`.
- `scripts/evaluate_checkpoint.py` loads saved checkpoints and writes exhaustive eval JSON.
- Evaluator now uses batched generation with progress output; the original one-by-one evaluator was too slow and looked stuck.
- Smoke run verified checkpoint creation: `runs/checkpoint-smoke`.
- Smoke eval verified evaluator command shape: `runs/checkpoint-smoke/eval_exhaustive_1digit.json`.
- Historical `*-ckpt` runs made before this code change still do not have checkpoints and must be rerun.

## Exhaustive Eval Result
- Run: `runs/stage4/digit2-depth6-2000-lr1e-3-ckpt2`.
- Checkpoint: `checkpoint_best_val.pt`.
- Evaluation: all `0..99 + 0..99` pairs.
- Correct: `9,978 / 10,000`.
- Accuracy: `0.9978`.
- Incorrect: `22`.
