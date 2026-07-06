# Todo

## Active Sprint
- `sprints/stage-8-multiop-grpo.md`

## Stage 8 Multi-Operation GRPO
- [x] Create Stage 8 sprint and plan.
- [x] Add generic arithmetic reward scoring.
- [x] Add operation-aware GRPO rollout configuration.
- [x] Add focused tests.
- [x] Run compile, unit, help, and smoke verification.
- [x] Report RL training command and artifact paths.

## Stage 7 Multi-Operation SFT Warmup
- [x] Create Stage 7 sprint plan.
- [x] Add explicit checkpoint vocab expansion for continuous SFT.
- [x] Add focused tests.
- [x] Run compile and unit verification.
- [x] Run short multi-operation SFT warmup under five minutes.
- [x] Report command, artifacts, and observed metrics.

## Stage 6 GRPO Current Addition
- [x] Create Stage 6 sprint plan.
- [x] Add stochastic sampled generation for RL completions.
- [x] Add deterministic reward, group advantage, log-prob, and GRPO loss helpers.
- [x] Add standalone GRPO addition trainer.
- [x] Add focused tests.
- [x] Run compile, unit, help, and smoke verification.
- [x] Report run command and log fields.

## Stage 1 Tiny Sanity
- [x] Create Stage 1 runner scaffold.
- [x] Implement tokenizer and synthetic addition data.
- [x] Implement tiny decoder-only transformer.
- [x] Implement train/eval/generation pipeline.
- [x] Persist experiment artifacts under `runs/stage1/<timestamp>/`.
- [x] Add focused unit tests.
- [x] Run local verification.
- [x] Run subagent verification.
- [x] Document how to run and where to track losses/experiments.

## Stage 2 Base Architecture
- [x] Change default runner config to Stage 2 base architecture.
- [x] Make default run artifacts stage-aware.
- [x] Add experiment comparison script.
- [x] Add focused tests for config defaults and comparison.
- [x] Update run/comparison docs.
- [x] Run local verification.

## Stage 3 Small Tuning
- [x] Change default runner config to Stage 3 2-digit baseline.
- [x] Keep Stage 2 architecture fixed.
- [x] Update tests for Stage 3 defaults.
- [x] Update run/comparison docs.
- [x] Run local verification.

## Stage 4 Scaling Ladder
- [x] Record Stage 3 baseline comparison.
- [x] Create Stage 4 scaling ladder sprint.
- [x] Run same-model longer-budget scaling.
- [x] Run width scaling.
- [x] Run depth scaling.
- [x] Compare Stage 4 runs against Stage 3 baseline.
- [x] Update experiment log with best Stage 4 result.
- [x] Run depth/budget confirmation before larger 4.7M model.
- [x] Run stronger sampled 2-digit evaluation before Stage 5.
- [x] Save best/final checkpoints from training runs.
- [x] Add exhaustive/checkpoint-based 2-digit evaluation before Stage 5.
- [x] Decide whether to run 4.7M checkpoint or enter Stage 5 with depth6-1500.
- [x] Promote depth6-2000-lr1e-3 after checkpoint/eval tooling is added.
- [x] Create Stage 5 final model selection sprint.

## Stage 5 Final Model Selection
- [x] Select final checkpoint.
- [x] Package checkpoint and run artifacts under `final/`.
- [x] Write final report.
- [x] Write final artifact manifest.
- [x] Update roadmap and docs.

## Hard-Case Curriculum Improvement
- [x] Add hard-case JSON parsing.
- [x] Add mixed hard-case/random batch generation.
- [x] Add CLI flags for hard-case training.
- [x] Add checkpoint initialization for fine-tuning.
- [x] Add focused tests.
- [x] Update commands in docs.
- [x] Run smoke verification.
- [x] Evaluate full 25% hard-case run.
- [ ] Optional: try smaller 5% hard-case fine-tune.

## Arithmetic Representation Loss
- [x] Add configurable number formats.
- [x] Add answer-weighted loss.
- [x] Thread number format through sampled and exhaustive evaluation.
- [x] Save per-eval step checkpoints.
- [x] Add focused tests.
- [x] Update experiment commands.
- [x] Run unit, compile, training smoke, and evaluator smoke verification.

## Local Chat Tester
- [x] Add checkpoint inference wrapper.
- [x] Add terminal chat mode.
- [x] Add browser chat mode.
- [x] Add focused tests.
- [x] Run unit, compile, help, and one-shot inference verification.

## Multi Operation Arithmetic
- [x] Add extended tokenizer vocab.
- [x] Preserve checkpoint-vocab loading.
- [x] Add operation formatting for `+`, `-`, `*`, `/`.
- [x] Add quotient/remainder division and divide-by-zero `ERR`.
- [x] Thread `--operations` through train/eval.
- [x] Update chat parser for all operations.
- [x] Add focused tests.
- [x] Run unit, compile, smoke train, evaluator smoke, and old-checkpoint compatibility checks.
