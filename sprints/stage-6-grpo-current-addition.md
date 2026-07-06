# Stage 6 GRPO Current Addition Sprint

## Scope
- Add an on-policy GRPO reinforcement-learning loop around the existing trained addition transformer.
- Fine-tune only the current model capability: 2-digit addition prompts in normal format, such as `37+48=`.
- Use a deterministic outcome verifier as the reward source.
- Keep the existing supervised training pipeline intact.

## Assumptions
- `final/checkpoint_best_val.pt` is the starting policy checkpoint.
- The final checkpoint was trained for addition only and uses the checkpoint vocab.
- GRPO can only reinforce correct answers that the policy can sample at least sometimes.
- Process rewards, multi-operation arithmetic, and 3-digit arithmetic are out of scope for this sprint.

## Architectural Decisions
- Add a separate trainer script instead of modifying `main.py`.
- Use two model instances:
  - `policy_model`: trainable on-policy model.
  - `reference_model`: frozen copy of the initial checkpoint for KL control.
- Add stochastic sampled generation for RL; keep greedy generation unchanged for eval/UI.
- Use outcome reward:

```text
reward = 1.0 if parsed_model_answer == canonical_answer else 0.0
```

- Compute group-relative advantages per prompt group.
- Apply a PPO-style clipped policy objective with a token-level KL penalty against the frozen reference model.

## Step-by-Step Tasks
- Create `src/sampling.py` for stochastic completion sampling.
- Create `src/grpo.py` for verifier rewards, group advantages, log-prob extraction, and GRPO loss.
- Create `scripts/train_grpo_addition.py` for the end-to-end training loop.
- Add focused tests for sampling, reward scoring, advantage behavior, log-prob masks, and script help.
- Update `tasks/todo.md` with the active Stage 6 work.
- Run unit tests, compile checks, script help, and a smoke GRPO run.

## Risks
- If all sampled answers are wrong for a prompt, GRPO has no positive signal for that group.
- The final model is already highly accurate, so measurable improvement may require hard-case sampling.
- Too high a learning rate or too weak a KL penalty can damage the already-good model.
- MPS/CPU stochastic sampling behavior can differ slightly, so tests must avoid flaky sampled expectations.

## Verification Strategy
- Unit tests for deterministic reward and loss helpers.
- `python -m compileall src scripts tests`.
- `python -m unittest`.
- `python scripts/train_grpo_addition.py --help`.
- Smoke run with a tiny number of steps and detailed logs enabled.
- Optional before/after exhaustive evaluation with `scripts/evaluate_checkpoint.py`.

## Verification Results
- `.venv/bin/python -m compileall src scripts tests` passed.
- `.venv/bin/python -m unittest discover -s tests -p 'test*.py'` passed: `26` tests.
- `.venv/bin/python scripts/train_grpo_addition.py --help` passed.
- Random-prompt smoke run passed and wrote:
  - `runs/stage6-grpo-current-addition/smoke/metrics.csv`
  - `runs/stage6-grpo-current-addition/smoke/sample_groups.jsonl`
  - `runs/stage6-grpo-current-addition/smoke/checkpoint_final.pt`
- Hard-case smoke run passed and showed nonzero RL signal:
  - `reward_mean=0.125`
  - `pass@8=0.500`
  - `grad_norm=5.500`
  - `active_groups=0.500`
  - `runs/stage6-grpo-current-addition/smoke-hard-cases/checkpoint_final.pt`
