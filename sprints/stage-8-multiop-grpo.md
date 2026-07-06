# Stage 8 Multi-Operation GRPO Sprint

## Scope
- Generalize the existing GRPO loop from addition-only to operation-aware arithmetic RL.
- Use the Stage 7 multi-operation SFT checkpoint as the starting policy/reference checkpoint.
- Make multiplication-only GRPO runnable first.
- Keep SFT code unchanged.

## Assumptions
- Current pre-RL checkpoint:
  `runs/stage7-multiop-sft-mul-correction/20260704-193008/checkpoint_final.pt`
- GRPO can improve only groups with mixed sampled rewards.
- Multiplication is the limiting operation after SFT.
- Existing arithmetic formatting, parsing, and canonical answer helpers are the verifier source of truth.

## Architectural Decisions
- Preserve the existing standalone script name, `scripts/train_grpo_addition.py`, to avoid unnecessary churn.
- Add operation-aware CLI configuration to the existing trainer.
- Keep addition reward scoring as a wrapper for compatibility.
- Reuse `mixed_arithmetic_problems(...)` for random rollout prompts.
- Add local operation-aware hard-case loading for RL without changing SFT hard-case behavior.

## Step-by-Step Tasks
- Add generic arithmetic reward scoring.
- Add operation-aware GRPO CLI config.
- Generate sampled rollout prompts for selected operations.
- Score completions with the selected operation.
- Thread selected operations through greedy accuracy.
- Add tests for multi-operation rewards and help output.
- Run compile, unit, help, and smoke verification.

## Risks
- Multiplication can have all-wrong or all-correct groups, producing no advantage.
- High temperature can create noisy invalid completions.
- Low temperature can hide useful sampled alternatives.
- Multiplication-only RL can regress other operations, so post-run all-op eval is required.

## Verification Strategy
- `.venv/bin/python -m compileall main.py src scripts tests`
- `.venv/bin/python -m unittest discover -s tests -p 'test*.py'`
- `.venv/bin/python scripts/train_grpo_addition.py --help`
- Addition smoke against the Stage 7 checkpoint.
- Multiplication-only smoke against the Stage 7 checkpoint.
- Full all-operation exhaustive eval after any serious RL run.

## Verification Results
- `.venv/bin/python -m compileall main.py src scripts tests` passed.
- `.venv/bin/python -m unittest discover -s tests -p 'test*.py'` passed: `32` tests.
- `.venv/bin/python scripts/train_grpo_addition.py --help` passed and exposes operation flags.
- Addition smoke passed:
  - run dir: `runs/stage8-multiop-grpo/addition-smoke`
  - final `reward_mean=1.000`, `pass@4=1.000`
- Multiplication smoke passed:
  - run dir: `runs/stage8-multiop-grpo/mul-smoke`
  - step 1: `reward_mean=0.719`, `pass@8=1.000`, `active_groups=0.500`
  - final checkpoint: `runs/stage8-multiop-grpo/mul-smoke/checkpoint_final.pt`
