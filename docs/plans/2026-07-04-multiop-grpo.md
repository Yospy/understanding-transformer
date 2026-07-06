# Multi-Operation GRPO Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generalize the existing GRPO training loop so the Stage 7 multi-operation checkpoint can run verifier-based RL on `+`, `-`, `*`, and `/`, starting with multiplication.

**Architecture:** Keep the existing standalone GRPO script and loss implementation. Replace addition-only rollout and reward wiring with operation-aware arithmetic helpers already used by SFT/eval, while preserving the addition helper as a compatibility wrapper.

**Tech Stack:** Python, PyTorch, existing `AdditionTransformer`, `src.data` arithmetic generation/parsing helpers, `unittest`.

---

### Task 1: Sprint Tracking

**Files:**
- Create: `sprints/stage-8-multiop-grpo.md`
- Modify: `tasks/todo.md`

**Steps:**
- Record scope, assumptions, decisions, risks, and verification strategy.
- Set the active sprint to Stage 8.
- Add the minimal checklist for implementation and verification.

### Task 2: Generic Reward Scoring

**Files:**
- Modify: `src/grpo.py`
- Test: `tests/test_grpo.py`

**Steps:**
- Add `score_arithmetic_completion(a, b, operation, completion_text, number_format)`.
- Use `canonical_answer(a, b, operation)` and `parse_formatted_answer(raw_answer, number_format, operation)`.
- Keep `score_addition_completion(...)` as a wrapper around the generic function.
- Test multiplication, negative subtraction, division remainder, divide-by-zero `ERR`, and invalid output.

### Task 3: Operation-Aware Rollouts

**Files:**
- Modify: `scripts/train_grpo_addition.py`
- Test: `tests/test_grpo.py`

**Steps:**
- Add CLI flags: `--operations`, `--operation-weights`, `--operand-sampling`.
- Store parsed `operations`, `operation_weights`, and `operand_sampling` in `GRPOTrainConfig`.
- Generate rollout problems through `mixed_arithmetic_problems(...)`.
- Format prompts with the sampled operation.
- Score completions with `score_arithmetic_completion(...)`.
- Include `operation` in sample group logs.
- Use `config.operations` in greedy accuracy instead of hardcoded addition.

### Task 4: Optional Operation-Aware Hard Cases

**Files:**
- Modify: `scripts/train_grpo_addition.py`

**Steps:**
- Accept old hard-case entries with `a`/`b` as addition cases.
- Accept new entries with `a`/`operation`/`b`.
- Mix hard cases into rollout problems before random arithmetic problems.

### Task 5: Verification

**Commands:**

```bash
.venv/bin/python -m compileall main.py src scripts tests
.venv/bin/python -m unittest discover -s tests -p 'test*.py'
.venv/bin/python scripts/train_grpo_addition.py --help
.venv/bin/python scripts/train_grpo_addition.py --checkpoint runs/stage7-multiop-sft-mul-correction/20260704-193008/checkpoint_final.pt --steps 2 --prompts-per-step 2 --group-size 4 --eval-interval 1 --accuracy-examples 16 --log-sample-groups 2 --run-dir runs/stage8-multiop-grpo/addition-smoke
.venv/bin/python scripts/train_grpo_addition.py --checkpoint runs/stage7-multiop-sft-mul-correction/20260704-193008/checkpoint_final.pt --operations "*" --operand-sampling diverse --steps 2 --prompts-per-step 4 --group-size 8 --temperature 1.1 --eval-interval 1 --accuracy-examples 32 --log-sample-groups 2 --run-dir runs/stage8-multiop-grpo/mul-smoke
```

**Expected:**
- Compile succeeds.
- Unit tests pass.
- Help text exposes the new operation flags.
- Addition smoke remains valid.
- Multiplication smoke logs `*` prompts and nonzero reward signal when sampled groups vary.
