# GRPO Current Addition Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an on-policy GRPO fine-tuning loop for the existing 2-digit addition checkpoint.

**Architecture:** Keep the supervised trainer untouched. Add stochastic sampling, deterministic verifier rewards, group-relative advantages, KL-controlled GRPO loss, and a standalone training script that saves normal checkpoint artifacts.

**Tech Stack:** Python, PyTorch, existing `AdditionTransformer`, existing arithmetic data/verifier helpers.

---

### Task 1: Sprint and Planning Docs

**Files:**
- Create: `sprints/stage-6-grpo-current-addition.md`
- Create: `docs/plans/2026-07-04-grpo-current-addition.md`
- Modify: `tasks/todo.md`

**Steps:**
- Record Stage 6 scope, assumptions, architectural decisions, risks, and verification.
- Set `tasks/todo.md` active sprint to Stage 6 and add checklist items.

### Task 2: Sampling Utility

**Files:**
- Create: `src/sampling.py`
- Test: `tests/test_grpo.py`

**Steps:**
- Add a sampled completion function that uses `softmax(logits / temperature)` and `torch.multinomial`.
- Stop on newline or context budget.
- Return generated token ids and decoded text.
- Test deterministic fake-model behavior with a peaked distribution.

### Task 3: GRPO Core

**Files:**
- Create: `src/grpo.py`
- Test: `tests/test_grpo.py`

**Steps:**
- Add reward scoring through `canonical_answer` and `parse_formatted_answer`.
- Add group-relative advantage computation.
- Add token log-prob extraction for completion tokens only.
- Add clipped GRPO loss with KL penalty.
- Test rewards, all-zero advantage behavior, masks, and finite loss values.

### Task 4: Training Script

**Files:**
- Create: `scripts/train_grpo_addition.py`
- Test: `tests/test_grpo.py`

**Steps:**
- Load policy and frozen reference model from checkpoint.
- Generate 2-digit addition prompts.
- Sample `K` completions per prompt.
- Score rewards, compute advantages, optimize, log metrics, and save checkpoints.
- Add `--help` script test.

### Task 5: Verification

**Commands:**

```bash
python -m compileall src scripts tests
python -m unittest
python scripts/train_grpo_addition.py --help
python scripts/train_grpo_addition.py --checkpoint final/checkpoint_best_val.pt --number-format normal --device cpu --steps 2 --prompts-per-step 2 --group-size 4 --eval-interval 1 --accuracy-examples 8 --log-sample-groups 2 --run-dir runs/stage6-grpo-current-addition/smoke
```

**Expected:**
- Compile succeeds.
- Unit tests pass.
- Help text prints.
- Smoke run prints reward, pass@K, KL, loss, clipping, and sample-group details.

