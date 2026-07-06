# Multi-Operation SFT Warmup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add safe checkpoint vocab expansion and run a short multi-operation SFT warmup from the final addition checkpoint.

**Architecture:** Preserve strict checkpoint loading by default. Add an explicit expansion path that copies shared vocab rows from the addition checkpoint into the extended tokenizer model and transfers all matching non-vocab weights.
Add optional operation sampling weights for short continuation runs that need more `-` and `*` examples without storing datasets.
Add optional operation-aware diverse operand sampling for the final pre-RL SFT pass.

**Tech Stack:** Python, PyTorch, existing `main.py` SFT runner, `AdditionTransformer`, synthetic arithmetic data.

---

### Task 1: Sprint Plan

**Files:**
- Create: `sprints/stage-7-multiop-sft-warmup.md`
- Create: `docs/plans/2026-07-04-multiop-sft-warmup.md`
- Modify: `tasks/todo.md`

**Steps:**
- Record scope, assumptions, architectural decisions, risks, and verification.
- Set active sprint to Stage 7.

### Task 2: Vocab Expansion Loader

**Files:**
- Modify: `main.py`
- Test: `tests/test_stage1.py`

**Steps:**
- Add `allow_vocab_expansion` config and CLI flag.
- Add helper logic to copy `token_embedding.weight` and `lm_head.weight` rows by token string.
- Keep strict loading behavior unchanged when the flag is not set.
- Validate non-vocab architecture dimensions match.

### Task 3: Verification

**Commands:**

```bash
.venv/bin/python -m compileall main.py src scripts tests
.venv/bin/python -m unittest discover -s tests -p 'test*.py'
```

### Task 4: Short SFT Warmup

**Command:**

```bash
.venv/bin/python main.py \
  --stage stage7-multiop-sft-warmup \
  --init-checkpoint final/checkpoint_best_val.pt \
  --allow-vocab-expansion \
  --d-model 128 \
  --num-heads 4 \
  --num-layers 6 \
  --ffn-hidden 512 \
  --context-length 16 \
  --digit-length 2 \
  --operations +,-,*,/ \
  --number-format normal \
  --steps 100 \
  --batch-size 64 \
  --learning-rate 5e-4 \
  --eval-interval 25 \
  --val-batches 2 \
  --accuracy-examples 50 \
  --samples 12 \
  --device auto
```

**Expected:**
- Completes under five minutes.
- Produces `runs/stage7-multiop-sft-warmup/<timestamp>/checkpoint_final.pt`.
- Gives an initial multi-op model for limitation/pass@K audit before RL.

### Task 5: Weighted Continuation Data

**Files:**
- Modify: `src/data.py`
- Modify: `src/train.py`
- Modify: `main.py`
- Test: `tests/test_stage1.py`

**Steps:**
- Add `--operation-weights` to bias synthetic operation sampling.
- Support both positional weights (`1,3,4,2`) and op-value weights (`+:1,-:3,*:4,/:2`).
- Keep the default behavior uniform when weights are omitted.
- Verify with compile, unit tests, and a 2-step smoke run.

### Task 6: Diverse Operand Sampling

**Files:**
- Modify: `src/data.py`
- Modify: `src/train.py`
- Modify: `main.py`
- Test: `tests/test_stage1.py`

**Steps:**
- Add `--operand-sampling {uniform,diverse}`.
- Keep default behavior as uniform.
- In diverse mode, sample operation-specific operand patterns:
  - addition carry/no-carry
  - subtraction positive/negative
  - multiplication small/one-small/round/full
  - division exact/remainder/zero-divisor
- Run compile and unit tests.
- Run a bounded final SFT continuation and per-operation eval before RL.
