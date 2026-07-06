# Arithmetic Representation Loss Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable arithmetic representations and answer-weighted training loss so 2-digit addition can learn carry behavior more directly.

**Architecture:** Keep the transformer architecture unchanged. Centralize formatting/parsing in `src/data.py`, thread `number_format` through training/evaluation, and add loss weights derived from target-token positions.

**Tech Stack:** Python, PyTorch, unittest, existing CLI scripts.

---

### Task 1: Number Formatting

**Files:**
- Modify: `src/data.py`
- Test: `tests/test_stage1.py`

**Steps:**
1. Add `NumberFormat` literals: `normal`, `fixed`, `reversed`, `fixed_reversed`.
2. Add `format_number`, `format_addition`, `format_prompt`, and `parse_formatted_answer`.
3. Update batch creation to accept `number_format`.
4. Add tests for all formats and batch compatibility.

### Task 2: Answer-Weighted Loss

**Files:**
- Modify: `src/train.py`
- Modify: `main.py`
- Test: `tests/test_stage1.py`

**Steps:**
1. Add `loss_prompt_weight` to experiment config and CLI.
2. Build per-token weights from inputs/targets using the `=` token boundary.
3. Add weighted `next_token_loss`.
4. Thread `number_format` into validation and sampled accuracy.

### Task 3: Evaluator Support

**Files:**
- Modify: `scripts/evaluate_checkpoint.py`
- Test: `tests/test_stage1.py`

**Steps:**
1. Add `--number-format`.
2. Build prompts/expected answers using shared formatter helpers.
3. Parse generated answers back to canonical numeric strings.
4. Keep old checkpoints evaluable with default `normal`.

### Task 4: Checkpoint Selection

**Files:**
- Modify: `main.py`
- Test: `tests/test_stage1.py`

**Steps:**
1. Save `checkpoint_step_<step>.pt` at each eval interval.
2. Include checkpoint paths in `summary.json`.
3. Verify smoke run writes per-step checkpoint.

### Task 5: Verification

**Commands:**
- `.venv/bin/python -m unittest discover -s tests -v`
- `.venv/bin/python -m compileall main.py src scripts tests`
- `.venv/bin/python main.py --stage stage5-smoke --digit-length 2 --d-model 32 --num-heads 2 --num-layers 1 --ffn-hidden 128 --steps 3 --eval-interval 1 --accuracy-examples 2 --val-batches 1 --samples 1 --number-format fixed_reversed --loss-prompt-weight 0.2 --run-dir runs/stage5/format-loss-smoke`

