# Multi Operation Arithmetic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the arithmetic transformer pipeline from addition-only to bounded multi-operation arithmetic.

**Architecture:** Preserve old checkpoint compatibility by loading tokenizer vocab from checkpoints, while new training uses an extended arithmetic vocab. Centralize operation semantics in `src/data.py`, then thread operation lists through training, sampled accuracy, exhaustive/random evaluation, and chat inference.

**Tech Stack:** Python, PyTorch, unittest, stdlib HTTP server.

---

### Task 1: Extended Tokenizer

**Files:**
- Modify: `src/tokenizer.py`
- Modify: checkpoint loaders in `main.py`, `scripts/evaluate_checkpoint.py`, `src/infer.py`

**Steps:**
1. Add extended tokens for `-`, `*`, `/`, `R`, `E`.
2. Allow tokenizer construction from checkpoint vocab.
3. Keep old checkpoints loadable.

### Task 2: Operation Semantics

**Files:**
- Modify: `src/data.py`

**Steps:**
1. Add operation literals for `+`, `-`, `*`, `/`.
2. Add formatting for operation prompts and answers.
3. Define division as quotient/remainder: `7/3=2R1`.
4. Define division by zero as `ERR`.
5. Update batch generation to sample configured operations.

### Task 3: Training and Evaluation

**Files:**
- Modify: `main.py`
- Modify: `src/train.py`
- Modify: `scripts/evaluate_checkpoint.py`

**Steps:**
1. Add `--operations`.
2. Thread operation list into loss/accuracy/evaluation.
3. Add random multi-operation evaluation for large operation spaces.

### Task 4: Chat Tester

**Files:**
- Modify: `src/infer.py`
- Modify: `scripts/chat_addition.py`

**Steps:**
1. Parse `+`, `-`, `*`, and `/`.
2. Return canonical operation answers.
3. Update UI copy to describe supported operations.

### Task 5: Verification

**Commands:**
- `.venv/bin/python -m unittest discover -s tests -v`
- `.venv/bin/python -m compileall main.py src scripts tests`
- Smoke train command with `--operations +,-,*,/`

