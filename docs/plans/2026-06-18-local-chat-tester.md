# Local Chat Tester Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Provide a local chatbot-style tester for the trained arithmetic transformer.

**Architecture:** Add a small inference wrapper that loads checkpoints, parses natural addition queries, formats prompts using the selected number representation, and decodes canonical answers. Expose it through a no-dependency CLI/web server script.

**Tech Stack:** Python, PyTorch, stdlib `http.server`, unittest.

---

### Task 1: Inference Wrapper

**Files:**
- Create: `src/infer.py`
- Test: `tests/test_stage1.py`

**Steps:**
1. Parse inputs like `37+48`, `37 + 48`, and `what is 37+48?`.
2. Validate operand range from `digit_length`.
3. Load checkpoints using stored model config and vocab.
4. Generate and parse canonical numeric answers.

### Task 2: Local Chat Script

**Files:**
- Create: `scripts/chat_addition.py`
- Test: `tests/test_stage1.py`

**Steps:**
1. Add CLI mode for terminal testing.
2. Add `--serve` mode using `ThreadingHTTPServer`.
3. Serve a compact chat UI and JSON `/api/ask` endpoint.
4. Default to the perfect fixed-reversed Stage 5 checkpoint.

### Task 3: Verification

**Commands:**
- `.venv/bin/python -m unittest discover -s tests -v`
- `.venv/bin/python -m compileall main.py src scripts tests`
- `.venv/bin/python scripts/chat_addition.py --help`
- `.venv/bin/python scripts/chat_addition.py --once "80+26"`

