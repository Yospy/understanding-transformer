# Local Chat Tester Sprint

## Scope
- Add a local chatbot-style interface for testing the trained addition model.
- Support normal user-facing addition prompts while preserving internal `fixed_reversed` model format.
- Avoid external web dependencies.

## Assumptions
- Current best checkpoint is `runs/stage5/digit2-fixed-reversed-answer-weighted/checkpoint_final.pt`.
- User wants local deployment for manual testing, not public hosting.
- The model supports 2-digit operands, `0` through `99`.

## Architectural Decisions
- Keep inference separate from training/evaluation.
- Use shared formatter/parser helpers from `src/data.py`.
- Provide both CLI and browser UI from one script.
- Use stdlib HTTP server for local deployment.

## Step-by-Step Tasks
1. Add `src/infer.py` for checkpoint loading and addition prediction.
2. Add `scripts/chat_addition.py` with CLI and web modes.
3. Add focused tests for input parsing and CLI help.
4. Run unit, compile, and one-shot inference verification.
5. Start the local server and provide the URL.

## Risks
- Free-form user text should be constrained to one addition expression.
- The model checkpoint is format-dependent; default to `fixed_reversed`.
- This is a local tester, not hardened production service.

## Verification Strategy
- Unit tests for parsing and validation.
- Compile check.
- One-shot real checkpoint inference.
- Manual local browser URL.

## Verification Results
- Unit tests passed: `.venv/bin/python -m unittest discover -s tests -v`.
- Compile check passed: `.venv/bin/python -m compileall main.py src scripts tests`.
- Help command passed: `.venv/bin/python scripts/chat_addition.py --help`.
- One-shot checkpoint inference passed: `.venv/bin/python scripts/chat_addition.py --once "80+26"` returned `80+26 = 106`.

## Usage
One-shot:

```bash
.venv/bin/python scripts/chat_addition.py --once "80+26"
```

Terminal chat:

```bash
.venv/bin/python scripts/chat_addition.py
```

Browser chat:

```bash
.venv/bin/python scripts/chat_addition.py --serve --host 127.0.0.1 --port 7860
```
