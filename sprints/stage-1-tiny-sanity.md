# Stage 1 Tiny Sanity Sprint

## Scope
- Build the first executable addition-transformer pipeline.
- Keep the model intentionally tiny and fast to run.
- Persist each run's config, metrics, summary, and generated samples.

## Assumptions
- Python runs from the repo-local `.venv`.
- PyTorch, NumPy, and Matplotlib are already installed.
- MPS should be used when available, with CPU fallback.
- Stage 1 focuses on pipeline correctness, not high final accuracy.

## Architectural Decisions
- `main.py` is a thin experiment runner.
- Core code lives under `src/` by responsibility:
  - tokenizer
  - synthetic data
  - model
  - training/evaluation
  - generation
- Use a decoder-only GPT-style transformer with RoPE, causal attention, RMSNorm, GELU FFN, and next-token loss from the start.
- Write experiment artifacts to `runs/stage1/<timestamp>/`.

## Tasks
1. Create the Stage 1 runner scaffold.
2. Implement character tokenizer and addition sample generation.
3. Implement tiny decoder-only transformer.
4. Implement training, validation, exact-answer evaluation, and generation.
5. Persist config, metrics, samples, and summary artifacts.
6. Add focused unit tests for tokenizer, data, model shape, and generation behavior.
7. Run local verification and independent subagent verification.

## Risks
- Tiny runs may show noisy accuracy even when the pipeline is correct.
- MPS support can vary by local PyTorch build, so device selection must gracefully fall back.
- Context length must be large enough for prompts, answers, and newline targets.

## Verification Strategy
- Run stdlib unit tests with `python -m unittest`.
- Run `main.py` with a short training budget.
- Confirm metrics are written to `metrics.csv`.
- Confirm generated samples are written to `samples.txt`.
- Confirm summary includes config, device, final losses, accuracy, runtime, and artifact paths.
- Review the diff for minimality and alignment with Stage 1 only.

## Verification Results
- Local unit tests passed: `.venv/bin/python -m unittest discover -s tests -v`.
- Local smoke run passed: loss decreased from `2.8160` to `0.8930` over 60 steps on MPS.
- Independent subagent verification passed: loss decreased from `2.8827` to `1.2710` over 30 steps.
- Run artifacts were written with `config.json`, `metrics.csv`, `samples.txt`, and `summary.json`.
- Exact-answer accuracy remains low at smoke-test budgets; Stage 1 currently verifies pipeline wiring, not mastery.
