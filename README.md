# Understanding Transformer

Learning project for building a small decoder-only transformer that learns arithmetic from scratch, then improves through supervised fine-tuning and verifier-based RL.

## What This Project Does

This repo trains a character-level transformer on arithmetic expressions such as:

```text
37+48=85
91-23=68
12*9=108
44/5=8R4
```

The goal was not to build a production calculator. The goal was to understand the training pipeline behind language models in a small, inspectable setting.

## Training Stages

1. **Base training / pre-training**
   - Train a small transformer from scratch on synthetic arithmetic examples.
   - Learn digits, operators, the equals sign, and answer formatting.

2. **Supervised fine-tuning**
   - Generate verified arithmetic examples on the fly.
   - Train with next-token prediction on correct answers.
   - Extend from addition into `+`, `-`, `*`, and `/`.

3. **RL with GRPO**
   - Sample multiple answers per prompt.
   - Score each answer with a deterministic Python verifier.
   - Reward correct answers with `1.0` and wrong answers with `0.0`.
   - Use GRPO-style group-relative advantages, clipping, and KL control.

## Key Results

- Final selected 2-digit addition checkpoint: `99.78%` exhaustive accuracy.
- Addition GRPO run improved exhaustive accuracy to `99.88%`.
- Multi-operation SFT reached roughly `85%` combined exhaustive accuracy across `+`, `-`, `*`, and `/`.
- Multiplication remained the hardest operation.
- GRPO helped most when the model could already sometimes sample correct answers.

## Important Files

- `main.py` - supervised training entrypoint.
- `src/model.py` - decoder-only transformer.
- `src/data.py` - synthetic arithmetic generation, formatting, parsing, and verifier helpers.
- `src/grpo.py` - reward scoring, group advantages, log-probs, and GRPO loss.
- `src/sampling.py` - stochastic rollout sampling for RL.
- `scripts/train_grpo_addition.py` - GRPO training script.
- `scripts/evaluate_checkpoint.py` - exhaustive checkpoint evaluation.
- `final/` - packaged final addition checkpoint and evaluation artifacts.
- `sprints/` and `docs/plans/` - experiment plans and learning notes.

## Run Checks

```bash
.venv/bin/python -m compileall main.py src scripts tests
.venv/bin/python -m unittest discover -s tests -p 'test*.py'
```

## Main Lesson

SFT and data representation produced the biggest gains. GRPO was useful as a learning exercise and as a targeted fine-tuning tool, but it only worked when sampled completions contained enough contrast between correct and wrong answers.
