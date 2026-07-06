# Project Roadmap

## Purpose
Keep the project moving in controlled stages from a tiny sanity run to the final addition-transformer experiment.

## Tracking Rules
- Work one stage at a time.
- Do not scale model size until the previous stage is verified.
- Keep architecture changes separate from experiment/config changes.
- Record completed runs with config, device, loss, accuracy, time, and notes.

## Stage 0: Environment
Status: Done

Requirements:
- Repo-local Python virtual environment
- PyTorch installed
- Matplotlib installed
- NumPy installed
- MPS available when supported, CPU fallback otherwise

Verification:
- `pip check`
- Import `torch`, `matplotlib`, and `numpy`
- Run a small tensor operation on selected device

## Stage 1: Tiny Sanity Run
Status: Done

Goal:
Prove the full pipeline works before optimizing accuracy.

Build:
- Character tokenizer for `<pad>`, digits, `+`, `=`, and newline
- Synthetic addition data generator
- Tiny decoder-only transformer
- Training loop
- Generation loop
- Minimal evaluation

Suggested config:
```text
d_model = 32
num_heads = 2
num_layers = 1
ffn_hidden = 128
context_length = 16
digit_length = 1
```

Verification:
- Batch shapes are correct
- Loss decreases over a short run
- Generated samples complete with newline
- Device selection uses MPS when available

## Stage 2: Base Architecture Run
Status: Done

Goal:
Move from tiny sanity model to the first serious documented config.

Config:
```text
d_model = 128
num_heads = 4
num_layers = 4
ffn_hidden = 512
context_length = 16 or 32
```

Requirements:
- RoPE applied to `Q` and `K` only
- Causal multi-head self-attention
- RMSNorm pre-norm blocks
- GELU FFN
- AdamW optimizer
- Next-token cross-entropy loss

Verification:
- Parameter count reported
- Training and validation loss reported
- Sample generations saved
- Exact-answer accuracy measured

## Stage 3: Small Tuning Runs
Status: Done

Goal:
Tune core training choices before scaling.

Vary:
- Learning rate
- Batch size
- Context length
- Digit length
- Training steps/tokens

Track:
- Train loss
- Validation loss
- Exact accuracy
- Training time
- Device used

## Stage 4: Scaling Ladder
Status: Done

Goal:
Scale deliberately using evidence from smaller runs.

Vary:
- `d_model`
- Number of layers
- Number of heads
- FFN hidden size
- Context length
- Training tokens

Plots:
- Parameter count vs validation loss
- Training tokens vs validation loss
- Digit length vs exact accuracy
- Model size vs training time

## Stage 5: Final Model Selection
Status: Done

Goal:
Choose the final project-stage configuration from experiment evidence.

Target:
- Final selected compact model: 1,188,736 parameters
- Same architecture family
- Chosen data scale and context length

Deliverables:
- Final checkpoint
- Exhaustive eval metrics
- Eval metrics
- Sample generations
- Short run summary

## Current Next Step
Final package is available under `final/`.
