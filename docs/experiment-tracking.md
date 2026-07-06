# Experiment Tracking

## Run Command
Use the repo-local environment:

```bash
.venv/bin/python main.py
```

The default config is the Stage 3 tuning baseline:

```text
stage = stage3
d_model = 128
num_heads = 4
num_layers = 4
ffn_hidden = 512
context_length = 16
digit_length = 2
steps = 500
eval_interval = 100
```

Useful Stage 3 baseline command:

```bash
.venv/bin/python main.py \
  --steps 500 \
  --eval-interval 100 \
  --batch-size 64 \
  --digit-length 2
```

For a quick Stage 3 smoke run:

```bash
.venv/bin/python main.py --steps 20 --eval-interval 10 --batch-size 16 --val-batches 2
```

To reproduce the Stage 2 baseline:

```bash
.venv/bin/python main.py \
  --stage stage2 \
  --digit-length 1 \
  --steps 300 \
  --eval-interval 50 \
  --run-dir runs/stage2/repro-300
```

To reproduce the Stage 1 tiny config:

```bash
.venv/bin/python main.py \
  --stage stage1 \
  --d-model 32 \
  --num-heads 2 \
  --num-layers 1 \
  --ffn-hidden 128 \
  --steps 300 \
  --eval-interval 50
```

## Artifact Location
Each run writes to:

```text
runs/<stage>/<YYYYMMDD-HHMMSS>/
```

Files:

- `config.json`: model config, experiment config, vocab, parameter count, selected device.
- `metrics.csv`: per-eval-step `train_loss`, `val_loss`, `exact_accuracy`, and elapsed seconds.
- `samples.txt`: generated examples after training.
- `summary.json`: final loss, accuracy, device, runtime, and artifact paths.
- `checkpoint_best_val.pt`: model and optimizer state from the lowest validation-loss eval step.
- `checkpoint_final.pt`: model and optimizer state from the final training step.

## Comparing Runs
Compare completed run directories with:

```bash
.venv/bin/python scripts/compare_runs.py \
  runs/stage2/20260617-142931 \
  runs/stage3/<run-dir>
```

The comparison table includes:

- config size: parameters, layers, heads, hidden size, context length
- training budget: steps and elapsed seconds
- result quality: final validation loss, best validation loss, final exact accuracy, best exact accuracy

## Stage 4 Scaling Ladder
Start with training budget before model size:

```bash
.venv/bin/python main.py \
  --stage stage4 \
  --digit-length 2 \
  --steps 1500 \
  --eval-interval 250 \
  --run-dir runs/stage4/digit2-budget-1500
```

Then test width and depth separately:

```bash
.venv/bin/python main.py \
  --stage stage4 \
  --digit-length 2 \
  --d-model 192 \
  --num-heads 6 \
  --num-layers 4 \
  --ffn-hidden 768 \
  --steps 1000 \
  --eval-interval 200 \
  --run-dir runs/stage4/digit2-width192-1000
```

```bash
.venv/bin/python main.py \
  --stage stage4 \
  --digit-length 2 \
  --d-model 128 \
  --num-heads 4 \
  --num-layers 6 \
  --ffn-hidden 512 \
  --steps 1000 \
  --eval-interval 200 \
  --run-dir runs/stage4/digit2-depth6-1000
```

Compare:

```bash
.venv/bin/python scripts/compare_runs.py \
  runs/stage3/digit2-baseline \
  runs/stage4/digit2-budget-1500 \
  runs/stage4/digit2-width192-1000 \
  runs/stage4/digit2-depth6-1000
```

## Durable Experiment Log
Use `docs/experiment-log.md` for concise records of meaningful runs. Keep raw run artifacts under `runs/`, which is intentionally ignored by git.

## Checkpoint Evaluation
Only runs made after checkpointing was added contain checkpoint files. Older runs must be rerun if reload/eval is needed.

Train the current best config with checkpointing:

```bash
.venv/bin/python main.py \
  --stage stage4 \
  --digit-length 2 \
  --d-model 128 \
  --num-heads 4 \
  --num-layers 6 \
  --ffn-hidden 512 \
  --steps 2000 \
  --eval-interval 250 \
  --accuracy-examples 500 \
  --learning-rate 0.001 \
  --run-dir runs/stage4/digit2-depth6-2000-lr1e-3-ckpt
```

Run exhaustive 2-digit evaluation:

```bash
.venv/bin/python scripts/evaluate_checkpoint.py \
  --checkpoint runs/stage4/digit2-depth6-2000-lr1e-3-ckpt/checkpoint_best_val.pt \
  --digit-length 2 \
  --batch-size 256 \
  --progress-every 1000 \
  --out runs/stage4/digit2-depth6-2000-lr1e-3-ckpt/eval_exhaustive_2digit.json
```

## Hard-Case Curriculum Training
Use the final exhaustive eval errors as a hard-case file:

```bash
.venv/bin/python main.py \
  --stage stage5 \
  --digit-length 2 \
  --d-model 128 \
  --num-heads 4 \
  --num-layers 6 \
  --ffn-hidden 512 \
  --steps 3000 \
  --eval-interval 500 \
  --accuracy-examples 1000 \
  --learning-rate 0.0005 \
  --init-checkpoint final/checkpoint_best_val.pt \
  --hard-case-file final/eval_exhaustive_2digit.json \
  --hard-case-ratio 0.25 \
  --run-dir runs/stage5/digit2-hardcase-lr5e-4
```

Then evaluate the best checkpoint:

```bash
.venv/bin/python scripts/evaluate_checkpoint.py \
  --checkpoint runs/stage5/digit2-hardcase-lr5e-4/checkpoint_best_val.pt \
  --digit-length 2 \
  --batch-size 256 \
  --progress-every 1000 \
  --out runs/stage5/digit2-hardcase-lr5e-4/eval_exhaustive_2digit.json
```

Result: this `0.25` hard-case ratio regressed exhaustive accuracy, so do not promote it. A safer next probe is a shorter, lower-LR fine-tune:

```bash
.venv/bin/python main.py \
  --stage stage5 \
  --digit-length 2 \
  --d-model 128 \
  --num-heads 4 \
  --num-layers 6 \
  --ffn-hidden 512 \
  --steps 500 \
  --eval-interval 100 \
  --accuracy-examples 1000 \
  --learning-rate 0.0001 \
  --init-checkpoint final/checkpoint_best_val.pt \
  --hard-case-file final/eval_exhaustive_2digit.json \
  --hard-case-ratio 0.05 \
  --run-dir runs/stage5/digit2-hardcase-r05-lr1e-4
```

Evaluate the final checkpoint, not only best-val, because val loss did not track exhaustive accuracy in the `0.25` run:

```bash
.venv/bin/python scripts/evaluate_checkpoint.py \
  --checkpoint runs/stage5/digit2-hardcase-r05-lr1e-4/checkpoint_final.pt \
  --digit-length 2 \
  --batch-size 256 \
  --progress-every 1000 \
  --out runs/stage5/digit2-hardcase-r05-lr1e-4/eval_exhaustive_2digit_final.json
```

## Carry-Local Representation Training
Train from scratch when changing `--number-format`; old normal-format checkpoints should not be used as initialization.

Recommended first run:

```bash
.venv/bin/python main.py \
  --stage stage5 \
  --digit-length 2 \
  --d-model 128 \
  --num-heads 4 \
  --num-layers 6 \
  --ffn-hidden 512 \
  --steps 2000 \
  --eval-interval 250 \
  --accuracy-examples 1000 \
  --learning-rate 0.001 \
  --number-format fixed_reversed \
  --loss-prompt-weight 0.2 \
  --run-dir runs/stage5/digit2-fixed-reversed-answer-weighted
```

Evaluate the final checkpoint:

```bash
.venv/bin/python scripts/evaluate_checkpoint.py \
  --checkpoint runs/stage5/digit2-fixed-reversed-answer-weighted/checkpoint_final.pt \
  --digit-length 2 \
  --batch-size 256 \
  --progress-every 1000 \
  --number-format fixed_reversed \
  --out runs/stage5/digit2-fixed-reversed-answer-weighted/eval_exhaustive_2digit_final.json
```

Result: `checkpoint_final.pt` reached `10,000 / 10,000` on exhaustive 2-digit evaluation with `--number-format fixed_reversed`.

## Multi-Operation Arithmetic Training
This target supports operands `0..99` and operations `+`, `-`, `*`, `/`.

Division semantics:

```text
80/7 = 11R3
80/0 = ERR
```

First full mixed-operation run:

```bash
.venv/bin/python main.py \
  --stage stage6 \
  --digit-length 2 \
  --context-length 24 \
  --d-model 128 \
  --num-heads 4 \
  --num-layers 6 \
  --ffn-hidden 512 \
  --steps 5000 \
  --eval-interval 500 \
  --accuracy-examples 2000 \
  --learning-rate 0.001 \
  --number-format fixed_reversed \
  --loss-prompt-weight 0.2 \
  --operations '+,-,*,/' \
  --run-dir runs/stage6/digit2-allops-fixed-reversed
```

Evaluate all `40,000` operation/input combinations:

```bash
.venv/bin/python scripts/evaluate_checkpoint.py \
  --checkpoint runs/stage6/digit2-allops-fixed-reversed/checkpoint_final.pt \
  --digit-length 2 \
  --batch-size 256 \
  --progress-every 2000 \
  --number-format fixed_reversed \
  --operations '+,-,*,/' \
  --out runs/stage6/digit2-allops-fixed-reversed/eval_allops_exhaustive_final.json
```

Serve the trained model in the local chat UI:

```bash
.venv/bin/python scripts/chat_addition.py \
  --checkpoint runs/stage6/digit2-allops-fixed-reversed/checkpoint_final.pt \
  --number-format fixed_reversed \
  --serve \
  --host 127.0.0.1 \
  --port 7860
```
