# Arithmetic Representation Loss Sprint

## Scope
- Add configurable arithmetic formatting for training and evaluation.
- Add answer-weighted loss to reduce wasted prompt-copying signal.
- Save per-eval checkpoints for exhaustive post-training selection.

## Assumptions
- Current best normal-format model remains `final/checkpoint_best_val.pt`.
- Reversed/fixed formats are not checkpoint-compatible with old runs, so train from scratch.
- Exhaustive evaluation is the promotion gate.

## Architectural Decisions
- Keep model architecture unchanged.
- Centralize arithmetic text formatting in `src/data.py`.
- Supported formats: `normal`, `fixed`, `reversed`, `fixed_reversed`.
- Default behavior remains `normal` with full next-token loss for backwards compatibility.

## Step-by-Step Tasks
1. Add shared formatting/parsing helpers.
2. Thread `number_format` through training, validation, samples, and exhaustive evaluation.
3. Add answer-weighted loss using `--loss-prompt-weight`.
4. Save checkpoints at every evaluation step.
5. Add focused tests.
6. Run unit, compile, and smoke verification.

## Risks
- Reversed outputs require evaluator parsing to compare canonical numeric answers.
- Answer-only loss may be too aggressive; use `0.2` prompt weight first.
- Sampled validation still may not track exhaustive accuracy.

## Verification Strategy
- Unit tests for formatting, parsing, loss weighting, CLI config, and evaluator behavior.
- Compile check.
- Smoke training run with `fixed_reversed` and answer-weighted loss.

## Verification Results
- Unit tests passed: `.venv/bin/python -m unittest discover -s tests -v`.
- Compile check passed: `.venv/bin/python -m compileall main.py src scripts tests`.
- Smoke train passed: `runs/stage5/format-loss-smoke`.
- Smoke train wrote `checkpoint_step_0001.pt`, `checkpoint_step_0002.pt`, and `checkpoint_step_0003.pt`.
- Smoke config recorded `number_format=fixed_reversed` and `loss_prompt_weight=0.2`.
- Evaluator smoke passed with `--number-format fixed_reversed`.
- Full run passed: `runs/stage5/digit2-fixed-reversed-answer-weighted`.
- Full final checkpoint exhaustive eval: `10,000 / 10,000` correct, `0` errors.
- Decision: this beats the packaged normal-format model `9,978 / 10,000` and is eligible for promotion after packaging.

## Recommended Full Run
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

Evaluate per-step checkpoints if final is close but not best:
```bash
for ckpt in runs/stage5/digit2-fixed-reversed-answer-weighted/checkpoint_step_*.pt; do
  name=$(basename "$ckpt" .pt)
  .venv/bin/python scripts/evaluate_checkpoint.py \
    --checkpoint "$ckpt" \
    --digit-length 2 \
    --batch-size 256 \
    --progress-every 0 \
    --number-format fixed_reversed \
    --out "runs/stage5/digit2-fixed-reversed-answer-weighted/eval_${name}.json"
done
```
