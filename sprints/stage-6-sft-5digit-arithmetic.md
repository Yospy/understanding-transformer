# Stage 6 SFT 5-Digit Arithmetic

## Scope
- Train a supervised arithmetic model for generated `+`, `-`, `*`, and `/` examples.
- Target operands up to 5 digits: `0..99999`.
- Use this SFT checkpoint as the base for later GRPO verifier training.
- Keep training data synthetic and programmatically verified.

## Assumptions
- No external dataset is needed because labels come from Python integer arithmetic.
- The strongest addition result came from `fixed_reversed` formatting plus answer-weighted loss, so keep that setup.
- The current code can already generate multi-operation examples with `--operations '+,-,*,/'`.
- True curriculum sampling across mixed digit lengths is a future improvement; the first run can train directly with `--digit-length 5`.

## Data Format
Use the existing next-token SFT format:

```text
prompt + completion
```

Example normal expression:

```text
12345 * 67890 = 838102050
```

Internal `fixed_reversed` training text:

```text
54321*09876=0502018380
```

Operation rules:
- `+`: answer width is `digit_length + 1`.
- `-`: signed answer, with formatted magnitude.
- `*`: answer width is `digit_length * 2`.
- `/`: answer is `quotientRremainder`.
- `/0`: answer is `ERR`.

## Architectural Decisions
- Train a new checkpoint with the extended arithmetic vocabulary.
- Keep the decoder-only transformer architecture unchanged for the first SFT run.
- Use `context_length=32`; 5-digit all-operation examples need at least `23`.
- Use `number_format=fixed_reversed` so carries and local digit operations are easier to learn.
- Use `loss_prompt_weight=0.2` to focus loss on answer generation while retaining prompt structure.
- Generate training batches on the fly instead of writing a huge dataset to disk.

## Step-by-Step Tasks
1. Verify generator formatting/parsing against Python arithmetic.
2. Run a short smoke SFT command.
3. Train the baseline 5-digit all-operation SFT model.
4. Evaluate sampled accuracy and inspect examples.
5. Save failed examples as hard cases for the next SFT or GRPO phase.

## Risks
- Direct 5-digit all-operation SFT may be too hard without curriculum.
- Multiplication and division have much larger output spaces than addition.
- Sampled accuracy can hide rare failures, so edge-case evaluation is required.
- Exhaustive 5-digit evaluation is infeasible.

## Verification Strategy
- Unit tests for formatting, parsing, tokenizer compatibility, and batch creation.
- Compile check before long training.
- Smoke train with the same flags but very few steps.
- Sampled evaluation during training.
- Edge-case evaluation for zeros, ones, powers of ten, large carries, borrows, division by zero, and large products.

## Smoke Command
Run this before a long job:

```bash
.venv/bin/python main.py \
  --stage stage6-sft-5digit-smoke \
  --digit-length 5 \
  --context-length 32 \
  --d-model 64 \
  --num-heads 4 \
  --num-layers 2 \
  --ffn-hidden 256 \
  --steps 5 \
  --eval-interval 1 \
  --accuracy-examples 8 \
  --val-batches 1 \
  --batch-size 8 \
  --learning-rate 0.001 \
  --number-format fixed_reversed \
  --loss-prompt-weight 0.2 \
  --operations '+,-,*,/' \
  --run-dir runs/stage6/sft-5digit-smoke
```

## Baseline Training Command
This command works with the current code and trains the first SFT baseline:

```bash
.venv/bin/python main.py \
  --stage stage6-sft-5digit-allops \
  --digit-length 5 \
  --context-length 32 \
  --d-model 128 \
  --num-heads 4 \
  --num-layers 6 \
  --ffn-hidden 512 \
  --steps 10000 \
  --eval-interval 500 \
  --accuracy-examples 2000 \
  --val-batches 8 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --number-format fixed_reversed \
  --loss-prompt-weight 0.2 \
  --operations '+,-,*,/' \
  --run-dir runs/stage6/sft-5digit-allops-fixed-reversed
```

## GRPO Boundary
Do not start GRPO until SFT can reliably produce valid formatted answers. The SFT output checkpoint becomes the GRPO reference/base policy, and Python arithmetic becomes the verifier reward.
