# Multi Operation Arithmetic Sprint

## Scope
- Extend arithmetic tasks to `+`, `-`, `*`, and `/`.
- Support operands `0..99` for the first multi-operation model.
- Keep local chat inference usable with normal expressions.

## Assumptions
- Division is integer quotient/remainder: `7/3=2R1`.
- Division by zero outputs `ERR`.
- Negative subtraction outputs use `-`, e.g. `3-7=-4`.
- New multi-operation checkpoints are not compatible with old 14-token checkpoints, but old checkpoints should remain loadable by reading checkpoint vocab.

## Architectural Decisions
- Add an extended tokenizer vocab for new runs.
- Preserve `fixed_reversed` operand formatting.
- Format operation-specific answers:
  - addition: result width `digit_length + 1`
  - subtraction: signed canonical integer
  - multiplication: result width `digit_length * 2`
  - division: quotient/remainder with formatted numeric fields, or `ERR`
- Use `context_length=24` for 2-digit multi-operation training.

## Step-by-Step Tasks
1. Add tokenizer vocab compatibility.
2. Add operation formatting and batch sampling.
3. Thread operations through training and sampled accuracy.
4. Add multi-operation random evaluator support.
5. Update chat parser and UI.
6. Add tests and run verification.

## Risks
- Mixed operations are harder than addition; first run may need more steps or capacity.
- Division is a different task shape than the other operations.
- Exhaustive evaluation for all operations is larger but still feasible for `0..99`: `40,000` cases.

## Verification Strategy
- Unit tests for operation formatting, parsing, and tokenizer compatibility.
- Compile check.
- Smoke training with all operations.
- Random/exhaustive evaluation after full training.

## Verification Results
- Unit tests passed: `.venv/bin/python -m unittest discover -s tests -v`.
- Compile check passed: `.venv/bin/python -m compileall main.py src scripts tests`.
- Multi-operation smoke train passed: `runs/stage6/multiop-smoke`.
- Multi-operation evaluator smoke ran successfully against the smoke checkpoint.
- Old fixed-reversed addition checkpoint still loads and evaluates correctly: `10,000 / 10,000` on 2-digit addition.

## Recommended Full Run
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
