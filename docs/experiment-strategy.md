# Experiment Strategy

## Goal
Build a small addition-learning transformer through a sequence of cheap experiments, then finish with a selected ~10M parameter model as the final model for this project stage.

## Core Principle
All experiments stay within the same architecture family:

- decoder-only transformer
- character-level arithmetic data
- next-token prediction
- PyTorch training loop
- MPS when available, CPU fallback

We do not redesign the system each time. We change the configuration, observe results, and scale deliberately.

## Experiment Flow
1. Tiny sanity run: prove the data, tokenizer, model, loss, training, and generation loop works.
2. Small runs: tune basic choices such as learning rate, batch size, context length, and digit difficulty.
3. Scaling ladder: increase model size and training tokens in controlled steps.
4. Plot results with `matplotlib`.
5. Pick the final configuration.
6. Train the final ~10M parameter model.

## What Changes Between Runs
- `d_model`
- number of layers
- number of heads
- FFN hidden size
- context length
- digit length
- batch size
- learning rate
- training tokens / steps

Each run starts from fresh random weights unless explicitly stated otherwise.

## What We Plot
- parameter count vs validation loss
- training tokens vs validation loss
- digit length vs exact accuracy
- model size vs training time

## Final Experiment
The last experiment is the finalized model for this project stage:

- around 10M parameters
- selected from earlier experiment evidence
- trained with the chosen data scale and context length
- saved with checkpoint, plots, eval metrics, and sample generations

This final model is not a general LLM. It is the project-stage final addition transformer.
