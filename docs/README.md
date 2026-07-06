# Project Docs

## Current Planning Docs
- [Project Roadmap](roadmap.md): stage-by-stage checklist from environment setup to final model.
- [Experiment Strategy](experiment-strategy.md): experiment ladder, plots, and final ~10M model plan.
- [Model Architecture](architecture.md): tokenizer, embeddings, RoPE, attention, FFN, training, and decoding design.
- [Experiment Tracking](experiment-tracking.md): run commands and artifact layout.
- [Experiment Log](experiment-log.md): concise record of meaningful runs.

## Final Package
- [`../final/final_report.md`](../final/final_report.md): final selected model report.
- [`../final/MANIFEST.md`](../final/MANIFEST.md): final artifact manifest and verification command.

## Current Direction
Build the addition transformer in controlled stages:

```text
tiny sanity run -> small tuning runs -> scaling ladder -> final ~10M model
```

The architecture family stays stable. Experiments change configuration and data scale.
