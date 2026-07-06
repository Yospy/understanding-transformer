# Final Package Manifest

## Source
```text
runs/stage4/digit2-depth6-2000-lr1e-3-ckpt2
```

## Files
| File | Purpose | SHA-256 |
|---|---|---|
| `checkpoint_best_val.pt` | Selected final checkpoint | `caa75c832a702e4705142468904629c2675ab0489c803a5b8830756dccd8754a` |
| `checkpoint_final.pt` | Final-step checkpoint | `1f3db3df4371c1e42600dad37324aafcb45cf1715093668a14900c9c1dbee156` |
| `config.json` | Model and experiment config | `50971d662e2151c4ab269bba37ff515b5274100f2af23992980f241b76ee52fc` |
| `eval_exhaustive_2digit.json` | Exhaustive 2-digit evaluation | `0b530259814f521cff8fcc701a5d71ba23da9e2eaac74b40c1c17fd13835ce3b` |
| `eval_exhaustive_2digit_verify.json` | Verification eval from copied final checkpoint | `86e7682457e35fd886858b1dc5414d8ed87a86305499845195ce2377941c7249` |
| `metrics.csv` | Training/eval curve | `ad15c26ef490d0e585ff19484c3ab46166b4b43529ba1e2f31d012b20086329e` |
| `samples.txt` | Final generated samples | `43a2f056bde1e2b829f012272afb8be06dd7f2fef9167d2c8122c882c25585b9` |
| `summary.json` | Final run summary | `1c676179995d87bcd54de1ecfece4fb2b552abdb662880518d309662f738f9a2` |
| `final_report.md` | Human-readable final report | `5ef5ff0d6608cd6e2504216ef8ece04cb35bc124b99fbab5836287e398093677` |

## Verification Commands
```bash
.venv/bin/python scripts/evaluate_checkpoint.py \
  --checkpoint final/checkpoint_best_val.pt \
  --digit-length 2 \
  --batch-size 256 \
  --progress-every 1000 \
  --out final/eval_exhaustive_2digit_verify.json
```
