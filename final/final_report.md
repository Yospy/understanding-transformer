# Final Addition Transformer Report

## Selected Model
The final selected model is the compact 6-layer Stage 4 checkpoint:

```text
source_run = runs/stage4/digit2-depth6-2000-lr1e-3-ckpt2
checkpoint = final/checkpoint_best_val.pt
parameters = 1,188,736
device = mps
```

## Architecture
```text
decoder-only transformer
character tokenizer
vocab_size = 14
d_model = 128
num_heads = 4
num_layers = 6
ffn_hidden = 512
context_length = 16
RoPE on Q/K
RMSNorm pre-norm blocks
GELU FFN
AdamW optimizer
next-token cross entropy
```

## Training Config
```text
digit_length = 2
steps = 2000
batch_size = 64
learning_rate = 0.001
weight_decay = 0.01
eval_interval = 250
accuracy_examples = 500
seed = 1337
```

## Final Metrics
```text
final_train_loss = 0.8637006879
final_val_loss = 0.8584643304
best_val_loss = 0.8584643304
sampled_exact_accuracy = 0.996
elapsed_s = 240.5609
```

## Exhaustive Evaluation
The final checkpoint was evaluated on every pair:

```text
0..99 + 0..99 = 10,000 examples
correct = 9,978
incorrect = 22
accuracy = 0.9978
```

This means the model performs 2-digit addition correctly on nearly all exhaustive cases, but it is not a formal arithmetic solver.

## Sample Generations
```text
80+26=106
35+5=40
5+95=100
85+45=130
27+25=52
53+26=79
64+91=155
81+54=135
```

## Known Failure Examples
From exhaustive evaluation:

```text
0+8=7      expected 8
1+7=9      expected 8
1+8=1      expected 9
2+7=10     expected 9
9+26=25    expected 35
92+41=132  expected 133
```

## Experiment Ladder Summary
- Stage 1 proved the full pipeline with a tiny model.
- Stage 2 solved 1-digit addition with the base architecture.
- Stage 3 showed 2-digit addition was not solved by the base model at 500 steps.
- Stage 4 showed that more depth plus longer training worked best.
- Width-heavy scaling and the 4.7M model did not beat the compact 1.19M model.

## Final Decision
Use `final/checkpoint_best_val.pt` as the final project-stage model.

The final model is good enough to close this stage: it reaches `99.78%` exhaustive 2-digit accuracy with a compact architecture.
