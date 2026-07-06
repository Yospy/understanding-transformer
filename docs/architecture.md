# Model Architecture

## Architecture Family
All experiments use the same base architecture family:

- decoder-only GPT-style transformer
- character-level tokenizer
- learned token embeddings
- RoPE positional encoding
- causal multi-head self-attention
- RMSNorm
- FFN
- residual stream
- LM head
- next-token prediction

Only the configuration changes between experiments.

## Tokenizer and Vocab
Use a tiny character-level arithmetic vocabulary:

```text
<pad>
0 1 2 3 4 5 6 7 8 9
+
=
\n
```

So:

```text
vocab_size = 14
```

No BPE is needed for this stage.

## Embeddings
The token embedding table starts randomly and is learned during training.

```text
token_embedding: [vocab_size x d_model]
```

For the first serious config:

```text
[14 x 128]
```

Each token ID selects one row from this table.

## Positional Encoding
Use **RoPE** from the beginning.

RoPE rules:

- not a learned position table
- applied to `Q` and `K` only
- not applied to `V`
- gives relative position information through rotation
- keeps the architecture closer to modern LLMs
- makes later context-extension experiments cleaner than learned absolute positions

Inside attention:

```text
X -> Q, K, V
apply RoPE to Q and K
attention = softmax(QK^T / sqrt(head_dim) + causal_mask) V
```

## Base Config
First serious practical config:

```text
d_model = 128
num_heads = 4
head_dim = 32
num_layers = 4
ffn_hidden = 512
context_length = 16 or 32
```

Invariant:

```text
d_model % num_heads == 0
```

## Transformer Block
Each block uses pre-norm residual structure:

```text
x = x + attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

Expanded:

```text
RMSNorm
-> causal multi-head attention with RoPE
-> residual add
-> RMSNorm
-> FFN
-> residual add
```

## Attention
For the base config:

```text
X [seq x 128]

Wq -> Q [seq x 128]
Wk -> K [seq x 128]
Wv -> V [seq x 128]

split into 4 heads:
Q/K/V -> [4 x seq x 32]

RoPE(Q, K)

scores:
[seq x 32] x [32 x seq] = [seq x seq]

causal mask
softmax
x V
concat heads
Wo projection
```

Attention mixes information across token positions.

## FFN
Use a simple dense FFN first:

```text
128 -> 512 -> 128
```

Activation:

```text
GELU
```

Later upgrades can test SwiGLU, but V1 stays simple.

## Output Head
Final hidden states:

```text
[seq x d_model]
```

LM head:

```text
[d_model x vocab_size]
```

For the base config:

```text
[128 x 14]
```

Output:

```text
logits [seq x 14]
```

Each row predicts the next character.

## Training Objective
Use next-token prediction.

Example:

```text
input:  1 2 + 7 = 1 9
target: 2 + 7 = 1 9 \n
```

Loss:

```text
cross_entropy(logits, targets)
```

Optimizer:

```text
AdamW
```

## Decoding
Prompt:

```text
12+7=
```

Autoregressive loop:

```text
predict "1"
append
predict "9"
append
predict "\n"
stop
```

## Flash Attention
Do not manually implement Flash Attention in V1.

Reason:

- sequence lengths are tiny
- normal attention is easier to inspect
- first goal is learning the architecture and training loop

Later we can test PyTorch scaled-dot-product attention or Flash-style kernels separately.

## Scaling Rule
The architecture family stays fixed across experiments:

```text
char vocab -> token embedding -> RoPE transformer blocks -> LM head
```

We scale only configuration values:

- `d_model`
- number of layers
- number of heads
- FFN hidden size
- context length
- data scale
- training budget
