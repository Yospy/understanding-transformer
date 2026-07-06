# LLM Internals — Complete Guide

> From the big-picture vision down to the last matrix dimension.
> A reference built end-to-end: attention, layers, hardware, training, serving.

**Running toy model used throughout:** `vocab = 10`, `d` (vector width) `= 4`, `sequence = 3 words`, `FFN hidden = 8`.
*(Real models: vocab ~100K+, d ~4,096–12,288, hidden ~4×d, 60–100+ layers.)*

---

## PART 1 — THE BIG PICTURE (why any of this matters)

- **Scale is solved; efficiency is the frontier.** We *can* build giant models. The game now is running them cheaply — that's where MSA, MoE, quantization, and the local-model dream all come from.
- **Decompose, don't compress.** A trillion-param model isn't a trillion params of intelligence — it's a small **reasoning core** drowning in a vast **knowledge store**. The path to local models is *separating* them: keep the small core hot, stream the cold knowledge on demand.
- **Two attacks on the 10-year-old softmax default:**
  - **Parallax** makes attention *more accurate* (adds a learned correction so it can extrapolate, not just average — unlocked by the Muon optimizer).
  - **MSA (MiniMax M3)** makes attention *more efficient* (selects relevant blocks instead of comparing everything — "select, don't compress").
- **The existence proof:** your brain runs frontier intelligence on ~20 watts. So "smart model, local, forever" isn't impossible — we're just doing it inefficiently for now.

---

## PART 2 — WHAT A MODEL *IS*

**A model = a stack of frozen weight matrices. Nothing else.** A forward pass is a **pipeline of matrix multiplications.** Everything else (softmax, activation, lookup) is cheap glue.

| Weight matrix | Size | Job |
|---|---|---|
| Embedding table `E` | `[vocab × d]` = `[10×4]` | word → vector (a lookup table) |
| `Wq, Wk, Wv, Wo` (per layer) | each `[d × d]` = `[4×4]` | attention projections |
| FFN `W1` (per layer) | `[d × hidden]` = `[4×8]` | expand |
| FFN `W2` (per layer) | `[hidden × d]` = `[8×4]` | contract |
| LM head | `[d × vocab]` = `[4×10]` | final vector → word scores |

**THE master anchor — every word is a ROW:**
- In the **data** (activations: X, Q, K, V, the stream, logits): **one row per word.**
- The **weights** are *not* per-word — they're **shared**, applied identically to every row.
- → *That's why one model handles any sentence length:* add a word = add a row; the fixed-size weights don't change.

**The matmul rule (#1 build-time bug):** `[a×b] × [c×d]` works only if **b = c** (inner dims match). Mismatch = it won't run.

---

## PART 3 — THE FORWARD PASS (using the model)

### Step 0–1: Text → vectors (a LOOKUP, not a matmul)
```
"I love you" → token IDs → pick those rows from E → X = [3 × 4]
```
One row per word. This is where input *becomes* a matrix.

### Step 2: Inside ONE layer — two halves

A layer = **Attention** (mix info *across* words = communication) then **FFN** (process *each* word = computation), each wrapped in a **residual add**.

**① Attention — the famous formula:**

```
Attention(Q, K, V) = softmax( (Q · Kᵀ) / √d ) · V
```

```
X[3×4] × Wq,Wk,Wv  → Q, K, V  each [3×4]   (3 different projections of the SAME X)
Q[3×4] × Kᵀ[4×3]   → scores [3×3]          ← TRANSPOSE: flips keys from rows→columns
                                              so the matmul gives all pairwise dot products
÷ √d               → keeps scores tame so softmax isn't too spiky
causal mask        → zero out the FUTURE (a word sees only itself + earlier words)
softmax (per row)  → attention weights [3×3] (each row sums to 1)
× V[3×4]           → attention output [3×4]  (each word = weighted blend of values)
+ residual (add back X)
```

- **Two matrices, don't confuse them:** scores `[seq×seq]` (every word vs every word — grows **quadratically**, the expensive monster MSA attacks) vs output `[seq×d]` (one enriched vector per word — grows **linearly**).

**② FFN — "the hop":**
```
[3×4] × W1[4×8] → [3×8]   EXPAND ("room to think")
   activation (e.g. ReLU: negatives→0)  ← the NONLINEARITY; without it, stacked
                                           layers collapse into one. This is what makes depth work.
[3×8] × W2[8×4] → [3×4]   CONTRACT (back to width d)
+ residual
```

- FFN processes **each word independently** (no cross-word mixing). It's also **where most of the model's knowledge lives** — which is exactly the part MoE splits into "experts."

### The unbreakable law: shape preserved, meaning enriched
> **`[seq × d]` in → `[seq × d]` out, every layer.** Width `d` never changes. Each layer just makes the numbers *richer*: early layers = grammar/surface, late layers = abstract meaning/reasoning.

### The residual stream (the "one matrix" insight)
> A **single `[seq × d]` matrix flows through the whole model.** It starts as the embedding X; each layer **reads it** (to make Q/K/V), computes an update, and **adds it back.** Q/K/V are temporary projections pulled out; the result is added back in.

### Step F: Final layer → next word
```
take the LAST word's row → [1×4]            ← a ROW-PICK (the last word predicts what comes next)
× LM head [4×10] → logits [1×10]            ← raw scores for each vocab word
softmax → probabilities → pick (or sample) the highest → NEXT TOKEN
```

---

## PART 4 — THE TWO MODES (same weights, different input shape)

| | **Prefill** (digest prompt) | **Decode** (write answer) |
|---|---|---|
| Input shape | **matrix** `[many × d]` | **vector** `[1 × d]` |
| Matmul type | matrix × matrix | vector × matrix |
| Bottleneck | **compute** (cores full) | **memory** (cores starved) |
| Speed/token | fast | slow |
| Parallel? | all tokens at once | one at a time (sequential) |

- **KV cache:** during prefill the K and V of every token are **stored in VRAM**; decode **reuses** them (so it doesn't recompute the past) and appends its own. It **grows** with context and **competes with the weights** for VRAM space.
- **The single root cause of everything:** a fat input matrix reuses each weight-load across many tokens; a single-row vector streams the whole model to make **one** token. **Decode is the sequential, memory-starved bottleneck.**

---

## PART 5 — THE HARDWARE REALITY

- **VRAM holds:** weights (permanent) + KV cache + activations. The **inference engine** (vLLM/TensorRT) is the *application* that runs them.
- **Math happens in the cores (Tensor Cores), NOT in VRAM.** Weights must **stream VRAM → cores** for every op.
- **Why decode starves (precisely):** the model is gigabytes; the on-chip scratchpad is megabytes → the model is too big to sit where math happens → it's **streamed past the cores every token.** Arithmetic intensity ≈ **1 math-op per weight delivered** → cores **wait**. "Loading the model into memory" was never the problem (it's already there); the cost is **VRAM→core streaming, per token.**
- **Work division:** layers run **sequentially**; within a layer, the **output matrix is chopped into tiles**, and a **hardware scheduler** hands tiles to cores dynamically. **Cores get *tiles*, never "layers."**
- **Multi-GPU (too big for one):** split **by layers** (pipeline) or **by matrix-slices** (tensor parallelism) — *this* is deliberately configured.

**The fixes (all attack the memory bottleneck):**
- **Batching** (the biggest lever) — stack many users' tokens → fatten the vector into a matrix → one weight-stream serves all.
- **Speculative decoding** — small model drafts several tokens, big model verifies in one parallel pass.
- **Shrink the KV cache** — compress (MLA) or select (MSA) or quantize → bigger batches + less to stream.

---

## PART 6 — TRAINING (the loop that *creates* the weights)

```
1. FORWARD PASS:   feed sentence → ALL positions predict at once → [seq × vocab]
                   (every word predicts its next — not just the last)

2. LABELS ARE FREE: correct answers = the sentence shifted LEFT by one
   input:  [I]   [love] [you]
   target: [love][you]  [...]        ← self-supervised; the internet labels itself

3. LOSS FUNCTION (separate CODE, runs AFTER the layers, not a layer):
   softmax(logits) → probabilities
   cross-entropy = −log(probability given to the TRUE word)
      gave truth 0.95 → loss 0.05 (good) ;  gave truth 0.05 → loss 3.0 (bad)
   → average over all positions → ONE loss number

4. BACKPROPAGATION: from that one number, compute every weight's
                    "share of the blame" = its gradient

5. OPTIMIZER STEP (AdamW / Muon): nudge every weight to reduce loss
                    ← this is the Muon that "unlocks" Parallax

   ↺ repeat over billions of sentences
```

- **One signal trains everything:** "was the next word right?" tunes the embeddings, attention scores, FFN, the MoE router, *and* the sparse-attention block-selector.
- **Backprop computes the changes; the optimizer applies them.**
- **Memory asymmetry:** training holds weights + gradients (1×) + optimizer states (~2×) + activations ≈ **4×+ the weights.** Serving holds **~1× + KV cache.** → *why a model you can run locally still needed a data center to train, and why "frontier on local" is a serving problem, not a training one.*

---

## PART 7 — SERVING (what's actually running)

> Training machinery (loss, gradients, optimizer) is **discarded.** The shipped model = **just the frozen weights.** At serving: VRAM = **weights + KV cache + activations**, executed by a lightweight **inference engine**. No labels, no comparison, no learning — just forward passes producing tokens.

**Storage detail:** the embedding table lives in the model file with all other weights, loaded into VRAM, accessed by a **row-read** (not a matmul). It's often **tied to the LM head** — the same `[vocab × d]` matrix, transposed, serves both the input lookup *and* the output projection. → *The machine's entrance and exit are the same matrix.*

---

## PART 8 — THE LAWS TO REMEMBER (your index)

1. **A model is a stack of matrices; a forward pass is a pipeline of matmuls.** Everything else is glue.
2. **Every word is a ROW** — in the data. The weights are shared and fixed-size, applied to every row. (Why any length works.)
3. **Matmul rule:** inner dimensions must match.
4. **Each layer: `[seq × d]` in → `[seq × d]` out.** Same shape, richer meaning. That's why layers stack.
5. **A layer = Attention (gather context across words) + FFN (think per word).** Communication, then computation.
6. **Attention formula:** `softmax(Q·Kᵀ / √d) · V`. Transpose turns keys into columns so the matmul makes all pairwise scores; causal mask hides the future; softmax makes weights sum to 1; ×V blends.
7. **One matrix flows through (the residual stream)** — each layer reads it, derives Q/K/V, computes its update, adds it back.
8. **Scores `[seq×seq]` are quadratic** (the cost monster → MSA selects blocks); **outputs `[seq×d]` are linear.**
9. **Prefill = matrix×matrix (compute-bound). Decode = vector×matrix (memory-bound).** One difference — input shape — explains the whole performance story.
10. **Decode's enemy is moving memory, not doing math** → batching amortizes it.
11. **Training = predict next word → cross-entropy loss → backprop → optimizer.** Labels are free (shifted text). One signal tunes everything.
12. **Weights are the product; training scaffolding is thrown away.** Serving needs a fraction of training's memory.
13. **Sparsity is the efficiency weapon:** MoE skips experts (weights), MSA skips blocks (context). Learn what to skip in training; skip it live.
14. **Decompose, don't compress** — the principle behind both MSA and the local-model dream.
15. **You can't learn without truth independent of the model.** Training on your own outputs = **model collapse.** Legitimate continual learning ingests *new external material* (self-supervised next-token), never its own hallucinations.
16. **Surprise = novelty, not truth.** Prediction error tells you *what's new*, not *what's right*. A lie is surprising too. (Forgetting has the most tools; **truth-grounding has a fundamental limit** — that's the real blocker.)
17. **Continual learning is ONE mechanism: delay, corroborate, consolidate.** Two stores at two speeds (fast/plastic + slow/stable), a quarantine before commit, and replay to protect the old. The same spine solves deception, memoization, *and* forgetting.

---

## THE ENTIRE MACHINE IN ONE BREATH

> Words become rows of numbers by a lookup; those rows flow through a stack of identical layers that each *gather context across words (attention: `softmax(QKᵀ/√d)·V`)* and *process each word (FFN: expand→nonlinearity→contract)*, keeping the same width but enriching meaning — all carried in one residual stream. The last word's final row is turned into vocabulary scores and the next word is chosen. This runs as a fast parallel **matrix** sweep over the prompt (prefill), then a slow **one-vector-at-a-time** memory-bound trickle (decode). And the weights that make it all work were forged by a separate training loop that simply predicted the next word billions of times, measured its error with cross-entropy, and nudged every number toward being right — after which the scaffolding was thrown away, leaving just the frozen matrices that serve you.

---

## PART 9 — THE FRONTIER: SELF-SUSTAINING CONTINUAL LEARNING

> *The dream:* don't throw the training loop away at deploy — let the model keep learning in production. Not by reproducing/labeling, and not as "model + a text database (RAG)." As **one weight-organism**: hot weights in VRAM, cold weights on SSD, that *observes new material, notices the edges of its own knowledge, and absorbs the **structure** of new topics into itself* — ingesting the **pattern**, not memorizing the **instances**.

### Why the training code is dropped at deploy
Loss + backprop + optimizer states are dropped — the optimizer alone ~**triples** memory. But that's the cheap reason. The real one: **where does the *label* come from?**

### The label trap (and the door around it)
- **Trap:** in production the model *generates* the next token — there's no external truth. Train on **its own outputs** → a feedback loop that amplifies its own errors → **model collapse.**
- **Door:** don't train on its outputs. Train on **new external material it observes** — same self-supervised objective as pretraining (*predict the next real token*). The truth comes from the new data, not the model grading itself. This is just **pretraining done continuously and selectively.**
- **The "what I don't know" detector is free:** `surprise = prediction error = loss = perplexity`. Predicts it well → already known, skip. Predicts it badly → *that spike is the unknown.* (= the predictive-coding / free-energy view of how brains learn.)

### The three real problems — and the solutions

**Problem 1 — Surprise ≠ truth (the deceiver).**
- **Two kinds of surprise, opposite meanings:** *high-entropy* surprise = "no prior" = genuinely novel → **candidate to learn.** *Confident-contradiction* surprise = model was sure of X, input says not-X → **red flag, resist.** Both readable from the logits.
- Stack filters a single lie can't beat: **corroboration** (learn only what *recurs* across independent sources — a lie told once dies here), **provenance weighting** (trust scales the learning rate; textbook ≠ random chat), **grounded verification** (run the code / check a tool — reality can't be social-engineered).
- **Honest residual:** for ungrounded claims, from untrusted sources, with no prior, that never recur — *you cannot tell truth from lie from text alone.* The only deep anchors are **grounding in reality** or **a trusted authority.** Policy: default skepticism; never let a single ungrounded assertion touch the stable core.

**Problem 2 — Memoization: instant vs. delayed (a write-triage policy = episodic vs. semantic memory).**

| | Write **instantly** (fast store) | Promote **slowly** (stable core) | **Never** |
|---|---|---|---|
| What | the *instance* ("user is Yash") | the *pattern* (recurred + trusted invariant) | noise, one-off trivia, quarantined lies |
| Form | verbatim, contextual, **reversible, decays** | abstracted, durable, expensive to change | forgotten |
| Brain | hippocampus (episodic) | neocortex (semantic) | — |

- Always write to the fast store immediately (cheap, reversible, usable now). Promotion to the permanent core is earned: `promotion_score = recurrence × trust × consequence` (a costly mistake burns in one-shot — the brain's emotional tagging). **Consolidation = converting episodic instances into semantic structure.**

**Problem 3 — Forgetting.**
- **Reframe the goal:** not "never forget" (impossible for a fixed-size net — information theory, and brains forget on purpose). Target: **forget gracefully — shed instances & the unused, protect structure & the load-bearing.**
- Techniques, cleanest first: **(1) Add, don't overwrite** — grow a new MoE expert for a new domain, freeze the rest (can't forget what you don't touch; maps to a new cold SSD expert). **(2) Protect load-bearing weights (EWC)** — penalize changing the weights that matter most for old knowledge. **(3) Replay during consolidation** — interleave old data so the gradient never drifts off the old manifold (= hippocampal replay in sleep; the most effective in practice). **(4) Low-rank/sparse updates (LoRA)** — small change footprint = small damage. **(5) Orthogonal-gradient projection** — learn the new only in directions that don't disturb the old.

### The unlock: it's ONE mechanism, seen three ways
> **Don't commit immediately. Hold new things in a cheap, reversible quarantine. Promote only what *recurs*, from *trusted* sources, while *replaying* the old to protect it. Append capacity for the genuinely new; touch the core rarely and gently.**

- Recurrence-gating kills the one-off lie → **Problem 1.**
- Recurrence-gating is also how an instance becomes a pattern → **Problem 2.**
- Quarantine + replay + append-don't-overwrite stops the new erasing the old → **Problem 3.**
- Shared spine: **delay, corroborate, consolidate** — exactly what a careful mind does (don't rewire on the first surprising claim from a stranger).

### The architecture (Complementary Learning Systems — two speeds, one organism)

| | Brain | Machine | Job |
|---|---|---|---|
| **Fast store** | hippocampus | small plastic weights, VRAM-hot (a LoRA / a few MoE experts) | grab the **instance** now, cheaply, reversibly |
| **Slow store** | neocortex | the big stable core, mostly SSD-cold | hold **structure**; change rarely & gently |
| **Consolidation** | sleep / replay | offline replay cycle, behind a gate | distill instances → pattern; promote only what passes a **retention check** |

**The loop:** observe → **surprise gate** (loss: is it new?) → **truth gate** (consistency + provenance: resist confident garbage) → **fast write** (instance, reversible, usable now) → **consolidate** (replay → slow core extracts only the recurring pattern; instances fade) → **promotion gate** (must not have *forgotten* vs. a fixed retention set) → repeat.

### Honest verdict
- Robust continual learning **is** a genuine holy grail — models that improve in deployment, personalize, and stay current without a from-scratch retrain.
- But: (a) it's chipped at **in pieces today** (LoRA, EWC, replay, MoE-grow all work *partially*); nobody has the unified self-running version — that's the prize. (b) The deepest blocker is **not forgetting** (most tools) — it's **Problem 1, truth-grounding** (a *fundamental*, not just engineering, limit). A model that can be *taught lies continuously* is worse than a frozen one.
- **Solve graceful forgetting → something powerful. Solve *trusted* continuous learning → something safe to run unsupervised.** The second is the realm.
- **Where to start:** build the **quarantine fast-store + recurrence-gated promotion** loop first — it's the shared backbone of all three problems.

---

## PART 10 — THE BUILD BLUEPRINT (creating an advanced model from scratch)

> Not notes to memorize — a **build order**. Each stage = decisions + components + the *why*. Parts 1–8 are the mechanics; this is the recipe that assembles them.

### STAGE 0 — The design dials (choose before anything)

| Dial | What it controls | Notes / coupling |
|---|---|---|
| `d` (model dim) | capacity per token vector | bigger `d` = more nuance per word (diminishing returns); sets *every* matrix's width |
| `L` (layers) | depth / abstraction | grammar→meaning as you go up |
| `h` (query heads) | parallel relationship "lenses" | `d` must divide by `h`; per-head dim = `d/h` |
| `g` (K/V heads) | KV-cache size (GQA) | `g < h` shrinks cache; `g=h` = full MHA, `g=1` = MQA |
| `vocab` | token count | bigger = shorter sequences but larger embed/LM-head + weaker per-token signal |
| `context` | max sequence length | pretrain short, extend later; sets KV-cache + compute (quadratic) ceiling |
| MoE? | total vs active params | scale total params while keeping per-token compute cheap |

**The two invariants everything obeys:** `d` is the highway width (every `[seq×d]` in → `[seq×d]` out); weights carry a **layer index but no word index** (different per layer, shared across all words → any length works).

### STAGE 1 — Tokenizer & vocabulary (build once, first)

- Run **BPE** on a corpus that matches your target domain/languages → a fixed **text↔ID dictionary** of subword pieces (common words whole, rare split, byte-fallback covers everything). **Not learned**; frozen for the model's life.
- Multilingual = **one fused vocab** (BPE on all languages mixed) → shared embedding space → cross-lingual transfer; rare languages pay a "token tax."
- You may **borrow** a tokenizer to train a new model, but the trained model is **locked** to it (IDs index learned vectors — swapping scrambles everything).

### STAGE 2 — The architecture (assemble the components)

**Embedding + LM head (weight-tied).** One `[vocab × d]` matrix: row-lookup at the entrance (ID→vector), transposed `[d × vocab]` at the exit (vector→word scores). Tying saves ~billions of params and aligns in/out meaning. Embeddings init **random**, learned in training.

**The attention block — exact shapes (per layer):**
```
X [seq × d]
  Wq [d × d]            → Q [seq × d]              (h query heads)
  Wk [d × g·(d/h)]      → K [seq × g·(d/h)]        (g K/V heads — GQA shrinks Wk/Wv)
  Wv [d × g·(d/h)]      → V [seq × g·(d/h)]
  RoPE rotates Q,K (per-head, fixed math, not learned, every layer, NOT V)
  per head:  softmax( Qᵢ·Kⱼᵀ / √(d/h)  + causal mask ) · Vⱼ   → [seq × d/h]
  concat h head outputs [seq × d]
  Wo [d × d]            → mix heads → output [seq × d]   (Wo is the 4th learned matrix)
+ residual
```
- **Multi-head:** split `d` into `h` slices (split the *dimension*, not the words); each head a different lens; same total compute as one head; `Wo` fuses them.
- **GQA:** `g` query-head groups **share** one K/V each → cache stores `g` K/V sets, not `h`. Q stays diverse; the "library" (K/V) is shared. Savings come from the smaller `Wk`/`Wv`.

**RoPE (position) — the precise model:**
- Rotates Q and K by an angle = position. **Length preserved, angle changes** → content rides in magnitude, position in angle; the dot product reads both, and only the **relative** gap survives.
- **Many dials (`d/2`):** fast dials cycle/repeat (fine local detail, harmless); **slow dials** sweep a unique non-repeating arc (long-range position) — *these are the constraint on context extension.*
- Fixed/parameter-free, applied **inside every layer to Q,K only**. Cached K is stored **already-rotated** (a token's position is fixed → rotation is permanent).

**FFN + norm + residual.** Full layer = `norm → attention → +residual → norm → FFN(expand→nonlinearity→contract) → +residual`. FFN is per-word and **where most knowledge lives** (what MoE splits into experts). Same `[seq×d]` shape preserved.

**Output path:** last word's final vector `[1×d]` → `· LM head [d×vocab]` → logits `[1×vocab]` → softmax → **argmax = index = token ID** → vocabulary → word. (Softmax optional for greedy; required for sampling.)

### STAGE 3 — Pretraining (forge the intelligence; raises the ceiling)

- **Objective:** predict next token. **Labels are free** (text shifted by one). All positions predict in **one parallel pass**; **causal mask** blinds each to its future; **teacher forcing** feeds the *true* past (so errors don't compound). Cross-entropy → backprop → optimizer (AdamW/Muon). Training memory ≈ 4× weights.
- **Chunk length = the initial context window.** Pretrain on **short chunks** — attention is **quadratic** (`N²`: every word vs every word, per layer per head), so long chunks are ~`(L_long/L_short)²` costlier per chunk; most knowledge needs only short context; genuine long data is scarce. Learn the cheap 99% first.
- **Sizing — scaling laws:** loss falls as a **smooth power law** in params/data/compute → predictable. **Chinchilla:** for compute-optimal training, balance params & data ≈ **20 tokens/param** (empirical, ~20). But you *serve* forever → **inference-optimal = over-train a smaller model** (100s–1000s tokens/param) for cheap serving. This is the path to a capable *local* model.
- **The cheap de-risking experiment:** train a **ladder of smaller models on proportionally less data** (scale params *and* data down together, matched ~20:1, same architecture/data), measure loss, **fit the power-law line on log-log, extrapolate** to target scale — forecast the $100M run before building it. Plus **small-scale ablations** (test architecture/data tweaks; the winner at small scale usually wins at large) and **muP** (tune hyperparameters small, transfer to big).

### STAGE 4 — Context extension (grow the window, cheaply)

- Keep the pretrained weights. **Rescale RoPE (position interpolation):** squeeze longer positions back into the **slow dials' known angle arc** (else they hit unseen angles and break). Positions become **fractional** (e.g. 0.28 apart) — valid, since the rotation is continuous.
- Then **continue training on genuinely long documents** with that scaling active (a brief phase, ~billions of tokens — *not* a from-scratch retrain). Do it in **stages** (8K→32K→128K).
- **Resolution floor:** squeeze too hard → adjacent fractional positions blur below what the model/float precision can distinguish → can't locate distant tokens → **hallucination + in-context "forgetting."** Compounded by **attention dilution** and "lost in the middle." So **advertised window > effective window**, and extension is finite. (YaRN/NTK: squeeze the *slow* dials hard, spare the *fast* ones → preserves local precision.)

### STAGE 5 — Post-training (shape behavior; moves *under* the ceiling)

- **SFT** (instruction-follow / chat format) → **RLHF / preference (DPO)** (helpful, honest, safe) → **RL on verifiable rewards** (reasoning, chain-of-thought, tools, agency — the test-time-compute axis, the biggest recent capability lever).
- These mostly **elicit and align** latent capability + add skills; they rarely raise the raw-intelligence ceiling (that needs a fresh, larger pretrain). Risk: **catastrophic forgetting** — mitigate with replay / LoRA / careful gating.

### STAGE 6 — Inference engineering (serve it efficiently)

- **Prefill** (prompt) = one parallel matrix pass; only the **last row** predicts the first new token; all K/V cached. **Decode** = sequential, one token per pass — **autoregressive dependency** (each token must exist before the next; the future doesn't exist until written). Decode is **memory-bound** (stream all weights per token).
- **KV cache:** stores **rotated K + V** per token, per layer, per **K/V head** → grows with context, competes with weights for VRAM. Shrink it via **GQA** (fewer K/V heads), **MLA** (compress), **MSA** (select blocks), or **quantization**.
- **Flash Attention:** never materialize the `[seq×seq]` grid. Compute it in **`~128×128` SRAM tiles** with **online (streaming) softmax** — keep running `max`, running `sum`, output accumulator; rescale as blocks arrive → **exact** result, **O(seq)** memory (no `N²` anywhere), 2–4× faster. (Exact reorder ≠ sparse approximation — orthogonal; use both.)
- **Sampling:** greedy (argmax) vs temperature/top-k/top-p (needs softmax). **Speculative decoding & batching** amortize the memory-bound decode.

### The generational-leap reality (set expectations)

Each major generation = a **fresh pretrain** (raises the ceiling); the GPT-3.5→5.5 jump = **scale + far better data + refined transformer (MoE, RoPE, efficient attention) + the RL/reasoning paradigm** — *not* one thing, and **no architectural revolution since the 2017 transformer.** Point releases & reasoning models (o1-style) *are* largely continued/RL training on a base. Pattern is **layered**: fresh pretrains + continued training, both ongoing.

### The build checklist (one screen)

```
0. DIALS        d, L, h, g, vocab, context, MoE?   (d divisible by h)
1. TOKENIZER    BPE on domain corpus → frozen text↔ID vocab
2. ARCHITECTURE embed⟷LM-head (tied) · per-layer {Wq[d×d], Wk/Wv[d×g·(d/h)], Wo[d×d], FFN}
                · RoPE on Q,K · norms · residual stream
3. PRETRAIN     next-token, short chunks, teacher forcing, cross-entropy
                · size via Chinchilla (~20:1) → over-train smaller for serving
                · DE-RISK with a small scaling ladder + ablations + muP
4. EXTEND       RoPE rescale (position interpolation) + long-doc continued training, staged
5. POST-TRAIN   SFT → RLHF/DPO → RL/reasoning   (elicit + align, watch forgetting)
6. SERVE        prefill∥ / decode-sequential · KV cache (GQA/MLA/quant) · Flash Attention · sampling
```

> **The blueprint in one breath:** *Choose the dials → freeze a BPE vocab → assemble a tied-embedding transformer of `L` layers, each doing GQA multi-head attention with RoPE-rotated Q/K and an FFN on a residual stream → pretrain next-token on short chunks (sized by Chinchilla, de-risked by a small scaling ladder) → extend context by rescaling RoPE on long data → post-train (SFT/RLHF/RL) to shape behavior → serve with a KV cache and Flash Attention.* Fresh pretrain forges the ceiling; everything after moves under it.

---

## PART 11 — FROM UNDERSTANDING TO BUILDING (roadmap & next steps)

### What we accomplished
A complete **conceptual** model of how an LLM works and how to build one — top to bottom:
tokenizer/vocab → embeddings (weight-tied to LM head) → attention (Q/Kᵀ, √d, mask, softmax, ·V) → **multi-head** (split `d`, concat, Wo) → **GQA** (shared K/V, smaller cache) → **RoPE** (rotate Q/K, relative position, slow/fast dials) → FFN/residual stack → output (argmax → ID → word) → **quadratic** cost → **prefill∥/decode-sequential** → **KV cache** → **Flash Attention** (SRAM tiles + online softmax, never store the grid) → **pretraining** (next-token, teacher forcing) → **scaling laws/Chinchilla** (~20:1, the cheap ladder) → **context extension** (RoPE rescale, resolution floor) → **post-training** (SFT/RLHF/RL) → continual-learning frontier.

**Honest level:** strong *conceptual* competence (above-average — most practitioners don't know RoPE/Flash at this depth). Untested *practically* — no code written yet. **The next 10% comes from the keyboard, not more reading.**

### The frontier-lab path (Vlad Feinberg article) — and how our learning maps to it
> Thesis: don't compete head-on for pretraining roles. **Work at the two edges.** Valued traits: **intent · mathematical maturity · grit.**

| The two edges | What it is | Our coverage |
|---|---|---|
| **Below the stack — kernels** | hardware/performance: roofline, CUDA/Triton, Flash Attention, quantization | ✅ conceptual: memory-bound vs compute-bound, VRAM↔SRAM, Flash Attention, KV cache |
| **Above the stack — agents** | rigorous experiments on LLM agent behavior | ✅ already doing it (agent-builder work) |

**Article's recommended study (the gaps to close, hands-on + rigorous):**
- **The Scaling Book** — every exercise, *paper-and-pencil* derivations (the "mathematical maturity" it insists on).
- **JAX** (official tutorials) — the framework labs use.
- **Roofline analysis** / hardware-accelerator modeling (FLOPs vs memory bandwidth — beyond theoretical FLOPS).
- **Papers:** Flash Attention series, quantization (LLM.int8, QuiP, AQLM), PL/DSL like ThunderKittens.
- **Mindset:** don't dismiss "solved" problems — examine the *unmodeled constraints*; keep zooming out.

### The concrete first project (article's pick = our build step)
> **Build a ~10M-param transformer that learns addition, and derive Chinchilla scaling laws for dense vs MoE variants.** (Article phrased it as JAX/Colab; we do **build #1 in PyTorch+MPS** on the M1 first — see below — then redo in JAX as build #2.)

This *is* the "build a small model" step — now with a target: apply the **scaling ladder** (train small models, plot loss, fit the line) and compare **dense vs MoE** curves. Everything in this guide becomes running code.

### Build #2 — the JAX project (the lab-flavor port) 🔭
> **Re-implement the *same* "learn addition" transformer in JAX (CPU locally, or free Colab TPU) — then do the dense-vs-MoE scaling ladder in JAX.**

- **Why second, not first:** by then the model is muscle memory, so JAX becomes purely about the *framework*, not the transformer — exactly the right way to learn it.
- **What JAX teaches that PyTorch hid:** `jit` (trace + compile the whole step), `vmap` (auto-batching), `grad` (functional autodiff), `pmap`/`shard_map` (multi-device parallelism), and pure-functional state (params passed in/out, no hidden mutation). This is the **TPU / frontier-lab** mental model.
- **Pair it with The Scaling Book** (which is JAX-based) — read a chapter, implement the matching piece. This is where "frontier-lab flavor" actually gets earned.
- **Stretch:** run it on a Colab **TPU** to feel `pmap` sharding for real; that's the closest thing to the lab environment without lab hardware.

### Building on the M1 Air (this machine: M1 · 8GB RAM · ~29GB free disk · Python 3.11)
- **Use PyTorch + `mps`** (the M1 GPU) — smooth on Mac. Skip JAX-on-Metal for now (flaky); do JAX later CPU-only or on Colab for the lab-flavor.
- **Data isn't the bottleneck:** 200M tokens ≈ ~400MB tokenized (uint16) → fits free disk easily; data is **memory-mapped/streamed**, not loaded whole → 8GB RAM is fine.
- **Real limits:** disk (~29GB free after cleanup; torch ≈ 2.5GB → plenty of room now) and **compute** (a full 200M-token Chinchilla run = *hours*, and the Air is **fanless → throttles**). Compute, not disk, is the binding constraint.
- **The fix:** don't do the full Chinchilla run to *learn*. **Generate synthetic "addition" data** (zero download, tiny) and train a **ladder of tiny models** (100K→5M params) — each run is minutes, derives Chinchilla cheaply, sidesteps every constraint.

### What to look forward to (the path)
```
NOW   → pip install torch · build "learn addition" GPT (PyTorch+MPS, ~1M params)
NEXT  → swap in the modern pieces we learned: RoPE, RMSNorm, GQA, KV cache
THEN  → scaling ladder: tiny models, plot loss, derive Chinchilla (dense vs MoE) yourself
EDGES → kernels (roofline, a Triton/CUDA kernel, the Scaling Book) · agents (already in flight)
LATER → BUILD #2: redo in JAX (lab-flavor, jit/vmap/pmap, Scaling Book, Colab TPU) · quantization · LoRA fine-tune
```

> **The verdict:** the conceptual half is done. The next step — for joy *or* a lab — is the **same small-model build** (10M-param "learn addition," dense-vs-MoE scaling ladder), on the M1 in PyTorch+MPS. Turn the blueprint into muscle memory; the edges (kernels, agents) follow.

---

## REFERENCE LINKS

- **Parallax** (local-linear attention correction): https://blog.tilderesearch.com/blog/parallax · paper https://arxiv.org/abs/2605.29157
- **MiniMax M3 / MSA** (sparse attention): https://www.minimax.io/blog/minimax-m3 · MSA paper https://arxiv.org/html/2606.13392
- **How to land a job at a frontier lab** (Vlad Feinberg): https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html
- **Build-along references:** Karpathy "Let's build GPT" / nanoGPT · The Scaling Book · JAX official tutorials
