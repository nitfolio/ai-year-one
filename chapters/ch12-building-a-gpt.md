# Chapter 12 — Building a GPT

**Time: 16–20 days** (Weeks 27–30 of the plan)

**Prerequisite:** Chapter 11 checkpoint — you can write a pre-norm transformer block from memory. Chapter 9's optimizers and schedules. Chapter 8's trainer.

**What you'll build:** a complete language model, from a byte-pair tokenizer you write yourself to a trained transformer that generates text. This is the capstone of Part III and the first artifact you'd show someone as evidence you can do this work.

**Companion resource:** Karpathy's nanoGPT and his GPT-2 reproduction video, and Stanford CS336 (*Language Modeling from Scratch*). Build your own first, then compare.

---

## 12.0 The objective

A language model does one thing: **predict the next token.**

From §6.4, the chain rule of probability factors any sequence exactly:

```
P(x₁...x_n) = ∏_t P(x_t | x_1...x_(t−1))
```

Nothing is approximated in the factorization — it's an identity. The model learns each conditional factor, and the loss is cross-entropy over the vocabulary at every position (§4.11).

Causal masking (§11.7) lets you train on all `T` positions **in parallel** in one forward pass, each seeing only its own past. One sequence of length 1024 gives you 1024 training signals.

---

## 12.1 Tokenization

The model needs integers. Turning text into integers is a bigger deal than it looks.

**Why not characters?** Sequences become very long, and attention is `O(n²)`. A 2,000-word document is ~10,000 characters — expensive — and the model spends capacity learning spelling.

**Why not words?** The vocabulary explodes, and any word not in it is out-of-vocabulary. Morphology is lost: "run", "running", "runner" become unrelated symbols.

**Byte-pair encoding** sits between. Start with individual bytes and repeatedly merge the most frequent adjacent pair into a new token. Common words end up as single tokens; rare words decompose into pieces; **nothing is ever out-of-vocabulary**, because you can always fall back to bytes.

### BPE from scratch

```python
def get_stats(ids):
    """Count adjacent pairs."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out


class BPETokenizer:
    def __init__(self):
        self.merges = {}                              # (a,b) -> new_id
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size, verbose=False):
        ids = list(text.encode("utf-8"))
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and i % 100 == 0:
                print(f"merge {i}: {pair} -> {new_id} "
                      f"({self.vocab[new_id]}) count {stats[pair]}")

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # apply the earliest-learned merge that's still present
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
```

**Merge order matters in `encode`.** You must apply merges in the order they were learned, because later merges were built on earlier ones. Applying them out of order produces a different — wrong — tokenization. This is the subtle bug in most from-scratch BPE implementations.

### Why tokenization causes so many visible model failures

Almost every strange LLM behaviour traces back here:

- **Arithmetic is hard** because numbers tokenize inconsistently — "1234" might be one token, "1235" three. The model never sees digits as digits.
- **Character-level tasks fail** ("how many r's in strawberry?") because the model never sees characters.
- **Non-English is more expensive**, sometimes 2–3× more tokens per word, since merges were learned on mostly-English data. Worse performance *and* higher cost.
- **Trailing whitespace breaks things** — " hello" and "hello" are different tokens, so a trailing space in a prompt shifts everything.
- **Glitch tokens.** Some tokens appear in the vocabulary but essentially never in training data, so their embeddings stay near initialization and produce bizarre behaviour when invoked.

**Tokenization is the least glamorous part of an LLM and the source of a surprising share of its failures.** Worth knowing well.

---

## 12.2 The data pipeline

```python
import numpy as np, torch

def prepare(text, tokenizer, path):
    ids = np.array(tokenizer.encode(text), dtype=np.uint16)   # <65536 vocab
    n = int(0.9 * len(ids))
    ids[:n].tofile(f"{path}/train.bin")
    ids[n:].tofile(f"{path}/val.bin")


def get_batch(split, block_size, batch_size, device, path):
    data = np.memmap(f"{path}/{split}.bin", dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
```

Two things to notice:

**`y` is `x` shifted by one.** That's the entire supervision signal — no labels needed, which is why language modelling scales to the whole internet.

**`np.memmap` reads from disk lazily**, so your dataset can be far larger than RAM. Standard practice for real corpora.

---

## 12.3 The model

```python
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False          # modern models drop biases in Linear/LayerNorm


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for t in (q, k, v))
        y = F.scaled_dot_product_attention(              # uses FlashAttention
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))       # pre-norm — §7.6
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.wte.weight            # weight tying — §12.4

        self.apply(self._init_weights)
        for name, p in self.named_parameters():          # residual scaling
            if name.endswith("c_proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / (2 * cfg.n_layer) ** 0.5)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)

        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                         # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))     # logits, not softmax — §8.5
        return logits, loss
```

That's a complete GPT. Roughly 100 lines, and every one of them is something you derived in an earlier chapter.

---

## 12.4 Two initialization details that matter

### Weight tying

```python
self.lm_head.weight = self.wte.weight
```

The input embedding and the output projection **share the same matrix**. Both map between token space and embedding space — one in each direction — so tying them is a reasonable inductive bias.

The practical argument is stronger: for `d = 768` and `V = 50,257`, that's **38.6 million parameters saved** — around 30% of a GPT-2-small. And it usually improves results.

### Residual scaling

```python
std = 0.02 / sqrt(2 * n_layer)     # applied to each residual branch's output projection
```

**Why:** the residual stream accumulates `2N` contributions (one per sublayer). If each has variance `σ²` and they're roughly independent, the stream's variance grows to `2Nσ²` — it drifts upward with depth, exactly the §7.2 problem in a new place.

Scaling each contribution by `1/√(2N)` keeps the stream's variance `O(1)` regardless of depth.

GPT-2 does this. It's four lines, it's rarely explained, and it matters more as models get deeper.

---

## 12.5 The training recipe

```python
def configure_optimizer(model, weight_decay, lr, betas):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas, fused=True)
```

The `p.dim() >= 2` test is a clean way to express §8.6's rule: matrices get decayed; biases and LayerNorm gains (both 1-D) don't.

### Standard settings

| Setting | Value | Why |
|---|---|---|
| Optimizer | AdamW | §9.6 |
| `β₁, β₂` | 0.9, **0.95** | Shorter second-moment memory is more stable at scale |
| Weight decay | 0.1 | Higher than typical vision values |
| Gradient clip | 1.0 | Global norm — §7.4 |
| Warmup | 1–2% of total steps | §9.7 — three reasons |
| Schedule | Cosine to 10% of peak | §9.7 |
| LR | `6e-4` (small) → `1.5e-4` (large) | Decreases with model size |
| Batch (tokens) | 0.5M for small models | Large batches; use gradient accumulation |
| Precision | bf16 | Same exponent range as fp32 — no loss scaling needed |

**bf16 over fp16.** `bfloat16` has fp32's exponent range with fewer mantissa bits, so it doesn't overflow and needs no `GradScaler`. `float16` has a narrow range and requires loss scaling to avoid underflow. If your hardware supports bf16, use it.

### The loop

```python
model = GPT(cfg).to(device)
model = torch.compile(model)
opt = configure_optimizer(model, 0.1, 6e-4, (0.9, 0.95))

for step in range(max_steps):
    lr = warmup_cosine(step, warmup_steps, max_steps, 6e-4, 6e-5)   # §9.7
    for g in opt.param_groups:
        g["lr"] = lr

    opt.zero_grad(set_to_none=True)
    for micro in range(grad_accum_steps):
        x, y = get_batch("train", cfg.block_size, batch_size, device, path)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        (loss / grad_accum_steps).backward()          # ← the /accum from §8.12

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {step}: train {losses['train']:.4f} "
              f"val {losses['val']:.4f} ppl {np.exp(losses['val']):.2f}")
```

---

## 12.6 Scaling laws

The most consequential empirical finding in modern AI, and it's worth understanding precisely.

### Kaplan et al. (2020)

Test loss follows smooth **power laws** in parameters `N`, data `D`, and compute `C`:

```
L(N) ≈ (N_c / N)^α
```

Straight lines on log-log axes, over many orders of magnitude. That predictability is why labs are willing to commit enormous budgets — you can forecast the result of a run you haven't done.

### Chinchilla (Hoffmann et al., 2022)

Kaplan's recommendations were wrong in an important way: **models were being badly undertrained.**

The compute cost of training is approximately

```
C ≈ 6 · N · D          FLOPs
```

(2 FLOPs per parameter for the forward multiply-add, ~4 for the backward pass.)

For a fixed budget `C` you choose how to split it between model size and data. Chinchilla's finding: **`N` and `D` should scale equally**, roughly

```
D ≈ 20 · N          tokens per parameter
```

**The demonstration:** Chinchilla (70B parameters, 1.4T tokens) outperformed Gopher (280B parameters, 300B tokens) using the *same* compute. A model four times smaller won, because it saw nearly five times the data.

**Why it matters practically:** a smaller model trained longer is also cheaper to serve, forever. That changed how the whole industry allocates compute.

**A caveat worth knowing:** Chinchilla optimizes *training* compute. If a model will serve billions of requests, inference cost dominates, and it's rational to train a smaller model far past the Chinchilla point. Most current open models are trained well beyond `20×`.

### Sizing your own run

```python
def token_budget(n_params, ratio=20):
    return n_params * ratio

def training_flops(n_params, n_tokens):
    return 6 * n_params * n_tokens

# a 10M-parameter model
print(f"{token_budget(10e6)/1e6:.0f}M tokens")            # 200M tokens
print(f"{training_flops(10e6, 200e6):.2e} FLOPs")         # 1.2e16
```

At roughly `10¹⁴` usable FLOPs/second on a consumer GPU, that's a few minutes of compute. **A Chinchilla-optimal 10M-parameter model is entirely feasible on free hardware.** That's your target for this chapter.

---

## 12.7 Efficiency

| Technique | Effect | Cost |
|---|---|---|
| bf16 autocast | ~2× faster, ~half activation memory | none on modern GPUs |
| `torch.compile` | 1.3–2× faster | slow first call |
| FlashAttention | Large memory saving at long context | none (use `scaled_dot_product_attention`) |
| Gradient checkpointing | Big activation-memory reduction | ~30% more compute |
| Gradient accumulation | Large effective batch on small memory | proportionally slower |
| Fused AdamW | Modest speedup | none |

### The memory budget

For AdamW in mixed precision, per parameter:

```
2 bytes   bf16 parameter
2 bytes   bf16 gradient
4 bytes   fp32 master copy
4 bytes   Adam m
4 bytes   Adam v
─────────
16 bytes per parameter
```

**A 1B-parameter model needs ~16GB before a single activation is stored.** This is why optimizer state, not the model, usually dictates what hardware you need — and why techniques like ZeRO sharding exist.

Activations add `≈ batch × seq_len × n_layer × n_embd × (some constant)`. Gradient checkpointing attacks this term.

---

## 12.8 Evaluating, and the loss milestones

Track **validation loss and perplexity** (§6.8). But know what the numbers mean:

| Loss (nats) | Perplexity | What it indicates |
|---|---|---|
| `ln(V) ≈ 10.8` | 50,257 | Untrained — uniform over the vocabulary |
| ~7 | 1,100 | Learned unigram frequencies |
| ~5 | 150 | Learned some local structure |
| ~3.5 | 33 | A small model doing real work |
| ~3.0 | 20 | Decent small model |

**Check your initial loss against `ln(V)`.** This is §7.9 step 1 applied to language models, and it catches broken initialization, wrong vocabulary size, and misaligned targets in ten seconds.

**A plateau near 7 means the model learned token frequencies and nothing else.** That's a specific, diagnosable state — usually too small, too low a learning rate, or a data bug.

**Loss isn't everything.** A model can have good perplexity and be useless downstream. Real evaluation also needs task benchmarks, and you must watch for **data contamination** — benchmark text leaking into training data, which makes results meaningless. This is a live problem in published results, and worth being skeptical about.

---

## 12.9 Generation and the KV cache

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]          # crop to context
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature             # last position only

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

Temperature, top-k and top-p are exactly §6.9 — you implemented all three in NumPy already.

### The KV cache

The loop above recomputes attention over the whole prefix for every new token: `O(n²)` total work. Caching `K` and `V` makes each step `O(n)`.

```python
class CachedAttention(nn.Module):
    def forward(self, x, cache=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
                   for t in (q, k, v))

        if cache is not None:
            k_prev, v_prev = cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        new_cache = (k, v)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=(cache is None))
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.c_proj(y), new_cache
```

**Note `is_causal=(cache is None)`.** During cached generation you're processing a single new token that legitimately attends to *all* cached positions — applying a causal mask there would be wrong. This is a genuinely easy bug to write and a hard one to notice.

**The cache is big:** `2 × n_layer × n_head × seq_len × d_head × batch × bytes`. It often exceeds the model itself at long context, which is why GQA (§11.11) exists.

---

## 12.10 What comes after pretraining

Brief orientation — you now have the base model, and there are three further stages in a modern pipeline:

1. **Supervised fine-tuning (SFT).** Train on instruction–response pairs. Same next-token objective, curated data. This turns a text-continuation engine into something that answers questions.
2. **Preference optimization.** Learn from comparisons between responses. RLHF trains a reward model and optimizes against it; **DPO** skips the reward model and optimizes a preference objective directly, which is much simpler and now more common.
3. **Evaluation and safety work.** Extensive, ongoing, and increasingly the bulk of the effort.

If you choose the LLM track in Chapter 14, this is where you'll go next. LoRA (§2.11) is how you'll do it on affordable hardware.

---

## 12.11 Debugging a language model run

| Symptom | Likely cause | Action |
|---|---|---|
| Initial loss ≠ `ln(V)` | Wrong vocab size; broken init; misaligned targets | Check before training |
| Loss flat at `ln(V)` | LR far too low; broken data pipeline | §7.9 protocol |
| Loss plateaus ~7 | Learned unigrams only | Bigger model; higher LR; check data |
| Loss spikes then recovers | Bad batch; LR slightly high | Usually fine; watch it |
| Loss spikes and never recovers | Diverged | Restart from checkpoint with lower LR |
| `nan` early | No warmup; LR too high; fp16 overflow | Add warmup; use bf16; clip |
| Val ≫ train | Overfitting (rare for LMs) | More data; you're past Chinchilla |
| Generation repeats endlessly | Greedy decoding | Temperature/top-p |
| Generation is incoherent | Undertrained; temperature too high | Train longer; lower temperature |

**Always overfit a tiny dataset first** (§4.15). Take 100 tokens and drive the loss near zero. If a language model can't memorize 100 tokens, the bug is in your code — and this test takes thirty seconds.

---

## 12.12 Exercises

**1.** Implement BPE from scratch. Train it on ~1MB of text with a 1,000-token vocabulary. Verify `decode(encode(text)) == text` for 100 random strings including emoji and non-English.

**2.** Show the merge-order bug: apply merges in frequency order instead of learned order and demonstrate the tokenization differs.

**3.** Compare token counts for the same passage in English, Hindi, and Chinese using a GPT-2 tokenizer. Report tokens-per-character for each and discuss the cost implication.

**4.** Tokenize the numbers 1–1000. Plot tokens-per-number. Explain why arithmetic is hard for LLMs.

**5.** Build the data pipeline with `memmap`. Verify that `y` is `x` shifted by one for random batches.

**6.** Implement `GPT` from scratch (you may use `scaled_dot_product_attention`). Verify: parameter count matches a hand calculation, forward shapes are right, and initial loss ≈ `ln(V)`.

**7.** Derive the parameter count formula for a GPT given `V, d, L`. Verify against your model for three configurations.

**8.** Implement weight tying. Report parameters saved. Train identical models with and without it and compare validation loss.

**9.** **Residual scaling.** Log the standard deviation of the residual stream at every layer, with and without the `1/√(2N)` initialization, at `n_layer = 24`. Plot both.

**10.** Overfit 100 tokens to near-zero loss. Then break something deliberately (shift targets by 2) and show it can no longer overfit.

**11.** Train a small GPT (~10M parameters) on a corpus of your choice to a validation loss below 3.5. Log everything. Plot the loss curve and mark the milestones from §12.8.

**12.** Compute the Chinchilla-optimal token budget for your model and train to it. Then train a 2× larger model on half the data at matched compute and compare. Which wins?

**13.** Ablate the training recipe on your small model: no warmup, no gradient clipping, `β₂ = 0.999` instead of `0.95`, constant LR instead of cosine. One at a time. Plot all five curves.

**14.** Implement generation with temperature, top-k, and top-p. Generate 20 samples at `T ∈ {0.2, 0.7, 1.0, 1.5}` and comment on the qualitative differences.

**15.** Implement the KV cache. Measure tokens/second with and without it at sequence lengths 128, 512, 2048. Plot both curves. Then compute the cache's memory footprint for your config.

**16.** Visualize attention patterns from a trained model. Pick three heads across different layers and characterize what each appears to attend to. Include the heatmaps.

**17.** **Chapter project.** Train the best language model you can on hardware you have access to. Requirements: your own tokenizer, your own model code, Chinchilla-optimal budgeting, full training recipe, logged diagnostics, an ablation of at least three recipe choices, scaling experiments at three model sizes with a log-log loss plot, generation samples, and attention visualizations. **Write it up as a technical report with a proper methods section.**

That report is the first artifact in your portfolio that reads as research rather than coursework. Treat it that way.

---

## 12.13 Solutions

<details>
<summary>Open only after attempting</summary>

```python
import numpy as np, torch, torch.nn.functional as F


# --- 2: the merge-order bug ---
# encode() must apply merges in LEARNED order. Merge #500 may have been built
# from the token produced by merge #12; applying #500 first means its inputs
# do not exist yet, so you get a different (wrong) tokenization that the model
# never saw in training. Selecting with
#     min(stats, key=lambda p: self.merges.get(p, inf))
# picks the earliest-learned applicable merge, which is correct.


# --- 4 ---
# "1234" may be one token while "1235" is three. Digit grouping is driven by
# corpus frequency, not by place value, so the model never sees a consistent
# positional representation of number. Every arithmetic algorithm it learns
# has to be re-learned per tokenization pattern — hence the unreliability.


# --- 7: parameter count ---
def gpt_params(V, d, L, block=1024, tied=True):
    emb  = V*d + block*d
    attn = L * 4*d*d                 # q,k,v,proj
    mlp  = L * 8*d*d                 # d->4d and 4d->d
    head = 0 if tied else V*d
    return emb + attn + mlp + head

print(f"{gpt_params(50257, 768, 12)/1e6:.1f}M")      # ≈124M — GPT-2 small
print("tying saves", 50257*768/1e6, "M params")      # 38.6M


# --- 9: residual scaling ---
# Each of the 2N sublayers adds a contribution to the residual stream. If each
# has variance σ² and they are roughly independent, the stream's variance is
# 2Nσ² — it grows with depth (§7.2's problem, relocated). Scaling each branch's
# output projection by 1/√(2N) restores O(1) variance. Without it, at L=24 the
# stream std climbs steadily layer by layer; with it, the curve is flat.


# --- 10 ---
# A working GPT memorizes 100 tokens to loss < 0.01 within a few hundred steps.
# Shift targets by 2 and it cannot: you have asked it to predict a token that
# is not determined by the context under a causal mask. Loss floors well above
# zero. That contrast is exactly what the overfit-tiny test is for — it
# separates "my code is broken" from "this task is hard".


# --- 12: Chinchilla ---
def compare(C=1.2e16):
    for N in (5e6, 10e6, 20e6):
        D = C / (6*N)
        print(f"N={N/1e6:5.1f}M  D={D/1e6:7.0f}M tokens  ratio={D/N:6.1f}")
compare()
# Matched compute, three splits. The one nearest 20 tokens/param generally wins
# on validation loss. Too-large-N is undertrained; too-small-N saturates.


# --- 15: KV cache ---
# Without cache: step t recomputes attention over t tokens -> O(n²) total.
# With cache: each step attends once over the cached prefix -> O(n) per step.
# Speedup grows linearly with sequence length; at 2048 it is typically >10x.
def kv_cache_bytes(n_layer, n_head, d_head, seq, batch, dtype_bytes=2):
    return 2 * n_layer * n_head * d_head * seq * batch * dtype_bytes
print(kv_cache_bytes(12, 12, 64, 1024, 8) / 1e9, "GB")
# This is why GQA exists — sharing K/V across heads divides this by the group
# factor, and at long context the cache dominates inference memory.


# --- 16 ---
# Common findings in small trained models:
#  - an early-layer head attending almost entirely to the previous token
#  - a head attending to the first token (an "attention sink" / no-op slot)
#  - a head matching repeated tokens (the induction-head pattern), which is
#    strongly associated with in-context learning ability
# Report what you actually see; do not assume these appear in every model.
```

</details>

---

## 12.14 Chapter 12 checkpoint

Cold — blank file, no notes.

- [ ] Explain why BPE rather than characters or words, and why merge order matters in `encode`.
- [ ] Give three model failures that trace back to tokenization.
- [ ] **Write a complete GPT forward pass from memory** — embeddings, blocks, final norm, head. **Target: 30 minutes.**
- [ ] Derive the parameter count formula from `V, d, L`.
- [ ] Explain weight tying and state what it saves for GPT-2 small.
- [ ] **Explain residual scaling** — why `1/√(2N)`, with the variance argument.
- [ ] State the standard training recipe from memory: optimizer, betas, decay, clip, warmup, schedule.
- [ ] State `C ≈ 6ND` and the Chinchilla ratio, and explain what Chinchilla corrected.
- [ ] Explain why bf16 is preferred to fp16.
- [ ] State the expected initial loss for a language model and what a plateau near 7 means.
- [ ] Explain what a KV cache does and why `is_causal` must be handled carefully with it.

Items 3, 6 and 8 are the ones that matter most.

### Anki cards

- Why BPE? Why not chars or words?
- BPE training algorithm, four steps
- Why does merge order matter in encoding?
- Three tokenization-caused failures
- GPT parameter count formula
- Weight tying — what and why
- Residual scaling — the factor and the reason
- Standard LLM training recipe (make several cards)
- `C ≈ 6ND`
- Chinchilla ratio and what it corrected
- bf16 vs fp16
- Memory per parameter for AdamW mixed precision
- Expected initial LM loss; the meaning of a plateau at ~7
- KV cache — what it saves and what it costs
- Loss milestones table

### Deliverables

```
gpt/tokenizer.py       BPE from scratch, round-trip tested
gpt/model.py           the full GPT
gpt/train.py           the complete training script
gpt/generate.py        sampling + KV cache
gpt/scaling.py         Chinchilla budgeting helpers
experiments/ablations/ exercise 13
reports/gpt_report.md  exercise 17 — the technical report
```

```bash
git add .
git commit -m "Chapter 12: GPT from scratch — tokenizer, model, training, generation"
git push
```

### Write-up

This one is different. **Write a technical report, not a blog post** — abstract, methods, results, ablations, limitations. Structure it the way a workshop paper is structured, because in Chapter 15 you'll write an actual one and this is the rehearsal.

Include: your scaling plot across three model sizes on log-log axes, your recipe ablations, your loss curve annotated with the §12.8 milestones, generation samples at several temperatures, and attention visualizations with honest interpretation.

**Part III is complete.** You have built, from primitives, the architecture that every current frontier model uses, and trained one.

Part IV is the transition from learning to researching: reading papers, reproducing them, and producing original work.

---

*Next: Chapter 13 — Reading and Reproducing Papers*
