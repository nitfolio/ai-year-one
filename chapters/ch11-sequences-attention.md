# Chapter 11 — Sequences and Attention

**Time: 14–16 days** (Weeks 23–26 of the plan)

**Prerequisite:** Chapter 10's residual connections. §7.4 (vanishing gradients), §4.9 (softmax saturation), §6.3 (variance of sums). All three get used directly.

**What you'll be able to do at the end:** derive attention from the problem it solves; explain the `√d` scaling with the actual variance argument; implement multi-head causal self-attention from scratch; understand RoPE; and build a complete transformer block. Together with Chapter 12, this is the core of modern AI.

---

## 11.0 What's different about sequences

Images have fixed size and a fixed spatial layout. Sequences don't:

- **Variable length.** A sentence might be 3 tokens or 3,000.
- **Order matters.** "Dog bites man" ≠ "man bites dog."
- **Long-range dependencies.** "The **keys** that I left on the table in the room upstairs **are** missing" — the verb agrees with a noun 11 words back.

An MLP needs fixed-size input. A CNN has a limited receptive field. Neither handles this well.

The historical answer was recurrence. The modern answer is attention. You need to understand why the first failed to understand why the second won.

---

## 11.1 Recurrent neural networks

Process one element at a time, carrying a hidden state:

```
h_t = tanh(W_hh · h_(t−1) + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

`h_t` is meant to summarize everything seen so far. The **same weights** are used at every timestep — weight sharing across time, exactly analogous to a CNN's weight sharing across space.

```python
def rnn_forward(X, Wxh, Whh, b, h0):
    """X: (T, d_in) — one sequence. Returns all hidden states."""
    h = h0
    hs = []
    for t in range(X.shape[0]):
        h = np.tanh(X[t] @ Wxh + h @ Whh + b)
        hs.append(h)
    return np.stack(hs)
```

Elegant, and it handles any length. Training uses **backpropagation through time** (BPTT): unroll the loop into a `T`-layer feedforward network sharing weights, then apply ordinary backprop.

---

## 11.2 Why RNNs fail

The gradient of a loss at step `t` with respect to a hidden state far back involves a product of Jacobians:

```
∂h_t/∂h_0 = ∏_(k=1)^t ∂h_k/∂h_(k−1)
          = ∏_(k=1)^t diag(tanh'(·)) · W_hhᵀ
```

Same exponential argument as §7.4 — but **worse**, and for a specific reason.

In a deep MLP, each layer has a *different* weight matrix, so there's some averaging across the product. In an RNN, it's the **same matrix repeated `t` times**. The product is essentially `W_hh^t`, so its largest eigenvalue `λ` governs everything:

```
λ < 1  →  gradient vanishes like λ^t
λ > 1  →  gradient explodes like λ^t
```

There is no averaging to save you. And with `tanh'` ≤ 1 multiplying in as well, vanishing is the default.

**The practical consequence:** a vanilla RNN reliably learns dependencies of about 10 steps and essentially nothing beyond that. For language, that's useless.

Exploding gradients are the easy half — clip them (§7.4). Vanishing gradients cannot be clipped; the information is simply gone.

---

## 11.3 LSTM: gating, and a familiar idea

The LSTM (1997) adds a **cell state** `c_t` that information can travel along with minimal interference, controlled by learned gates.

```
f_t = σ(W_f·[h_(t−1), x_t] + b_f)        forget gate  — what to discard
i_t = σ(W_i·[h_(t−1), x_t] + b_i)        input gate   — what to write
g_t = tanh(W_g·[h_(t−1), x_t] + b_g)     candidate    — what could be written
o_t = σ(W_o·[h_(t−1), x_t] + b_o)        output gate  — what to expose

c_t = f_t ⊙ c_(t−1) + i_t ⊙ g_t          cell state update
h_t = o_t ⊙ tanh(c_t)                    hidden state
```

`⊙` is elementwise multiplication. The sigmoid gates output values in `(0,1)` — soft switches.

### Why it fixes vanishing gradients

Look at the cell state path:

```
∂c_t/∂c_(t−1) = f_t
```

**If the forget gate is near 1, the gradient passes through unchanged.** No matrix multiplication, no `tanh'` factor — just elementwise multiplication by something close to 1.

Compare §10.9: a residual block has `∂y/∂x = 1 + ∂F/∂x`, a path with derivative exactly 1.

**The LSTM cell state is a residual connection, invented eighteen years before ResNets.** Same insight — give gradient an additive highway — arrived at independently in a different subfield. Worth noticing: good ideas in this field are often the same idea wearing different clothes.

**One practical detail:** initialize the forget gate bias to 1 (or 2). This starts the gate open, so information flows by default and the network learns what to forget rather than having to learn to remember. It's a small change with a large effect on trainability.

### GRU

A simplification with two gates instead of three, merging cell and hidden state:

```
z_t = σ(W_z·[h_(t−1), x_t])              update gate
r_t = σ(W_r·[h_(t−1), x_t])              reset gate
h̃_t = tanh(W·[r_t ⊙ h_(t−1), x_t])
h_t = (1 − z_t) ⊙ h_(t−1) + z_t ⊙ h̃_t
```

Fewer parameters, comparable performance. Note the last line is an interpolation between old and new state — again an additive path.

---

## 11.4 The problems gating didn't solve

LSTMs worked well enough to power a decade of NLP. Three limitations killed them anyway:

**1. Sequential computation.** You cannot compute `h_t` before `h_(t−1)`. Training requires `T` sequential steps, and GPUs — which are massively parallel — sit mostly idle. This is the decisive one. It caps how much data you can train on, and scale turned out to matter more than architecture.

**2. The fixed-size bottleneck.** Everything about the first 500 tokens has to fit into one vector `h_500`. Information is necessarily lost, and the model must decide what to discard before knowing what will be asked of it.

**3. Long-range dependencies remain hard.** Better than vanilla RNNs, still not good. Gradient paths of length 1,000 are difficult even with gates.

The fix for all three is the same: **stop compressing history into a single state, and let every position look directly at every other position.**

---

## 11.5 Attention, derived

Start from the bottleneck problem. You want position `i`'s output to depend on *all* positions, not on a compressed summary.

The simplest thing that could work is a weighted average:

```
out_i = Σ_j α_ij · v_j        with   Σ_j α_ij = 1
```

Each output is a blend of information from every position. Now: how should the weights `α_ij` be computed?

**They should be large when position `i` is "interested in" position `j`.** That's a similarity question — and from §2.3, the dot product is exactly the operation that measures how much two vectors point the same way.

So: give each position a vector saying what it's looking for, and a vector saying what it contains, and take the dot product. Normalize with softmax so the weights sum to 1.

That's attention. Three learned projections give the three roles:

```
Q = X·W_Q        queries — "what am I looking for?"
K = X·W_K        keys    — "what do I contain?"
V = X·W_V        values  — "what do I contribute if selected?"
```

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

### The database analogy

It's a **soft dictionary lookup**. A hard lookup matches one key exactly and returns its value. Attention matches a query against *all* keys, converts the match scores to weights, and returns a weighted blend of *all* values.

Everything is differentiable, so the network learns what to look for.

### Why three separate projections?

Because the roles genuinely differ. What a token *asks about* (query), what it *advertises* (key), and what it *transfers* (value) are three different things. A pronoun's query might be "find me a nearby noun"; its key advertises "I am a pronoun"; its value carries pronoun-ness forward. Collapsing these into one vector loses expressiveness.

---

## 11.6 The `√d_k` scaling, derived

The most commonly hand-waved detail in the transformer paper, and it's a three-line derivation using §6.3.

Suppose `q` and `k` are `d_k`-dimensional with components that are independent, mean 0, variance 1. Then:

```
q·k = Σ_i q_i k_i

E[q·k] = 0
Var[q·k] = Σ_i Var[q_i k_i] = d_k · 1 · 1 = d_k
```

So the dot products have **standard deviation `√d_k`**. With `d_k = 64`, scores typically range over ±8; with `d_k = 512`, ±23.

**Why that's a problem:** feed values with a large spread into softmax and it saturates — nearly all mass goes to the single largest score, and the output approaches one-hot. From §4.9, a saturated softmax has near-zero gradient. Training stalls.

Dividing by `√d_k` rescales the variance back to 1, keeping the softmax in its responsive range.

```
Var[q·k / √d_k] = d_k / d_k = 1  ✓
```

**Verify it yourself** (exercise 6): sample random `q, k` at several `d_k`, histogram the raw dot products and the scaled ones, then plot the resulting softmax distributions and their gradients. The saturation is dramatic and seeing it makes the reason permanent.

---

## 11.7 Self-attention, cross-attention, masking

**Self-attention:** `Q`, `K`, `V` all come from the same sequence. Each position attends to its own sequence. This is what a GPT block does.

**Cross-attention:** `Q` from one sequence, `K, V` from another. Used in encoder–decoder models (translation), and in multimodal models where text queries attend to image features.

### Causal masking

A language model predicting token `t` must not see tokens after `t` — that would be cheating, and the model would learn nothing useful.

Set the disallowed scores to `−∞` **before** the softmax:

```python
mask = np.triu(np.ones((T, T)), k=1).astype(bool)     # True above the diagonal
scores = np.where(mask, -np.inf, scores)
```

**Why `−∞` before softmax rather than zeroing after?** Because `exp(−∞) = 0`, and then the softmax normalizes over only the *allowed* positions. Zeroing afterwards would leave the weights not summing to 1 — the remaining mass would be wrong.

This masking is what lets a transformer train on all positions **in parallel**. Every position predicts its next token simultaneously in one forward pass, while each only sees its own past. That parallelism is the entire reason transformers beat RNNs — same objective, `T` times fewer sequential steps.

---

## 11.8 Multi-head attention

One softmax produces one attention pattern. Usually you want several — one head tracking syntax, another coreference, another local position.

Split `d_model` into `h` heads of size `d_k = d_model/h`, attend independently, concatenate, project:

```
head_i = Attention(X·W_Q^i, X·W_K^i, X·W_V^i)
MHA(X) = Concat(head_1, ..., head_h) · W_O
```

**Note the parameter count doesn't grow.** Eight heads of size 64 use the same parameters as one head of size 512 — you've partitioned the space, not enlarged it. You get multiple attention patterns for free.

```python
def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=True):
    """X: (B, T, d_model). Wq/Wk/Wv: (d_model, d_model). Wo: (d_model, d_model)."""
    B, T, d = X.shape
    dk = d // n_heads

    Q = (X @ Wq).reshape(B, T, n_heads, dk).transpose(0, 2, 1, 3)   # (B,h,T,dk)
    K = (X @ Wk).reshape(B, T, n_heads, dk).transpose(0, 2, 1, 3)
    V = (X @ Wv).reshape(B, T, n_heads, dk).transpose(0, 2, 1, 3)

    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dk)              # (B,h,T,T)

    if causal:
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)

    A = softmax(scores)                                             # over last axis
    out = A @ V                                                     # (B,h,T,dk)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, d)                # concat heads
    return out @ Wo
```

**Track the shapes carefully** — this function is where most people's first transformer breaks. Write the shape of every intermediate in a comment, exactly as above.

---

## 11.9 Positional encoding

Here's a fact that surprises people: **attention has no notion of order.**

Shuffle the input tokens and the output shuffles identically. Attention is permutation-equivariant — it sees a *set*, not a sequence. Nothing in `softmax(QKᵀ/√d)V` refers to position.

So position must be injected explicitly.

### Sinusoidal (original transformer)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Added to the token embeddings. Different dimensions oscillate at different frequencies — like a binary encoding in continuous form. Fixed, not learned; extrapolates somewhat beyond training length.

### Learned

Just an embedding table indexed by position. Simple, works well, **cannot extrapolate** past the maximum length seen in training. Used by GPT-2 and BERT.

### RoPE — rotary position embedding

The modern standard (Llama and most of what followed). Instead of *adding* position, **rotate** `q` and `k` by an angle proportional to their position.

Treat consecutive dimension pairs as 2-D vectors and rotate the pair at position `m` by angle `mθ`.

**Why this is elegant:** for rotation matrices,

```
(R(mθ)q) · (R(nθ)k) = qᵀ R(mθ)ᵀ R(nθ) k = qᵀ R((n−m)θ) k
```

The attention score depends **only on the relative distance `n − m`**, never on absolute positions. Relative positioning falls straight out of the algebra, with no extra parameters, and it extrapolates far better than learned embeddings.

```python
def rope(x, positions, base=10000.0):
    """x: (..., T, d) with d even. Rotates dimension pairs by position."""
    d = x.shape[-1]
    half = d // 2
    freqs = 1.0 / (base ** (np.arange(half) * 2 / d))       # (half,)
    ang = positions[:, None] * freqs[None, :]               # (T, half)
    cos, sin = np.cos(ang), np.sin(ang)

    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin,
                           x1 * sin + x2 * cos], axis=-1)
```

Apply it to `Q` and `K` only — never to `V`. Position should affect *which* tokens you attend to, not *what* information they carry.

---

## 11.10 The transformer block

Assemble everything:

```
x ← x + MHA(LN(x))          attention sublayer
x ← x + MLP(LN(x))          feedforward sublayer
```

with

```
MLP(x) = W₂ · GELU(W₁x + b₁) + b₂        where W₁: d → 4d,  W₂: 4d → d
```

Four things to notice, each of which you've already derived:

1. **Residual connections** (§10.9) around both sublayers. This is what makes 96-layer stacks trainable.
2. **Pre-norm** (§7.6) — LayerNorm inside the residual branch, leaving a clean unnormalized path from input to output.
3. **LayerNorm, not BatchNorm** (§7.6) — no batch dependence, works with variable-length sequences.
4. **The 4× expansion** in the MLP. Empirical, near-universal, and it holds most of the model's parameters.

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

**That's the whole architecture.** Stack `N` of these, add token embeddings at the bottom and a linear projection to vocabulary size at the top, and you have GPT. Chapter 12 does exactly that.

### Where the parameters live

For one block with `d = d_model`:

- Attention: `4d²` (Q, K, V, and the output projection)
- MLP: `8d²` (`d→4d` and `4d→d`)

**The MLP holds two-thirds of the parameters.** Attention gets the attention, but most of the model's capacity is in the feedforward layers. There's a growing body of interpretability work suggesting they act as key-value memories storing factual knowledge.

---

## 11.11 The `O(n²)` problem

Attention computes an `n × n` score matrix:

```
time:   O(n²·d)
memory: O(n²) per head per layer
```

| Sequence length | Attention matrix entries |
|---|---|
| 1,000 | 10⁶ |
| 10,000 | 10⁸ |
| 100,000 | 10¹⁰ |

**This quadratic cost is the central engineering constraint of modern LLMs**, and most systems work in the field is about managing it.

### FlashAttention

Doesn't reduce FLOPs at all. It reduces **memory traffic** by tiling the computation and never materializing the full `n × n` matrix in slow memory — computing softmax in a streaming fashion instead.

The lesson generalizes: **on modern hardware, memory bandwidth is usually the bottleneck, not arithmetic.** An algorithm doing the same math with better memory access patterns can be several times faster.

### KV cache

At generation time, each new token re-attends to all previous ones. Recomputing every `K` and `V` each step is `O(n²)` total; caching them makes each step `O(n)`.

**The cache is large** — for every layer, every head, every previous token. It often dominates inference memory, which motivates:

### Multi-query and grouped-query attention

- **MQA** — all heads share one `K` and one `V`. Shrinks the cache by a factor of `h`.
- **GQA** — heads are grouped, each group sharing `K`/`V`. The compromise; used in most current models.

Small quality cost, large memory saving. Essentially every deployed LLM uses GQA.

### Long-context approaches

Sparse attention (attend to a subset), sliding windows (local only, plus a few global tokens), linear-attention approximations, and state-space models (Mamba and relatives) that revisit recurrence with parallelizable formulations. An active area — worth tracking, none yet a universal replacement.

---

## 11.12 Exercises

**1.** Implement an RNN forward pass and BPTT from scratch. Gradient-check it on a 5-step sequence.

**2.** **Demonstrate vanishing gradients.** Compute `‖∂h_t/∂h_0‖` for `t = 1…100` with `W_hh` scaled so its largest singular value is 0.5, 1.0, and 1.5. Plot on a log scale. Explain the three curves.

**3.** Train an RNN on the copy task: read a sequence, then reproduce it after a delay of `k` steps. Vary `k ∈ {5, 10, 20, 50}`. Find where it fails.

**4.** Implement an LSTM cell from scratch. Repeat exercise 3. Report the new failure point.

**5.** Show that `∂c_t/∂c_(t−1) = f_t` for the LSTM. Then train with the forget-gate bias at 0 and at 1, and compare learning curves.

**6.** **The `√d` demonstration.** For `d ∈ {8, 64, 512}`, sample 10,000 random `q·k` pairs. Histogram the raw and scaled dot products. Then plot the softmax output and its gradient magnitude for both. Explain what saturation does to learning.

**7.** Implement single-head scaled dot-product attention. Verify against `torch.nn.functional.scaled_dot_product_attention`.

**8.** Implement causal masking. Verify that position `i`'s output is unchanged when tokens after `i` are altered — this is the actual test that your mask is correct.

**9.** Implement multi-head attention from scratch. Verify against `nn.MultiheadAttention` with matched weights.

**10.** **Permutation equivariance.** Feed a shuffled sequence to attention without positional encoding and show the output is the same shuffle of the original output. Then add positional encoding and show it isn't.

**11.** Implement sinusoidal positional encoding. Plot it as a heatmap over positions and dimensions. Explain the frequency structure.

**12.** Implement RoPE. **Verify the key property**: show that `(R(mθ)q)·(R(nθ)k)` depends only on `n − m`, by holding `n − m` fixed and varying `m`.

**13.** Implement a full `TransformerBlock` with pre-norm. Verify shapes and gradient flow. Then implement the post-norm variant and compare gradient norms at depth 24.

**14.** Compute and plot attention memory usage against sequence length for a realistic configuration (12 layers, 12 heads). Mark where it exceeds 8GB, 24GB, and 80GB.

**15.** Implement a KV cache for generation. Measure tokens-per-second with and without it across sequence lengths 128 to 2048. Plot both.

**16.** **Chapter project.** Build a complete decoder-only transformer from scratch in PyTorch — no `nn.MultiheadAttention`, no `nn.Transformer`. Train it on a character-level task (arithmetic, sorting, or a small text corpus). Requirements: pre-norm blocks, RoPE, causal masking, weight tying, gradient clipping, warmup+cosine schedule, and **attention pattern visualizations** for at least three heads with an interpretation of what each appears to do. Write it up.

---

## 11.13 Solutions

<details>
<summary>Open only after attempting</summary>

```python
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
rng = np.random.default_rng(0)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


# --- 2: vanishing gradients ---
def jacobian_norm(sv, T=100, n=32):
    W = rng.standard_normal((n, n))
    U, S, Vt = np.linalg.svd(W); W = U @ np.diag(S/S.max()*sv) @ Vt
    J = np.eye(n); norms = []
    for _ in range(T):
        J = J @ W                       # tanh' <= 1 only makes it worse
        norms.append(np.linalg.norm(J))
    return norms

for sv in (0.5, 1.0, 1.5):
    print(sv, f"{jacobian_norm(sv)[-1]:.3e}")
# 0.5 -> ~1e-30 (vanished), 1.0 -> O(1), 1.5 -> ~1e17 (exploded).
# Because it is the SAME matrix repeated, the largest singular value governs
# exactly — no averaging across differing layers to soften it (§7.4).


# --- 5 ---
# c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t.  Treating the gates as given,
# ∂c_t/∂c_{t-1} = f_t elementwise. With f_t ≈ 1 the gradient passes through
# untouched — an additive highway, the same trick as a residual block (§10.9).
# Initializing b_f = 1 opens the gate at the start so the default is REMEMBER.


# --- 6: THE √d DEMONSTRATION ---
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 3, figsize=(14, 6))
for c, d in enumerate((8, 64, 512)):
    q = rng.standard_normal((10_000, d)); k = rng.standard_normal((10_000, d))
    raw = np.sum(q*k, axis=1); scaled = raw/np.sqrt(d)
    ax[0,c].hist(raw, bins=60, alpha=.6, label="raw")
    ax[0,c].hist(scaled, bins=60, alpha=.6, label="/√d")
    ax[0,c].set_title(f"d={d}  std raw={raw.std():.1f}  scaled={scaled.std():.2f}")
    ax[0,c].legend()
    s_raw = softmax(raw[:8]); s_sc = softmax(scaled[:8])
    ax[1,c].plot(s_raw, "o-", label="raw"); ax[1,c].plot(s_sc, "s-", label="/√d")
    ax[1,c].legend()
plt.tight_layout(); plt.show()
# Var[q·k] = d exactly. At d=512 raw scores span ±60, softmax becomes one-hot,
# and its Jacobian p_i(δ_ij − p_j) (§4.11) goes to zero — no gradient.
# Dividing by √d restores unit variance and keeps softmax responsive.


# --- 7, 8, 9 ---
def attention(Q, K, V, causal=False):
    dk = Q.shape[-1]
    s = Q @ np.swapaxes(K, -1, -2) / np.sqrt(dk)
    if causal:
        T = Q.shape[-2]
        s = np.where(np.triu(np.ones((T,T), bool), 1), -np.inf, s)
    return softmax(s) @ V, softmax(s)

B, T, d = 2, 6, 16
Q, K, V = (rng.standard_normal((B,T,d)) for _ in range(3))
mine, _ = attention(Q, K, V, causal=True)
ref = F.scaled_dot_product_attention(*(torch.tensor(x) for x in (Q,K,V)), is_causal=True)
assert np.allclose(mine, ref.numpy(), atol=1e-10)

# ex 8: the real test of a causal mask
V2 = V.copy(); V2[:, 4:] = rng.standard_normal(V2[:, 4:].shape)
out1, _ = attention(Q, K, V,  causal=True)
out2, _ = attention(Q, K, V2, causal=True)
assert np.allclose(out1[:, :4], out2[:, :4])       # positions 0-3 unaffected
print("causal mask verified")


# --- 10: permutation equivariance ---
perm = rng.permutation(T)
o_plain, _ = attention(Q, K, V)
o_perm, _  = attention(Q[:,perm], K[:,perm], V[:,perm])
assert np.allclose(o_perm, o_plain[:, perm], atol=1e-10)
print("attention sees a SET, not a sequence — hence positional encoding")


# --- 12: RoPE's key property ---
def rope(x, pos, base=10000.):
    d = x.shape[-1]; h = d//2
    fr = 1./(base ** (np.arange(h)*2/d))
    a = np.asarray(pos)[:,None]*fr[None,:]
    c, s = np.cos(a), np.sin(a)
    x1, x2 = x[...,:h], x[...,h:]
    return np.concatenate([x1*c - x2*s, x1*s + x2*c], -1)

q0, k0 = rng.standard_normal((1,8)), rng.standard_normal((1,8))
for m in (0, 3, 7, 20):                     # hold n-m = 5 fixed
    qm = rope(q0, [m]); kn = rope(k0, [m+5])
    print(f"m={m:3d}  score={float(qm @ kn.T):.10f}")
# Identical for every m. R(mθ)ᵀR(nθ) = R((n−m)θ), so the score depends only on
# relative distance. Relative positioning for free, and it extrapolates.


# --- 13 ---
class CausalSelfAttention(nn.Module):
    def __init__(self, d, h, dropout=0.):
        super().__init__()
        assert d % h == 0
        self.h, self.dk = h, d//h
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (t.view(B, T, self.h, self.dk).transpose(1,2) for t in (q,k,v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1,2).contiguous().view(B, T, d)
        return self.drop(self.proj(o))

class TransformerBlock(nn.Module):
    def __init__(self, d, h, dropout=0., prenorm=True):
        super().__init__()
        self.prenorm = prenorm
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, h, dropout)
        self.mlp = nn.Sequential(nn.Linear(d,4*d), nn.GELU(),
                                 nn.Linear(4*d,d), nn.Dropout(dropout))
    def forward(self, x):
        if self.prenorm:
            x = x + self.attn(self.ln1(x)); x = x + self.mlp(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x)); x = self.ln2(x + self.mlp(x))
        return x

for pre in (True, False):
    net = nn.Sequential(*[TransformerBlock(128, 8, prenorm=pre) for _ in range(24)])
    xb = torch.randn(4, 32, 128, requires_grad=True)
    net(xb).sum().backward()
    print(f"prenorm={pre}  input grad norm {xb.grad.norm():.3e}")
# Pre-norm keeps the residual path unnormalized, so gradient reaches layer 0
# intact. Post-norm rescales at every layer and needs warmup to train at depth.


# --- 14 ---
for T in (512, 1024, 2048, 8192, 32768):
    gb = 12 * 12 * T * T * 2 / 1e9          # layers × heads × T² × bf16 bytes
    print(f"T={T:6d}  attention matrices ≈ {gb:8.2f} GB")
# Quadratic growth is why FlashAttention (never materialize the matrix) and
# GQA (shrink the KV cache) exist.
```

</details>

---

## 11.14 Chapter 11 checkpoint

Cold — blank file, no notes.

- [ ] Explain why RNN gradients vanish, and why repeating the *same* matrix makes it worse than a deep MLP.
- [ ] State the LSTM equations and show `∂c_t/∂c_(t−1) = f_t`. Explain the connection to residual connections.
- [ ] Give three reasons transformers replaced RNNs, and say which was decisive.
- [ ] **Derive attention from the bottleneck problem** — weighted average, similarity, softmax. In writing.
- [ ] **Derive the `√d_k` scaling** with the variance argument. **On paper, 5 minutes.**
- [ ] Explain what Q, K, V each represent and why three projections rather than one.
- [ ] Explain why the causal mask uses `−∞` before softmax rather than zeroing after.
- [ ] **Implement multi-head causal self-attention from scratch.** **Target: 25 minutes.**
- [ ] Explain why attention needs positional encoding at all, and state RoPE's key property with the rotation identity.
- [ ] **Write a complete pre-norm transformer block from memory.** **Target: 10 minutes.**
- [ ] State attention's time and memory complexity, and what FlashAttention and GQA each address.

Items 5, 8 and 10 are mandatory before Chapter 12.

### Anki cards

- RNN update equation
- Why RNN gradients vanish — the repeated-matrix argument
- LSTM's four gates and the cell state update
- `∂c_t/∂c_(t−1) = ?` and why that matters
- Why did transformers replace RNNs? (three reasons)
- Attention formula
- `Var[q·k] = ?` and why divide by `√d_k`
- What do Q, K, V mean?
- Why `−∞` before softmax for masking?
- Multi-head attention — why the parameter count doesn't grow
- Why does attention need positional encoding?
- RoPE's key identity
- Pre-norm transformer block, both lines
- Where do a transformer block's parameters live? (the 4d²/8d² split)
- Attention complexity; what FlashAttention fixes; what GQA fixes

### Deliverables

```
attention/attention.py     scaled dot-product, causal mask, gradient-checked
attention/mha.py           multi-head, verified against PyTorch
attention/rope.py          RoPE with the relative-position test
attention/block.py         TransformerBlock, pre-norm and post-norm
rnn/lstm.py                LSTM cell from scratch
experiments/copy_task.py   exercises 3-4
experiments/sqrt_d.py      exercise 6, with plots
projects/mini_transformer/ exercise 16
```

```bash
git add .
git commit -m "Chapter 11: RNNs, LSTM, attention from scratch, transformer block"
git push
```

### Write-up

900 words: **"Attention, derived from the problem."** Start with the RNN bottleneck, build up the weighted average, motivate the similarity score from the dot product, then the `√d` variance derivation with your exercise 6 plots, then masking, multi-head, and positional encoding. Include your permutation-equivariance demonstration from exercise 10.

This is the most valuable post in your portfolio so far. Nearly every attention explanation online presents the formula and then explains what the parts do. Yours will show why the formula had to look like that.

**You now have the architecture that everything runs on.** Chapter 12 assembles these blocks into a GPT, trains it, and takes you through tokenization, scaling laws, and generation.

---

*Next: Chapter 12 — Building a GPT*
