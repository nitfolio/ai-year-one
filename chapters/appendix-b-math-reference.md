# Appendix B — Math Reference Card

Everything derived in this book, compressed. Each entry has the formula, a one-line reminder of *why*, and the chapter to reread.

**This is a reminder card, not a substitute for the derivations.** If you can't reconstruct an entry from scratch, go back to its section.

---

## B.1 Linear algebra

| | |
|---|---|
| **Dot product** | `a·b = Σᵢaᵢbᵢ = \|a\|\|b\|cos θ` — how much two vectors point the same way (§2.3) |
| **Zero dot product** | Perpendicular. Basis of projection, least squares, orthogonality |
| **Cosine similarity** | `(a·b)/(\|a\|\|b\|)` — direction only, scale-free. Embedding similarity |
| **Norms** | `\|a\|₂=√(a·a)`, `\|a\|₁=Σ\|aᵢ\|`, `\|a\|∞=max\|aᵢ\|` (§2.4) |
| **Matrix = transformation** | Columns are where the basis vectors land (§2.5) |
| **Matmul shape** | `(m,k) @ (k,n) → (m,n)`; inner dims must match and vanish |
| **`AB` order** | Apply `B` first, then `A`. `AB ≠ BA` |
| **Transpose** | `(AB)ᵀ = BᵀAᵀ` — **order reverses** (§2.9) |
| **Rank** | Dimensions of the output space. `rank(AB) ≤ min(rank A, rank B)` (§2.11) |
| **LoRA** | `ΔW = BA` with `B:(d,r)`, `A:(r,k)` → rank ≤ r, few parameters |
| **det = 0** | Collapses a dimension. Information destroyed. Not invertible (§2.12) |
| **Eigenvector** | `Av = λv` — direction unchanged by the transformation (§2.13) |
| **Condition number** | `λmax/λmin` of the Hessian. High → ravine → slow gradient descent |
| **SVD** | `A = UΣVᵀ` — every linear map is rotate, stretch, rotate (§2.14) |
| **Low-rank approx** | Keep top `k` singular values. Provably optimal (Eckart–Young) |
| **Projection** | `proj_b(a) = ((a·b)/(b·b))b`; residual ⊥ `b` (§2.15) |
| **Normal equations** | `AᵀAx = Aᵀy` — least squares. Use `solve`, never `inv` |

### The layer-collapse theorem (§2.8)

```
W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂) = W'x + b'
```

Stacked linear layers collapse to one. **This is why activation functions exist.**

---

## B.2 Calculus and gradients

| | |
|---|---|
| **Derivative (the view that matters)** | `f(x+Δ) ≈ f(x) + f'(x)Δ` — local linear approximation (§3.1) |
| **Chain rule** | `dy/dx = (dy/du)(du/dx)` — rates multiply |
| **Multi-path rule** | `∂L/∂x = Σᵢ(∂L/∂yᵢ)(∂yᵢ/∂x)` — **paths ADD** (§3.5) |
| **Gradient** | Vector of partials. Points in the direction of steepest ascent |
| **Why uphill** | `D_u f = ∇f·u = \|∇f\|cos θ`, maximized at `θ=0` (§3.8) |
| **Gradient descent** | `θ ← θ − η∇L(θ)` |
| **Central difference** | `[f(x+h) − f(x−h)]/(2h)`, error `O(h²)`. Use `h≈1e-5` for float64 (§3.12) |
| **Relative error** | `\|a−b\|/(\|a\|+\|b\|)`. `<1e-7` good, `>1e-3` broken |

### Derivatives to know cold (§3.3)

| `f(x)` | `f'(x)` |
|---|---|
| `xⁿ` | `n·xⁿ⁻¹` |
| `eˣ` | `eˣ` |
| `ln x` | `1/x` |
| `σ(x)` | `σ(1−σ)` — **max 0.25** |
| `tanh x` | `1 − tanh²x` — max 1.0 |
| `max(0,x)` | `1` if `x>0` else `0` |

---

## B.3 Probability and information theory

| | |
|---|---|
| **Expectation linearity** | `E[aX+bY] = aE[X]+bE[Y]` — holds even when dependent |
| **Variance** | `Var[X] = E[X²] − E[X]²`; `Var[aX] = a²Var[X]` |
| **Sum of independents** | `Var[X+Y] = Var[X]+Var[Y]` |
| **Gradient noise** | Averaging `B` samples → std `∝ 1/√B`. Quadrupling batch halves noise (§6.3) |
| **Bayes** | `P(A\|B) = P(B\|A)P(A)/P(B)` — small prior ⟹ strong evidence still leaves doubt |
| **Chain rule of probability** | `P(x₁…xₙ) = ∏ₜ P(xₜ\|x_<t)` — justifies autoregressive LMs |
| **NLL** | `−Σᵢ log P(yᵢ\|xᵢ,θ)`. **Every loss is an NLL under some assumption** (§6.5) |
| **Entropy** | `H(p) = −Σpᵢlog pᵢ` — expected surprise; max `log k` for `k` outcomes |
| **Cross-entropy** | `H(p,q) = −Σpᵢlog qᵢ` |
| **KL divergence** | `KL(p‖q) = H(p,q) − H(p) ≥ 0`, **not symmetric** (§6.7) |
| **Why CE works** | `H(p)` is fixed ⟹ minimizing CE = minimizing KL. Loss floor is `H(p)` |
| **Perplexity** | `exp(cross-entropy in nats)` — effective number of choices (§6.8) |
| **Temperature** | `pᵢ ∝ exp(zᵢ/T)`. `T→0` greedy, `T>1` flatter |

### Loss derivations (§6.5)

```
Gaussian noise         ⟹  mean squared error
Bernoulli              ⟹  binary cross-entropy
Categorical            ⟹  softmax cross-entropy
```

For any exponential-family distribution with the natural parameter, `∂NLL/∂z = predicted − observed`. That's why `p − y` appears three times.

### High-dimensional facts (§6.10)

- Random vectors are nearly orthogonal: `cos θ ~ 1/√d`
- Gaussian mass concentrates on a shell at radius `√d`
- Ball volume concentrates at the boundary: `(1−ε)ᵈ → 0`
- Distances become uninformative — the curse of dimensionality

---

## B.4 Layers and their derivatives

### Linear layer

```
forward:   Y = XW + b        X:(N,in)  W:(in,out)  b:(out,)
backward:  dX = dY Wᵀ        dW = Xᵀ dY        db = dY.sum(axis=0)
```

Each formula is the only shape-legal arrangement. **When you forget, write the shapes.** (§2.9, §5.8)

### Matmul

```
C = A @ B    ⟹    dA = dC Bᵀ,    dB = Aᵀ dC
```

### Broadcasting

**Forward broadcast ⟹ backward sum** over the broadcast axes. It's the multi-path rule again (§5.8).

### Loss gradients

| Model + loss | `∂L/∂z` | `∂L/∂w` |
|---|---|---|
| Linear + MSE | `2(ŷ−y)/n` | `(2/n)Xᵀ(Xw+b−y)` |
| Sigmoid + BCE | `p − y` | `(1/n)Xᵀ(p−y)` |
| Softmax + CE | `p − y` | `(1/n)Xᵀ(P−Y)` |

The `σ'` factor cancels in the last two. That cancellation is why cross-entropy is paired with sigmoid/softmax, and why frameworks fuse them (§4.10, §4.11).

### Softmax Jacobian

```
∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ − pⱼ)
```

### Numerically stable forms

```
softmax:        subtract the row max before exp
BCE(logits):    max(z,0) − z·y + log(1 + exp(−|z|))
```

---

## B.5 Initialization (§7.2–7.3)

**Variance through a layer:**

```
Var[y] = n_in · σ_w² · σ_x²
```

After `L` layers the signal is multiplied by `(n_in·σ_w²)^L` — **exponential in depth.**

| Activation | Init | Scale |
|---|---|---|
| tanh / sigmoid | Xavier | `σ_w² = 2/(n_in + n_out)`, uniform limit `√(6/(n_in+n_out))` |
| ReLU family | He | `σ_w² = 2/n_in` |
| Biases | — | zero |

**Why ReLU needs the 2:** `E[relu(z)²] = σ²/2` for `z ~ 𝒩(0,σ²)` — exactly half the variance destroyed.

**Never initialize weights to zero** — every unit stays identical forever.

**Residual scaling** (§12.4): scale each residual branch's output projection by `1/√(2N)` for `N` layers, so the residual stream's variance stays `O(1)`.

---

## B.6 Optimizers (§9)

```
SGD:        θ ← θ − ηg

Momentum:   v ← βv + g
            θ ← θ − ηv                    effective step η/(1−β)

RMSProp:    G ← ρG + (1−ρ)g²
            θ ← θ − ηg/(√G + ε)

Adam:       m ← β₁m + (1−β₁)g
            v ← β₂v + (1−β₂)g²
            m̂ = m/(1−β₁ᵗ),  v̂ = v/(1−β₂ᵗ)
            θ ← θ − η·m̂/(√v̂ + ε)

AdamW:      θ ← θ − η·m̂/(√v̂ + ε) − ηλθ    ← decay OUTSIDE the adaptive scaling
```

**Bias correction, derived:** for constant `g`, `m_t = g(1−β₁ᵗ)`. At `t=1` with `β₂=0.999` the second moment is 1000× too small. Dividing by `(1−βᵗ)` removes the bias exactly.

**Why AdamW ≠ Adam + L2:** L2 adds `λθ` to the gradient, which then gets divided by `√v̂` — so large-gradient parameters get *less* decay. Backwards.

**Schedules:**

```
cosine:         lr = lr_min + ½(lr_max−lr_min)(1 + cos(πt/T))
warmup+cosine:  linear ramp over first W steps, then cosine
```

**Why warmup:** Adam's `v` is unreliable early; early gradients are large; post-norm transformers destabilize (§9.7).

**Linear scaling rule:** batch ×k ⟹ LR ×k, up to a critical batch size.

---

## B.7 Convolutions (§10)

```
output size:      out = ⌊(in + 2p − d(k−1) − 1)/s⌋ + 1
"same" padding:   p = (k−1)/2 with s=1
parameters:       k²·C_in·C_out + C_out
receptive field:  r_l = r_(l−1) + (k_l − 1)·∏_(i<l) s_i
```

| | |
|---|---|
| **Three 3×3 vs one 7×7** | Same RF (7), `27C²` vs `49C²` params, 3× the nonlinearity |
| **1×1 conv** | Mixes channels, not space. A per-pixel linear layer |
| **Depthwise separable** | `k²C + C·C_out` vs `k²·C·C_out` |
| **Max pool backward** | Gradient routes only to the argmax |
| **Residual** | `y = F(x) + x` ⟹ `∂y/∂x = 1 + ∂F/∂x` — a path with derivative exactly 1 |

**The degradation problem:** deeper plain nets had *higher training* error. Not overfitting — a deeper net can represent a shallower one via identity layers, so the solution exists and SGD failed to find it. That diagnosis is what led to residuals.

---

## B.8 Attention and transformers (§11–12)

```
Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V
```

**The `√d_k`:** `Var[q·k] = d_k` for unit-variance components, so scores have std `√d_k`. Large spread saturates softmax → near-zero gradient. Dividing restores unit variance.

| | |
|---|---|
| **Q, K, V** | What I'm looking for / what I contain / what I contribute |
| **Causal mask** | Set scores to `−∞` **before** softmax, so normalization covers only allowed positions |
| **Multi-head** | Split `d_model` into `h` heads of `d_k = d_model/h`. Parameter count unchanged |
| **Permutation equivariance** | Attention sees a **set** — hence positional encoding |
| **RoPE** | `(R(mθ)q)·(R(nθ)k) = qᵀR((n−m)θ)k` — depends only on relative distance |
| **Complexity** | `O(n²d)` time, `O(n²)` memory per head per layer |
| **FlashAttention** | Same FLOPs, less memory traffic — never materializes the `n×n` matrix |
| **KV cache** | Caches `K,V` at generation: `O(n²)` → `O(n)` per token |
| **GQA/MQA** | Share `K,V` across heads to shrink the cache |

**Transformer block (pre-norm):**

```
x ← x + MHA(LN(x))
x ← x + MLP(LN(x))          MLP: d → 4d → GELU → d
```

**Parameters per block:** attention `4d²`, MLP `8d²`. **The MLP holds two-thirds.**

**Full GPT:**

```
params ≈ V·d + block·d + 12·L·d²        (with weight tying)
```

---

## B.9 Scaling and compute (§12.6–12.7)

```
training FLOPs:      C ≈ 6·N·D
Chinchilla optimal:  D ≈ 20·N tokens per parameter
```

Chinchilla's correction: models were badly undertrained. 70B on 1.4T tokens beat 280B on 300B tokens at matched compute.

**Caveat:** Chinchilla optimizes *training* compute. If inference dominates, train smaller models far past `20×`.

**Memory per parameter, AdamW mixed precision:**

```
2  bf16 parameter
2  bf16 gradient
4  fp32 master copy
4  Adam m
4  Adam v
──
16 bytes/param        → a 1B model needs ~16GB before activations
```

**bf16 over fp16:** same exponent range as fp32, so no loss scaling needed.

**Loss milestones for an LM** (vocab 50k): start at `ln(V) ≈ 10.8`; ~7 means unigrams only; ~3.5 is a working small model.

---

## B.10 Shapes cheat sheet

```
NumPy → PyTorch:   axis → dim,  keepdims → keepdim

Linear (math):     x:(in,)         W:(out,in)      y:(out,)
Linear (code):     X:(N,in)        W:(in,out)      Y:(N,out)
nn.Linear stores:  weight:(out,in) — computes X @ W.T + b

Conv2d:            X:(N,C_in,H,W)  W:(C_out,C_in,kh,kw)  → (N,C_out,H',W')

Attention:         X:(B,T,d)  →  Q,K,V:(B,h,T,d_k)
                   scores:(B,h,T,T)  →  out:(B,T,d)

CrossEntropyLoss:  input:(N,C) logits,  target:(N,) indices
LM logits:         (B,T,V);  targets:(B,T)  → flatten to (B·T,V) and (B·T,)
```

---

## B.11 The derivations you should be able to reproduce cold

If you can do these ten from a blank page, you understand this book:

1. **Two linear layers collapse to one** — three lines (§2.8)
2. **`∂L/∂x = Wᵀ∂L/∂y`** from shapes and from the chain rule (§2.9)
3. **`σ'(x) = σ(1−σ)`** (§3.3)
4. **The gradient points uphill**, via the dot product (§3.8)
5. **`∂L/∂z = p − y`** for sigmoid + BCE (§4.10)
6. **MSE from maximum likelihood** under Gaussian noise (§6.5)
7. **`Var[y] = n_in·σ_w²·σ_x²`** and therefore `σ_w = 1/√n_in` (§7.2)
8. **Adam's bias correction**: `m_t = g(1−β₁ᵗ)` (§9.5)
9. **`Var[q·k] = d_k`** and therefore the `√d_k` scaling (§11.6)
10. **`∂y/∂x = 1 + ∂F/∂x`** for a residual block, and why it fixes gradient flow (§10.9)

Test yourself on these monthly. They rot faster than you'd expect, and they're the load-bearing ones.

---

*Next: Appendix C — Free Resource Directory*
