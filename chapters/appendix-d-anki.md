# Appendix D — Anki Card Bank

A starter deck so you're not building one from nothing on day one.

**These are not a substitute for making your own cards.** Cards you write while learning stick better than cards you import. Use these as a floor and a format example, and add your own as you go — especially for the things *you* keep forgetting, which nobody else can predict.

---

## D.0 The rules (from §0.5)

- **Never card something you haven't understood.** Cards retain understanding; they don't create it.
- **One fact per card.** If the answer has three parts, make three cards.
- **15 minutes a day, every day.** Including bad days. The schedule *is* the method.
- **Add ~10 cards a day as you learn**, not in batches later.
- **Delete cards you keep failing** and relearn the underlying concept instead. A card you fail twenty times is telling you the understanding isn't there.
- **Don't card long code.** You learn code by writing it.

---

## D.1 Importing

Copy the tables below into a text file as tab-separated `Front⇥Back`, then Anki → *File* → *Import*, field separator: Tab.

Quick conversion from this markdown:

```python
import re, sys

rows = []
for line in open("appendix-d-anki.md"):
    if line.startswith("| ") and " | " in line and "---" not in line:
        parts = [p.strip() for p in line.strip().strip("|").split(" | ")]
        if len(parts) == 2 and parts[0] not in ("Front", "Question"):
            rows.append("\t".join(parts))

open("deck.tsv", "w").write("\n".join(rows))
print(len(rows), "cards written to deck.tsv")
```

Then tag by chapter inside Anki so you can review selectively.

---

## D.2 Chapter 0 — Method

| Front | Back |
|---|---|
| What is a falsifiable checkpoint? | A test of understanding that can fail — implementing from a blank file, not "feeling like I get it" |
| The five levels of evidence of understanding | 1 explain aloud, 2 explain in writing, 3 derive on paper, 4 implement from blank file, 5 find a planted bug |
| What is a cold rebuild? | Rebuild a chapter's central artifact from a blank editor, no notes, timed |
| Unstuck Protocol, steps 1–4 | 1 name it precisely, 2 check the prerequisite, 3 shrink it, 4 find a second explanation |
| Unstuck Protocol, steps 5–8 | 5 print everything, 6 sleep on it, 7 park it and move on, 8 ask a human or model |
| Total time budget for the Unstuck Protocol | ~2 focused hours across 2 days — never a 6-hour night grind |
| The build-first rule | Never use a library for something you haven't implemented yourself once |
| The 48-hour rule | Any concept learned must appear in code you wrote within 48 hours |
| Minimum viable study day | Block B (build, 90 min) + Block D (Anki, 15 min) |
| Three verification methods with no teacher | Numerical gradient check; compare to a reference implementation; overfit a tiny dataset |

---

## D.3 Chapters 1–2 — NumPy and linear algebra

| Front | Back |
|---|---|
| What does `axis=k` mean in a NumPy reduction? | The axis that disappears. `(2,3).sum(axis=0)` → `(3,)` |
| Broadcasting rule, one sentence | Align shapes from the right; dims are compatible if equal, or one is 1, or missing |
| Difference between `*` and `@` | `*` is elementwise; `@` is matrix multiplication |
| Why subtract the max before `exp` in softmax? | Prevents overflow. Softmax is invariant to adding a constant to a row |
| What does `keepdims=True` do, and when do you need it? | Keeps the collapsed axis at size 1 — needed for broadcasting back against the original |
| Shape rule for `@` | `(n,k) @ (k,m) → (n,m)`; inner dims must match and vanish |
| Dot product, geometric form | `a·b = \|a\|\|b\|cos θ` — how much two vectors point the same way |
| What does a zero dot product mean? | The vectors are perpendicular |
| Columns of a matrix are ___ | Where the basis vectors land |
| `(AB)ᵀ = ?` | `BᵀAᵀ` — the order reverses |
| Why do neural networks need nonlinearities? | `W₂(W₁x+b₁)+b₂ = W'x+b'` — stacked linear layers collapse to one |
| In backprop, if `y = Wx`, then `∂L/∂x = ?` | `Wᵀ ∂L/∂y` |
| The shape trick for backprop formulas | Write the shapes; usually only one arrangement type-checks |
| `rank(AB) ≤ ?` | `min(rank A, rank B)` — multiplying can only lose rank |
| What does LoRA exploit? | `ΔW = BA` with small `r` gives a big matrix from few parameters, since rank ≤ r |
| SVD: `A = ?` and what each factor does | `UΣVᵀ` — rotate, stretch, rotate |
| What does `det = 0` mean geometrically? | The transformation collapses a dimension; information destroyed, not invertible |
| Eigenvector definition | `Av = λv` — direction unchanged, only scaled |
| Why never compute an explicit matrix inverse in numerical code? | Slower and less stable; use `np.linalg.solve` |

---

## D.4 Chapter 3 — Calculus

| Front | Back |
|---|---|
| Three views of the derivative | Slope; rate of change; **local linear approximation** |
| Which view of the derivative matters most for ML, and why? | Local linear approximation — gradient descent is repeated local linearization |
| `σ'(x) = ?` | `σ(x)(1 − σ(x))`, maximum 0.25 |
| `tanh'(x) = ?` | `1 − tanh²(x)`, maximum 1.0 |
| ReLU derivative | 1 for `x>0`, 0 for `x<0`, undefined at 0 (convention: 0) |
| Chain rule | `dy/dx = (dy/du)(du/dx)` — rates multiply through a chain |
| The multi-path rule | If `x` reaches the output by several routes, **sum** the contributions |
| Why do gradients from multiple paths add rather than one being chosen? | The node changing by 1 affects every consumer simultaneously; all effects happen |
| Why does the gradient point uphill? | `D_u f = ∇f·u = \|∇f\|cos θ`, maximized when `u` aligns with `∇f` |
| Gradient descent update rule | `θ ← θ − η∇L(θ)` |
| Central difference formula and error order | `[f(x+h) − f(x−h)]/(2h)`, error `O(h²)` |
| Good `h` for gradient checking in float64 | ~`1e-5` |
| Why is too small an `h` a problem? | Catastrophic cancellation — subtracting nearly-equal numbers destroys precision |
| Relative error thresholds for a gradient check | `<1e-7` correct; `1e-5`–`1e-3` suspicious; `>1e-3` broken |
| Why can't we train with numerical gradients? | `2N` forward passes for `N` parameters; backprop gets all of them in ~1 |
| `∂L/∂z` for sigmoid + BCE | `p − y` |

---

## D.5 Chapter 4 — First learning algorithm

| Front | Back |
|---|---|
| The four components of supervised learning | A model, a loss, a gradient, an update rule |
| `∂L/∂w` for linear regression + MSE | `(2/n) Xᵀ(Xw + b − y)` |
| Why does the `σ'` factor cancel in sigmoid + BCE? | Cross-entropy's `1/p` and sigmoid's `p(1−p)` cancel exactly, giving `p − y` |
| Why is MSE bad for classification? | With sigmoid, gradient `∝ p(1−p)` → near zero exactly when confidently wrong |
| `∂L/∂z` for softmax + cross-entropy | `p − y` |
| Softmax Jacobian | `∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ − pⱼ)` |
| Normal equations, and four reasons not to use them | `XᵀXw = Xᵀy`. Doesn't scale, needs all data in memory, exists for almost nothing, numerically fragile |
| Why standardize features? | Unequal scales create a high condition number → ravine → slow convergence |
| Rule for computing standardization statistics | Training set only; apply the same values to val and test |
| L1 vs L2 regularization — what differs in the result? | L1 drives weights exactly to zero (sparse); L2 shrinks smoothly |
| Why exclude the bias from weight decay? | It sets the baseline output; shrinking it just biases predictions toward zero |
| The overfit-tiny-dataset test | 10 examples, no regularization, train to ~0 loss. If it fails, the code is broken |
| Three causes of `nan` loss | LR too high; `log(0)`; numerical overflow |
| Minibatch vs full batch vs SGD | Full: exact, slow. SGD: noisy, escapes saddles. Minibatch: hardware-efficient compromise |

---

## D.6 Chapter 5 — Autograd

| Front | Back |
|---|---|
| Forward vs reverse mode — which for neural nets and why? | Reverse. One sweep per **output**; NNs have billions of inputs and one scalar loss |
| Why `+=` and not `=` in `_backward`? | A node can feed multiple consumers; gradients from all paths must accumulate (§3.5) |
| Why is topological order required in backward? | A node's gradient must be fully accumulated before it passes anything on |
| Local derivative of `+` | 1 to both inputs — addition routes gradient unchanged |
| Local derivative of `*` | Each input gets the **other** input's value times the incoming gradient |
| Why does the backward pass reuse forward values? | Local derivatives are expressed in terms of already-computed outputs (e.g. `σ' = σ(1−σ)`) |
| Matmul backward: `C = A@B` | `dA = dC Bᵀ`, `dB = Aᵀ dC` |
| Broadcasting forward implies ___ backward | Summing over the broadcast axes |
| Why must the output layer be linear? | An activation there clamps the logits (e.g. ReLU zeroes negatives), capping accuracy |
| What does `zero_grad` do, and what breaks without it? | Clears accumulated gradients; without it each step uses the sum of all previous ones |
| The test that catches the `=` vs `+=` bug | Gradient-check an expression where one node is used twice, e.g. `a*b + a*a` |

---

## D.7 Chapter 6 — Probability

| Front | Back |
|---|---|
| Bayes' rule | `P(A\|B) = P(B\|A)P(A)/P(B)` |
| Why is a positive test for a rare condition usually a false positive? | With a small prior, false positives from the large healthy population outnumber true positives |
| Negative log-likelihood | `−Σ log P(yᵢ\|xᵢ,θ)`. Every loss is an NLL under some assumption |
| Gaussian noise assumption implies which loss? | Mean squared error |
| Bernoulli assumption implies which loss? | Binary cross-entropy |
| Entropy formula and what it measures | `H(p) = −Σpᵢlog pᵢ` — expected surprise; max is `log k` |
| `H(p,q) = ?` in terms of `H(p)` and KL | `H(p) + KL(p‖q)` |
| Why is minimizing cross-entropy the same as minimizing KL? | `H(p)` is fixed by the data, so only the KL term can change |
| What is the floor of cross-entropy loss? | `H(p)` — the irreducible entropy of the data |
| Is KL symmetric? What does each direction encourage? | No. Forward KL is mode-covering; reverse KL is mode-seeking |
| Perplexity definition and interpretation | `exp(cross-entropy in nats)` — the effective number of choices per step |
| How does gradient noise scale with batch size? | Standard error `∝ 1/√B` — quadrupling the batch halves the noise |
| Variance of a Bernoulli, and where else you've seen it | `p(1−p)` — also the sigmoid derivative |
| Chain rule of probability | `P(x₁…xₙ) = ∏ₜ P(xₜ\|x_<t)` — justifies autoregressive LMs |
| Three high-dimensional surprises | Random vectors nearly orthogonal; Gaussian mass on a shell at `√d`; volume at the boundary |
| Temperature in sampling | `pᵢ ∝ exp(zᵢ/T)`. `T→0` greedy, `T>1` flatter |
| Top-k vs top-p | Top-k keeps a fixed number; top-p keeps the smallest set reaching cumulative probability p |

---

## D.8 Chapter 7 — Making networks trainable

| Front | Back |
|---|---|
| Variance through a layer | `Var[y] = n_in · σ_w² · σ_x²` |
| Why does initialization scale matter exponentially? | Each layer multiplies variance by `n_in·σ_w²`; after L layers it's that factor to the L |
| Xavier initialization | `σ_w² = 2/(n_in + n_out)` — for tanh/sigmoid |
| He initialization | `σ_w² = 2/n_in` — for ReLU |
| Why does ReLU need the factor of 2? | `E[relu(z)²] = σ²/2` — exactly half the variance is destroyed |
| Why not initialize weights to zero? | Every unit in a layer computes the same thing and receives the same gradient — forever |
| Maximum of `σ'`, and the 20-layer consequence | 0.25; `0.25²⁰ ≈ 10⁻¹²` gradient at the early layers |
| Three historical fixes for vanishing gradients | ReLU; careful initialization; residual connections |
| BatchNorm vs LayerNorm — which axis? | BatchNorm over the batch (`axis=0`); LayerNorm over features (`axis=-1`) |
| Two reasons transformers use LayerNorm | No batch dependence; works with variable-length sequences |
| RMSNorm — what's dropped? | Mean subtraction and the bias. Just `γ·x/RMS(x)` |
| Pre-norm vs post-norm | Pre: `x + Sub(LN(x))` — clean residual path, trains at depth. Post: `LN(x + Sub(x))` |
| Residual: `∂y/∂x = ?` | `1 + ∂F/∂x` — a path with derivative exactly 1 |
| Dead ReLU: cause and fix | Always-negative pre-activation → zero gradient forever. Fix: LeakyReLU, lower LR, He init |
| Expected initial loss for `k` classes | `ln(k)`. 10 classes → 2.30 |
| Why global rather than per-parameter gradient clipping? | Global rescaling preserves the gradient direction; per-parameter distorts it |
| The seven debugging steps | Initial loss; overfit 10; gradient check; activation stats; gradient norms; LR sweep; add regularization back |

---

## D.9 Chapter 8 — PyTorch

| Front | Back |
|---|---|
| `.detach()` vs `.data` vs `.item()` | `.detach()` cuts from the graph safely; `.data` is raw and unsafe; `.item()` gives a Python float |
| When and why `torch.no_grad()`? | Inference and evaluation — builds no graph, saves memory and time |
| `CrossEntropyLoss` input and target shapes | input `(N,C)` raw logits; target `(N,)` class indices |
| Why does `CrossEntropyLoss` take logits? | It fuses LogSoftmax + NLL; the combined gradient is `p − y`, stable and clean |
| What does `model.eval()` change? | Disables dropout; BatchNorm switches to running statistics |
| `nn.Linear` weight shape | `(out, in)` — it computes `x @ W.T + b` |
| Which parameters should skip weight decay? | Biases and normalization parameters — all the 1-D ones |
| Gradient accumulation — why divide by the accumulation steps? | Otherwise you sum instead of average, multiplying the effective learning rate |
| Why save `state_dict` rather than the model object? | Pickling ties the file to your class definitions and directory layout |
| Why save the optimizer state? | Adam's momentum buffers; resuming without them causes a loss spike |
| PyTorch default dtype, and the consequence | `float32` — use `h≈1e-3` for gradient checks, or cast to `float64` |

---

## D.10 Chapter 9 — Optimization

| Front | Back |
|---|---|
| Momentum update rule | `v ← βv + g`; `θ ← θ − ηv` |
| Why does momentum fix the ravine? | Oscillating directions cancel in the sum; consistent directions accumulate |
| Effective step amplification with momentum | `η/(1−β)` — at `β=0.9` that's 10× |
| AdaGrad's flaw, and RMSProp's fix | `G` grows forever so LR decays to zero; RMSProp uses an exponential moving average instead |
| Adam, four lines | `m←β₁m+(1−β₁)g`; `v←β₂v+(1−β₂)g²`; bias-correct both; `θ ← θ − η m̂/(√v̂+ε)` |
| Bias correction — the derivation | For constant `g`, `m_t = g(1−β₁ᵗ)`; divide by `(1−β₁ᵗ)` to remove the bias |
| Adam bias correction factor at `t=1` with `β₂=0.999` | 1000× — the second moment is a thousand times too small without it |
| Adam vs AdamW — the actual difference | AdamW applies decay outside the adaptive scaling, so it isn't divided by `√v̂` |
| Why is Adam so insensitive to learning rate? | `m̂/√v̂` divides out the gradient magnitude, giving roughly unit-scale steps |
| Three reasons warmup exists | Adam's `v` unreliable early; large early gradients; post-norm instability |
| Cosine schedule | `lr_min + ½(lr_max−lr_min)(1 + cos(πt/T))` |
| Linear scaling rule, and where it breaks | Batch ×k ⟹ LR ×k; breaks past a critical batch size |
| Why no second-order methods? | The Hessian is `N×N` — 4.9×10¹⁹ entries at 7B parameters |
| Standard LLM settings | AdamW, `β=(0.9,0.95)`, wd 0.1, clip 1.0, warmup + cosine |
| Optimization failure vs generalization failure | Train loss bad → optimization. Train good, val bad → generalization; don't tune the optimizer |

---

## D.11 Chapters 10–11 — CNNs and attention

| Front | Back |
|---|---|
| The three ideas behind convolution | Local connectivity; weight sharing; translation equivariance |
| Conv output size formula | `⌊(in + 2p − d(k−1) − 1)/s⌋ + 1` |
| Conv parameter count | `k²·C_in·C_out + C_out` |
| How do channels work in a conv layer? | Each of `C_out` filters spans **all** input channels and produces one output channel |
| What is im2col and why does it help? | Lays every patch out as a matrix row so convolution becomes one matmul |
| Receptive field formula | `r_l = r_(l−1) + (k_l − 1)·∏_(i<l) s_i` |
| Three 3×3 convs vs one 7×7 | Same RF; `27C²` vs `49C²` params; 3× the nonlinearity |
| Max pool backward — where does gradient go? | Only to the argmax position |
| The degradation problem and what it proved | Deeper plain nets had higher **training** error — an optimization failure, not overfitting |
| Why is learning `F(x) → 0` easier than learning the identity? | Driving weights to zero is what gradient descent and weight decay do naturally |
| What does a 1×1 convolution do? | Mixes channels without mixing space — a per-pixel linear layer |
| Depthwise separable cost | `k²C + C·C_out` vs `k²·C·C_out` |
| Two rules for data augmentation | Never augment val/test; only use label-preserving transforms |
| Why do RNN gradients vanish worse than a deep MLP's? | The **same** matrix is repeated, so its largest eigenvalue governs exactly — no averaging |
| LSTM: `∂c_t/∂c_(t−1) = ?` | `f_t` — with the forget gate near 1, gradient passes through unchanged |
| What is the LSTM cell state, conceptually? | A residual/additive highway — the same idea as a ResNet, from 1997 |
| Why did transformers replace RNNs? | Parallel over time (decisive); no fixed bottleneck; better long-range |
| Attention formula | `softmax(QKᵀ/√d_k)·V` |
| `Var[q·k] = ?` and why divide by `√d_k` | `d_k`. Large spread saturates softmax → near-zero gradient |
| What do Q, K, V mean? | What I'm looking for / what I contain / what I contribute |
| Why `−∞` before softmax for masking? | So normalization covers only the allowed positions |
| Multi-head — why doesn't the parameter count grow? | You partition `d_model` into heads rather than adding capacity |
| Why does attention need positional encoding? | Attention is permutation-equivariant — it sees a set, not a sequence |
| RoPE's key identity | `(R(mθ)q)·(R(nθ)k) = qᵀR((n−m)θ)k` — depends only on relative distance |
| Pre-norm transformer block | `x ← x + MHA(LN(x))`; `x ← x + MLP(LN(x))` |
| Where do a transformer block's parameters live? | Attention `4d²`, MLP `8d²` — the MLP holds two-thirds |
| Attention complexity | `O(n²d)` time, `O(n²)` memory per head per layer |
| What does FlashAttention fix? What does GQA fix? | FlashAttention: memory traffic. GQA: KV cache size |

---

## D.12 Chapter 12 — GPT

| Front | Back |
|---|---|
| Why BPE rather than characters or words? | Characters make sequences too long; words explode the vocabulary and have OOV. BPE has neither |
| BPE training algorithm | Start from bytes; count adjacent pairs; merge the most frequent; repeat to target vocab size |
| Why does merge order matter when encoding? | Later merges were built from earlier ones; out of order gives a different, wrong tokenization |
| Three failures caused by tokenization | Arithmetic; character-level tasks; non-English cost and quality |
| GPT parameter count | `V·d + block·d + 12·L·d²` with weight tying |
| Weight tying — what and why | Input embedding = output projection. Saves `V·d` params (38.6M for GPT-2 small) and usually helps |
| Residual scaling — the factor and the reason | `1/√(2N)`; the residual stream accumulates `2N` contributions, so its variance would grow with depth |
| `C ≈ ?` for training FLOPs | `6·N·D` |
| Chinchilla ratio and what it corrected | ~20 tokens per parameter; models had been badly undertrained |
| bf16 vs fp16 | bf16 has fp32's exponent range — no overflow, no loss scaling needed |
| Memory per parameter for AdamW mixed precision | ~16 bytes — a 1B model needs ~16GB before activations |
| Expected initial LM loss | `ln(V)` — 10.8 for a 50k vocabulary |
| What does an LM loss plateau near 7 mean? | It learned unigram frequencies and nothing more |
| What does a KV cache save, and cost? | Turns `O(n²)` generation into `O(n)`; costs memory that can exceed the model |
| KV cache gotcha with `is_causal` | With a cache, the new token legitimately attends to all cached positions — don't re-apply the mask |

---

## D.13 Chapters 13–15 — Research practice

| Front | Back |
|---|---|
| The three passes, and the question each answers | 1 (10 min): what's the claim, do I care? 2 (1 hr): what's the method, does evidence support it? 3 (hours): could I have written this? |
| Reading order within a paper | Abstract → Figure 1 → Tables → Conclusion → Method → Experiments → Appendix |
| The most common flaw in ML papers | The baseline wasn't tuned as hard as the method |
| Five things papers systematically omit | LR schedule details; init; failed variants; seed count and variance; actual compute |
| Why reproduce the baseline first? | If your baseline doesn't match, nothing downstream means anything |
| The ablation ladder, rungs 0–5 | 0 reproduce; 1 + a new ablation; 2 new scale/data; 3 find where it breaks; 4 explain why; 5 fix it |
| Which rung is already novel? | Rung 1 — one ablation the authors didn't run |
| Five patterns that generate research questions | "They only tested on X"; mechanism claimed but not shown; "this should break when"; papers disagree; unjustified standard practice |
| DDP vs FSDP | DDP: full model per device, gradients all-reduced. FSDP: shards params, grads, optimizer states |
| The test for a good research question | Is it interesting **either way**? If only one outcome is publishable, you'll fool yourself |
| What is a negative control, and why does it matter? | Plant a known effect and confirm your setup detects it — otherwise "no effect" may just mean "no power" |
| Minimum seeds for a reported result | 3, preferably 5, with variance reported |
| Order to write a paper in | Figures → Method → Experiments → Related Work → Limitations → Intro → Abstract last |
| Why write figures first? | Making the plot forces you to know what your result actually is |
| How to tell a reviewer's misunderstanding from your writing problem | If two reviewers misunderstood the same thing, it's your writing |

---

## D.14 The derivation cards

These aren't recall cards — they're **do it** prompts. Put them in a separate deck and review monthly. The answer is always "go to a blank page and derive it."

| Front | Back |
|---|---|
| DERIVE: two stacked linear layers collapse to one | §2.8 — three lines |
| DERIVE: `∂L/∂x = Wᵀ∂L/∂y`, both by shapes and by the chain rule | §2.9 |
| DERIVE: `σ'(x) = σ(1−σ)` | §3.3 |
| DERIVE: the gradient points in the direction of steepest ascent | §3.8 — four lines via the dot product |
| DERIVE: `∂L/∂z = p − y` for sigmoid + BCE | §4.10 |
| DERIVE: MSE from maximum likelihood under Gaussian noise | §6.5 |
| DERIVE: `Var[y] = n_in·σ_w²·σ_x²`, and therefore `σ_w = 1/√n_in` | §7.2 |
| DERIVE: `E[relu(z)²] = σ²/2` | §7.3 |
| DERIVE: Adam's bias correction, `m_t = g(1−β₁ᵗ)` | §9.5 |
| DERIVE: `Var[q·k] = d_k`, and therefore the `√d_k` scaling | §11.6 |
| DERIVE: `∂c_t/∂c_(t−1) = f_t` for an LSTM | §11.3 |
| DERIVE: `∂y/∂x = 1 + ∂F/∂x` for a residual block, and why it fixes gradient flow | §10.9 |

**If you can do all twelve from a blank page, you understand this book.**

---

## D.15 Growing the deck

The imported cards cover what the book emphasizes. **Your own cards should cover what *you* keep forgetting** — and only you can know what that is.

Signals that something needs a card:

- You looked it up twice
- You got it wrong in a cold rebuild
- It's a "why," not just a "what" — those are the valuable ones
- It's a default value or a shape convention you keep second-guessing

Signals that a card should go:

- You've failed it 20 times — the underlying concept isn't there; go relearn it, then remake the card
- It's something you now derive in seconds
- It's code

**Target: ~10 new cards a day while learning, 15 minutes of review a day forever.**

---

*This is the end of the book.*
