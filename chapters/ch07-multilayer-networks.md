# Chapter 7 — Multilayer Networks

**Time: 12–14 days** (Weeks 12–14 of the plan)

**Prerequisite:** Chapter 5 checkpoint — you can write the autograd engine cold. Chapter 6's MLE derivations.

**What you'll be able to do at the end:** derive the correct initialization scale for any layer instead of copying it; explain vanishing and exploding gradients with the actual exponential argument; choose an activation function for a reason; implement LayerNorm and RMSNorm from scratch; and diagnose a network that won't train using a fixed protocol rather than guessing.

---

## 7.0 The gap between "runs" and "trains"

Your Chapter 5 engine is correct. It computes exact gradients. Yet if you build a 20-layer network with it and initialize the weights from `N(0,1)`, it will not train. The loss will sit flat or blow up to `nan`, and every gradient check will pass.

This is the single most disorienting experience in learning deep learning: **your code is right and the network still doesn't work.**

The gap is that correct gradients aren't sufficient. Signals have to survive twenty multiplications going forward and twenty going backward, and whether they survive is determined by things that look like arbitrary details — the scale you drew the weights from, which activation you picked, whether you normalized.

They aren't arbitrary. Each one has a derivation. This chapter is those derivations.

---

## 7.1 The architecture

An MLP with `L` layers:

```
h⁰ = x
hˡ = φ(Wˡ hˡ⁻¹ + bˡ)      for l = 1 … L−1
ŷ  = W^L h^(L−1) + b^L      (no activation on the output — §5.6)
```

- `Wˡ` has shape `(nˡ, nˡ⁻¹)` in math convention, `(nˡ⁻¹, nˡ)` in code convention (§2.10)
- `φ` is the activation function
- **Width** = units per layer. **Depth** = number of layers.

That's the whole architecture. Everything in this chapter is about making it trainable.

---

## 7.2 Why initialization is not a detail

Consider what happens to the variance of the signal as it moves through layers.

Take one layer `y = Wx`, with `n_in` inputs. Assume the weights are drawn independently with mean 0 and variance `σ_w²`, and the inputs are independent with variance `σ_x²`. Then each output is a sum of `n_in` independent products:

```
Var[yᵢ] = Var[ Σⱼ Wᵢⱼxⱼ ]
        = Σⱼ Var[Wᵢⱼxⱼ]           (independence — §6.3)
        = n_in · σ_w² · σ_x²
```

So each layer **multiplies the signal variance by `n_in · σ_w²`.**

Call that factor `g`. After `L` layers the variance has been multiplied by `g^L`.

| `g` | After 20 layers |
|---|---|
| 0.5 | `0.5²⁰ ≈ 10⁻⁶` — signal has vanished |
| 1.0 | unchanged — exactly what you want |
| 2.0 | `2²⁰ ≈ 10⁶` — signal has exploded |

**This is exponential in depth.** There is no learning rate that fixes it, no amount of training. Get `g` wrong and a deep network is dead before the first step.

The fix is to force `g = 1`:

```
n_in · σ_w² = 1    ⟹    σ_w² = 1/n_in    ⟹    σ_w = 1/√n_in
```

**That's where the `1/√n` you saw in Chapter 5's `Neuron.__init__` comes from.** It isn't a convention; it's the unique scale that preserves signal variance through a layer.

---

## 7.3 Xavier and He initialization

### The backward pass wants something different

Run the same analysis on gradients. From §2.9, `∂L/∂x = Wᵀ ∂L/∂y`, and by identical reasoning the gradient variance is multiplied by `n_out · σ_w²`. So the backward pass wants `σ_w² = 1/n_out`.

Forward wants `1/n_in`; backward wants `1/n_out`. Unless the layer is square you can't have both.

**Xavier/Glorot initialization** splits the difference with the harmonic-mean-style compromise:

```
σ_w² = 2 / (n_in + n_out)
```

This was derived for `tanh` and sigmoid networks, which are (approximately) linear near zero — which is what the analysis above assumed.

### ReLU changes the arithmetic

ReLU zeroes every negative input. If `z ~ 𝒩(0, σ²)` then

```
E[relu(z)²] = ∫₀^∞ z² p(z) dz = σ²/2
```

**Exactly half the variance is destroyed.** So to preserve variance you need twice as much:

```
σ_w² = 2/n_in            ← He initialization
```

**Use He for ReLU-family activations, Xavier for tanh/sigmoid.** That's the whole rule, and now you can derive it rather than remember it.

```python
import numpy as np

def he_normal(n_in, n_out, rng):
    """For ReLU and friends."""
    return rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)

def xavier_normal(n_in, n_out, rng):
    """For tanh / sigmoid."""
    return rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / (n_in + n_out))

def xavier_uniform(n_in, n_out, rng):
    limit = np.sqrt(6.0 / (n_in + n_out))     # matches the variance above
    return rng.uniform(-limit, limit, (n_in, n_out))
```

### Biases

Initialize to **zero**. There's no symmetry problem — the weights already break symmetry — and any nonzero bias just shifts the pre-activations off center for no reason.

### Why not initialize weights to zero?

Because every unit in a layer would compute the same thing, receive the same gradient, and update identically. They'd stay identical forever. A 512-unit layer would behave as one unit.

**Random initialization exists to break symmetry.** The scale exists to preserve variance. Two different jobs, often conflated.

**Verify all of this yourself** — exercise 2. Push a signal through 30 layers at three different initialization scales and print the variance per layer. Watching `1e-6` and `1e+6` appear in your own output makes this permanent in a way reading cannot.

---

## 7.4 Vanishing and exploding gradients

Same exponential argument, applied backward, with the activation included.

The gradient at layer `l` involves a product of Jacobians from the output back to `l`. Each layer contributes roughly a factor of `Wᵀ` times the activation derivative `φ'`. So the gradient magnitude scales like

```
∏ (typical singular value of W) × (typical φ')
```

Multiply `L` numbers together and the result is exponential in `L`. Slightly under 1 → vanishing. Slightly over 1 → exploding.

### The sigmoid disaster, quantified

From §3.3, `σ'(x) = σ(1−σ)`, whose **maximum value is 0.25** (at `x = 0`), and it's much smaller away from zero.

So even in the best case, every sigmoid layer multiplies the gradient by at most 0.25:

| Depth | Best-case gradient scale |
|---|---|
| 5 | `0.25⁵ ≈ 10⁻³` |
| 10 | `0.25¹⁰ ≈ 10⁻⁶` |
| 20 | `0.25²⁰ ≈ 10⁻¹²` |

**A 20-layer sigmoid network's early layers receive gradients around `10⁻¹²`.** They never learn anything. This is not a tuning problem; it's arithmetic.

This is the concrete reason deep networks were considered untrainable before roughly 2010, and why the three fixes that broke the impasse — **ReLU** (`φ' = 1`, no decay), **careful initialization** (§7.3), and **residual connections** (a path with derivative exactly 1) — mattered so much.

### Exploding gradients

The opposite failure: gradients grow, weights take enormous steps, loss goes to `nan`. Common in RNNs and in early transformer training.

The standard fix is **gradient clipping** — rescale the whole gradient if its norm exceeds a threshold:

```python
def clip_grad_norm(grads, max_norm=1.0):
    total = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        grads = [g * scale for g in grads]
    return grads, total
```

**Clip the global norm, not each parameter individually.** Per-parameter clipping distorts the gradient *direction*; global clipping only shortens it, leaving the direction intact. Every serious training script uses the global version.

---

## 7.5 Activation functions

| Name | `φ(x)` | `φ'(x)` | Notes |
|---|---|---|---|
| Sigmoid | `1/(1+e⁻ˣ)` | `σ(1−σ)`, max 0.25 | Saturates badly. Output layers only. |
| Tanh | `tanh(x)` | `1 − tanh²`, max 1.0 | Zero-centered. Still saturates. |
| **ReLU** | `max(0,x)` | `1` or `0` | Cheap, no positive saturation. Can die. |
| LeakyReLU | `max(αx, x)`, `α≈0.01` | `1` or `α` | Fixes dead units. |
| ELU | `x` or `α(eˣ−1)` | smooth | Smooth, negative saturation. |
| **GELU** | `x·Φ(x)` | smooth | Transformer standard. |
| **SiLU/Swish** | `x·σ(x)` | smooth | Modern LLM standard. |

### Why zero-centered output matters

Sigmoid outputs are always positive. That means every weight into the next layer receives a gradient of the same sign, so the entire weight vector can only move in one diagonal direction at a time — producing a characteristic zigzag. Tanh fixes this; it's one of the reasons tanh beat sigmoid historically.

### Dead ReLU

If a unit's pre-activation is negative for **every** input in the dataset, its output is always 0, its gradient is always 0, and its weights never change again. It's permanently dead — it contributes nothing for the rest of training.

Causes: a learning rate high enough to knock the bias far negative in one step; bad initialization.

Fixes: LeakyReLU (gradient `α` instead of 0 on the negative side), lower learning rate, proper He init.

**Monitor this.** The fraction of dead units per layer is a genuinely useful diagnostic and almost nobody logs it:

```python
def dead_fraction(activations):
    """Fraction of units that are zero for every example in the batch."""
    return float(np.mean(np.all(activations == 0, axis=0)))
```

Above ~20% dead in a layer means something's wrong.

### GELU and SiLU

Both are smooth approximations to ReLU that allow small negative outputs.

```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def silu(x):
    return x / (1 + np.exp(-x))
```

GELU has a probabilistic story — `x·Φ(x)` is `x` times the probability that a standard Gaussian is below `x`, so it's a "soft" gate. Honestly, the main reason both are used is empirical: they work slightly better, and the smoothness helps optimization. Modern LLMs mostly use SiLU inside a gated variant (SwiGLU). **Don't over-theorize activation choice** — the differences are small and the field settled on these by experiment.

**Default advice:** ReLU when you're building something from scratch and want to understand it. GELU/SiLU when you're matching a modern architecture.

---

## 7.6 Normalization

Even with good initialization, activation statistics **drift** during training as weights change. Normalization forces them back to a controlled range at every step.

### Batch normalization

Normalize each feature across the batch:

```
μ = mean over batch,   σ² = variance over batch
x̂ = (x − μ)/√(σ² + ε)
y  = γ·x̂ + β           γ, β learnable
```

The learnable `γ` and `β` matter: without them, normalization would remove the network's ability to represent non-zero-mean, non-unit-variance features. They let the network undo the normalization if that's what it needs.

**BatchNorm has real problems:**

1. **Batch-size dependent.** With batch size 2 the statistics are garbage.
2. **Train/eval mismatch.** At inference there's no batch, so you use running averages accumulated during training. Forgetting to switch modes is one of the most common bugs in applied deep learning.
3. **Awkward for sequences.** Variable lengths make batch statistics ill-defined.
4. **Awkward when distributed.** Statistics must be synchronized across devices.

### Layer normalization

Normalize across **features**, per example:

```
μᵢ = mean over features of example i
σᵢ² = variance over features of example i
x̂ᵢ = (xᵢ − μᵢ)/√(σᵢ² + ε)
yᵢ = γ·x̂ᵢ + β
```

**No batch dependence at all.** Same behaviour at train and eval, works with batch size 1, works with variable-length sequences, needs no cross-device synchronization.

**This is why every transformer uses LayerNorm.** It's not that it normalizes better — it's that it removes the batch from the equation.

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """x:(batch, features) — normalize each row independently."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta
```

Note `axis=-1` — the feature axis — versus BatchNorm's `axis=0`. **That one character is the entire difference between the two methods.**

### RMSNorm

Modern LLMs (Llama and most of what followed) use a simplification: skip the mean subtraction entirely.

```
RMS(x) = √(mean(x²))
y = γ · x/RMS(x)
```

```python
def rms_norm(x, gamma, eps=1e-6):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return gamma * x / rms
```

Cheaper (no mean, no `β`), and empirically just as good. The re-centering turns out not to have been doing much work.

### Pre-norm vs post-norm

Where you put the normalization inside a residual block matters a lot:

```
post-norm:   x ← LayerNorm(x + Sublayer(x))       original transformer
pre-norm:    x ← x + Sublayer(LayerNorm(x))       modern standard
```

Pre-norm keeps a **clean, unnormalized residual path** from input to output, so gradients flow straight through without being rescaled at every layer. Post-norm networks deeper than about 12 layers need careful learning-rate warmup to train at all; pre-norm ones mostly just work.

You'll implement both in Chapter 12. Remember this section when you do.

---

## 7.7 Residual connections (preview)

```
y = x + F(x)
```

Instead of learning the output, learn the *change* to the input.

Why this fixes vanishing gradients, in one line: `∂y/∂x = 1 + ∂F/∂x`. There is a path with derivative exactly **1**, so the gradient can flow from the loss to any layer without being multiplied by anything small.

That's the whole idea, and it's the reason 100+ layer networks became trainable. Full treatment in Chapter 10.

---

## 7.8 Depth versus width

**Universal approximation theorem:** a network with a single hidden layer and enough units can approximate any continuous function on a compact domain to arbitrary accuracy.

**What it does not say**, and this matters:

1. **How many units.** "Enough" can mean exponentially many.
2. **That you can find the weights.** It's an existence result, silent about optimization.
3. **That it generalizes.** Fitting the training data isn't the goal.

So the theorem is reassuring and nearly useless for practice.

**The practically important fact is that depth is exponentially more efficient than width** for many function classes. A function representable by a deep network with `O(n)` units may need `O(2ⁿ)` units in a shallow one. Depth composes features hierarchically — edges into shapes into objects — and reuses intermediate results, which is exactly what shallow networks can't do.

**Practical guidance:** start deeper and narrower rather than shallow and wide. Add depth until optimization becomes difficult, then add residual connections and normalization to keep going.

---

## 7.9 The debugging protocol

This is the most immediately useful section in the chapter. When a network won't train, **run these in order.** Don't skip ahead to hyperparameter tuning — that's the last resort, not the first.

### Step 1 — Check the initial loss

Before any training, a randomly initialized `k`-class classifier should predict roughly uniformly, giving

```
initial loss ≈ ln(k)
```

For 10 classes: `ln(10) ≈ 2.30`. For 2 classes: `ln(2) ≈ 0.69`.

**If your initial loss is far from this, stop.** It means your loss function is wrong, your initialization is broken, or your labels are misaligned. This ten-second check catches an enormous class of bugs before you waste an hour.

### Step 2 — Overfit 10 examples

From §4.15. Disable regularization, take 10 examples, train until the loss is ~0. If you can't, the bug is in your code, not your hyperparameters.

### Step 3 — Gradient check

Your Chapter 3 tool, on a small version of the network.

### Step 4 — Log activation statistics per layer

```python
def activation_report(activations_by_layer):
    print(f"{'layer':>6} {'mean':>9} {'std':>9} {'dead%':>7}")
    for i, a in enumerate(activations_by_layer):
        dead = 100 * np.mean(np.all(a == 0, axis=0))
        print(f"{i:>6} {a.mean():>9.4f} {a.std():>9.4f} {dead:>7.1f}")
```

**What you want:** std roughly constant across layers, near 1. Mean near 0. Dead fraction low.

**What the failures look like:**

- std shrinking layer by layer → initialization too small, or saturating activations
- std growing → initialization too large
- dead% climbing → learning rate too high, or ReLU with bad init
- std ≈ 0 in a deep layer → signal has vanished; nothing downstream can learn

### Step 5 — Log gradient norms per layer

```python
def gradient_report(grads_by_layer):
    for i, g in enumerate(grads_by_layer):
        print(f"layer {i:>3}  ‖grad‖ = {np.linalg.norm(g):.3e}")
```

Gradient norms should be within an order of magnitude or two of each other. A steady decay from last layer to first is vanishing gradients, visible directly.

### Step 6 — Learning-rate sweep

Only now. Try `1e-1, 1e-2, 1e-3, 1e-4` for a few hundred steps each and plot all four loss curves. The best one usually sits just below the largest rate that doesn't diverge.

### Step 7 — Add regularization back

Get it training first. Then add weight decay, dropout, augmentation. **Never debug with regularization on** — it masks the symptoms you're trying to read.

---

## 7.10 Exercises

**1.** Derive `Var[y] = n_in · σ_w² · σ_x²` on paper. State every independence assumption you used.

**2.** **The initialization experiment.** Build a 30-layer linear network (no activation). Push a random input through it with weights initialized at scale `0.5/√n`, `1/√n`, and `2/√n`. Print the activation std at every layer. Plot all three on a log scale. Explain the three curves in one sentence each.

**3.** Repeat exercise 2 with tanh, and again with ReLU, using both Xavier and He init on each. Which pairing preserves variance? Confirm the ReLU/He result matches the `√(2/n)` derivation.

**4.** Prove that `E[relu(z)²] = σ²/2` for `z ~ 𝒩(0, σ²)`. (Hint: symmetry — the integral over the positive half is half the total.) Verify numerically.

**5.** Implement `he_normal`, `xavier_normal`, `xavier_uniform`. Verify each produces the intended variance empirically.

**6.** Initialize a network with all-zero weights and train it. Show that every unit in a layer remains identical forever. Print the weight matrix to demonstrate it.

**7.** Plot sigmoid, tanh, ReLU, LeakyReLU, GELU and SiLU, and their derivatives, on the same axes over `[−5, 5]`. Mark the maximum of each derivative.

**8.** Build a 15-layer sigmoid network. Log the gradient norm at every layer after one backward pass. Plot on a log scale. Compare the observed decay against the predicted `0.25^L`.

**9.** **Dead ReLU demonstration.** Train a ReLU network with a deliberately excessive learning rate. Log the dead fraction per layer per epoch. Show units dying and never recovering. Then rerun with LeakyReLU and compare.

**10.** Implement `layer_norm` and `rms_norm` from scratch, with gradients, in your Chapter 5 tensor engine. Gradient-check both.

**11.** Implement `batch_norm` with train and eval modes and running statistics. Demonstrate the failure: train with batch size 64, then evaluate in *training* mode with batch size 2, and show the accuracy collapse.

**12.** Train the same MLP three ways — no normalization, LayerNorm, BatchNorm — at depth 20. Plot all three loss curves.

**13.** Implement `clip_grad_norm` with global norm. Show that per-parameter clipping changes the gradient direction while global clipping does not (compute the cosine similarity to the unclipped gradient in each case).

**14.** **Depth vs width.** With a fixed parameter budget (~50,000 parameters), train networks shaped `[512]`, `[128,128]`, `[64,64,64]`, `[32]×6`. Compare final validation accuracy. Then repeat at ~500,000 parameters. What changes?

**15.** **Chapter project.** Train an MLP on MNIST (`sklearn.datasets.fetch_openml('mnist_784')` or the Keras loader — **data loading only**) using your own tensor autograd engine. Requirements: He init, ReLU, LayerNorm, gradient clipping, minibatch training, the full §7.9 diagnostic suite logged every epoch, and a written analysis of what the activation and gradient statistics showed. Target: >97% test accuracy. Write it up.

---

## 7.11 Solutions

<details>
<summary>Open only after attempting</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)


# --- 1 ---
# Var[yᵢ] = Var[Σⱼ Wᵢⱼxⱼ]. Assumptions: (a) W and x independent of each other,
# (b) entries of W iid with mean 0, (c) entries of x iid, (d) all cross terms
# vanish because E[W]=0. Then Var of a sum of independent terms is the sum of
# variances, and Var[Wx] = E[W²]E[x²] = σ_w²σ_x². Hence n_in·σ_w²·σ_x².


# --- 2, 3 ---
def propagate(scale_fn, act=lambda z: z, depth=30, n=256):
    x = rng.standard_normal((512, n))
    stds = []
    for _ in range(depth):
        W = rng.standard_normal((n, n)) * scale_fn(n)
        x = act(x @ W)
        stds.append(x.std())
    return stds

for label, s in [("0.5/√n", lambda n: 0.5/np.sqrt(n)),
                 ("1.0/√n", lambda n: 1.0/np.sqrt(n)),
                 ("2.0/√n", lambda n: 2.0/np.sqrt(n))]:
    plt.semilogy(propagate(s), label=label)
plt.legend(); plt.xlabel("layer"); plt.ylabel("activation std"); plt.show()
# 0.5/√n: g=0.25, variance dies exponentially -> ~1e-9 by layer 30.
# 1.0/√n: g=1, variance preserved -> flat. This is the whole point.
# 2.0/√n: g=4, variance explodes -> ~1e9 by layer 30.

relu = lambda z: np.maximum(z, 0)
print("relu + xavier:", propagate(lambda n: np.sqrt(2/(2*n)), relu)[-1])   # decays
print("relu + he    :", propagate(lambda n: np.sqrt(2/n), relu)[-1])       # ~stable
print("tanh + xavier:", propagate(lambda n: np.sqrt(2/(2*n)), np.tanh)[-1])


# --- 4 ---
# z ~ N(0,σ²). E[relu(z)²] = ∫₀^∞ z²p(z)dz. Since z²p(z) is symmetric about 0,
# the half-line integral is exactly half the full one, = E[z²]/2 = σ²/2. ∎
z = rng.standard_normal(2_000_000) * 3.0
print(np.mean(np.maximum(z,0)**2), 9.0/2)          # ≈ 4.5


# --- 6 ---
W = np.zeros((4, 3)); b = np.zeros(3)
X = rng.standard_normal((20, 4)); y = rng.standard_normal(20)
for _ in range(50):
    h = np.maximum(X @ W + b, 0)
    d = np.ones_like(h) * (h > 0)
    W -= 0.01 * X.T @ d; b -= 0.01 * d.sum(0)
print(W)     # all three columns identical, forever — symmetry never breaks


# --- 7, 8 ---
x = np.linspace(-5, 5, 400)
sig = 1/(1+np.exp(-x))
acts = {"sigmoid": (sig, sig*(1-sig)),
        "tanh":    (np.tanh(x), 1-np.tanh(x)**2),
        "relu":    (np.maximum(x,0), (x>0).astype(float)),
        "leaky":   (np.where(x>0,x,.01*x), np.where(x>0,1,.01)),
        "gelu":    (0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3))), None),
        "silu":    (x*sig, None)}
for k,(f,d) in acts.items():
    plt.plot(x, f, label=k)
plt.legend(); plt.show()
print("max sigmoid' =", (sig*(1-sig)).max())    # 0.25

def sigmoid_depth_grads(L=15, n=64):
    Ws = [rng.standard_normal((n,n))*np.sqrt(2/(2*n)) for _ in range(L)]
    h = rng.standard_normal((128,n)); hs=[h]
    for W in Ws:
        h = 1/(1+np.exp(-(h@W))); hs.append(h)
    g = np.ones_like(h)/h.size; norms=[]
    for l in range(L-1,-1,-1):
        g = g * hs[l+1]*(1-hs[l+1])
        norms.append(np.linalg.norm(g)); g = g @ Ws[l].T
    return norms[::-1]

n = sigmoid_depth_grads()
plt.semilogy(n, "o-"); plt.semilogy([n[-1]*0.25**(len(n)-1-i) for i in range(len(n))], "--")
plt.show()      # observed decay tracks the 0.25^L prediction closely


# --- 10 ---
def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return g * (x-mu)/np.sqrt(var+eps) + b

def rms_norm(x, g, eps=1e-6):
    return g * x / np.sqrt((x**2).mean(-1, keepdims=True) + eps)

xa = rng.standard_normal((4, 8))
print(layer_norm(xa, 1., 0.).mean(-1).round(9))   # ~0
print(layer_norm(xa, 1., 0.).std(-1).round(4))    # ~1


# --- 13 ---
gs = [rng.standard_normal((10,10))*10 for _ in range(3)]
flat = np.concatenate([g.ravel() for g in gs])

def global_clip(gs, m=1.0):
    t = np.sqrt(sum((g**2).sum() for g in gs))
    return [g*(m/(t+1e-6)) for g in gs] if t>m else gs

def per_param_clip(gs, m=1.0):
    out=[]
    for g in gs:
        t=np.linalg.norm(g); out.append(g*(m/(t+1e-6)) if t>m else g)
    return out

def cos(a,b):
    a=np.concatenate([x.ravel() for x in a]); b=np.concatenate([x.ravel() for x in b])
    return a@b/(np.linalg.norm(a)*np.linalg.norm(b))

print("global  cos:", cos(gs, global_clip(gs)))      # exactly 1.0
print("per-param  :", cos(gs, per_param_clip(gs)))   # < 1 — direction changed
# Global clipping is a pure scalar rescale of the whole gradient, so direction
# is preserved exactly. Per-parameter clipping rescales blocks differently,
# which rotates the update away from the true gradient direction.


# --- 14 ---
# Fixed budget: deeper generally wins at moderate budgets, but past a depth the
# optimization gets harder without residuals/normalization and it reverses.
# At larger budgets the depth advantage widens, because depth buys hierarchical
# feature reuse that width cannot replicate (§7.8).
```

</details>

---

## 7.12 Chapter 7 checkpoint

Cold — blank file, no notes.

- [ ] **Derive the initialization scale** `σ_w = 1/√n_in` from the variance analysis. On paper. **10 minutes.**
- [ ] Explain why ReLU needs `√(2/n_in)` and prove the factor of 2.
- [ ] Explain vanishing gradients with the exponential argument, and compute the best-case gradient scale for a 20-layer sigmoid network.
- [ ] State the difference between BatchNorm and LayerNorm in one sentence, and give two concrete reasons transformers use LayerNorm.
- [ ] Implement `layer_norm` and `rms_norm` from scratch. **Target: 10 minutes.**
- [ ] Explain why global gradient clipping is preferred to per-parameter clipping.
- [ ] **Recite the seven-step debugging protocol of §7.9 in order.**
- [ ] Given a 10-class classifier, state the expected initial loss and what it means if the real one is 5.0.

Item 7 is the one you'll use most. Item 1 is the one that proves you understand the chapter.

### Anki cards

- Variance through a layer: `Var[y] = ?`
- Init scale for tanh (Xavier), for ReLU (He) — and the derivation of each
- Why does ReLU need a factor of 2?
- Why not initialize weights to zero?
- Max value of `σ'` and the 20-layer consequence
- Three historical fixes for vanishing gradients
- BatchNorm vs LayerNorm — which axis, and why transformers use LayerNorm
- RMSNorm — what's dropped and why it's fine
- Pre-norm vs post-norm
- Residual: `∂y/∂x = ?` and why that fixes gradient flow
- Dead ReLU: cause and fix
- Expected initial loss for `k` classes
- The seven debugging steps
- Global vs per-parameter gradient clipping

### Deliverables

```
nn/init.py           he_normal, xavier_normal, xavier_uniform
nn/activations.py    all six, with derivatives
nn/norm.py           layer_norm, rms_norm, batch_norm (train/eval)
nn/utils.py          clip_grad_norm, activation_report, gradient_report
projects/mnist.py    exercise 15
reports/mnist.md     the diagnostic write-up
```

```bash
git add .
git commit -m "Chapter 7: initialization, activations, normalization, MNIST >97%"
git push
```

### Write-up

700 words: **"Why your deep network doesn't train."** Lead with the variance-propagation derivation, include your log-scale plot from exercise 2 showing the three initialization regimes, the sigmoid gradient decay from exercise 8, and close with the seven-step protocol.

This is the most *practically useful* post you'll have written. Almost every explanation of initialization online states `He init = √(2/n)` without deriving it, and states "vanishing gradients" without computing `0.25²⁰`. Yours will do both.

**You can now build networks that actually train.** Chapter 8 hands you PyTorch — and because you built all of this yourself, PyTorch will read as a set of things you recognize rather than a set of things you memorize.

---

*Next: Chapter 8 — PyTorch*
