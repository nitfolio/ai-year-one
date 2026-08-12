# Chapter 9 — Optimization

**Time: 10–12 days** (Weeks 17–19 of the plan)

**Prerequisite:** Chapter 8's trainer working. You'll also need §2.13 (eigenvalues, condition number) and exercise 3.10 (the ravine plot).

**What you'll be able to do at the end:** implement every major optimizer from scratch and explain what problem each one solves; derive Adam's bias correction; explain why AdamW replaced Adam; know why transformers need learning-rate warmup; and choose hyperparameters from reasoning rather than copying.

---

## 9.0 The problem

Back in exercise 3.10 you ran gradient descent on `f(x,y) = x² + 100y²` and plotted a path that zigzagged violently across a narrow valley while creeping along it. That picture is the whole motivation for this chapter.

The cause was §2.13: **curvature differs enormously by direction.** The Hessian's eigenvalues were 2 and 200, a condition number of 100. Your learning rate is capped by the steepest direction (or you diverge), so the shallow direction — which is the one you actually need to travel along — progresses 100× too slowly.

Real loss surfaces are far worse. Condition numbers in the thousands are routine. Plain gradient descent, applied naively, is unusably slow.

Everything in this chapter is a response to that one problem.

---

## 9.1 What the landscape looks like

Some facts about high-dimensional loss surfaces that correct common intuitions:

**Local minima are not the main problem.** In `N` dimensions, a critical point is a local minimum only if *all* `N` Hessian eigenvalues are positive. If signs were roughly independent, that's about `2⁻ᴺ` — vanishingly unlikely for large `N`.

**Saddle points are the main problem.** Almost every critical point in high dimensions is a saddle: a minimum in some directions, a maximum in others. They're surrounded by large near-flat regions where the gradient is tiny and progress stalls.

**Gradient noise helps.** This is why minibatch SGD often *generalizes better* than full-batch: the noise knocks you off saddles and out of sharp minima. Noise isn't merely tolerated — it does useful work.

**Flat minima appear to generalize better than sharp ones.** A minimum in a wide basin is robust to small parameter perturbations, which correlates with robustness to distribution shift. This is an empirical regularity with contested theory — hold it loosely, but it explains why several techniques that add noise also improve test accuracy.

---

## 9.2 SGD and where it fails

```python
def sgd(params, grads, lr):
    for p, g in zip(params, grads):
        p -= lr * g
```

Three failure modes:

1. **Ravines** — zigzagging, as above. Slow along the direction that matters.
2. **Saddles and plateaus** — near-zero gradient means near-zero step. You stall.
3. **One global learning rate** — but parameters differ wildly in gradient scale. A rate that suits the embedding layer may be far too large for the output layer.

Momentum addresses 1 and 2. Adaptive methods address 3.

---

## 9.3 Momentum

The fix is to accumulate a velocity instead of stepping on the raw gradient:

```
v ← βv + g
θ ← θ − ηv
```

with `β ≈ 0.9`.

### Why it fixes the ravine

Look at what happens in each direction separately.

**Across the ravine** (the steep, oscillating direction): successive gradients point in *opposite* directions. In the sum `βv + g`, they partially cancel. Oscillation is damped.

**Along the ravine** (the shallow, consistent direction): successive gradients point the *same* way. They accumulate.

Momentum suppresses the direction you're wasting steps on and amplifies the direction you want. That's exactly the right response to a high condition number.

### How much amplification?

Suppose the gradient is constant at `g`. The velocity is a geometric series:

```
v = g + βg + β²g + ... = g/(1 − β)
```

With `β = 0.9`, that's **10× the plain gradient step** in any consistently-pointed direction. With `β = 0.99`, 100×.

This is why raising momentum usually means lowering the learning rate — the effective step size is `η/(1−β)`, not `η`. People who bump momentum from 0.9 to 0.99 and keep `η` fixed are quietly multiplying their step size by ten, and then wondering why the loss diverged.

### The physical picture

`v` is velocity; `g` is a force; `β` is friction. A ball rolling down the surface carries momentum through flat regions and small bumps instead of stopping in them. That's also how it escapes saddle points — it doesn't need a gradient to keep moving.

```python
class SGDMomentum:
    def __init__(self, params, lr=0.01, beta=0.9):
        self.params, self.lr, self.beta = params, lr, beta
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.v[i] = self.beta * self.v[i] + g
            p -= self.lr * self.v[i]
```

### Nesterov momentum

A refinement: evaluate the gradient at where momentum is *about to take you*, rather than where you are.

```
θ_lookahead = θ − ηβv
g = ∇L(θ_lookahead)
v ← βv + g
θ ← θ − ηv
```

The intuition is that this lets the step "see" that it's about to overshoot and correct in advance. It gives a better theoretical convergence rate for convex problems and a small, real improvement in practice.

---

## 9.4 Adaptive learning rates

Different parameters need different step sizes. Momentum doesn't address that. Adaptive methods do, by tracking each parameter's gradient history.

### AdaGrad — the idea, and its fatal flaw

Accumulate squared gradients per parameter, and divide by their root:

```
G ← G + g²
θ ← θ − η·g/(√G + ε)
```

Parameters with historically large gradients get smaller steps; rarely-updated ones get larger steps. Good for sparse features.

**The flaw:** `G` only ever grows. So the effective learning rate decays monotonically to zero and training halts prematurely. AdaGrad works for convex problems and dies on deep networks.

### RMSProp — the fix

Replace the running sum with an exponential moving average:

```
G ← ρG + (1 − ρ)g²
θ ← θ − η·g/(√G + ε)
```

with `ρ ≈ 0.9`. Now `G` reflects *recent* gradient magnitude and can shrink again. One-character conceptual change; entirely different behaviour.

```python
class RMSProp:
    def __init__(self, params, lr=1e-3, rho=0.9, eps=1e-8):
        self.params, self.lr, self.rho, self.eps = params, lr, rho, eps
        self.G = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.G[i] = self.rho * self.G[i] + (1 - self.rho) * g**2
            p -= self.lr * g / (np.sqrt(self.G[i]) + self.eps)
```

---

## 9.5 Adam

Momentum solves ravines and saddles. RMSProp solves per-parameter scaling. **Adam is both at once**, plus one correction.

```
m ← β₁m + (1 − β₁)g          first moment  (mean of gradients)
v ← β₂v + (1 − β₂)g²         second moment (mean of squared gradients)

m̂ = m / (1 − β₁ᵗ)            bias correction
v̂ = v / (1 − β₂ᵗ)

θ ← θ − η·m̂/(√v̂ + ε)
```

Defaults: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`.

### The bias correction, derived

This is the part everyone skips, and it's a clean derivation worth doing.

`m` starts at zero. Suppose the gradient is constant at `g`. Then after `t` steps:

```
m_t = (1−β₁) Σᵢ₌₁ᵗ β₁^(t−i) g
    = (1−β₁) g · (1 + β₁ + β₁² + ... + β₁^(t−1))
    = (1−β₁) g · (1 − β₁ᵗ)/(1 − β₁)
    = g(1 − β₁ᵗ)
```

So `m_t` estimates `g` but is scaled down by `(1 − β₁ᵗ)`.

**At `t = 1` with `β₁ = 0.9`, that factor is 0.1** — the estimate is ten times too small. The first steps would be ten times too short, and with `β₂ = 0.999` the second moment is a *thousand* times too small, making `√v̂` far too small and the step far too *large*.

Dividing by `(1 − β₁ᵗ)` and `(1 − β₂ᵗ)` removes the bias exactly. As `t` grows, both correction factors approach 1 and the correction fades out on its own. ∎

**Without bias correction, Adam takes wildly wrong steps for its first few hundred updates**, which is precisely when a network is most fragile.

```python
class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr = params, lr
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g**2
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

### What Adam is really doing

`√v̂` estimates the typical magnitude of the gradient for that parameter, so `m̂/√v̂` is roughly a **unit-scale step regardless of gradient magnitude.** That's why Adam is so insensitive to learning rate compared to SGD: the raw gradient scale has been divided out.

It's also a crude diagonal approximation to second-order information — using per-parameter gradient variance as a stand-in for curvature. Cheap, and good enough.

---

## 9.6 AdamW

Adam plus L2 regularization behaves badly, for a specific and understandable reason.

**With L2 regularization**, you add `λθ` to the gradient. Adam then divides everything by `√v̂`:

```
θ ← θ − η·(m̂ + λθ)/(√v̂ + ε)
```

So the decay term also gets divided by `√v̂`. **Parameters with large gradients receive less weight decay.** That is backwards — those are typically the ones you'd most want constrained. The regularization strength ends up coupled to gradient magnitude, which nobody intended.

**AdamW decouples them**: apply decay directly to the parameter, outside the adaptive scaling.

```
θ ← θ − η·m̂/(√v̂ + ε) − ηλθ
```

Now every parameter is decayed at the same rate, independent of its gradient history.

```python
class AdamW(Adam):
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
        super().__init__(params, lr, b1, b2, eps)
        self.wd = wd

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g**2
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * p)
```

**This is why `AdamW` is the default in every modern training script**, and why PyTorch's `Adam(weight_decay=...)` is not the same thing as `AdamW`. Don't confuse them; the difference shows up in final accuracy.

Remember §8.6: exclude biases and normalization parameters from decay.

---

## 9.7 Learning rate schedules

A constant learning rate is almost never optimal. You want large steps early (cover ground) and small steps late (settle precisely).

### Step decay

```python
def step_decay(lr0, epoch, drop=0.1, every=30):
    return lr0 * (drop ** (epoch // every))
```

Simple, historically standard for vision. Produces the characteristic staircase loss curve.

### Cosine annealing

```python
def cosine(lr_max, step, total, lr_min=0.0):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * step / total))
```

Smooth decay from `lr_max` to `lr_min`. **The modern default.** No tuning of drop points, no discontinuities.

### Warmup

Start near zero and ramp up linearly over the first few hundred to few thousand steps:

```python
def warmup_cosine(step, warmup, total, lr_max, lr_min=0.0):
    if step < warmup:
        return lr_max * step / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * prog))
```

**Warmup + cosine is the standard schedule for training transformers.** You'll use it in Chapter 12.

### Why warmup is necessary

Three reasons, all real:

1. **Adam's moment estimates are unreliable early.** With `β₂ = 0.999`, `v` averages over roughly the last 1,000 gradients — but at step 5 it has seen five. `√v̂` is a noisy estimate, so the adaptive step size has enormous variance. Warmup keeps steps small until the estimate stabilizes.

2. **Early gradients are large and the landscape is badly conditioned.** A full-size step from a random initialization can land somewhere much worse.

3. **Post-norm transformers are unstable at depth.** Large early updates disrupt the residual stream (§7.6). Warmup was originally introduced for exactly this, and pre-norm architectures reduced but did not eliminate the need.

**If a transformer training run diverges in the first few hundred steps, insufficient warmup is the first thing to check.**

---

## 9.8 Choosing hyperparameters

### Sensible defaults

| Setting | Optimizer | LR | Other |
|---|---|---|---|
| Transformers / LLMs | AdamW | `3e-4` (small), `1e-4`–`6e-4` (larger) | `β₂=0.95`, `wd=0.1`, warmup + cosine, clip 1.0 |
| Vision (CNNs) | SGD + momentum 0.9 | `0.1` | `wd=5e-4`, cosine or step decay |
| Fine-tuning | AdamW | `1e-5`–`5e-5` | short warmup, low or no decay |
| "I have no idea" | AdamW | `3e-4` | it works surprisingly often |

Note `β₂ = 0.95` rather than `0.999` for large language models — shorter memory for the second moment gives faster adaptation and is more stable at scale.

### The LR range test

Rather than guessing: increase the learning rate exponentially over a few hundred steps and plot loss against LR.

```python
def lr_range_test(model, loader, loss_fn, lr_min=1e-7, lr_max=10, steps=200):
    lrs = np.logspace(np.log10(lr_min), np.log10(lr_max), steps)
    losses = []
    for lr, (xb, yb) in zip(lrs, itertools.cycle(loader)):
        for gparam in opt.param_groups:
            gparam["lr"] = lr
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
        losses.append(loss.item())
    return lrs, losses
```

Plot on a log-x axis. The loss falls, reaches a minimum, then explodes. **Pick roughly an order of magnitude below where it explodes** — around the steepest descent point, not the minimum.

This takes two minutes and beats hours of manual guessing.

### Tuning order

When time is limited, tune in this order — it's roughly the order of impact:

1. **Learning rate.** Dominates everything else.
2. **Batch size** (subject to memory).
3. **Schedule** — warmup length and decay shape.
4. **Weight decay.**
5. **Everything else.** `β₁`, `β₂`, `ε` almost never need touching.

**Change one thing at a time, and change it by a factor of 3 or 10, not by 10%.** A 10% change is within run-to-run noise; you'll fool yourself.

---

## 9.9 Batch size and the scaling rule

From §6.3, gradient noise scales as `1/√B`. Larger batches mean cleaner gradients, which means you can afford larger steps.

**The linear scaling rule:** multiply batch size by `k`, multiply learning rate by `k`.

It works well up to a point, then breaks. Beyond some critical batch size, `√k` scaling fits better, and past that, larger batches stop helping at all — you're spending compute for no gain. The critical size depends on the task and grows during training.

**Practically:** use the largest batch that fits in memory, apply linear scaling, and add warmup — large-batch runs are more unstable at the start.

---

## 9.10 Why not second-order methods?

Newton's method uses curvature directly:

```
θ ← θ − H⁻¹g
```

This is far better than gradient descent — it accounts for the ravine exactly and converges in very few steps on quadratic surfaces.

Nobody uses it for neural networks:

1. **`H` is `N × N`.** For 7 billion parameters that's 4.9 × 10¹⁹ entries. It cannot be stored, let alone inverted.
2. **Inversion is `O(N³)`.**
3. **Stochastic gradients make Hessian estimates extremely noisy.**
4. **Non-convexity.** Where the Hessian has negative eigenvalues, Newton's method walks *toward* the saddle.

Quasi-Newton methods (L-BFGS) approximate `H⁻¹` from gradient history. Still too expensive for large models, and they assume relatively clean gradients.

**Adam is the practical compromise:** a diagonal approximation to curvature, costing two extra numbers per parameter. That's the trade the field settled on.

Newer optimizers (Lion, Shampoo, Muon and others) revisit this territory with cleverer structure. Worth reading about later; AdamW remains the safe default.

---

## 9.11 Diagnosing optimization

| Symptom | Likely cause | Action |
|---|---|---|
| Loss flat from step 0 | LR too low; broken gradients | §7.9 protocol first, then raise LR ×10 |
| Loss `nan` immediately | LR too high; no warmup; overflow | Lower LR ×10; add warmup; clip |
| Loss falls then spikes | LR too high for the later landscape | Add or extend decay |
| Loss oscillates around a floor | LR too high late | Decay schedule |
| Loss plateaus mid-training | Saddle or plateau | More momentum; warm restart; verify it isn't just converged |
| Adam works, SGD doesn't | Bad conditioning | Fine — but try SGD + momentum + schedule for final accuracy |
| Train fine, val bad | **Not an optimization problem** | Regularization, more data — don't tune the optimizer |

That last row is worth internalizing. Optimization failures and generalization failures look different and need completely different responses. **Train loss tells you about optimization. The train–val gap tells you about generalization.** Tuning the optimizer to fix an overfitting problem is a common waste of a week.

---

## 9.12 Exercises

**1.** Implement `SGD`, `SGDMomentum`, `Nesterov`, `AdaGrad`, `RMSProp`, `Adam`, `AdamW` from scratch in NumPy. Common interface: `step(grads)`.

**2.** **The ravine, revisited.** Run all seven on `f(x,y) = x² + 100y²` from `(10, 10)`. Plot each path over the contours. Count steps to reach `|θ| < 0.01`. Rank them.

**3.** Derive the geometric series result `v = g/(1−β)` for constant gradient. Verify numerically for `β = 0.9` and `β = 0.99`. Explain the practical consequence for tuning.

**4.** **Derive Adam's bias correction** on paper: show `m_t = g(1 − β₁ᵗ)` for constant `g`. Then plot the correction factors `1/(1−β₁ᵗ)` and `1/(1−β₂ᵗ)` for the first 1,000 steps.

**5.** Train the same model with Adam with and without bias correction. Plot the first 200 steps of both loss curves. Quantify the damage.

**6.** Demonstrate AdaGrad's death: plot its effective learning rate `η/√G` over 10,000 steps and show it decaying to near zero. Then show RMSProp doesn't.

**7.** **The AdamW difference.** Train the same model with `Adam(weight_decay=0.01)` and `AdamW(weight_decay=0.01)`. Compare final validation accuracy and the weight-norm trajectories. Explain the difference using §9.6.

**8.** Implement `step_decay`, `cosine`, and `warmup_cosine`. Plot all three over 10,000 steps.

**9.** **Warmup necessity.** Train a 12-layer post-norm transformer block stack (or a deep MLP with LayerNorm) with and without warmup at a high LR. Show one diverges. Then repeat with pre-norm and show it's more forgiving.

**10.** Implement the LR range test. Run it on your MNIST model and pick a learning rate from the plot. Compare against your previous hand-chosen value.

**11.** Verify the linear scaling rule: train at batch sizes 32, 64, 128, 256 with LR scaled linearly, and plot loss against *epochs* (not steps). They should roughly overlap — until they don't. Find where it breaks.

**12.** Show that Adam's step size is roughly scale-invariant: multiply the loss function by 1000 and confirm Adam's trajectory barely changes while SGD's changes drastically. Explain using `m̂/√v̂`.

**13.** Implement gradient clipping and train an intentionally unstable model with and without it. Plot both loss curves and the gradient-norm history.

**14.** **Optimizer bake-off.** On MNIST with a fixed architecture and a fixed step budget, tune the learning rate for each of SGD, SGD+momentum, RMSProp, Adam and AdamW using the LR range test. Report best validation accuracy for each, and wall-clock time. Write up which you'd pick and why.

**15.** **Chapter project.** Build `optim.py` containing all seven optimizers plus all three schedules, with a common interface and tests verifying each matches PyTorch's version to `1e-6` on a fixed sequence of gradients. This is a real, verifiable artifact — matching a production implementation exactly is a strong correctness claim.

---

## 9.13 Solutions

<details>
<summary>Open only after attempting</summary>

```python
import numpy as np, matplotlib.pyplot as plt


# --- 1: the optimizers ---
class Base:
    def __init__(self, lr): self.lr = lr; self.t = 0
class SGD(Base):
    def step(self, p, g): return p - self.lr * g

class Momentum(Base):
    def __init__(self, lr, beta=0.9): super().__init__(lr); self.beta=beta; self.v=None
    def step(self, p, g):
        self.v = g if self.v is None else self.beta*self.v + g
        return p - self.lr*self.v

class Nesterov(Momentum):
    def step(self, p, g):                      # g evaluated at lookahead by caller
        self.v = g if self.v is None else self.beta*self.v + g
        return p - self.lr*(self.beta*self.v + g)

class AdaGrad(Base):
    def __init__(self, lr, eps=1e-8): super().__init__(lr); self.G=None; self.eps=eps
    def step(self, p, g):
        self.G = g**2 if self.G is None else self.G + g**2
        return p - self.lr*g/(np.sqrt(self.G)+self.eps)

class RMSProp(Base):
    def __init__(self, lr, rho=0.9, eps=1e-8):
        super().__init__(lr); self.rho=rho; self.eps=eps; self.G=None
    def step(self, p, g):
        self.G = (1-self.rho)*g**2 if self.G is None else self.rho*self.G+(1-self.rho)*g**2
        return p - self.lr*g/(np.sqrt(self.G)+self.eps)

class Adam(Base):
    def __init__(self, lr, b1=.9, b2=.999, eps=1e-8, correct=True):
        super().__init__(lr); self.b1,self.b2,self.eps=b1,b2,eps
        self.m=None; self.v=None; self.correct=correct
    def step(self, p, g):
        self.t += 1
        self.m = (1-self.b1)*g   if self.m is None else self.b1*self.m+(1-self.b1)*g
        self.v = (1-self.b2)*g**2 if self.v is None else self.b2*self.v+(1-self.b2)*g**2
        if self.correct:
            mh = self.m/(1-self.b1**self.t); vh = self.v/(1-self.b2**self.t)
        else:
            mh, vh = self.m, self.v
        return p - self.lr*mh/(np.sqrt(vh)+self.eps)

class AdamW(Adam):
    def __init__(self, lr, wd=0.01, **kw): super().__init__(lr, **kw); self.wd=wd
    def step(self, p, g):
        return super().step(p, g) - self.lr*self.wd*p


# --- 2: the ravine ---
f    = lambda v: v[0]**2 + 100*v[1]**2
grad = lambda v: np.array([2*v[0], 200*v[1]])

def run(opt, steps=200, x0=(10.,10.)):
    p = np.array(x0); path=[p.copy()]
    for _ in range(steps):
        p = opt.step(p, grad(p)); path.append(p.copy())
    return np.array(path)

opts = {"SGD": SGD(.004), "Momentum": Momentum(.002), "AdaGrad": AdaGrad(1.5),
        "RMSProp": RMSProp(.3), "Adam": Adam(.5), "AdamW": AdamW(.5)}
xs = np.linspace(-11,11,300); ys = np.linspace(-11,11,300)
Xg,Yg = np.meshgrid(xs,ys)
for name, o in opts.items():
    pth = run(o)
    plt.figure(); plt.contour(Xg,Yg,Xg**2+100*Yg**2,levels=40)
    plt.plot(pth[:,0],pth[:,1],".-"); plt.title(name); plt.show()
    hits = np.argmax(np.linalg.norm(pth,axis=1) < .01)
    print(f"{name:10s} steps to |θ|<0.01: {hits if hits else '>200'}")
# SGD zigzags badly. Momentum damps the oscillation and accelerates along x.
# The adaptive methods rescale each coordinate by its own gradient magnitude,
# which removes the conditioning problem almost entirely.


# --- 3, 4 ---
g = 1.0
for beta in (.9, .99):
    v = 0.
    for _ in range(500): v = beta*v + g
    print(f"beta={beta}: v={v:.3f}  predicted {g/(1-beta):.3f}")
# Practical consequence: effective step is η/(1-β). Raising β from .9 to .99
# multiplies the step by 10 — lower η to compensate.

# m_t = (1-β)Σᵢ β^(t-i) g = (1-β)g(1-βᵗ)/(1-β) = g(1-βᵗ).  ∎
t = np.arange(1, 1001)
plt.semilogy(t, 1/(1-.9**t), label="1/(1-β₁ᵗ), β₁=.9")
plt.semilogy(t, 1/(1-.999**t), label="1/(1-β₂ᵗ), β₂=.999")
plt.legend(); plt.show()
print("t=1 corrections:", 1/(1-.9), 1/(1-.999))     # 10x and 1000x


# --- 6 ---
rng = np.random.default_rng(0)
ada, rms = AdaGrad(.1), RMSProp(.1)
ea, er = [], []
p1 = p2 = np.array([1.0])
for _ in range(10_000):
    gg = rng.standard_normal(1)
    p1 = ada.step(p1, gg); ea.append(.1/np.sqrt(ada.G[0]+1e-8))
    p2 = rms.step(p2, gg); er.append(.1/np.sqrt(rms.G[0]+1e-8))
plt.semilogy(ea, label="AdaGrad"); plt.semilogy(er, label="RMSProp")
plt.ylabel("effective LR"); plt.legend(); plt.show()
# AdaGrad decays like 1/√t forever. RMSProp levels off.


# --- 8 ---
def step_decay(lr0, e, drop=.1, every=30): return lr0*(drop**(e//every))
def cosine(lr_max, s, T, lr_min=0.): return lr_min+.5*(lr_max-lr_min)*(1+np.cos(np.pi*s/T))
def warmup_cosine(s, w, T, lr_max, lr_min=0.):
    if s < w: return lr_max*s/w
    q = (s-w)/max(1, T-w)
    return lr_min+.5*(lr_max-lr_min)*(1+np.cos(np.pi*q))

S = np.arange(10_000)
plt.plot(S,[step_decay(3e-4,s//100) for s in S],label="step")
plt.plot(S,[cosine(3e-4,s,10_000) for s in S],label="cosine")
plt.plot(S,[warmup_cosine(s,500,10_000,3e-4) for s in S],label="warmup+cosine")
plt.legend(); plt.show()


# --- 12: Adam's scale invariance ---
for mult in (1., 1000.):
    gA = lambda v: mult*grad(v)
    a = run(Adam(.5), 60); 
    pa = np.array([10.,10.]); o = Adam(.5)
    for _ in range(60): pa = o.step(pa, gA(pa))
    ps = np.array([10.,10.]); s = SGD(.004)
    for _ in range(60): ps = s.step(ps, gA(ps))
    print(f"mult={mult:6.0f}  adam→{np.round(pa,4)}  sgd→{np.round(ps,4)}")
# Adam's update is m̂/√v̂. Scaling the loss scales m̂ by c and √v̂ by c, so the
# ratio is unchanged — the trajectory is identical. SGD's step scales by c and
# immediately diverges. This is why Adam needs so little LR tuning.
```

</details>

---

## 9.14 Chapter 9 checkpoint

Cold — blank file, no notes.

- [ ] **Implement Adam from scratch**, including bias correction. **Target: 15 minutes.**
- [ ] **Derive the bias correction** — show `m_t = g(1−β₁ᵗ)`. On paper.
- [ ] Explain why momentum fixes the ravine, in terms of what happens in each direction.
- [ ] Derive the effective step amplification `1/(1−β)` and state the tuning consequence.
- [ ] Explain AdaGrad's failure and RMSProp's one-line fix.
- [ ] **Explain the difference between Adam+L2 and AdamW**, and why the coupling is wrong.
- [ ] Give three reasons warmup exists.
- [ ] Explain why second-order methods aren't used, with the number for a 7B model.
- [ ] Given "train loss good, val loss bad," say whether it's an optimization problem and what to do.

Items 2 and 6 are the ones almost nobody can do. They're worth the extra time.

### Anki cards

- Momentum update rule, and why it damps oscillation
- Effective step amplification with momentum
- AdaGrad's flaw; RMSProp's fix
- Adam's four lines
- Bias correction — the formula and the derivation in one line
- Correction factor at `t=1` for `β₂=0.999`
- Adam vs AdamW — the actual difference
- Why does Adam need so little LR tuning?
- Warmup — three reasons
- Cosine schedule formula
- Linear scaling rule, and where it breaks
- Why no second-order methods?
- Standard LLM settings: optimizer, LR, `β₂`, wd, schedule, clip
- Optimization failure vs generalization failure — how to tell

### Deliverables

```
optim/optimizers.py    all seven, matching PyTorch to 1e-6
optim/schedules.py     step, cosine, warmup_cosine
optim/lr_finder.py     the LR range test
experiments/ravine.py  exercise 2, with plots
reports/bakeoff.md     exercise 14
```

```bash
git add .
git commit -m "Chapter 9: optimizers from scratch, schedules, LR finder, bake-off"
git push
```

### Write-up

700 words: **"Adam, derived."** Build it up in the order the chapter does — the ravine problem, momentum's fix, AdaGrad's death, RMSProp's repair, then Adam as the combination, then the bias correction derivation, then AdamW's decoupling. Include your ravine plots from exercise 2 and the correction-factor plot from exercise 4.

Most explanations present Adam as four lines to memorize. Yours will show that every term solves a problem you can name.

---

*Next: Chapter 10 — Convolutional Networks*
