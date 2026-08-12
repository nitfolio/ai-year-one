# Chapter 4 — Your First Learning Algorithm

**Time: 10–12 days** (Weeks 5–6 of the plan)

**Prerequisite:** Chapters 2 and 3 checkpoints passed cold. You need `utils/gradcheck.py` working — you'll use it throughout.

**What you'll be able to do at the end:** build, from nothing, a model that learns from data. Derive its gradients by hand, verify them mechanically, train it with minibatch gradient descent, diagnose why a training run is failing from the shape of its loss curve, and extend it to multiclass classification. Everything after this chapter is a variation on what you build here.

---

## 4.0 What "learning" actually means

Three chapters of tools. Now they combine into something that works.

Supervised learning, stripped to its skeleton, is four things:

1. **A model** — a function with adjustable parameters. `f(x; θ)`
2. **A loss** — a single number saying how wrong the model is. `L(θ)`
3. **A gradient** — how the loss changes with each parameter. `∇L(θ)`
4. **An update rule** — move the parameters downhill. `θ ← θ − η∇L`

Repeat 2–4 until the loss stops dropping.

**That is all of deep learning.** GPT-5 and the linear regression you write today differ only in the complexity of step 1. Steps 2, 3 and 4 are identical in kind. If you internalize this loop now, every later architecture is just a new `f`.

---

## 4.1 The setup

You have `n` examples. Each has features `xᵢ ∈ ℝᵈ` and a target `yᵢ`.

Stack them:

```
X : (n, d)     one example per row
y : (n,)       one target per row
```

Two problem types:

- **Regression** — `y` is a real number (price, temperature). Measured by squared error.
- **Classification** — `y` is a category (spam/not, digit 0–9). Measured by cross-entropy.

You'll build both.

---

## 4.2 Linear regression

The simplest useful model. Predict a weighted sum of the features, plus an offset:

```
ŷ = Xw + b
```

- `w : (d,)` — one weight per feature
- `b : scalar` — the bias (intercept)
- `ŷ : (n,)` — one prediction per example

From §2.6: each prediction is a dot product between that example's features and `w`. The model asks *how much does this example point in the direction I care about?* Training finds the direction.

```python
import numpy as np

def linear_forward(X, w, b):
    """X:(n,d)  w:(d,)  b:scalar  ->  (n,)"""
    assert X.shape[1] == w.shape[0], f"{X.shape} vs {w.shape}"
    return X @ w + b
```

---

## 4.3 The loss: mean squared error

```
L(w, b) = (1/n) Σᵢ (ŷᵢ − yᵢ)²
```

```python
def mse_loss(pred, y):
    return np.mean((pred - y) ** 2)
```

### Why squared, and not absolute?

A fair question with four real answers:

1. **Differentiable everywhere.** `|x|` has a corner at zero; `x²` doesn't. Gradient descent needs smoothness.
2. **Punishes large errors disproportionately.** An error of 10 costs 100× an error of 1, not 10×. Usually what you want.
3. **Convex** in `w` — one global minimum, no local traps. Rare and valuable.
4. **It's the maximum-likelihood loss under Gaussian noise.** If you assume `y = Xw + ε` with `ε ~ 𝒩(0, σ²)`, then maximizing the likelihood of your data is *exactly* minimizing squared error. You'll derive this in Chapter 6.

Reason 4 is the deep one. Loss functions are not arbitrary design choices — each corresponds to an assumption about how your data was generated.

Absolute error (`L1`) is used when you want robustness to outliers, since it doesn't let one bad point dominate.

---

## 4.4 Deriving the gradient

Do this on paper before reading further.

Write the residual `r = Xw − y`, shape `(n,)`. Then `L = (1/n) rᵀr`.

**Step 1 — differentiate the loss with respect to the residual:**

```
∂L/∂r = (2/n) r
```

**Step 2 — the residual depends on `w` through `X`.** Apply the transpose rule from §2.9 — since `r = Xw − y` is linear in `w` with matrix `X`, the gradient flows back as `Xᵀ`:

```
∂L/∂w = Xᵀ · (2/n) r = (2/n) Xᵀ(Xw − y)
```

**Step 3 — the bias.** `b` is added to every prediction, so it reaches the loss through all `n` residuals. By the multi-path rule of §3.5, sum over them:

```
∂L/∂b = (2/n) Σᵢ rᵢ
```

**Shape check** — always do this:

```
Xᵀ is (d,n),  r is (n,)   →   (d,n) @ (n,) = (d,)   ✓ matches w
```

```python
def linear_grads(X, y, w, b):
    """Returns (dw, db) with shapes (d,) and scalar."""
    n = X.shape[0]
    r = X @ w + b - y                # (n,)
    dw = (2 / n) * (X.T @ r)         # (d,)
    db = (2 / n) * r.sum()           # scalar
    return dw, db
```

**Now verify it.** This is the moment your Chapter 3 tool pays for itself:

```python
from utils.gradcheck import gradient_check

rng = np.random.default_rng(0)
X = rng.standard_normal((20, 5))
y = rng.standard_normal(20)

def loss_of_params(p):
    return mse_loss(linear_forward(X, p[:-1], p[-1]), y)

def grad_of_params(p):
    dw, db = linear_grads(X, y, p[:-1], p[-1])
    return np.append(dw, db)

gradient_check(loss_of_params, grad_of_params, rng.standard_normal(6))
# max relative error: 3.2e-11   PASS
```

**Never write a gradient without checking it.** Make this reflexive. An unverified gradient that's subtly wrong will train to a mediocre result and you will spend two weeks blaming your architecture.

---

## 4.5 Training it

```python
def train_linear(X, y, lr=0.1, epochs=200):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    history = []

    for epoch in range(epochs):
        pred = linear_forward(X, w, b)
        loss = mse_loss(pred, y)
        history.append(loss)

        dw, db = linear_grads(X, y, w, b)
        w -= lr * dw
        b -= lr * db

        if epoch % 20 == 0:
            print(f"epoch {epoch:4d}  loss {loss:.6f}")

    return w, b, history
```

**Verify on synthetic data with a known answer.** This is the single best way to test any learning algorithm — you know what it *should* find:

```python
rng = np.random.default_rng(0)
X_true = rng.standard_normal((500, 3))
w_true = np.array([2.0, -1.0, 0.5])
b_true = 3.0
y_true = X_true @ w_true + b_true + 0.1 * rng.standard_normal(500)

w, b, hist = train_linear(X_true, y_true, lr=0.1, epochs=500)
print(f"recovered w = {w}   (true {w_true})")
print(f"recovered b = {b:.3f} (true {b_true})")
```

If it doesn't recover the true parameters to two decimals, something is broken — and you know it's your code, not the data. **Always test new learning code on synthetic data with a known answer first.** Real data can't tell you whether a mediocre result means a bug or a hard problem.

---

## 4.6 The closed-form solution (and why nobody uses it)

Linear regression is one of the very few models that can be solved exactly. Set the gradient to zero:

```
(2/n) Xᵀ(Xw − y) = 0
⟹  XᵀXw = Xᵀy                  ← the normal equations
⟹  w = (XᵀX)⁻¹Xᵀy
```

This is the same result you reached geometrically in §2.15 by projecting `y` onto the column space of `X`. Calculus and geometry agreeing is a good sign.

```python
def linear_closed_form(X, y):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])   # absorb bias as a column
    theta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)     # NOT inv() — see §2.12
    return theta[:-1], theta[-1]
```

**So why bother with gradient descent?**

1. **It doesn't scale.** `XᵀX` is `(d, d)`; solving costs roughly `O(d³)`. At `d = 100,000` that's hopeless.
2. **It needs all data in memory at once.** Gradient descent works on minibatches.
3. **It only exists for a handful of models.** Neural networks have no closed form. Not "we haven't found it" — there isn't one.
4. **It's numerically fragile** when features are correlated (`XᵀX` becomes near-singular).

Gradient descent is worse for this one problem and works for everything. That's the trade the whole field made.

---

## 4.7 Feature scaling

Try training on features with wildly different scales — say one in `[0, 1]` and another in `[0, 100000]` — and convergence will be miserable.

**Why**, precisely: the loss surface becomes a long narrow ravine. Its curvature along the large-scale feature is enormous; along the small-scale one it's tiny. That's exactly the high-condition-number situation from §2.13, and exactly the zigzag you plotted in exercise 3.10. Your learning rate is capped by the steepest direction, so the shallow directions crawl.

**Fix — standardize:**

```python
def standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std
```

Every feature now has mean 0 and standard deviation 1. The ravine becomes a bowl and convergence can improve by orders of magnitude.

**Critical rule that people get wrong constantly:** compute `mean` and `std` on the **training set only**, then apply those same values to validation and test. Computing them on the full dataset leaks information from your test set into training and inflates your reported score. This is one of the most common silent errors in applied ML.

```python
X_train, mu, sd = standardize(X_train)
X_val, _, _ = standardize(X_val, mu, sd)     # reuse training statistics
X_test, _, _ = standardize(X_test, mu, sd)
```

---

## 4.8 Classification, and why linear regression fails at it

Now `y ∈ {0, 1}`. Why not just fit a line and threshold at 0.5?

Three reasons it breaks:

1. **Outputs aren't probabilities.** A linear model happily predicts −3.7 or 8.2.
2. **Distant correct points distort the fit.** A point far on the correct side still contributes large squared error, dragging the boundary to reduce an error that doesn't matter.
3. **Squared error is the wrong measure of wrongness for categories.** Being 0.9 confident and right versus 0.9 confident and wrong should not be symmetric.

The fix: squash the output into `(0, 1)` and use a loss designed for probabilities.

---

## 4.9 Logistic regression

```
z = Xw + b            the logit — any real number
p = σ(z)              squashed to (0,1) — a probability
```

```python
def sigmoid(z):
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))    # stable for large |z|
```

That two-branch form avoids `exp` of a large positive number, which overflows to `inf`. The naive one-liner will bite you on real data.

### Binary cross-entropy

```
L = −(1/n) Σᵢ [ yᵢ·ln(pᵢ) + (1 − yᵢ)·ln(1 − pᵢ) ]
```

Read it one example at a time. If `y = 1`, only the first term survives: `−ln(p)`. Predict `p = 0.99` → loss ≈ 0.01. Predict `p = 0.01` → loss ≈ 4.6. **Confident and wrong is punished enormously**, which is exactly the behaviour you want from a probabilistic classifier.

### Why not MSE here? — the concrete reason

This deserves a demonstration rather than an assertion.

With MSE and a sigmoid, `L = (p − y)²`, and by the chain rule:

```
∂L/∂z = 2(p − y) · σ'(z) = 2(p − y) · p(1 − p)
```

Suppose the model is **confidently wrong**: `y = 1` but `p = 0.001`. Then `p(1−p) ≈ 0.001`, so

```
∂L/∂z ≈ 2(−0.999)(0.001) ≈ −0.002
```

A near-zero gradient at the moment the model is *most* wrong. It will barely move. This is the **saturation problem**, and it makes sigmoid+MSE networks nearly untrainable.

With cross-entropy (derivation in §4.10):

```
∂L/∂z = p − y = 0.001 − 1 = −0.999
```

Full-strength gradient exactly when it's needed most. **The `σ'` factor cancels.** That cancellation is why cross-entropy and sigmoid are always paired.

---

## 4.10 The derivation worth doing by hand

Show that `∂L/∂z = p − y` for one example.

```
L = −[y·ln(p) + (1−y)·ln(1−p)],    p = σ(z)
```

**Step 1 — loss with respect to `p`:**

```
∂L/∂p = −[ y/p − (1−y)/(1−p) ]
```

**Step 2 — sigmoid derivative** (§3.3):

```
∂p/∂z = p(1 − p)
```

**Step 3 — chain them, and watch the cancellation:**

```
∂L/∂z = −[ y/p − (1−y)/(1−p) ] · p(1−p)
      = −[ y(1−p) − (1−y)p ]
      = −[ y − yp − p + yp ]
      = p − y                        ∎
```

Extending to all `n` examples and back through `X`:

```
∂L/∂w = (1/n) Xᵀ(p − y)
∂L/∂b = (1/n) Σᵢ (pᵢ − yᵢ)
```

**Compare to linear regression:** `(2/n)Xᵀ(ŷ − y)`. Structurally identical — `Xᵀ` times the prediction error. Different model, different loss, same gradient shape. That's not coincidence; it's a property of the exponential family, and Chapter 6 explains it.

```python
def logistic_grads(X, y, w, b):
    n = X.shape[0]
    p = sigmoid(X @ w + b)
    err = p - y
    return (X.T @ err) / n, err.sum() / n
```

### Numerical stability of the loss

`ln(0)` is `−inf`. If `p` ever hits exactly 0 or 1 in floating point, your loss becomes `nan` and the run is destroyed.

Naive fix — clip:

```python
def bce_loss(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
```

Better fix — compute the loss directly from logits, never forming `p`:

```python
def bce_with_logits(z, y):
    """Numerically stable for any z. Algebraically identical to bce(σ(z), y)."""
    return np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))
```

Derivation, if you want it: substitute `ln σ(z) = −ln(1+e⁻ᶻ)` into the loss and simplify to `z − yz + ln(1+e⁻ᶻ)`, then use `ln(1+e⁻ᶻ) = max(−z,0) + ln(1+e⁻|ᶻ|)` to keep every exponent negative.

**This is what PyTorch's `BCEWithLogitsLoss` does**, and it's why the docs tell you to use it instead of `Sigmoid` followed by `BCELoss`. You now know why.

---

## 4.11 Multiclass: softmax and cross-entropy

For `k` classes, produce `k` logits and normalize them into a probability distribution:

```
pᵢ = exp(zᵢ) / Σₖ exp(zₖ)
```

```python
def softmax(Z):
    """Z:(n,k) -> (n,k), each row sums to 1."""
    Z = Z - Z.max(axis=1, keepdims=True)       # stability — see §1.9
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)
```

Cross-entropy with a one-hot target `y`:

```
L = −Σᵢ yᵢ ln(pᵢ) = −ln(p_correct)
```

Only the correct class's probability matters. The loss is the negative log of how much probability mass you put on the right answer.

### The gradient — same beautiful result

The softmax Jacobian is:

```
∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ − pⱼ)          where δᵢⱼ = 1 if i=j else 0
```

Chaining through cross-entropy:

```
∂L/∂zⱼ = −Σᵢ (yᵢ/pᵢ) · pᵢ(δᵢⱼ − pⱼ)
       = −Σᵢ yᵢ(δᵢⱼ − pⱼ)
       = −(yⱼ − pⱼ·Σᵢyᵢ)
       = pⱼ − yⱼ                 (since Σyᵢ = 1 for one-hot)     ∎
```

**`p − y` again.** Third time: linear+MSE, sigmoid+BCE, softmax+CE all produce the same clean form. This is why every framework fuses softmax and cross-entropy into a single operation — the combined gradient is trivial, while each part alone is messy.

```python
def softmax_ce_loss(Z, Y):
    """Z:(n,k) logits, Y:(n,k) one-hot"""
    P = softmax(Z)
    return -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))

def softmax_ce_grads(X, Y, W, b):
    """X:(n,d) Y:(n,k) W:(d,k) b:(k,)"""
    n = X.shape[0]
    P = softmax(X @ W + b)
    G = (P - Y) / n                   # (n,k)
    return X.T @ G, G.sum(axis=0)     # (d,k), (k,)
```

---

## 4.12 Minibatch gradient descent

So far you've used the whole dataset for every step — **full-batch** gradient descent. Three variants exist:

| Variant | Batch size | Character |
|---|---|---|
| Full batch | all `n` | Exact gradient. Slow steps. Gets stuck at saddles. |
| Stochastic (SGD) | 1 | Very noisy. Fast steps. Noise escapes saddles. |
| **Minibatch** | 32–512 | The practical answer. |

Minibatch wins because:

- The gradient estimate is good enough — averaging 64 examples removes most of the noise.
- It's **hardware-efficient**. A GPU processing 128 examples takes barely longer than 1.
- The residual noise is *useful*: it helps escape saddle points and appears to improve generalization.

```python
def iterate_minibatches(X, y, batch_size, rng):
    n = X.shape[0]
    idx = rng.permutation(n)              # reshuffle every epoch — important
    for start in range(0, n, batch_size):
        batch = idx[start:start + batch_size]
        yield X[batch], y[batch]
```

**Shuffle every epoch.** If your data is sorted by label and you don't shuffle, each batch contains a single class and training will be chaotic. This is a real bug people ship.

---

## 4.13 Train, validation, test

Split your data three ways:

- **Train (~70%)** — gradients computed here.
- **Validation (~15%)** — hyperparameter choices and early stopping.
- **Test (~15%)** — touched **once**, at the very end.

**Overfitting** — the model memorizes training data. Train loss keeps falling; validation loss turns and rises. The gap between them is the diagnostic.

**Underfitting** — the model isn't expressive enough. Both losses plateau high.

**The rule people break:** if you tune anything based on test performance, the test set has become a validation set and your reported number is optimistic. Split it once, put it away, and don't look.

```python
def split_data(X, y, rng, train=0.7, val=0.15):
    n = len(X)
    idx = rng.permutation(n)
    i1, i2 = int(n * train), int(n * (train + val))
    tr, va, te = idx[:i1], idx[i1:i2], idx[i2:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])
```

---

## 4.14 Regularization

Constrain the weights to prevent memorization.

**L2 (weight decay)** — add `λ|w|²` to the loss:

```
L_total = L_data + λ·Σⱼwⱼ²
∂/∂w    = ∂L_data/∂w + 2λw
```

Shrinks all weights smoothly toward zero. The default choice.

**L1** — add `λ·Σ|wⱼ|`:

```
∂/∂w = ∂L_data/∂w + λ·sign(w)
```

Drives many weights to *exactly* zero, producing a sparse model — useful when you want feature selection.

```python
def l2_penalty(w, lam):     return lam * np.sum(w ** 2)
def l2_penalty_grad(w, lam): return 2 * lam * w
```

**Do not regularize the bias.** The bias sets the model's baseline output; shrinking it just biases predictions toward zero for no benefit. Every framework excludes it by default.

---

## 4.15 Reading a loss curve

**Plot your loss every single run.** A loss curve tells you more in one glance than an hour of staring at code. Learn to read it:

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss becomes `nan` | LR too high; `log(0)`; overflow | Lower LR ×10; use logit-based losses; clip gradients |
| Flat from step 0 | LR far too low; bug in gradient; all-zero init in a symmetric model | Gradient-check; raise LR ×10 |
| Wild oscillation | LR too high; batch too small | Lower LR; larger batch |
| Smooth but glacial | Poor conditioning; unscaled features | Standardize features (§4.7) |
| Plateaus high, train ≈ val | Underfitting | More features or capacity |
| Train ↓, val ↑ | Overfitting | Regularize; more data; early stopping |
| Loss drops then jumps up | LR too high late in training | LR schedule (Ch. 9) |

### The single best debugging test

From §0.4, and worth repeating because you'll use it every chapter from here:

> **Take 10 examples. Turn off regularization. Train until the loss is essentially zero.**

Any correctly implemented model with enough capacity *can* memorize 10 examples. If it can't, your implementation is broken — the gradient, the loss, the data pipeline, something. Stop tuning hyperparameters and go find the bug.

This one test will save you weeks over the year. Run it before every serious training run.

---

## 4.16 Putting it together

```python
def train(X_tr, y_tr, X_va, y_va, forward, loss_fn, grads_fn,
          init_params, lr=0.1, epochs=100, batch_size=64, lam=0.0, seed=0):
    """Generic training loop — reused for every model in this chapter."""
    rng = np.random.default_rng(seed)
    w, b = init_params
    hist = {"train": [], "val": []}

    for epoch in range(epochs):
        for Xb, yb in iterate_minibatches(X_tr, y_tr, batch_size, rng):
            dw, db = grads_fn(Xb, yb, w, b)
            if lam:
                dw = dw + 2 * lam * w          # bias excluded
            w -= lr * dw
            b -= lr * db

        hist["train"].append(loss_fn(forward(X_tr, w, b), y_tr))
        hist["val"].append(loss_fn(forward(X_va, w, b), y_va))

        if epoch % 10 == 0:
            print(f"epoch {epoch:4d}  train {hist['train'][-1]:.4f}  "
                  f"val {hist['val'][-1]:.4f}")

    return w, b, hist
```

One loop, swappable model. That structure — forward, loss, grads, update — is the skeleton of every training script you will ever write, including the one in Chapter 12 that trains a transformer.

---

## 4.17 Exercises

**1.** Implement `linear_forward`, `mse_loss`, `linear_grads`. Gradient-check all of them.

**2.** Generate synthetic data with a known `w` and `b`. Train and verify you recover them to two decimal places. Plot the loss curve.

**3.** Implement `linear_closed_form`. Compare its solution against gradient descent on the same data. Then time both for `d = 10, 100, 1000`.

**4.** Take your synthetic data and multiply one feature column by 10,000. Train with and without standardization, same learning rate. Plot both loss curves. Explain the difference in terms of §2.13.

**5.** Implement stable `sigmoid`, `bce_loss`, and `bce_with_logits`. Verify the last two agree for `z` in `[−5, 5]`, then compare at `z = ±800` — one will produce `nan`.

**6.** Derive `∂L/∂z = p − y` for sigmoid+BCE on paper, from scratch, without looking. Then implement `logistic_grads` and gradient-check it.

**7.** **The saturation demonstration.** For `y = 1` and `p` ranging over `[0.001, 0.999]`, plot `∂L/∂z` for MSE+sigmoid and for BCE+sigmoid on the same axes. Explain what happens near `p = 0` and why it makes MSE unusable for classification.

**8.** Train logistic regression on a synthetic 2-D two-class dataset. Plot the data and the decision boundary (`w·x + b = 0`).

**9.** Implement `softmax` (numerically stable) and `softmax_ce_loss`. Verify each row of the softmax output sums to 1, and that the loss for a perfect prediction is ≈ 0.

**10.** Derive `∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ − pⱼ)` on paper. Then derive `∂L/∂z = p − y` for softmax+CE. Gradient-check `softmax_ce_grads`.

**11.** Implement `iterate_minibatches`. Train the same model with batch sizes 1, 32, and full-batch. Plot all three loss curves on one axis and describe the noise difference.

**12.** Implement `split_data`. Create a dataset with 20 examples and 50 features (far more features than data). Train without regularization and plot train vs validation loss — you should see clear overfitting. Then sweep `λ ∈ {0, 0.001, 0.01, 0.1, 1, 10}` and plot final validation loss against `λ`. Which wins?

**13.** **The overfit-tiny test.** Take 10 examples from any dataset. Turn off regularization. Train until the loss is below `1e-6`. Then deliberately introduce a bug — flip the sign in `linear_grads` — and confirm the test now fails. This teaches you what the test catches.

**14.** **Chapter project.** Build a multiclass classifier from scratch on a real dataset (`sklearn.datasets.load_digits` is a good choice — **use sklearn only to load data, never for the model**). Requirements: proper 3-way split, feature standardization using training statistics only, softmax + cross-entropy, minibatch SGD, L2 regularization, gradient-checked gradients, plotted train/val curves, and a final test accuracy reported exactly once. Write it up.

---

## 4.18 Solutions

<details>
<summary>Open only after attempting</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from utils.gradcheck import gradient_check

rng = np.random.default_rng(0)


# --- 1, 2 ---
def linear_forward(X, w, b): return X @ w + b
def mse_loss(pred, y):       return np.mean((pred - y) ** 2)

def linear_grads(X, y, w, b):
    n = len(y); r = X @ w + b - y
    return (2/n) * (X.T @ r), (2/n) * r.sum()

X = rng.standard_normal((500, 3))
w_true, b_true = np.array([2., -1., .5]), 3.
y = X @ w_true + b_true + .1 * rng.standard_normal(500)

gradient_check(lambda p: mse_loss(linear_forward(X, p[:-1], p[-1]), y),
               lambda p: np.append(*linear_grads(X, y, p[:-1], p[-1])),
               rng.standard_normal(4))

w, b = np.zeros(3), 0.
for _ in range(2000):
    dw, db = linear_grads(X, y, w, b)
    w -= .1 * dw; b -= .1 * db
print(w, b)          # ≈ [2, -1, .5], 3


# --- 3 ---
def linear_closed_form(X, y):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    th = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
    return th[:-1], th[-1]
print(linear_closed_form(X, y))     # essentially identical


# --- 4 ---
Xs = X.copy(); Xs[:, 0] *= 10_000
for label, data in (("raw", Xs), ("standardized", (Xs - Xs.mean(0)) / Xs.std(0))):
    w, b, h = np.zeros(3), 0., []
    for _ in range(300):
        dw, db = linear_grads(data, y, w, b)
        w -= 1e-3 * dw; b -= 1e-3 * db
        h.append(mse_loss(data @ w + b, y))
    plt.semilogy(h, label=label)
plt.legend(); plt.show()
# Raw: condition number ~1e8, so the usable LR is capped by the huge-curvature
# direction and the others barely move. Standardizing makes curvature uniform.


# --- 5 ---
def sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-np.abs(z))), np.exp(-np.abs(z))/(1+np.exp(-np.abs(z))))

def bce_loss(p, y, eps=1e-12):
    p = np.clip(p, eps, 1-eps)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

def bce_with_logits(z, y):
    return np.mean(np.maximum(z, 0) - z*y + np.log1p(np.exp(-np.abs(z))))

z = np.linspace(-5, 5, 50); yb = (rng.random(50) > .5).astype(float)
assert np.isclose(bce_loss(sigmoid(z), yb), bce_with_logits(z, yb))
print(bce_loss(sigmoid(np.array([800.])), np.array([0.])))   # inf / nan territory
print(bce_with_logits(np.array([800.]), np.array([0.])))     # 800.0 — correct


# --- 6 ---
def logistic_grads(X, y, w, b):
    n = len(y); err = sigmoid(X @ w + b) - y
    return (X.T @ err)/n, err.sum()/n

yc = (X @ w_true + b_true > 3).astype(float)
gradient_check(lambda p: bce_with_logits(X @ p[:-1] + p[-1], yc),
               lambda p: np.append(*logistic_grads(X, yc, p[:-1], p[-1])),
               rng.standard_normal(4))


# --- 7 ---  THE KEY DEMONSTRATION
p = np.linspace(.001, .999, 500)
plt.plot(p, 2*(p-1)*p*(1-p), label="MSE + sigmoid")
plt.plot(p, p - 1,           label="BCE + sigmoid")
plt.axhline(0, c="k", lw=.5); plt.xlabel("predicted p (true y=1)")
plt.ylabel("dL/dz"); plt.legend(); plt.show()
# Near p=0 (confidently wrong) MSE's gradient →0 because of the p(1-p) factor.
# BCE's is -1: maximal. The sigmoid derivative cancels. That cancellation is
# the entire reason cross-entropy is the standard classification loss.


# --- 9, 10 ---
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z); return e / e.sum(axis=1, keepdims=True)

def softmax_ce_loss(Z, Y):
    return -np.mean(np.sum(Y * np.log(softmax(Z) + 1e-12), axis=1))

def softmax_ce_grads(X, Y, W, b):
    G = (softmax(X @ W + b) - Y) / len(X)
    return X.T @ G, G.sum(axis=0)

n, d, k = 40, 5, 3
Xm = rng.standard_normal((n, d))
lab = rng.integers(0, k, n); Ym = np.eye(k)[lab]
assert np.allclose(softmax(rng.standard_normal((7, k))).sum(axis=1), 1)

def pack(W, b):   return np.concatenate([W.ravel(), b])
def unpack(p):    return p[:d*k].reshape(d, k), p[d*k:]
gradient_check(lambda p: softmax_ce_loss(Xm @ unpack(p)[0] + unpack(p)[1], Ym),
               lambda p: pack(*softmax_ce_grads(Xm, Ym, *unpack(p))),
               rng.standard_normal(d*k + k))


# --- 12 ---
Xo = rng.standard_normal((20, 50))
yo = Xo @ rng.standard_normal(50) + .1*rng.standard_normal(20)
tr, va = slice(0, 14), slice(14, 20)
finals = []
for lam in (0, .001, .01, .1, 1, 10):
    w, b = np.zeros(50), 0.
    for _ in range(3000):
        dw, db = linear_grads(Xo[tr], yo[tr], w, b)
        w -= .01 * (dw + 2*lam*w); b -= .01 * db
    finals.append(mse_loss(Xo[va] @ w + b, yo[va]))
print(finals)   # U-shaped: too little λ overfits, too much underfits.


# --- 13 ---
Xt, yt = X[:10], y[:10]
w, b = np.zeros(3), 0.
for _ in range(20000):
    dw, db = linear_grads(Xt, yt, w, b)
    w -= .05*dw; b -= .05*db
print("overfit-tiny loss:", mse_loss(Xt @ w + b, yt))    # ~1e-9 -> code is sound
# With the sign flipped in linear_grads the loss diverges instead. That is the
# whole value of the test: it separates "broken code" from "hard problem".
```

</details>

---

## 4.19 Chapter 4 checkpoint

Cold — blank file, no notes.

- [ ] State the four components of supervised learning.
- [ ] Derive `∂L/∂w` for linear regression + MSE on paper. **5 minutes.**
- [ ] Derive `∂L/∂z = p − y` for sigmoid + BCE on paper. **10 minutes.**
- [ ] Explain, with the saturation argument, why MSE fails for classification.
- [ ] Implement logistic regression end to end — forward, loss, gradients, training loop — and gradient-check it. **Target: 40 minutes.**
- [ ] Explain why feature standardization matters, in terms of condition number.
- [ ] State the overfit-tiny-dataset test and what it proves.
- [ ] Given a loss curve that goes `nan`, list three causes.

Items 3, 4 and 5 are mandatory. Chapter 5 assumes them.

### Anki cards

- Four components of supervised learning
- `∂L/∂w` for linear regression + MSE
- `∂L/∂z` for sigmoid + BCE — and why the `σ'` cancels
- `∂L/∂z` for softmax + CE
- Why is MSE bad for classification?
- Normal equations — and four reasons not to use them
- Why standardize features?
- Rule for computing standardization statistics
- L1 vs L2 regularization — what differs in the result?
- Why exclude the bias from regularization?
- Minibatch vs full batch vs SGD
- The overfit-tiny-dataset test
- Three causes of `nan` loss

### Deliverables

```
models/linear.py       forward, loss, grads, gradient-checked
models/logistic.py     stable sigmoid, BCE-with-logits, grads
models/softmax.py      stable softmax, CE, grads
train.py               the generic loop from §4.16
project_digits.py      exercise 14, written up
```

```bash
git add .
git commit -m "Chapter 4: linear + logistic + softmax regression from scratch"
git push
```

### Write-up

600 words: **"Why cross-entropy and not squared error."** Include your derivation of `∂L/∂z = p − y`, the saturation plot from exercise 7, and the observation that the gradient is `Xᵀ(p − y)` in all three models. This is a strong post — most explanations of cross-entropy stop at "it works better," and yours will show the exact mechanism.

**You now have a working machine learning system built entirely from primitives.** Chapter 5 removes the last piece of manual labour: instead of deriving gradients by hand for every new model, you'll build a machine that derives them automatically. That's the chapter where you stop implementing models and start implementing the thing that implements models.

---

*Next: Chapter 5 — Backpropagation and Autograd*
