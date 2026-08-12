# Appendix A — Debugging Playbook

**How to use this:** find your symptom, work the causes in the order listed. They're ranked by frequency, not by how interesting they are. The boring cause is usually the cause.

**Before anything else, run §A.1.** It takes five minutes and resolves a large fraction of problems without further thought.

---

## A.1 The five-minute triage

Run these in order, always, before doing anything clever.

**1. Check the initial loss.**

```python
print(loss.item(), "should be ≈", np.log(n_classes))
```

10 classes → 2.30. 2 classes → 0.69. Vocabulary of 50,257 → 10.82.

If it's far off: broken initialization, wrong loss function, misaligned targets, or wrong vocabulary size. **Stop and fix this before training.** (§7.9)

**2. Can it overfit 10 examples?**

Turn off regularization, dropout, and augmentation. Take 10 examples. Train to loss ≈ 0.

If it can't, the bug is in your code — not your hyperparameters. Stop tuning and go find it. (§4.15)

**3. Print every shape.**

```python
print(f"{x.shape=} {W.shape=} {out.shape=} {target.shape=}")
```

**4. Gradient-check a small version.**

```python
from utils.gradcheck import gradient_check
gradient_check(loss_fn, grad_fn, small_params)
```

**5. Print per-layer gradient norms.**

```python
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"{name:40s} {p.grad.norm().item():.3e}")
```

Should be within an order of magnitude or two of each other. A monotonic decay from last layer to first is vanishing gradients, visible directly.

---

## A.2 Shape errors

**These are the most common bugs in deep learning and the easiest to fix.** Read the error bottom-up (§1.4).

| Error | Cause | Fix |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied (AxB and CxD)` | `B ≠ C` | Check `@` operand order; a transpose is usually missing |
| `The size of tensor a (X) must match tensor b (Y)` | Broadcasting mismatch | Print both shapes; align from the right (§1.6) |
| `Expected input batch_size (X) to match target batch_size (Y)` | Flattened wrong | For `CrossEntropyLoss`: input `(N,C)`, target `(N,)` |
| `Expected target size [A, B], got [C]` | One-hot passed to `CrossEntropyLoss` | Pass class **indices**, not one-hot |
| `index out of range in self` | Token id ≥ `vocab_size` | Check tokenizer output range against embedding size |
| Silent wrong answer, no error | **Unintended broadcasting** | See below — the dangerous one |

### The silent broadcasting bug

```python
a = np.array([1, 2, 3])          # (3,)
b = np.array([[1], [2], [3]])    # (3, 1)
a + b                            # (3, 3)  — no error, wrong answer
```

**Defence:** assert shapes at function boundaries.

```python
def layer(X, W, b):
    assert X.ndim == 2, f"expected 2-D, got {X.shape}"
    out = X @ W + b
    assert out.shape == (X.shape[0], W.shape[1]), f"got {out.shape}"
    return out
```

Costs nothing, catches bugs that would cost a day. (§1.6, §2.10)

---

## A.3 NaN and inf

**Find where it first appears:**

```python
torch.autograd.set_detect_anomaly(True)     # slow — temporary only
```

Or manually:

```python
for name, p in model.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        print("non-finite grad in", name)
```

| Cause | Check | Fix |
|---|---|---|
| Learning rate too high | Does step 1 already produce inf? | Lower LR ×10 |
| No warmup (transformers) | Diverges in first ~100 steps? | Add warmup, 1–2% of steps (§9.7) |
| `log(0)` | Any probability exactly 0 or 1? | Use `BCEWithLogitsLoss` / `cross_entropy` on logits (§4.10) |
| Division by zero | Any variance/norm denominators? | Add `eps` (1e-8 for fp32, 1e-5 for LayerNorm) |
| `exp` overflow | Softmax without max subtraction? | Subtract row max first (§1.9) |
| fp16 overflow | Using `float16`? | Switch to `bfloat16`, or use `GradScaler` (§12.5) |
| Exploding gradients | Grad norm climbing before the NaN? | `clip_grad_norm_(params, 1.0)` (§7.4) |
| Bad data | Any NaN in the inputs? | `assert torch.isfinite(x).all()` in the loader |

**Rule of thumb:** NaN at step 1 is a code bug. NaN at step 5,000 is a learning-rate or stability problem.

---

## A.4 Loss won't decrease

Work top to bottom.

**Flat from step 0, exactly at `ln(k)`:**

1. **Gradients aren't reaching the parameters.** Check `p.grad is not None` for every parameter. Common causes: missing `super().__init__()`; parameters created outside `nn.Module`; something detached.
2. **`optimizer.zero_grad()` missing or `optimizer.step()` not called.**
3. **Learning rate far too low.** Raise ×10, then ×100. If nothing changes at any LR, it's not the LR.
4. **All-zero initialization** — every unit computes the same thing forever (§7.3).
5. **Data pipeline broken** — labels shuffled independently of inputs. Print a few `(x, y)` pairs and eyeball them.

**Falls slightly then flattens high:**

1. **Model too small** for the task.
2. **Learning rate too low** — try the LR range test (§9.8).
3. **Saturating activations** — check activation std per layer (§7.9 step 4).
4. **Dead ReLUs** — check the dead fraction; >20% in a layer is a problem (§7.5).
5. **Features unscaled** — standardize (§4.7).
6. **For language models:** a plateau near 7 nats means it learned unigram frequencies and nothing else (§12.8).

**Decreases far too slowly:**

1. **Poor conditioning** — standardize inputs, add normalization layers.
2. **Wrong optimizer** — try AdamW at `3e-4` as a baseline.
3. **Batch too small** — gradient noise dominating.

---

## A.5 Loss breaks partway through

| Pattern | Cause | Fix |
|---|---|---|
| Spikes and recovers | Bad batch, or LR slightly high | Usually fine. Watch it. |
| Spikes and never recovers | Diverged | Restart from last checkpoint at lower LR |
| Oscillates around a floor | LR too high for this stage | Add decay schedule (§9.7) |
| Steadily rises | LR much too high; or sign error | Check the update direction |
| Sudden jump at an epoch boundary | Data-loader state, or BatchNorm mode | Check `drop_last`, `model.train()` |
| Jump after resuming | Optimizer state not restored | Save and load `optimizer.state_dict()` (§8.9) |

---

## A.6 Train good, validation bad

**This is a generalization problem, not an optimization problem.** Do not tune the optimizer. (§9.11)

In order of what to try:

1. **More data.** Almost always the strongest fix.
2. **Data augmentation** (§10.10) — often the single biggest return on small datasets.
3. **Weight decay** — try `0.01`, `0.1`.
4. **Dropout** — `0.1` to `0.3`.
5. **Early stopping** on validation loss.
6. **Smaller model.**

**But first, rule out these three:**

- **Data leakage** — train and validation overlap. Check for duplicates.
- **Distribution mismatch** — validation set drawn differently from training.
- **Normalization computed on the full dataset** rather than training-only statistics (§4.7). Very common, silently inflates results.

**Note for language models:** they're usually *under*trained, not overfit. A large train/val gap in an LM more often means a data problem than a regularization problem.

---

## A.7 Gradient problems

**Vanishing** — early-layer gradients orders of magnitude below late-layer ones:

1. Add residual connections (§10.9)
2. Switch to ReLU/GELU from sigmoid/tanh (§7.5)
3. Fix initialization — He for ReLU, Xavier for tanh (§7.3)
4. Add normalization layers (§7.6)
5. Use pre-norm rather than post-norm (§7.6)

**Exploding** — gradient norm climbing:

1. `clip_grad_norm_(params, 1.0)` — **global** norm, not per-parameter (§7.4)
2. Lower the learning rate
3. Check initialization scale
4. Add warmup

**Diagnostic:**

```python
def grad_report(model):
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"{name:40s} {p.grad.norm().item():.3e}")
```

---

## A.8 Custom layers and autograd

| Symptom | Cause |
|---|---|
| Simple gradient checks pass, complex ones fail | **`=` instead of `+=` in `_backward`** (§5.3) |
| Gradient check fails everywhere | Wrong local derivative |
| Fails only near `x=0` for ReLU | The kink — checker artifact, not your bug (§3.12) |
| Fails only with dropout on | Randomness differs between `f(x+h)` and `f(x−h)` — fix the seed |
| Fails only in `float32` | Precision. Use `float64` and `h≈1e-5` |
| `RecursionError` in backward | Deep graph — use an iterative topological sort |
| Memory grows every step | Holding references to old graphs — use `.item()` (§8.3) |

**The test that catches the `+=` bug:**

```python
# f = a*b + a*a  — 'a' has two paths to the output
gradient_check(lambda v: v[0]*v[1] + v[0]*v[0], grad_fn, np.array([2.0, -3.0]))
```

An engine with `=` passes simple tests and fails this one. Write it first.

---

## A.9 PyTorch gotchas

| # | Mistake | Symptom |
|---|---|---|
| 1 | Softmax before `cross_entropy` | Trains slowly, no error (§8.5) |
| 2 | Missing `optimizer.zero_grad()` | Gradients accumulate |
| 3 | Missing `model.eval()` | Dropout active at eval; BatchNorm uses batch stats |
| 4 | `total += loss` not `.item()` | Memory leak |
| 5 | Device mismatch | `Expected all tensors on the same device` |
| 6 | Forgot `super().__init__()` | Parameters not registered |
| 7 | Activation on the output layer | Accuracy capped for no visible reason |
| 8 | `shuffle=True` on validation loader | Noisy metrics |
| 9 | `Adam(weight_decay=)` when you meant `AdamW` | Worse final accuracy (§9.6) |
| 10 | Gradient accumulation without `/accum` | Effective LR multiplied by `accum` (§8.12) |
| 11 | `is_causal=True` with a KV cache | Masks the cached positions wrongly (§12.9) |
| 12 | In-place op on a leaf tensor | `a leaf Variable that requires grad...` |

---

## A.10 Data pipeline

**Always eyeball your data before training.** Print five examples, decode them, look at them. An hour here saves days.

| Check | Command |
|---|---|
| Inputs and labels aligned? | Print 5 `(x, y)` pairs and verify by hand |
| Labels in range? | `assert y.min() >= 0 and y.max() < n_classes` |
| Any NaN? | `assert torch.isfinite(x).all()` |
| Class balance? | `np.bincount(y)` |
| Train/val overlap? | Hash examples, check the intersection |
| Normalization stats from training only? | Inspect where `mean`/`std` were computed |
| Shuffled? | Print the labels of one batch — all the same class means unshuffled |
| For LMs: `y` is `x` shifted by 1? | `assert (x[1:] == y[:-1]).all()` |

---

## A.11 Out of memory

**Quick fixes, in order of what to try:**

1. Reduce batch size; add gradient accumulation to compensate (§8.12)
2. `torch.autocast` with `bfloat16` — roughly halves activation memory
3. `torch.no_grad()` around evaluation
4. `optimizer.zero_grad(set_to_none=True)`
5. Gradient checkpointing — ~30% more compute, large memory saving
6. Shorten sequence length (attention memory is quadratic — §11.11)
7. Smaller model

**The memory budget** for AdamW in mixed precision is ~**16 bytes per parameter** before any activations (§12.7). A 1B model needs ~16GB just for states.

```python
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
print(torch.cuda.max_memory_allocated() / 1e9, "GB peak")
torch.cuda.reset_peak_memory_stats()
```

---

## A.12 Can't reproduce a run

```python
def set_seed(s=0):
    import random, numpy as np, torch
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Still differs? Check: `num_workers > 0` (worker seeding), non-deterministic CUDA kernels, `torch.compile` caching, and any unseeded augmentation.

**Note:** exact bitwise reproducibility across different hardware is usually not achievable. Aim for statistical reproducibility — same result within seed variance.

---

## A.13 The toolkit

Keep these in `utils/`. You'll use them all year.

```python
def initial_loss_check(loss, n_classes):
    import numpy as np
    print(f"initial loss {loss:.4f}, expected ≈ {np.log(n_classes):.4f}")

def can_overfit_tiny(model, x, y, loss_fn, opt, steps=2000):
    for _ in range(steps):
        opt.zero_grad(); l = loss_fn(model(x), y); l.backward(); opt.step()
    print(f"final loss on tiny set: {l.item():.2e}  (want < 1e-3)")

def grad_report(model):
    for n, p in model.named_parameters():
        if p.grad is not None:
            print(f"{n:40s} {p.grad.norm().item():.3e}")

def activation_report(acts):
    print(f"{'layer':>6} {'mean':>9} {'std':>9} {'dead%':>7}")
    for i, a in enumerate(acts):
        dead = 100 * (a == 0).all(dim=0).float().mean().item()
        print(f"{i:>6} {a.mean():>9.4f} {a.std():>9.4f} {dead:>7.1f}")

def hook_activations(model, store):
    import torch.nn as nn
    def make(name):
        def fn(m, i, o): store[name] = o.detach()
        return fn
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.ReLU, nn.GELU)):
            mod.register_forward_hook(make(name))
```

---

## A.14 When nothing works

Run the Unstuck Protocol (§0.3):

1. Name the problem in one precise sentence
2. Check the prerequisite — is this a gap from three steps back?
3. Shrink it — smallest possible version, print everything
4. Second explanation — different author, different medium
5. Print all intermediates
6. Sleep on it
7. Park it in `PARKED.md`, move on, review Sunday
8. Ask a human or a model — with the precise sentence, the minimal reproduction, and what you already tried

**Two focused hours across two days.** Not six hours in one night — that's how runs end, not how bugs get fixed.

---

*Next: Appendix B — Math Reference Card*
