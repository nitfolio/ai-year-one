# Chapter 8 — PyTorch

**Time: 10–12 days** (Weeks 15–16 of the plan)

**Prerequisite:** Chapter 5 checkpoint (you can write the autograd engine cold) and Chapter 7's initialization derivation.

**What you'll be able to do at the end:** use PyTorch fluently, recognize every piece of it as something you already built, write a training script you'll reuse for the rest of the year, and avoid the dozen gotchas that cost everyone else a week.

---

## 8.0 Why you waited

You could have started here in week one. Most courses do, and most people who take them end up able to *use* PyTorch without knowing what it does.

Because you built the engine yourself, this chapter is mostly renaming. `loss.backward()` is your `backward()`. `optimizer.zero_grad()` is your `zero_grad()`, and you know exactly why it's necessary — because gradients accumulate with `+=`, because a node can have multiple consumers, because of §3.5.

**That difference compounds.** When a PyTorch training run goes wrong in an unusual way, someone who learned PyTorch as an API is stuck googling. You'll be reasoning about the graph.

This chapter is where the last three chapters get cashed in.

---

## 8.1 Setup

```bash
pip install torch torchvision
```

Then figure out what hardware you have:

```python
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()      # Apple Silicon
    else "cpu"
)
print(device, torch.__version__)
```

### On getting compute

You do not need to buy a GPU this year. Practical options, roughly in order:

- **Kaggle Notebooks** — around 30 GPU-hours per week, free, no card required. The most generous free tier available and badly underused. Start here.
- **Google Colab** — free tier gives you a GPU with session limits; fine for experiments up to a few hours.
- **Your own CPU** — genuinely sufficient through Chapter 11. MLPs on MNIST, small CNNs, and every exercise in this chapter run fine on a laptop.
- **Rented GPUs** (Vast.ai, RunPod, Lambda) — only needed for Chapter 12's larger training runs. Budget a small amount for a handful of runs late in the year.

**Do not let hardware become a reason to stall.** Everything up to Chapter 12 is CPU-feasible if you keep models small, and keeping models small while learning is correct anyway.

---

## 8.2 Tensors

A `torch.Tensor` is your `Tensor` class from §5.8, with GPU support and several hundred more operations.

```python
import torch

torch.tensor([1., 2., 3.])            # from data
torch.zeros(2, 3)
torch.ones(2, 3)
torch.arange(6).reshape(2, 3)
torch.randn(2, 3)                     # standard normal
torch.linspace(0, 1, 5)

x = torch.randn(3, 4)
x.shape          # torch.Size([3, 4])
x.dtype          # torch.float32  ← note: NOT float64 like numpy
x.device         # cpu
```

**The `float32` default is a real difference from NumPy** and it matters for gradient checking: `float32` has ~7 decimal digits of precision, so use `h ≈ 1e-3` and expect relative errors around `1e-3`, not `1e-10`. Cast to `float64` when you need a strict check.

Everything from Chapters 1 and 2 transfers directly:

```python
A @ B            # matmul
A * B            # elementwise
A.T              # transpose
x.sum(dim=1)     # 'dim' instead of numpy's 'axis' — same meaning
x.mean(dim=0, keepdim=True)
x.view(-1, 8)    # reshape, requires contiguous memory
x.reshape(-1, 8) # reshape, copies if needed — prefer this
x[:, None]       # add an axis, same as numpy
```

Two NumPy names change: `axis` → `dim`, `keepdims` → `keepdim`. That's it.

Conversion is cheap and shares memory on CPU:

```python
t = torch.from_numpy(np_array)
a = tensor.numpy()             # fails if it requires grad — use .detach().numpy()
```

---

## 8.3 Autograd

```python
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([3.0], requires_grad=True)

y = w * x + 1
z = y ** 2

z.backward()

print(x.grad)     # tensor([42.])
print(w.grad)     # tensor([28.])
```

Check it by hand: `z = (3·2+1)² = 49`, `dz/dx = 2(wx+1)·w = 2·7·3 = 42` ✓, `dz/dw = 2·7·2 = 28` ✓.

**This is your engine.** `requires_grad=True` marks a leaf node. The graph builds during the forward pass. `.backward()` seeds `∂z/∂z = 1` and sweeps in reverse topological order.

You can even inspect the graph:

```python
print(z.grad_fn)                    # <PowBackward0>
print(z.grad_fn.next_functions)     # its children
```

`grad_fn` is your `_backward`. `next_functions` is your `_prev`. Same design, industrial implementation.

### The correspondence

| Your engine | PyTorch |
|---|---|
| `Value` / `Tensor` | `torch.Tensor` |
| `.data` | `.data` (raw) or `.detach()` (safe) |
| `.grad` | `.grad` |
| `._backward` | `.grad_fn` |
| `._prev` | `.grad_fn.next_functions` |
| `.backward()` | `.backward()` |
| `zero_grad()` | `optimizer.zero_grad()` |
| `Module` | `nn.Module` |
| `parameters()` | `.parameters()` |
| `Layer(n_in, n_out)` | `nn.Linear(n_in, n_out)` |
| `.relu()` | `F.relu()` / `nn.ReLU()` |
| manual `p.data -= lr * p.grad` | `optimizer.step()` |

Print this table. When PyTorch confuses you, translate back to the code you wrote.

### Three context managers

```python
with torch.no_grad():          # build no graph — inference, evaluation
    preds = model(x)

y = x.detach()                 # same values, cut out of the graph

loss_value = loss.item()       # Python float — releases the graph
```

**`.item()` matters more than it looks.** Writing `total_loss += loss` inside a loop keeps every graph alive, and memory grows until the process dies. Writing `total_loss += loss.item()` frees them. This is the most common memory leak in beginner PyTorch code.

---

## 8.4 nn.Module

```python
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()                      # required — don't forget it
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.ln  = nn.LayerNorm(n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = F.relu(self.ln(self.fc1(x)))
        return self.fc2(x)                      # logits — no activation


model = MLP(784, 256, 10)
print(model)
print(sum(p.numel() for p in model.parameters()), "parameters")
```

`nn.Module` tracks submodules and parameters automatically. Assigning `self.fc1 = nn.Linear(...)` registers it; `model.parameters()` then yields everything recursively — exactly your `Module.parameters()`, done with `__setattr__` magic.

**Call `model(x)`, not `model.forward(x)`.** The `__call__` wrapper runs hooks that some features depend on.

### nn.Linear's weight shape

```python
layer = nn.Linear(64, 128)
layer.weight.shape      # torch.Size([128, 64])  — (out, in)
layer.bias.shape        # torch.Size([128])
```

`(out, in)`, and the input is `(batch, in)`, so internally it computes `x @ W.T + b`. This is §2.10 exactly. You already know why it looks backwards.

### Default initialization

`nn.Linear` uses a Kaiming-uniform scheme by default — reasonable for ReLU. To match Chapter 7 exactly:

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")   # He
        nn.init.zeros_(m.bias)

model.apply(init_weights)
```

---

## 8.5 Losses — and why the fused ones exist

```python
nn.MSELoss()                # regression
nn.BCEWithLogitsLoss()      # binary — takes LOGITS
nn.CrossEntropyLoss()       # multiclass — takes LOGITS
nn.NLLLoss()                # multiclass — takes LOG-PROBABILITIES
```

**The single most common PyTorch mistake:**

```python
# ✗ WRONG — double softmax, model trains badly, no error
logits = model(x)
probs = F.softmax(logits, dim=1)
loss = F.cross_entropy(probs, y)

# ✓ RIGHT
logits = model(x)
loss = F.cross_entropy(logits, y)
```

`CrossEntropyLoss` **is** `LogSoftmax + NLLLoss` fused. You know from §4.11 why they're fused: the combined gradient is `p − y`, clean and stable, while computing them separately risks `log(0)`. Same for `BCEWithLogitsLoss` versus `Sigmoid` + `BCELoss` — that's §4.10's stable formulation, implemented for you.

**Shapes for `CrossEntropyLoss`:**

```
input:  (N, C)   raw logits
target: (N,)     integer class indices — NOT one-hot
```

Passing one-hot targets is the second most common mistake.

---

## 8.6 Optimizers

```python
import torch.optim as optim

optim.SGD(model.parameters(), lr=0.01)
optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optim.Adam(model.parameters(), lr=1e-3)
optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)     # the default choice
```

An optimizer holds references to the parameters and does the update in `step()`. It replaces your hand-written `p.data -= lr * p.grad`.

**Reasonable starting points**, to be understood properly in Chapter 9:

- **AdamW, `lr=3e-4`** — the default for almost everything, transformers especially
- **SGD + momentum 0.9, `lr=0.1` with a schedule** — often better final accuracy for vision
- Halve or double the learning rate to tune; changing it by 10% does nothing

### Excluding parameters from weight decay

From §4.14: don't decay biases. Norm parameters shouldn't be decayed either.

```python
decay, no_decay = [], []
for name, p in model.named_parameters():
    if p.ndim < 2 or "bias" in name:     # biases and norm gains are 1-D
        no_decay.append(p)
    else:
        decay.append(p)

opt = optim.AdamW([
    {"params": decay,    "weight_decay": 0.01},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4)
```

Nearly every serious training script does this. Almost no tutorial mentions it.

---

## 8.7 Data loading

```python
from torch.utils.data import Dataset, DataLoader, TensorDataset

ds = TensorDataset(X_tensor, y_tensor)          # for in-memory arrays

loader = DataLoader(ds, batch_size=64, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)
```

- `shuffle=True` **for training only** — §4.12's reshuffle-every-epoch rule
- `num_workers` — parallel loading processes; set to your core count, or 0 on Windows/macOS if it hangs
- `pin_memory=True` — faster CPU→GPU transfer; only useful with CUDA
- `drop_last=True` — drops a ragged final batch; useful when BatchNorm is involved

Custom datasets need exactly two methods:

```python
class MyDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = load(self.paths[idx])
        if self.transform:
            x = self.transform(x)
        return x, self.labels[idx]
```

---

## 8.8 The canonical training loop

Learn this shape. You'll write variations of it for the rest of your career.

```python
def train_epoch(model, loader, loss_fn, optimizer, device, clip=None):
    model.train()                                   # ← enables dropout/BN training
    total, n = 0.0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)       # ← clear old gradients
        logits = model(xb)                          # forward
        loss = loss_fn(logits, yb)
        loss.backward()                             # backward

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)   # §7.4

        optimizer.step()                            # update

        total += loss.item() * xb.size(0)           # ← .item(), not loss
        n += xb.size(0)

    return total / n


@torch.no_grad()                                    # ← no graph during eval
def evaluate(model, loader, loss_fn, device):
    model.eval()                                    # ← disables dropout, BN uses running stats
    total, correct, n = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total += loss_fn(logits, yb).item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)

    return total / n, correct / n
```

Every marked line is a bug if you omit it. Specifically:

- No `zero_grad()` → gradients accumulate across steps (§5.6)
- No `model.train()`/`model.eval()` → dropout active at eval, BatchNorm using batch stats at inference
- No `torch.no_grad()` during eval → builds graphs you never use; memory and time wasted
- `loss` instead of `loss.item()` → the memory leak from §8.3

`set_to_none=True` sets gradients to `None` rather than zero. Slightly faster, slightly less memory, and now the default in recent PyTorch.

---

## 8.9 Saving, loading, reproducibility

```python
# save the state dict, not the model object
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
}, "checkpoint.pt")

ckpt = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
```

**Save `state_dict`, never `torch.save(model)`.** Pickling the whole object ties the file to your exact class definitions and directory layout, and it breaks the moment you refactor.

**Save the optimizer state too.** Adam carries momentum buffers; resuming without them causes a visible loss spike.

```python
def set_seed(seed=0):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Set the seed at the top of every experiment. **A result you can't reproduce isn't a result** — that's a research habit, not a coding one, and it's worth forming now.

---

## 8.10 The gotchas

Ordered by how much time they cost people.

| # | Mistake | Symptom | Fix |
|---|---|---|---|
| 1 | Softmax before `CrossEntropyLoss` | Trains, but badly. No error. | Pass raw logits |
| 2 | Missing `optimizer.zero_grad()` | Loss behaves strangely, then diverges | Zero every step |
| 3 | Missing `model.eval()` | Val accuracy noisy or much worse than train | Switch modes |
| 4 | `total += loss` not `.item()` | Memory grows until OOM | Use `.item()` |
| 5 | Device mismatch | `Expected all tensors on same device` | `.to(device)` everything |
| 6 | One-hot targets to `CrossEntropyLoss` | Shape error, or silent nonsense | Pass class indices |
| 7 | Forgot `super().__init__()` | Parameters not registered; model won't train | Add it |
| 8 | In-place op on a leaf | `a leaf Variable that requires grad...` | Avoid `x += 1` on parameters |
| 9 | `shuffle=True` on the val loader | Not a bug, but noise in your metrics | Only shuffle training data |
| 10 | Activation on the output layer | Accuracy capped for no visible reason | Logits are unactivated (§5.6) |
| 11 | Learning rate off by 10× | Flat or `nan` loss | §7.9 protocol |
| 12 | Normalizing with test statistics | Optimistic scores that don't hold up | Training stats only (§4.7) |

### Debugging in PyTorch

```python
torch.autograd.set_detect_anomaly(True)     # locates the op producing nan — slow, temporary

for name, p in model.named_parameters():    # per-layer gradient norms — §7.9 step 5
    if p.grad is not None:
        print(f"{name:30s} {p.grad.norm().item():.3e}")
```

The whole §7.9 protocol applies unchanged. The initial-loss check is one line:

```python
print(loss_fn(model(xb), yb).item(), "should be ≈", np.log(n_classes))
```

---

## 8.11 Experiment tracking

From now on you run many experiments, and you will not remember which produced which number. Track from the start.

```python
import wandb
wandb.init(project="year-one", config={"lr": 3e-4, "batch_size": 64})
wandb.log({"train_loss": tr, "val_loss": va, "val_acc": acc, "epoch": epoch})
```

Free for personal use. TensorBoard is the offline alternative.

**If you want zero dependencies, a CSV is genuinely fine:**

```python
import csv, json, pathlib

def log_run(path, config, rows):
    p = pathlib.Path(path); p.mkdir(parents=True, exist_ok=True)
    (p / "config.json").write_text(json.dumps(config, indent=2))
    with open(p / "metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader(); wr.writerows(rows)
```

What matters is not the tool. **What matters is that every run writes its config and its metrics somewhere, automatically.** Otherwise, three weeks from now, you'll have a good result and no idea how you got it. That has happened to every researcher at least once and it's entirely avoidable.

---

## 8.12 Performance essentials

Enough to not be slow; the details come in Chapter 12.

```python
model = model.to(device)                     # model and data on the same device

# mixed precision — roughly 2× faster on modern GPUs
scaler = torch.amp.GradScaler()
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = loss_fn(model(xb), yb)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

model = torch.compile(model)                 # graph optimization; slow first call
```

**Gradient accumulation** — simulate a large batch on small memory:

```python
accum = 4
for i, (xb, yb) in enumerate(loader):
    loss = loss_fn(model(xb), yb) / accum     # ← scale, or your LR is 4× too big
    loss.backward()
    if (i + 1) % accum == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

The `/accum` is required. Without it you're summing four batches of gradient instead of averaging, which is equivalent to quadrupling the learning rate — a subtle and popular bug.

---

## 8.13 Exercises

**1.** Reproduce the `z = (wx+1)²` example. Verify `x.grad` and `w.grad` by hand.

**2.** Inspect `z.grad_fn` and walk `next_functions` recursively to print the whole graph. Compare against your `print_graph` from §5.5.

**3.** **The parity test.** Build a 2-layer MLP in your Chapter 5 engine and in PyTorch. Copy the weights across so they're identical. Feed the same input. Assert the forward outputs match to `1e-6` and every gradient matches to `1e-5`. *This is the exercise that makes the chapter land.*

**4.** Demonstrate gradient accumulation: call `.backward()` twice without zeroing and show `.grad` doubles. Explain it using §3.5.

**5.** Show the `.item()` memory leak. Accumulate `loss` for 1000 steps and watch memory grow; then use `.item()` and show it doesn't. (`torch.cuda.memory_allocated()` on GPU, `tracemalloc` on CPU.)

**6.** Train the same model with and without `model.eval()` at validation time, using a model containing dropout. Plot both validation curves.

**7.** Demonstrate gotcha #1: train a classifier passing softmax outputs to `cross_entropy`, and again passing logits. Plot both loss curves and report both accuracies.

**8.** Verify `nn.Linear`'s weight shape is `(out, in)`, and reimplement its forward pass manually with `x @ W.T + b`, asserting equality.

**9.** Write `set_seed`. Run the same training twice and assert identical losses. Then remove the seed and show they diverge.

**10.** Implement `train_epoch` and `evaluate` from §8.8. Train an MLP on MNIST to >97% test accuracy. Compare wall-clock time against your Chapter 7 NumPy implementation.

**11.** Implement the parameter-group split so biases and norm parameters skip weight decay. Verify the groups contain what you expect by printing names and counts.

**12.** Run the §7.9 protocol entirely in PyTorch: check initial loss against `ln(k)`, overfit 10 examples, gradient-check a small model with `torch.autograd.gradcheck`, and log per-layer activation and gradient statistics using forward hooks.

**13.** Implement gradient accumulation. Show that accumulating 4 steps of batch 16 gives gradients matching one step of batch 64 (to floating-point tolerance). Then omit the `/accum` and show the mismatch.

**14.** Save a checkpoint mid-training, restart the process, load it, and confirm training resumes with a continuous loss curve. Then reload *without* the optimizer state and show the spike.

**15.** **Chapter project.** Rebuild every model from Chapters 4–7 in PyTorch: linear regression, logistic regression, softmax regression, and the MNIST MLP. For each, verify it reaches the same result as your from-scratch version. Package it as a reusable `trainer.py` with config, seeding, logging, checkpointing, and early stopping. **You will use this file for the rest of the year — make it good.**

---

## 8.14 Solutions

<details>
<summary>Open only after attempting</summary>

```python
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np


# --- 1, 4 ---
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([3.0], requires_grad=True)
z = (w * x + 1) ** 2
z.backward()
assert torch.allclose(x.grad, torch.tensor([42.]))
assert torch.allclose(w.grad, torch.tensor([28.]))

z2 = (w * x + 1) ** 2          # no zeroing
z2.backward()
print(x.grad)                  # tensor([84.]) — doubled
# Gradients accumulate with += exactly as in your engine (§3.5, §5.3).
# PyTorch does not clear them for you; that is what zero_grad() is for.


# --- 3: THE PARITY TEST ---
from autograd.tensor import Tensor      # your Chapter 5 engine

rng = np.random.default_rng(0)
W1 = rng.standard_normal((4, 8)) * 0.5;  b1 = rng.standard_normal(8) * 0.1
W2 = rng.standard_normal((8, 3)) * 0.5;  b2 = rng.standard_normal(3) * 0.1
Xd = rng.standard_normal((6, 4))

# --- yours ---
tW1, tb1 = Tensor(W1), Tensor(b1)
tW2, tb2 = Tensor(W2), Tensor(b2)
out_mine = (((Tensor(Xd) @ tW1) + tb1).relu() @ tW2 + tb2)
out_mine.sum().backward()

# --- pytorch ---
pW1 = torch.tensor(W1, requires_grad=True); pb1 = torch.tensor(b1, requires_grad=True)
pW2 = torch.tensor(W2, requires_grad=True); pb2 = torch.tensor(b2, requires_grad=True)
out_pt = (torch.tensor(Xd) @ pW1 + pb1).relu() @ pW2 + pb2
out_pt.sum().backward()

assert np.allclose(out_mine.data, out_pt.detach().numpy(), atol=1e-10)
for mine, pt in [(tW1, pW1), (tb1, pb1), (tW2, pW2), (tb2, pb2)]:
    assert np.allclose(mine.grad, pt.grad.numpy(), atol=1e-8)
print("parity: forward and all gradients match")
# This is the payoff. PyTorch is doing exactly what you wrote.


# --- 7: the softmax gotcha ---
def run(double_softmax):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 5))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    X = torch.randn(1000, 20); y = torch.randint(0, 5, (1000,))
    hist = []
    for _ in range(300):
        opt.zero_grad()
        out = m(X)
        if double_softmax:
            out = F.softmax(out, dim=1)      # WRONG
        loss = F.cross_entropy(out, y)
        loss.backward(); opt.step()
        hist.append(loss.item())
    acc = (m(X).argmax(1) == y).float().mean().item()
    return hist, acc

h_bad, a_bad = run(True); h_ok, a_ok = run(False)
print(f"double softmax: acc {a_bad:.3f}   correct: acc {a_ok:.3f}")
# The wrong version still trains — that is why the bug survives. Applying
# softmax twice compresses the logits into a narrow range, so the effective
# gradient is far smaller and learning is much slower. No error is raised.


# --- 8 ---
lin = nn.Linear(6, 3)
xb = torch.randn(4, 6)
assert lin.weight.shape == (3, 6)                       # (out, in) — §2.10
assert torch.allclose(lin(xb), xb @ lin.weight.T + lin.bias, atol=1e-6)


# --- 9 ---
def set_seed(s=0):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 11 ---
model = nn.Sequential(nn.Linear(64,128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128,10))
decay, no_decay = [], []
for n_, p in model.named_parameters():
    (no_decay if p.ndim < 2 or "bias" in n_ else decay).append(p)
print(len(decay), "decayed;", len(no_decay), "not")
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                         {"params": no_decay, "weight_decay": 0.0}], lr=3e-4)


# --- 12: forward hooks for activation statistics ---
stats = {}
def hook(name):
    def fn(mod, inp, out):
        stats[name] = (out.mean().item(), out.std().item(),
                       (out == 0).float().mean().item())
    return fn
for name, mod in model.named_modules():
    if isinstance(mod, nn.ReLU):
        mod.register_forward_hook(hook(name))
model(torch.randn(64, 64))
print(stats)          # mean, std, dead-fraction per ReLU — §7.9 step 4

# initial-loss check — §7.9 step 1
xb, yb = torch.randn(256, 64), torch.randint(0, 10, (256,))
print(F.cross_entropy(model(xb), yb).item(), "should be ≈", np.log(10))


# --- 13 ---
torch.manual_seed(0)
m1 = nn.Linear(8, 4); m2 = nn.Linear(8, 4); m2.load_state_dict(m1.state_dict())
X = torch.randn(64, 8); y = torch.randn(64, 4)

F.mse_loss(m1(X), y).backward()                     # one batch of 64
for i in range(4):                                  # four batches of 16
    sl = slice(i*16, (i+1)*16)
    (F.mse_loss(m2(X[sl]), y[sl]) / 4).backward()
assert torch.allclose(m1.weight.grad, m2.weight.grad, atol=1e-6)
print("accumulation matches full batch")
# Without the /4 the accumulated gradient is 4x too large — silently
# equivalent to a 4x learning rate.
```

</details>

---

## 8.15 Chapter 8 checkpoint

Cold — blank file, no notes.

- [ ] **Write the canonical training loop** of §8.8 from memory, including `zero_grad`, `train()`/`eval()`, `no_grad`, and `.item()`. **Target: 10 minutes.**
- [ ] Explain what `optimizer.zero_grad()` does and why it's necessary, referencing your own engine.
- [ ] State what `nn.Linear`'s weight shape is and why it looks transposed.
- [ ] Explain why `CrossEntropyLoss` takes logits, referencing §4.11.
- [ ] Explain what breaks without `model.eval()` — name both mechanisms.
- [ ] Explain the `.item()` memory leak.
- [ ] Pass the parity test: same forward and same gradients between your engine and PyTorch.
- [ ] Run the §7.9 protocol entirely in PyTorch, including the `ln(k)` initial-loss check.

Item 7 is the one that proves the last four chapters worked.

### Anki cards

- Your engine ↔ PyTorch correspondence (make several cards)
- `.detach()` vs `.data` vs `.item()`
- `torch.no_grad()` — when and why
- `CrossEntropyLoss` input/target shapes and dtypes
- Why does `CrossEntropyLoss` take logits?
- What does `model.eval()` change?
- `nn.Linear` weight shape
- Which parameters should skip weight decay?
- Gradient accumulation — and why divide by the accumulation steps
- Why save `state_dict` and not the model?
- Why save the optimizer state?
- PyTorch default dtype, and the consequence for gradient checking

### Deliverables

```
trainer.py             THE reusable training harness (exercise 15)
tests/test_parity.py   your engine vs PyTorch
projects/mnist_torch.py
utils/seed.py, utils/logging.py
```

```bash
git add .
git commit -m "Chapter 8: PyTorch, parity with own engine, reusable trainer"
git push
```

### Write-up

600 words: **"PyTorch stopped being magic when I rebuilt it."** Show the correspondence table, the parity test with real numbers, and the softmax gotcha from exercise 7 with both loss curves. The argument to make: the reason you can debug PyTorch is that you know what it's doing underneath.

**Part II is complete.** You've built an autograd engine, derived the loss functions from probability, learned to make deep networks trainable, and mapped all of it onto the industry-standard framework.

Part III is modern deep learning: optimization, convolutions, attention, and a transformer you train yourself.

---

*Next: Chapter 9 — Optimization*
