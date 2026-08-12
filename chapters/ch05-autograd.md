# Chapter 5 — Backpropagation and Autograd

**Time: 14–18 days** (Weeks 7–9 of the plan)

**Prerequisite:** Chapters 3 and 4 checkpoints passed cold. Specifically: the chain rule, the multi-path rule of §3.5, and the two-layer hand derivation of §3.11. If any of those are shaky, go back now. This chapter is pure chain rule and it will be miserable otherwise.

**This is the hardest chapter in the book.** Budget more time than you think. It is also the one that changes what you are: after this, you will have written the thing that PyTorch does, and neural networks will stop being magic permanently.

**What you'll build:** a working automatic differentiation engine — your own miniature PyTorch — and a neural network library on top of it, trained end to end.

---

## 5.0 The problem with what you did in Chapter 4

In Chapter 4 you derived every gradient by hand. It worked because the models were tiny.

Now consider a real network: 96 layers, attention, normalization, residual connections, 7 billion parameters. Deriving those gradients by hand is not merely tedious — it's impossible in practice, and it would need redoing every time you changed the architecture.

Worse for a researcher: **it makes experimentation expensive.** If trying a new architecture means a week of calculus, you try very few architectures. The whole pace of the field depends on this being free.

So: build a machine that does the calculus.

The insight is that you already know the algorithm. §3.11 walked backward through a two-layer network mechanically — each step used only the step before it and one local derivative. Nothing about that procedure was specific to two layers, or to sigmoid, or to squared error.

**Automatic differentiation is that procedure, automated.**

---

## 5.1 The computational graph

Any computation decomposes into a graph of elementary operations.

Take `L = (w·x + b − y)²` with `x = 2, w = 3, b = 1, y = 5`:

```
   w=3   x=2
     \   /
      [*]  →  z₁ = 6
        \      b=1
         \    /
          [+]  →  z₂ = 7
            \      y=5
             \    /
              [-]  →  z₃ = 2
                \
               [**2]  →  L = 4
```

Every node holds a value. Every edge carries a dependency. The forward pass fills in the values, moving up.

The backward pass moves down, and at each node it needs only one thing: **the derivative of that node's output with respect to its own inputs.** The `*` node knows that if you nudge `w`, `z₁` moves by `x`. That's a purely local fact — the `*` node has no idea a squaring operation exists three levels above.

This is the design insight that makes autodiff possible:

> **Every operation knows only its own local derivative. The chain rule composes them into global gradients. No operation needs global knowledge.**

That's why you can add a new operation to PyTorch by writing twenty lines, and it immediately works inside any architecture.

---

## 5.2 Forward mode vs reverse mode

Two ways to traverse the graph. Understanding why one wins is genuinely important — it explains the shape of the entire field.

**Forward mode.** Pick one *input*. Sweep forward, computing how every node changes as that input changes. One sweep gives you `∂(everything)/∂(one input)`.

**Reverse mode.** Pick one *output*. Sweep backward, computing how the output changes as every node changes. One sweep gives you `∂(one output)/∂(everything)`.

Now count the cost for a neural network:

- Inputs (parameters): **N**, often billions
- Outputs (the loss): **1**

| Mode | Sweeps needed | Cost for N = 7×10⁹ |
|---|---|---|
| Forward | one per input → N | 7 billion sweeps |
| **Reverse** | one per output → **1** | **1 sweep** |

Reverse mode gets every gradient in a single backward pass, costing roughly the same as one forward pass — **regardless of how many parameters there are.**

That is the entire reason deep learning is computationally feasible. It's not an optimization; it's the difference between possible and impossible.

(Forward mode isn't useless — it wins when you have few inputs and many outputs. That's just never the shape of a training problem.)

**"Backpropagation" is reverse-mode automatic differentiation applied to neural networks.** Same thing, different name, from a different research community.

---

## 5.3 The algorithm

Three steps.

**1. Build the graph** during the forward pass. Each operation records which nodes it consumed and how to compute its local derivative.

**2. Topologically sort** the nodes. A node's gradient must be fully accumulated before it passes anything on — and it isn't fully accumulated until *every* consumer downstream has contributed. Topological order guarantees that.

**3. Sweep backward.** Seed the output node with `grad = 1` (since `∂L/∂L = 1`), then visit nodes in reverse topological order, each pushing its gradient to its inputs.

### The rule that everything hinges on

From §3.5: **if a node feeds multiple consumers, its gradient is the sum of the contributions from all of them.**

In code this means `+=`, never `=`.

```python
self.grad += local_derivative * out.grad     # ✓ correct
self.grad  = local_derivative * out.grad     # ✗ silently wrong
```

**This is the single most common bug in every hand-written autograd engine ever built, including the one you're about to write.** It produces no error. Your network trains — badly — and you have no idea why. Write it in your notes now.

---

## 5.4 Building the engine

Build this incrementally, testing at every step. Do not type the whole class and then debug it.

### Step 1: the node

```python
import math


class Value:
    """A scalar with a gradient and a record of how it was produced."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None    # set by whichever op created this
        self._prev = set(_children)      # the nodes this one came from
        self._op = _op                   # for debugging / visualization

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

Note `self._prev` is a `set`, which requires `Value` to be hashable. It is by default, because we never define `__eq__`. **If you later add an `__eq__` method, Python silently makes the class unhashable and this breaks.** Don't.

### Step 2: addition

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out
```

**Why `1.0`?** For `c = a + b`, `∂c/∂a = 1` and `∂c/∂b = 1`. Addition passes gradient through unchanged to both inputs — it's a gradient *router*.

The closure captures `self`, `other`, and `out`, so calling `out._backward()` later still has everything it needs. If closures are unfamiliar, stop and read about them for twenty minutes — this pattern is the whole engine.

### Step 3: multiplication

```python
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad

        out._backward = _backward
        return out
```

For `c = a·b`, `∂c/∂a = b` and `∂c/∂b = a`. **Multiplication swaps the values.** Worth remembering — it's why gradients through a multiply are large when the *other* operand is large, which is the seed of exploding gradients.

### Step 4: powers, and the operators built from these two

```python
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only numeric powers"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):          return self * -1
    def __sub__(self, other):   return self + (-other)
    def __truediv__(self, other): return self * other ** -1

    # reflected operators — these make  2 * a  and  1 - a  work
    def __radd__(self, other):  return self + other
    def __rmul__(self, other):  return self * other
    def __rsub__(self, other):  return (-self) + other
    def __rtruediv__(self, other): return (self ** -1) * other
```

**Why the `__r*__` methods matter.** When Python evaluates `2 * a`, it first asks `int.__mul__(2, a)`, which returns `NotImplemented`, then falls back to `a.__rmul__(2)`. Without these, half your expressions raise `TypeError` — and the error message is confusing enough that people lose an hour to it.

Notice that subtraction and division are *defined in terms of* add, mul and pow. You never write their backward passes. **Every operation you express through existing ones gets its derivative for free.** That compositionality is the reason autodiff scales to enormous libraries.

### Step 5: nonlinearities

```python
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad      # §3.3

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad                 # d/dx eˣ = eˣ = out.data

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out
```

Look at `tanh` and `exp`: both reuse a value computed in the forward pass (`t`, `e`). This is the pattern you noticed twice in Chapter 3 — **the backward pass is cheap because the forward pass already did the work.**

### Step 6: the backward sweep

```python
    def backward(self):
        # 1. topological order — children before parents
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        # 2. seed:  ∂L/∂L = 1
        self.grad = 1.0

        # 3. sweep in reverse
        for node in reversed(topo):
            node._backward()
```

**Why topological order is non-negotiable.** Consider `a → b → d` and `a → c → d`. Node `a` receives gradient through both `b` and `c`. If you called `a._backward()` before `c` had contributed, `a` would push an incomplete gradient downstream — and every node below it would be wrong.

Topological order guarantees that when you reach a node, every consumer of it has already run.

The recursive `build` will hit Python's recursion limit on very deep graphs. Exercise 12 asks you to write an iterative version.

---

## 5.5 Testing the engine

**Do not proceed until these pass.** Your Chapter 3 gradient checker is exactly the right tool.

```python
import numpy as np
from utils.gradcheck import numerical_gradient, relative_error


def test_against_numeric(build_expr, x0, names=None):
    """
    build_expr : takes a list of Values, returns a single output Value
    x0         : np.array of starting values
    """
    vals = [Value(v) for v in x0]
    out = build_expr(vals)
    out.backward()
    analytic = np.array([v.grad for v in vals])

    def f(arr):
        return build_expr([Value(v) for v in arr]).data

    numeric = numerical_gradient(f, x0.copy())
    err = relative_error(analytic, numeric).max()
    print(f"max rel error {err:.3e}  {'PASS' if err < 1e-5 else 'FAIL'}")
    return err < 1e-5


# a simple expression
test_against_numeric(lambda v: v[0] * v[1] + v[0], np.array([2.0, -3.0]))

# THE IMPORTANT TEST: a node used twice
# f = a*b + a*a  — 'a' has two paths to the output.
# If you wrote '=' instead of '+=', this fails and the simple test above passes.
test_against_numeric(lambda v: v[0] * v[1] + v[0] * v[0], np.array([2.0, -3.0]))

# deep chain with nonlinearities
test_against_numeric(
    lambda v: ((v[0] * v[1]).tanh() + v[2].relu() * v[0]).exp() * v[1],
    np.array([0.5, -1.2, 0.8]),
)
```

**The second test is the one that matters.** A broken `=`-instead-of-`+=` engine passes the first test and fails the second. That's exactly why the bug survives: casual testing doesn't catch it.

### Seeing the graph

Useful when something is wrong:

```python
def print_graph(node, indent=0, seen=None):
    seen = seen if seen is not None else set()
    tag = f"{'  ' * indent}{node._op or 'leaf':>6} = {node.data:8.4f}  grad {node.grad:8.4f}"
    if id(node) in seen:
        print(tag + "   (shared)")
        return
    seen.add(id(node))
    print(tag)
    for child in node._prev:
        print_graph(child, indent + 1, seen)
```

(If you want proper pictures, `pip install graphviz` and render `_prev` edges. Optional but genuinely helpful for building intuition — seeing a node marked `(shared)` makes the accumulation rule obvious.)

---

## 5.6 A neural network library on top

Now the payoff. With gradients automatic, defining a network is just writing the forward pass.

```python
import random


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    def __init__(self, n_in, nonlin=True):
        # scaled init — see §7 for why the 1/√n_in factor matters
        scale = n_in ** -0.5
        self.w = [Value(random.gauss(0, 1) * scale) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, n_in, n_out, **kw):
        self.neurons = [Neuron(n_in, **kw) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, n_in, n_outs):
        sizes = [n_in] + n_outs
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=(i != len(n_outs) - 1))
            for i in range(len(n_outs))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
```

Note the last layer has `nonlin=False`. **Output layers are almost always linear** — you don't want a ReLU clamping your logits at zero. Getting this wrong is a common beginner bug that caps your model's accuracy for no visible reason.

### Training it on XOR

XOR is the right first test, and not arbitrarily: it is **not linearly separable**. No single linear layer can solve it, at any width, ever. It is the concrete demonstration of the theorem you proved in §2.8.

```python
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0.0, 1.0, 1.0, 0.0]

model = MLP(2, [8, 8, 1])
print(f"{len(model.parameters())} parameters")

for step in range(300):
    # forward
    preds = [model(x) for x in X]
    loss = sum((p - y) ** 2 for p, y in zip(preds, Y)) * (1.0 / len(Y))

    # backward  — zero first!
    model.zero_grad()
    loss.backward()

    # update
    lr = 0.1
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 50 == 0:
        print(f"step {step:4d}  loss {loss.data:.6f}")

print([round(model(x).data, 3) for x in X])     # ≈ [0, 1, 1, 0]
```

**`model.zero_grad()` before every `backward()`.** Gradients accumulate by design — that's what `+=` does. Forget to clear them and each step uses the sum of all previous gradients, which produces a loss curve that looks vaguely like it's training and never converges properly. This is bug #2 on the list below.

**Then prove the point:** rerun with `MLP(2, [8, 8, 1])` where every layer has `nonlin=False`. It will fail, permanently, at loss ≈ 0.25 — predicting 0.5 for everything. That's §2.8 made concrete. Do it; it's worth five minutes.

---

## 5.7 The bugs you will hit

Written in the order of how often they occur and how long they cost.

| # | Bug | Symptom | Fix |
|---|---|---|---|
| 1 | `=` instead of `+=` in a `_backward` | Trains badly; no error. Simple tests pass | Test with a node used twice (§5.5) |
| 2 | Forgot `zero_grad()` | Loss decreases oddly then stalls or explodes | Zero before every backward |
| 3 | Missing `__radd__`/`__rmul__` | `TypeError` on `2 * a` | Add the reflected operators |
| 4 | `sum()` without a `Value` start | `int + Value` error | `sum(gen, Value(0.0))` or start with `self.b` |
| 5 | ReLU on the output layer | Accuracy plateaus; half the outputs stuck at 0 | `nonlin=False` on the last layer |
| 6 | Graph rebuilt but old one retained | Memory grows every step | Don't keep references to old loss nodes |
| 7 | `RecursionError` on deep graphs | Crash in `build()` | Iterative topological sort (ex. 12) |
| 8 | Comparing Values (`if a > b`) | `TypeError` or nonsense | Compare `.data` |
| 9 | Defined `__eq__` | `unhashable type` | Don't; `_prev` is a set |
| 10 | Learning rate wildly wrong | `nan`, or no movement | §4.15's table |

**When something is wrong, the procedure is fixed:** gradient-check the smallest expression that reproduces it. Not the network — a two-node expression. Shrink until it's trivially inspectable. That's Unstuck Protocol step 3, and this chapter is where it earns its keep.

---

## 5.8 Scaling up: tensors

The scalar engine is correct but slow — a `(128, 784) @ (784, 256)` layer would create 25 million `Value` objects. Real frameworks operate on whole arrays, where one node holds an entire tensor.

The structure is identical. Two things change.

### Matmul backward

For `C = A @ B` with `A:(n,k)`, `B:(k,m)`, `C:(n,m)`:

```
dA = dC @ Bᵀ        (n,m) @ (m,k) → (n,k)  ✓
dB = Aᵀ @ dC        (k,n) @ (n,m) → (k,m)  ✓
```

You derived the first of these in §2.9. The second follows the same way. **And notice: the shapes leave only one legal arrangement each.** When you can't remember the formula, write the shapes and there's exactly one thing that type-checks. Use this constantly.

### Broadcasting backward — the genuinely tricky part

If the forward pass broadcast a `(d,)` bias across a `(n, d)` batch, the incoming gradient is `(n, d)` but the bias needs `(d,)`.

**The rule: broadcasting in the forward pass becomes summing in the backward pass.** Which is just the multi-path rule again — one bias value influenced `n` outputs, so its gradient is the sum over all of them.

```python
def unbroadcast(grad, shape):
    """Reduce `grad` down to `shape`, undoing numpy broadcasting."""
    while grad.ndim > len(shape):          # drop leading axes that were added
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):        # collapse axes that were size 1
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
```

Then every binary op ends with `unbroadcast(g, self.data.shape)`. Getting this right is the main work of exercise 13, and it's the part of tensor autograd people underestimate.

### A tensor node

```python
import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad  += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad  += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    # backward() is identical to the scalar version, except the seed is
    # np.ones_like(self.data) for a scalar output.
```

---

## 5.9 What real frameworks add

You've now built the core. PyTorch is this plus engineering:

- **GPU kernels** — the same operations, executed on thousands of cores.
- **Fused operations** — softmax+cross-entropy as one kernel (§4.11), avoiding a memory round-trip.
- **Memory management** — freeing the graph after backward; gradient checkpointing to trade compute for memory.
- **A large operation library** — convolutions, attention, normalizations, each with a hand-written backward.
- **`torch.compile`** — tracing the graph and optimizing it ahead of time.
- **Distributed training** — synchronizing gradients across many machines.

None of that changes the idea. When you open PyTorch in Chapter 8, you'll recognize `Tensor.grad`, `.backward()`, `zero_grad()` and `requires_grad` as things you built. **That recognition is the whole point of this chapter.**

---

## 5.10 Exercises

**1.** Build `Value` with `__add__`, `__mul__`, and `backward()`. Test on `f = a*b + c` at `a=2, b=-3, c=10`, by hand and against the checker.

**2.** Add `__pow__`, `__neg__`, `__sub__`, `__truediv__` and all the reflected operators. Verify `2*a`, `a-1`, `1-a`, `a/2`, `2/a` all work.

**3.** Add `tanh`, `exp`, `log`, `relu`. Gradient-check each in isolation.

**4.** **The accumulation test.** Gradient-check `f = a*b + a*a`. Then deliberately change one `+=` to `=` and confirm it now fails. Keep the broken version's output in your notes — knowing what the bug *looks like* is worth as much as knowing the fix.

**5.** Implement `sigmoid()` two ways: as a primitive with its own `_backward`, and as a composition `(1 + (-x).exp()) ** -1` using only existing ops. Verify they give identical gradients. This demonstrates why compositionality matters.

**6.** Write `print_graph`. Run it on a graph where a node is used twice. Confirm you can see the sharing.

**7.** Build `Module`, `Neuron`, `Layer`, `MLP`. Count parameters for `MLP(2, [8, 8, 1])` and verify by hand.

**8.** Train on XOR to a loss below `0.001`. Plot the loss curve.

**9.** **The §2.8 demonstration.** Train the same MLP with every layer linear (`nonlin=False`). Show it cannot get below ≈0.25 loss. Explain why in one paragraph, referencing the collapse proof.

**10.** Reproduce bugs 1, 2 and 5 from §5.7 deliberately. Record what each looks like in the loss curve. This is worth far more than reading about them.

**11.** Implement `zero_grad`. Then train *without* it and plot the loss curve alongside the correct one.

**12.** Rewrite the topological sort iteratively (explicit stack, no recursion). Verify it handles a 5,000-node chain that crashes the recursive version.

**13.** **The big one.** Build the `Tensor` version: `+`, `*`, `@`, `sum`, `relu`, with correct `unbroadcast`. Gradient-check a full linear layer `Y = X @ W + b` with `X:(8,4)`, `W:(4,3)`, `b:(3,)`. Verify `b`'s gradient has shape `(3,)` and equals the column sums of `dY`.

**14.** Rebuild Chapter 4's logistic regression using your engine — no hand-derived gradients. Confirm it reaches the same solution. This is the moment the chapter's value becomes obvious.

**15.** **Chapter project.** Train an MLP on `sklearn.datasets.make_moons` (200 points, `noise=0.1`) using only your engine. Requirements: train/val split, a loss curve, a plotted decision boundary, and an architecture comparison across `[8,1]`, `[16,16,1]`, `[64,64,1]`. Write up which won and why.

---

## 5.11 Solutions

<details>
<summary>Open only after attempting — this chapter especially</summary>

```python
import math, random
import numpy as np
import matplotlib.pyplot as plt
from utils.gradcheck import numerical_gradient, relative_error


# ---------- 1-3: the engine ----------
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data); self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children); self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _b(): self.grad += out.grad; other.grad += out.grad
        out._backward = _b; return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _b():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _b; return out

    def __pow__(self, k):
        assert isinstance(k, (int, float))
        out = Value(self.data ** k, (self,), f"**{k}")
        def _b(): self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _b; return out

    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,), "tanh")
        def _b(): self.grad += (1 - t*t) * out.grad
        out._backward = _b; return out

    def exp(self):
        e = math.exp(self.data); out = Value(e, (self,), "exp")
        def _b(): self.grad += e * out.grad
        out._backward = _b; return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")
        def _b(): self.grad += out.grad / self.data
        out._backward = _b; return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def _b(): self.grad += (out.data > 0) * out.grad
        out._backward = _b; return out

    def sigmoid(self):                      # ex 5, primitive version
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")
        def _b(): self.grad += s * (1 - s) * out.grad
        out._backward = _b; return out

    def __neg__(self):            return self * -1
    def __sub__(self, o):         return self + (-o)
    def __truediv__(self, o):     return self * o ** -1
    def __radd__(self, o):        return self + o
    def __rmul__(self, o):        return self * o
    def __rsub__(self, o):        return (-self) + o
    def __rtruediv__(self, o):    return (self ** -1) * o
    def __repr__(self):           return f"Value({self.data:.4f}, grad={self.grad:.4f})"

    def backward(self):
        topo, seen, stack = [], set(), [(self, False)]
        while stack:                                   # ex 12: iterative
            node, expanded = stack.pop()
            if expanded:
                topo.append(node); continue
            if node in seen: continue
            seen.add(node); stack.append((node, True))
            for c in node._prev:
                stack.append((c, False))
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# ---------- 4: the accumulation test ----------
def check(expr, x0):
    vs = [Value(v) for v in x0]; expr(vs).backward()
    an = np.array([v.grad for v in vs])
    nu = numerical_gradient(lambda a: expr([Value(v) for v in a]).data, x0.copy())
    e = relative_error(an, nu).max()
    print(f"  rel err {e:.2e}  {'PASS' if e < 1e-5 else 'FAIL'}")

print("simple :"); check(lambda v: v[0]*v[1] + v[2], np.array([2., -3., 10.]))
print("shared :"); check(lambda v: v[0]*v[1] + v[0]*v[0], np.array([2., -3.]))
# With '=' instead of '+=' the first PASSES and the second FAILS badly (~0.6).


# ---------- 5 ----------
a = Value(0.7); s1 = a.sigmoid(); s1.backward(); g1 = a.grad
a2 = Value(0.7); s2 = (1 + (-a2).exp()) ** -1; s2.backward()
assert abs(g1 - a2.grad) < 1e-12


# ---------- 6-9: the network ----------
class Module:
    def parameters(self): return []
    def zero_grad(self):
        for p in self.parameters(): p.grad = 0.0

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        s = nin ** -0.5
        self.w = [Value(random.gauss(0, 1) * s) for _ in range(nin)]
        self.b = Value(0.0); self.nonlin = nonlin
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act
    def parameters(self): return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout, **kw):
        self.neurons = [Neuron(nin, **kw) for _ in range(nout)]
    def __call__(self, x):
        o = [n(x) for n in self.neurons]; return o[0] if len(o) == 1 else o
    def parameters(self): return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
    def __init__(self, nin, nouts, nonlin=True):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1],
                             nonlin=(nonlin and i != len(nouts) - 1))
                       for i in range(len(nouts))]
    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x
    def parameters(self): return [p for l in self.layers for p in l.parameters()]


def train_xor(nonlin=True, steps=400, lr=0.1):
    random.seed(0)
    X = [[0,0],[0,1],[1,0],[1,1]]; Y = [0.,1.,1.,0.]
    m = MLP(2, [8, 8, 1], nonlin=nonlin); hist = []
    for _ in range(steps):
        loss = sum((m(x) - y)**2 for x, y in zip(X, Y)) * 0.25
        m.zero_grad(); loss.backward()
        for p in m.parameters(): p.data -= lr * p.grad
        hist.append(loss.data)
    return m, hist

m, h_nl = train_xor(True)
_, h_lin = train_xor(False)
print("relu MLP  :", [round(m(x).data, 3) for x in [[0,0],[0,1],[1,0],[1,1]]])
print("final loss:", h_nl[-1], "vs linear:", h_lin[-1])
plt.semilogy(h_nl, label="with ReLU"); plt.semilogy(h_lin, label="all linear")
plt.legend(); plt.show()
# ex 9: the linear stack collapses to one affine map (§2.8). XOR is not
# linearly separable, so the best any affine map can do is predict the mean,
# 0.5 everywhere, giving MSE 0.25. It is stuck there by mathematics, not by
# optimization difficulty — more steps, more width, and lower LR change nothing.


# ---------- 13: tensors ----------
def unbroadcast(g, shape):
    while g.ndim > len(shape): g = g.sum(axis=0)
    for i, d in enumerate(shape):
        if d == 1: g = g.sum(axis=i, keepdims=True)
    return g

class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children); self._op = _op

    def __add__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        out = Tensor(self.data + o.data, (self, o), "+")
        def _b():
            self.grad += unbroadcast(out.grad, self.data.shape)
            o.grad    += unbroadcast(out.grad, o.data.shape)
        out._backward = _b; return out

    def __matmul__(self, o):
        out = Tensor(self.data @ o.data, (self, o), "@")
        def _b():
            self.grad += out.grad @ o.data.T
            o.grad    += self.data.T @ out.grad
        out._backward = _b; return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum")
        def _b(): self.grad += np.ones_like(self.data) * out.grad
        out._backward = _b; return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,), "relu")
        def _b(): self.grad += (self.data > 0) * out.grad
        out._backward = _b; return out

    def backward(self):
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            n, ex = stack.pop()
            if ex: topo.append(n); continue
            if n in seen: continue
            seen.add(n); stack.append((n, True))
            for c in n._prev: stack.append((c, False))
        self.grad = np.ones_like(self.data)
        for n in reversed(topo): n._backward()

rng = np.random.default_rng(0)
Xd, Wd, bd = rng.standard_normal((8,4)), rng.standard_normal((4,3)), rng.standard_normal(3)

def loss_np(flat):
    W = flat[:12].reshape(4,3); b = flat[12:]
    return np.maximum(Xd @ W + b, 0).sum()

Xt, Wt, bt = Tensor(Xd), Tensor(Wd), Tensor(bd)
((Xt @ Wt) + bt).relu().sum().backward()
an = np.concatenate([Wt.grad.ravel(), bt.grad])
nu = numerical_gradient(loss_np, np.concatenate([Wd.ravel(), bd]))
print("tensor engine rel err:", relative_error(an, nu).max())
assert bt.grad.shape == (3,)
# b's gradient is the column-sum of dY: one bias value fed 8 outputs, so the
# multi-path rule sums over the batch. That is what unbroadcast implements.
```

</details>

---

## 5.12 Chapter 5 checkpoint

Cold — blank file, no notes, no internet. This is the most important checkpoint in the book.

- [ ] **Write the full `Value` class** — `add`, `mul`, `pow`, `tanh`, `relu`, reflected operators, topological `backward()`. **Target: 45 minutes.**
- [ ] Gradient-check it on an expression where a node is used twice. Must pass.
- [ ] Explain why reverse mode beats forward mode for neural networks, with the cost argument.
- [ ] Explain why gradients accumulate with `+=`, referencing §3.5.
- [ ] Explain why topological order is required, with a concrete example of what breaks.
- [ ] Build `Neuron`, `Layer`, `MLP` on top and train XOR to loss < 0.01. **Target: 30 minutes.**
- [ ] State the matmul backward formulas and derive them from shapes alone.
- [ ] Explain what `unbroadcast` does and why broadcasting forward means summing backward.

**Do not advance until the first item passes.** Everything after this chapter assumes automatic differentiation is intuitive to you. If it takes three attempts across a week, that's normal and fine — this is the chapter people repeat.

### Anki cards

- Forward vs reverse mode — which for NNs and why
- Why `+=` and not `=` in `_backward`?
- Why topological order?
- Local derivative of `+`, of `*`
- `d/dx tanh`, `d/dx relu`, `d/dx exp`
- Why does the backward pass reuse forward values?
- matmul backward: `dA = ?`, `dB = ?`
- Broadcasting forward ⟹ ? backward
- Why must the output layer be linear?
- What does `zero_grad` do and what breaks without it?

### Deliverables

```
autograd/engine.py     the Value class
autograd/nn.py         Module, Neuron, Layer, MLP
autograd/tensor.py     the Tensor version with unbroadcast
tests/test_engine.py   gradient checks, including the shared-node test
projects/moons.py      exercise 15, written up
```

```bash
git add .
git commit -m "Chapter 5: autograd engine, MLP library, XOR and moons"
git push
```

### Write-up

800 words: **"I wrote PyTorch's core in 100 lines."** Walk through the graph, the local gradient principle, the reverse-mode cost argument, and the `+=` bug with the exact test that catches it. Include your loss curves from exercise 9 showing the linear network stuck at 0.25.

This is the best blog post you will have written so far, and it's a genuinely credible artifact — being able to explain autodiff clearly is a real signal to anyone evaluating you.

**You have now built the engine that every modern AI system runs on.** From here the chapters get easier for a while: Chapter 6 gives you the probability behind the loss functions you've been using, and Chapter 7 turns your MLP into something that trains well rather than merely trains.

---

*Next: Chapter 6 — Probability and Information Theory*
