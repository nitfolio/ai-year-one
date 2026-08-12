# Chapter 1 — Setup, Python, and NumPy

**Time: 8–10 days** (Weeks 1–2 of the plan)

**What you'll be able to do at the end:** write vectorized NumPy that a professional would recognize as competent, predict the shape of any array expression before running it, and debug a shape error in under two minutes. That last skill will save you literal weeks over this year.

---

## 1.1 Environment setup

Do this once, carefully. A broken environment causes bugs that look like conceptual failures, and you'll waste days chasing the wrong thing.

### Install Python 3.12

Check what you have:

```bash
python3 --version
```

If it's below 3.10, install a newer one.

- **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install python3.12 python3.12-venv python3-pip`
- **macOS:** install [Homebrew](https://brew.sh), then `brew install python@3.12`
- **Windows:** install from python.org, **and check "Add Python to PATH"** during installation. Or use WSL2 and follow the Linux instructions — recommended, because nearly all ML tooling assumes Linux and you'll hit fewer weird problems all year.

### Create your project and a virtual environment

A virtual environment is a per-project sandbox for packages. Without one, projects fight over package versions and you eventually break everything.

```bash
mkdir ai-year-one
cd ai-year-one
python3 -m venv .venv
```

Activate it:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Your prompt should now show `(.venv)`. **You must activate the venv every time you open a new terminal.** Forgetting this is the #1 cause of "but I installed it!"

Install what you need:

```bash
pip install numpy matplotlib jupyter ipython
pip freeze > requirements.txt
```

### Editor

VS Code, with the Python extension (by Microsoft) and Jupyter extension. Set the interpreter to your venv: `Ctrl+Shift+P` → "Python: Select Interpreter" → pick the one in `.venv`.

Two settings worth turning on immediately: format on save, and the Pylance type checker in "basic" mode. They'll catch a class of mistakes before you run anything.

### Sanity check

```python
# save as check.py, run with: python check.py
import numpy as np
import matplotlib
print("numpy", np.__version__)
print("2x3 matrix:\n", np.arange(6).reshape(2, 3))
```

If that runs, you're set.

---

## 1.2 Git — the minimum that matters

You need about six commands. Ignore everything else for now.

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

git init                      # start tracking this folder
git add .                     # stage all changes
git commit -m "message"       # save a snapshot
git status                    # what's changed?
git log --oneline             # history
git push                      # send to GitHub
```

Create a `.gitignore` before your first commit — this stops you committing hundreds of megabytes of junk:

```
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/
*.pt
*.pth
.DS_Store
```

Then create an empty repo on GitHub called `ai-year-one` and connect it:

```bash
git remote add origin https://github.com/YOURNAME/ai-year-one.git
git branch -M main
git add .
git commit -m "Day 1: start"
git push -u origin main
```

**Commit at the end of every study day.** Even broken code. The commit graph becomes a year-long record of showing up, and on the days you feel like you've achieved nothing it is genuinely useful evidence to the contrary.

---

## 1.3 The Python you actually need

You do not need all of Python. You need this list, fluently.

### Types and basics

```python
x = 5                    # int
y = 3.14                 # float
name = "gradient"        # str
flag = True              # bool
nums = [1, 2, 3]         # list — ordered, mutable
point = (1, 2)           # tuple — ordered, immutable (shapes are tuples!)
config = {"lr": 0.01}    # dict — key/value
```

`tuple` matters more than beginners expect: **array shapes in NumPy are tuples**, and you'll be reading and comparing them constantly.

### Control flow

```python
for i in range(5):              # 0,1,2,3,4
    print(i)

for i, v in enumerate(['a','b']):   # index AND value
    print(i, v)

for a, b in zip([1,2], [10,20]):    # walk two lists together
    print(a, b)

while loss > 0.01:
    loss = step()

if x > 0:
    ...
elif x == 0:
    ...
else:
    ...
```

`enumerate` and `zip` appear in essentially every training loop you will ever write.

### Functions

```python
def mse(predictions, targets):
    """Mean squared error between two lists of numbers."""
    total = 0.0
    for p, t in zip(predictions, targets):
        total += (p - t) ** 2
    return total / len(predictions)


def train(data, lr=0.01, epochs=100):     # lr and epochs have defaults
    ...
```

Type hints are optional but will save you real time later:

```python
def dot(a: list[float], b: list[float]) -> float:
    ...
```

### Comprehensions

You'll read these constantly, so you need to be able to write them:

```python
squares = [x**2 for x in range(10)]
evens   = [x for x in range(20) if x % 2 == 0]
pairs   = [(i, j) for i in range(3) for j in range(3)]
lookup  = {word: len(word) for word in ["a", "bb", "ccc"]}
```

A comprehension is just a `for` loop that builds a list. If one confuses you, rewrite it as a loop — they're always equivalent.

### Classes

You need these for Chapter 5, where you build an autograd engine.

```python
class Neuron:
    def __init__(self, n_inputs):        # constructor
        self.weights = [0.0] * n_inputs   # instance attribute
        self.bias = 0.0

    def forward(self, x):                 # method
        total = self.bias
        for w, xi in zip(self.weights, x):
            total += w * xi
        return total

    def __repr__(self):                   # what print() shows
        return f"Neuron(n={len(self.weights)})"


n = Neuron(3)
print(n.forward([1.0, 2.0, 3.0]))
print(n)                                  # Neuron(n=3)
```

`self` is just "this particular object." Every method takes it as the first argument, automatically.

**Dunder methods** (double-underscore) let your objects work with Python's operators. This is exactly how you'll make `a + b` work on your own `Value` class in Chapter 5:

```python
class Value:
    def __init__(self, data):
        self.data = data

    def __add__(self, other):             # enables:  a + b
        return Value(self.data + other.data)

    def __mul__(self, other):             # enables:  a * b
        return Value(self.data * other.data)


a, b = Value(2), Value(3)
print((a + b).data)     # 5
print((a * b).data)     # 6
```

### f-strings

Use these for all printing.

```python
loss, epoch = 0.03271, 12
print(f"epoch {epoch}: loss={loss:.4f}")     # epoch 12: loss=0.0327
print(f"{loss=}")                            # loss=0.03271   <- debug shorthand
```

That `{loss=}` form prints both the variable name and its value. Use it constantly while debugging.

---

## 1.4 Debugging: the skill nobody teaches

You will spend more time debugging this year than writing new code. Being systematic about it is worth more than any single concept in this book.

### Read the traceback correctly

```
Traceback (most recent call last):
  File "train.py", line 42, in <module>
    loss = model.forward(x)
  File "model.py", line 18, in forward
    return self.W @ x
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0
```

Read it **bottom-up**:

1. **Last line = what went wrong.** Shapes don't line up for a matmul.
2. **Second-to-last file/line = where.** `model.py` line 18.
3. **Lines above = how you got there.** The call chain.

Beginners read the top first and get overwhelmed. Start at the bottom.

### Print shapes before anything else

Ninety percent of your bugs this year will be shape bugs. Make this reflexive:

```python
print(f"{x.shape=}  {W.shape=}")
```

### The three-line debugger

```python
import pdb; pdb.set_trace()     # execution stops here, you get a prompt
```

At the prompt: `n` (next line), `c` (continue), `p variable` (print), `q` (quit). That's enough.

### Binary search a bug

If you don't know where the problem is: comment out half the code. Still broken? The bug is in the remaining half. Repeat. Five rounds narrows 1000 lines to about 30. Boring, mechanical, always works.

### Rubber duck it

Explain the code out loud, line by line, to an inanimate object. This sounds like a joke. It has a very high hit rate, because the act of forcing the explanation into words exposes the assumption you skipped over silently.

---

## 1.5 NumPy: arrays and shapes

Everything from here to the end of the book runs on NumPy or its descendants. Slow down and get this properly.

### Why NumPy exists

Python lists are boxes of pointers to objects scattered around memory. NumPy arrays are contiguous blocks of raw numbers, operated on by compiled C loops. That's the whole difference, and it's worth 50–200×.

```python
import numpy as np

a = np.array([1, 2, 3])                       # 1-D, shape (3,)
b = np.array([[1, 2, 3], [4, 5, 6]])          # 2-D, shape (2, 3)

np.zeros((2, 3))          # 2x3 of zeros
np.ones((3,))             # [1., 1., 1.]
np.arange(6)              # [0 1 2 3 4 5]
np.arange(6).reshape(2,3) # 2x3
np.linspace(0, 1, 5)      # [0., 0.25, 0.5, 0.75, 1.]
np.random.randn(2, 3)     # 2x3 from standard normal
np.eye(3)                 # 3x3 identity
```

### Shape is the thing to track

```python
b = np.array([[1, 2, 3], [4, 5, 6]])

b.shape     # (2, 3)  — always a tuple
b.ndim      # 2       — number of dimensions
b.size      # 6       — total elements
b.dtype     # dtype('int64')
```

Read `(2, 3)` as "2 rows, 3 columns." For higher dimensions, read right-to-left as innermost-to-outermost. A batch of images with shape `(32, 3, 224, 224)` is 32 images × 3 colour channels × 224 height × 224 width.

**Habit to build now:** whenever you write a line involving arrays, predict the output shape *before* running it. Then check. Being wrong is informative; you'll be calibrated within a week.

### Indexing and slicing

```python
a = np.arange(10)         # [0 1 2 3 4 5 6 7 8 9]

a[0]        # 0
a[-1]       # 9
a[2:5]      # [2 3 4]      — start inclusive, stop exclusive
a[:3]       # [0 1 2]
a[5:]       # [5 6 7 8 9]
a[::2]      # [0 2 4 6 8]  — every 2nd
a[::-1]     # reversed

M = np.arange(12).reshape(3, 4)

M[1, 2]     # single element, row 1 col 2
M[1]        # entire row 1, shape (4,)
M[:, 2]     # entire column 2, shape (3,)
M[0:2, 1:3] # submatrix, shape (2, 2)
```

**Boolean masking** — used constantly for filtering:

```python
a = np.array([1, -2, 3, -4])
a > 0                 # [True False True False]
a[a > 0]              # [1 3]
a[a < 0] = 0          # ReLU, in one line: [1 0 3 0]
```

### Elementwise operations vs. matrix multiplication

This distinction trips up everyone once. Learn it now and never lose a day to it.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A * B      # ELEMENTWISE:  [[5, 12], [21, 32]]
A @ B      # MATRIX MULT:  [[19, 22], [43, 50]]
```

`*` multiplies corresponding entries. `@` is real matrix multiplication (sum over the shared dimension). In neural networks you want `@` for layers and `*` for things like masks and gating.

Rule for `@`: `(n, k) @ (k, m) → (n, m)`. The inner dimensions must match and they disappear.

### Reductions and the `axis` argument

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])       # shape (2, 3)

M.sum()             # 21            — everything
M.sum(axis=0)       # [5 7 9]       — shape (3,)
M.sum(axis=1)       # [6 15]        — shape (2,)
```

**The rule that makes `axis` finally make sense:** `axis=k` means *the axis that disappears*. `M` has shape `(2, 3)`; summing over `axis=0` removes the 2, leaving `(3,)`. Summing over `axis=1` removes the 3, leaving `(2,)`.

Say it to yourself as "collapse axis k." Same rule applies to `mean`, `max`, `min`, `argmax`, `std`.

`keepdims=True` keeps the collapsed axis as size 1, which matters enormously for broadcasting:

```python
M.sum(axis=1)                    # shape (2,)
M.sum(axis=1, keepdims=True)     # shape (2, 1)
```

You'll want the second form when normalizing rows — as in softmax, which you'll write in Chapter 4.

---

## 1.6 Broadcasting

This is the most important section in the chapter. Broadcasting is NumPy's rule for combining arrays of different shapes, it's used in every neural network, and its failure mode is *silent wrong answers* rather than errors.

### The rule

Align shapes **from the right**. For each dimension, they're compatible if:

- they're equal, **or**
- one of them is 1 (that one gets stretched), **or**
- one array has fewer dimensions (missing dims are treated as 1)

Worked examples:

```
(3, 4)  and  (4,)          →  (3, 4)     ✓  the (4,) is applied to each row
(3, 4)  and  (3, 1)        →  (3, 4)     ✓  the column is applied to each column
(3, 1)  and  (1, 4)        →  (3, 4)     ✓  both stretch — outer-product shaped
(3, 4)  and  (3,)          →  ERROR      ✗  aligning right: 4 vs 3
(2,3,4) and  (3, 4)        →  (2, 3, 4)  ✓
(2,3,4) and  (4,)          →  (2, 3, 4)  ✓
```

In code:

```python
X = np.arange(12).reshape(3, 4)      # (3, 4)
b = np.array([10, 20, 30, 40])       # (4,)
X + b                                 # (3, 4) — adds b to every row
```

That is exactly the bias addition in a neural network layer: one bias vector, added to every example in the batch, with no loop.

### The trap

```python
a = np.array([1, 2, 3])          # (3,)
b = np.array([[1], [2], [3]])    # (3, 1)
a + b                            # (3, 3)  !!!
```

You probably wanted `[2, 4, 6]`. You got a 3×3 matrix, no error, and the wrong answer flows silently into your loss. **This is the single most common silent bug in numerical code.**

Defense:

1. Print shapes whenever an operation surprises you.
2. Be deliberate with `reshape(-1, 1)` and `keepdims=True` — know which one you want.
3. Assert shapes at function boundaries:

```python
def layer(X, W, b):
    assert X.ndim == 2, f"expected 2-D X, got {X.shape}"
    out = X @ W + b
    assert out.shape == (X.shape[0], W.shape[1])
    return out
```

Those asserts cost nothing and will catch bugs that would otherwise cost you a day.

### Adding and removing axes

```python
a = np.array([1, 2, 3])          # (3,)

a.reshape(-1, 1)                 # (3, 1)  — column. -1 means "infer it"
a.reshape(1, -1)                 # (1, 3)  — row
a[:, None]                       # (3, 1)  — same as reshape(-1,1), common in real code
a[None, :]                       # (1, 3)
np.newaxis                       # an alias for None, sometimes clearer
a.squeeze()                      # removes all size-1 axes
```

You'll see `[:, None]` constantly in research code. It's just "insert an axis here."

---

## 1.7 Vectorization

Vectorization means replacing Python loops with array operations. It's not only about speed — vectorized code is usually shorter and closer to the math.

```python
# Slow: explicit loop
def dot_slow(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

# Fast: vectorized
def dot_fast(a, b):
    return np.sum(a * b)     # or simply a @ b
```

The mental move: **stop thinking about individual elements, start thinking about whole arrays.**

| Loop version | Vectorized |
|---|---|
| `for i: out[i] = a[i] + b[i]` | `out = a + b` |
| `for i: if a[i] < 0: a[i] = 0` | `a = np.maximum(a, 0)` |
| `for i: total += a[i]` | `total = a.sum()` |
| `for i: for j: C[i,j] = sum(A[i,:]*B[:,j])` | `C = A @ B` |

Useful vectorized primitives to know:

```python
np.maximum(a, 0)          # elementwise max — this is ReLU
np.where(cond, x, y)      # elementwise if/else
np.clip(a, lo, hi)        # bound values
np.exp(a), np.log(a)      # elementwise
np.sqrt(a), np.abs(a)
np.argmax(a, axis=1)      # index of max — used for predictions
np.linalg.norm(a)         # vector length
```

### Measuring it

```python
import time

A = np.random.randn(200, 200)
B = np.random.randn(200, 200)

t0 = time.perf_counter()
C = A @ B
t1 = time.perf_counter()
print(f"numpy: {t1 - t0:.6f}s")
```

Use `time.perf_counter()`, not `time.time()` — it's the correct clock for measuring short durations.

---

## 1.8 Exercises

Do these before reading the solutions. Struggling on them is the point; reading them is worth nothing.

**1.** Write `dot(a, b)` in pure Python (no NumPy) taking two lists. Raise a `ValueError` if lengths differ.

**2.** Write `matvec(M, v)` in pure Python. `M` is a list of lists, `v` is a list. Return a list.

**3.** Write `matmul(A, B)` in pure Python. Raise `ValueError` if the inner dimensions don't match.

**4.** Write `transpose(M)` in pure Python.

**5.** Write a `verify()` function that generates random matrices, runs your pure-Python functions and the NumPy equivalents, and asserts they agree to within `1e-9`.

**6.** Benchmark `matmul` (yours) vs `A @ B` (NumPy) on 200×200 matrices. Print both times and the ratio.

**7.** Predict the shape of each, *on paper first*, then verify:
```python
a = np.ones((3, 4)); b = np.ones((4,)); c = np.ones((3, 1)); d = np.ones((1, 4))
a + b     # ?
a + c     # ?
c + d     # ?
a @ b     # ?
a.T @ c   # ?
a.sum(axis=0)                  # ?
a.sum(axis=1, keepdims=True)   # ?
```

**8.** Without loops: given `X` of shape `(n, d)`, normalize each **row** to have mean 0 and standard deviation 1.

**9.** Without loops: given `scores` of shape `(n, k)`, compute row-wise softmax. Each row must sum to 1. (Hint: subtract each row's max before exponentiating — otherwise `np.exp` overflows. This trick appears in every real implementation.)

**10.** Given `y_true` of shape `(n,)` containing class indices and `y_pred` of shape `(n, k)` of scores, compute classification accuracy without loops.

---

## 1.9 Solutions

<details>
<summary>Click only after attempting</summary>

```python
import numpy as np
import time


# --- 1 ---
def dot(a, b):
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


# --- 2 ---
def matvec(M, v):
    if len(M[0]) != len(v):
        raise ValueError(f"shape mismatch: {len(M[0])} vs {len(v)}")
    return [dot(row, v) for row in M]


# --- 3 ---
def matmul(A, B):
    n, k = len(A), len(A[0])
    k2, m = len(B), len(B[0])
    if k != k2:
        raise ValueError(f"inner dims differ: {k} vs {k2}")
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            C[i][j] = s
    return C


# --- 4 ---
def transpose(M):
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]
    # equivalently: return [list(row) for row in zip(*M)]


# --- 5 ---
def verify():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 4))
    B = rng.standard_normal((4, 3))
    v = rng.standard_normal(4)

    assert np.allclose(matvec(A.tolist(), v.tolist()), A @ v, atol=1e-9)
    assert np.allclose(matmul(A.tolist(), B.tolist()), A @ B, atol=1e-9)
    assert np.allclose(transpose(A.tolist()), A.T, atol=1e-9)
    assert abs(dot(v.tolist(), v.tolist()) - v @ v) < 1e-9
    print("all correct")


# --- 6 ---
def benchmark(n=200):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    Al, Bl = A.tolist(), B.tolist()

    t0 = time.perf_counter(); matmul(Al, Bl); t1 = time.perf_counter()
    t2 = time.perf_counter(); A @ B;          t3 = time.perf_counter()

    pure, fast = t1 - t0, t3 - t2
    print(f"pure python: {pure:.4f}s")
    print(f"numpy:       {fast:.6f}s")
    print(f"speedup:     {pure / fast:.0f}x")


# --- 7 ---
# a+b  -> (3,4)   b is (4,), aligns with last axis
# a+c  -> (3,4)   c is (3,1), the 1 stretches to 4
# c+d  -> (3,4)   both stretch
# a@b  -> (3,)    (3,4)@(4,) contracts the 4
# a.T@c-> (4,1)   (4,3)@(3,1)
# a.sum(axis=0)                -> (4,)
# a.sum(axis=1, keepdims=True) -> (3,1)


# --- 8 ---
def normalize_rows(X):
    mu = X.mean(axis=1, keepdims=True)     # (n,1)
    sd = X.std(axis=1, keepdims=True)      # (n,1)
    return (X - mu) / (sd + 1e-8)          # epsilon guards divide-by-zero


# --- 9 ---
def softmax(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)   # numerical stability
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


# --- 10 ---
def accuracy(y_true, y_pred):
    return (y_pred.argmax(axis=1) == y_true).mean()


if __name__ == "__main__":
    verify()
    benchmark()
```

**Why the softmax shift works.** Softmax is invariant to adding a constant to every element of a row: `exp(x−c)/Σexp(x−c) = exp(x)exp(−c) / (exp(−c)Σexp(x))` and the `exp(−c)` cancels. Choosing `c = max(x)` makes the largest exponent exactly `exp(0)=1`, so nothing overflows. Every production implementation does this. Remember it — you'll need it in Chapter 4 and again in Chapter 12.

</details>

---

## 1.10 Chapter 1 checkpoint

Cold — blank file, no notes, no internet, timer running:

- [ ] Implement `dot`, `matvec`, `matmul`, `transpose` in pure Python. **Target: 20 minutes.**
- [ ] Implement row-wise softmax in NumPy, numerically stable, no loops. **Target: 5 minutes.**
- [ ] Given `(3,4)` and `(4,)`, `(3,1)`, `(1,4)`, `(3,)` — state which broadcast and to what shape. **All correct, from memory.**
- [ ] Explain in writing why NumPy beats pure Python by ~100×.
- [ ] Explain in writing what `axis=1` means, using the "which axis disappears" framing.

If any fails, re-work the chapter for two days and retest. Do not advance — Chapter 2 will be twice as hard on a shaky foundation, and Chapter 5 will be impossible.

### Anki cards to make now

- What does `axis=k` mean in a NumPy reduction?
- Broadcasting rule, stated in one sentence
- Difference between `*` and `@`
- Why subtract the max before `exp` in softmax?
- What does `keepdims=True` do and when do you need it?
- Shape rule for `@`: `(n,k) @ (k,m) → ?`

### Commit

```bash
git add .
git commit -m "Chapter 1: pure-python linalg, numpy fundamentals, exercises"
git push
```

### Write-up

Post 300 words: what broadcasting is, the trap in §1.6, and your measured speedup number. Explaining it publicly is a Level-2 test from Chapter 0 and it's how you find out what you only *think* you understood.

---

*Next: Chapter 2 — Linear Algebra for Neural Networks*
