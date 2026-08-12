# Chapter 2 — Linear Algebra for Neural Networks

**Time: 10–12 days** (Weeks 2–3 of the plan)

**Prerequisite:** Chapter 1 checkpoint passed cold. If you can't write `matmul` from a blank file in 20 minutes, go back. This chapter will not work otherwise.

**What you'll be able to do at the end:** look at `W @ x + b` and see a geometric operation rather than a formula; predict and debug any shape in a neural network; explain from first principles why neural networks need nonlinear activation functions; and understand what LoRA is doing, which is one of the most important practical techniques in modern AI.

---

## 2.0 Why this chapter matters more than it looks

Almost everything a neural network does is linear algebra. Not "uses" linear algebra — *is* linear algebra, with a small nonlinear function sprinkled between the layers.

A GPT forward pass is: multiply by a matrix, apply a nonlinearity, multiply by a matrix, apply a nonlinearity, repeated ninety-six times. That's it. If matrices feel like abstract grids of numbers to you, the entire field will feel like abstract grids of numbers. If matrices feel like *operations that do something to space*, the field becomes visual and intuitive.

That transformation — from "grid of numbers" to "thing that does something" — is the whole point of this chapter. Do not rush it.

**Companion resource:** watch 3Blue1Brown's *Essence of Linear Algebra*, the entire series. Watch it once before starting this chapter, and once again after finishing. It is the best mathematical exposition on the internet and it is free. This chapter assumes you've seen it and goes further into the parts that matter for neural networks specifically.

---

## 2.1 Vectors: three ways to see the same thing

A vector is one object with three interpretations. Fluency means switching between them without noticing.

**View 1 — a list of numbers.**

```
v = [3, 4]
```

This is the programmer's view. It's what's in memory.

**View 2 — an arrow in space.**

`[3, 4]` is an arrow from the origin, going 3 right and 4 up. This is the geometer's view, and it's where intuition lives.

**View 3 — a point in space.**

`[3, 4]` is the location (3, 4). This is the data scientist's view. When you have a dataset of 1,000 images each with 784 pixels, you have 1,000 points floating in 784-dimensional space, and machine learning is the study of the shape of that cloud.

### On high dimensions

You cannot visualize 784 dimensions. Nobody can. What actually works is:

> **Reason in 2 or 3 dimensions, then trust the algebra.**

Almost every fact you'll learn in 2-D generalizes exactly. The dot product means the same thing in 784 dimensions as it does in 2. Draw the 2-D picture, get the intuition, then write the general formula.

There are exceptions — high-dimensional space is genuinely weird, and you'll meet that in Chapter 6 — but for linear algebra the 2-D picture is reliable.

```python
import numpy as np

v = np.array([3.0, 4.0])          # shape (2,)
w = np.array([1.0, 2.0, 3.0])     # shape (3,)
```

**Convention note:** in mathematical writing, a vector is usually a *column*. In NumPy, a 1-D array of shape `(n,)` has no orientation at all — it's neither row nor column, and NumPy decides based on context. This is convenient and occasionally confusing. When orientation matters, be explicit: `v.reshape(-1, 1)` for a column, `v.reshape(1, -1)` for a row.

---

## 2.2 Addition and scaling

Two operations. Everything else is built from them.

**Addition** — elementwise:

```
[3, 4] + [1, 2] = [4, 6]
```

Geometrically: place the tail of the second arrow at the head of the first. The sum is the arrow from the original start to the final end.

**Scalar multiplication** — multiply every component:

```
2 * [3, 4] = [6, 8]
```

Geometrically: stretch the arrow by that factor. A negative scalar also flips it around.

```python
v = np.array([3.0, 4.0])
w = np.array([1.0, 2.0])

v + w        # [4., 6.]
2 * v        # [6., 8.]
v - w        # [2., 2.]
-1 * v       # [-3., -4.]
```

### Linear combinations

Combine both operations and you get the central concept of linear algebra:

```
a·v + b·w      for any scalars a, b
```

This is a **linear combination** of `v` and `w`. Take a couple of directions, scale them however you like, add them up.

Why this matters: **the set of all linear combinations of some vectors is called their span**, and this single idea explains what a neural network layer can and cannot represent. Hold onto it — we return to it in §2.11.

---

## 2.3 The dot product

The most important operation in this book. Read this section twice.

### Algebraic definition

Multiply corresponding components, sum the results. One number out.

```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ = Σᵢ aᵢbᵢ
```

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

np.dot(a, b)     # 32.0
a @ b            # 32.0  — preferred
np.sum(a * b)    # 32.0  — same thing, spelled out
```

### Geometric definition

The remarkable fact is that this same quantity equals:

```
a · b = |a| |b| cos θ
```

where `|a|` is the length of `a` and `θ` is the angle between the two vectors.

Two completely different-looking formulas, always equal. That's not obvious — it's a theorem — but the consequence is what you need:

> **The dot product measures how much two vectors point in the same direction, scaled by their lengths.**

| Situation | cos θ | a · b |
|---|---|---|
| Same direction | 1 | maximum positive |
| 60° apart | 0.5 | positive, smaller |
| Perpendicular | 0 | **exactly zero** |
| Opposite directions | −1 | maximum negative |

**Zero dot product means perpendicular.** That single fact underlies projections, least squares, orthogonal initialization, and half the geometry in machine learning.

### Why this is the operation neural networks are made of

A neuron holds a weight vector `w`. Given an input `x`, it computes `w · x`.

By the geometric formula, that computation is literally asking: *how much does this input point in the direction I'm tuned to?*

- Input strongly matches the pattern → large positive number → neuron "fires"
- Input unrelated to the pattern → near zero
- Input is the opposite of the pattern → large negative

Training a network means adjusting each `w` so it points toward a pattern worth detecting. **That's the whole conceptual content of a neural network.** Everything else — depth, attention, normalization — is engineering on top of this.

It also explains why **attention** works, which you'll meet in Chapter 11. Attention computes `query · key` for every pair of tokens. It's asking "how relevant is this token to that one?" using exactly this similarity measure.

### Cosine similarity

Often you care about direction but not magnitude. Divide out the lengths:

```
cos_sim(a, b) = (a · b) / (|a| |b|)
```

This ranges from −1 to 1 regardless of scale. It's how embedding similarity is measured throughout modern AI — semantic search, retrieval, clustering.

```python
def cosine_similarity(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### Properties worth memorizing

```
a · b = b · a                      (commutative)
a · (b + c) = a·b + a·c            (distributive)
(ka) · b = k(a · b)                (scalars pull out)
a · a = |a|²                       (dot with self = length squared)
```

That last one is used constantly. `a · a` is how you get squared length without a square root — cheaper, and differentiable at zero, which matters for loss functions.

---

## 2.4 Norms

A norm measures the size of a vector. Several definitions exist and each shows up somewhere in ML.

**L2 norm (Euclidean) — the default.**

```
|a|₂ = √(a₁² + a₂² + ... + aₙ²) = √(a · a)
```

Ordinary straight-line length. When someone says "norm" without qualification, this is it.

**L1 norm (Manhattan).**

```
|a|₁ = |a₁| + |a₂| + ... + |aₙ|
```

Distance if you can only move along the axes. Used for L1 regularization, which produces sparse weights — many exactly zero — because of the shape of its constraint region.

**L∞ norm (max).**

```
|a|∞ = max(|a₁|, |a₂|, ..., |aₙ|)
```

Largest single component. Shows up in adversarial robustness.

```python
np.linalg.norm(a)            # L2 (default)
np.linalg.norm(a, ord=1)     # L1
np.linalg.norm(a, ord=np.inf)  # L∞
```

### Where norms show up for you

- **Weight decay / L2 regularization:** add `λ|w|₂²` to the loss to penalize large weights.
- **Gradient clipping:** if `|g|₂ > threshold`, rescale `g` to that threshold. This is what stops training runs from exploding, and you will use it constantly from Chapter 12 on.
- **Normalization:** `v / |v|` produces a unit vector — same direction, length 1.
- **Distance between points:** `|a − b|₂`.

```python
def unit(v):
    return v / np.linalg.norm(v)

def clip_by_norm(g, max_norm):
    n = np.linalg.norm(g)
    return g * (max_norm / n) if n > max_norm else g
```

---

## 2.5 Matrices as transformations

Here is the reframe that makes linear algebra click. Most people are taught a matrix is a grid of numbers. That's true and useless.

> **A matrix is a function that takes a vector and returns a vector. Specifically, it's a function that moves all of space in a way that keeps grid lines parallel and evenly spaced, and keeps the origin fixed.**

Such a function is called a **linear transformation**. Rotations, stretches, shears, reflections, projections — all of them. Not: bending, curving, or translating (translation is what the bias vector `b` is for, and it's precisely why layers are `Wx + b` and not just `Wx`).

### Reading a matrix

The key to reading any matrix instantly:

> **The columns of a matrix are where the basis vectors land.**

In 2-D, the basis vectors are `î = [1,0]` (one step right) and `ĵ = [0,1]` (one step up). Given

```
M = [ a  b ]
    [ c  d ]
```

`î` lands on `[a, c]` (first column) and `ĵ` lands on `[b, d]` (second column). That's all a matrix is — a record of where the basis vectors go.

Examples worth knowing by sight:

```
[ 1  0 ]   identity — nothing moves
[ 0  1 ]

[ 2  0 ]   stretch x by 2, y by 3
[ 0  3 ]

[ 0 -1 ]   rotate 90° counter-clockwise
[ 1  0 ]     (î → [0,1], ĵ → [-1,0])

[ 1  1 ]   shear — î stays, ĵ slides right
[ 0  1 ]

[ 1  0 ]   project onto the x-axis — flattens 2-D into 1-D
[ 0  0 ]
```

That last one is important: it **loses information**. Everything on a vertical line collapses to one point, and you can't undo it. Matrices that destroy dimensions are called *singular* or *rank-deficient*, and we come back to this in §2.11.

---

## 2.6 Matrix–vector product: two views

```
Mv
```

Both views are correct and you need both. Different situations make different views obvious.

### View A — rows are dot products

Each output component is the dot product of a row of `M` with `v`.

```
[ 1  2 ] [ 5 ]   [ 1·5 + 2·6 ]   [ 17 ]
[ 3  4 ] [ 6 ] = [ 3·5 + 4·6 ] = [ 39 ]
```

This is the neural network view: **each row is one neuron's weight vector, and each output is that neuron's response to the input.** A layer with 128 neurons taking 64 inputs is a `(128, 64)` matrix.

### View B — columns get combined

The output is a linear combination of `M`'s columns, weighted by `v`'s entries.

```
[ 1  2 ] [ 5 ]      [ 1 ]      [ 2 ]   [ 5 ]   [ 12 ]   [ 17 ]
[ 3  4 ] [ 6 ] = 5· [ 3 ] + 6· [ 4 ] = [ 15] + [ 24 ] = [ 39 ]
```

This is the geometric view: the transformation sends `î` to column 1 and `ĵ` to column 2, so a vector `[5, 6]` — which is `5î + 6ĵ` — must land on `5·(col 1) + 6·(col 2)`.

**View B is why the "columns are where basis vectors land" rule works**, and it's how you'll reason about rank, span, and what a layer can represent.

```python
M = np.array([[1., 2.],
              [3., 4.]])
v = np.array([5., 6.])

M @ v                                  # [17., 39.]
np.array([M[0] @ v, M[1] @ v])          # View A
v[0] * M[:, 0] + v[1] * M[:, 1]         # View B — same answer
```

### Shape rule

```
(m, n) @ (n,) → (m,)
```

The `n` must match, and it disappears. An `(out, in)` matrix maps an `in`-dimensional vector to an `out`-dimensional one.

---

## 2.7 Matrix–matrix product: composition

`AB` means: **apply `B` first, then `A`.**

Right to left. This ordering feels backwards and it trips up everyone at first. The reason is that `(AB)v = A(Bv)` — `v` touches `B` first because it's adjacent to it.

```
(m, k) @ (k, n) → (m, n)
```

Inner dimensions must match and vanish. Outer dimensions survive.

Computing entry `(i, j)` of `AB` = dot product of row `i` of `A` with column `j` of `B`.

### Properties — memorize these

```
(AB)C = A(BC)         ✓ associative
A(B + C) = AB + AC    ✓ distributive
AB ≠ BA               ✗ NOT commutative
```

The failure of commutativity is real geometry, not a technicality. Rotate-then-stretch is a different transformation from stretch-then-rotate. Try it with your hands.

```python
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

(A @ B).shape        # (3, 5)
# (B @ A) would be an error — 5 doesn't match 3
```

---

## 2.8 The theorem: why neural networks need nonlinearity

This is the most important result in the chapter. It's short, provable in three lines, and it justifies the existence of activation functions — which otherwise look like an arbitrary hack.

**Claim:** stacking linear layers without a nonlinearity between them gives you nothing beyond a single linear layer.

**Proof.** Take two layers:

```
h = W₁x + b₁
y = W₂h + b₂
```

Substitute:

```
y = W₂(W₁x + b₁) + b₂
  = W₂W₁x + W₂b₁ + b₂
```

Now define `W' = W₂W₁` (a matrix) and `b' = W₂b₁ + b₂` (a vector). Then

```
y = W'x + b'
```

which is exactly one linear layer. ∎

**Consequence:** a hundred stacked linear layers has exactly the same expressive power as one. All that depth buys you nothing — it can only ever represent linear functions, and the real world is not linear.

The fix is to insert a nonlinear function `σ` between layers:

```
h = σ(W₁x + b₁)
y = W₂h + b₂
```

Now the collapse is impossible, because `σ` doesn't distribute out. With enough width, such a network can approximate essentially any continuous function — the **universal approximation theorem**.

Common choices for `σ`:

```python
def relu(x):    return np.maximum(x, 0)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x):    return np.tanh(x)
def gelu(x):    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
```

ReLU is the workhorse — trivially cheap, and it doesn't saturate for positive inputs, which matters for gradient flow (Chapter 5). GELU is what transformers actually use.

**Verify the theorem yourself in code today.** It's exercise 7. Seeing two random linear layers collapse into one matrix numerically makes the abstract argument concrete.

---

## 2.9 Transpose, and why backprop is full of it

Transposing flips a matrix across its diagonal: rows become columns.

```
A = [ 1  2  3 ]        Aᵀ = [ 1  4 ]
    [ 4  5  6 ]             [ 2  5 ]
                            [ 3  6 ]
```

`(m, n)` becomes `(n, m)`.

```python
A = np.arange(6).reshape(2, 3)
A.T          # shape (3, 2)
```

### The one rule to memorize

```
(AB)ᵀ = BᵀAᵀ
```

**The order reverses.** This catches everyone at least once.

Also: `(Aᵀ)ᵀ = A`, and `a · b = aᵀb` when both are column vectors.

### Why transpose appears everywhere in backprop

Here's the fact that will save you when Chapter 5 gets hard.

Suppose a layer computes `y = Wx`, and backpropagation has handed you `∂L/∂y` — how the loss changes with the output. You need `∂L/∂x` — how the loss changes with the input — to pass it further back.

The answer is:

```
∂L/∂x = Wᵀ (∂L/∂y)
```

**Why?** Write out one component. `yᵢ = Σⱼ Wᵢⱼxⱼ`, so `∂yᵢ/∂xⱼ = Wᵢⱼ`. By the chain rule, `xⱼ` affects the loss through every `yᵢ`, so we sum:

```
∂L/∂xⱼ = Σᵢ (∂L/∂yᵢ)(∂yᵢ/∂xⱼ) = Σᵢ (∂L/∂yᵢ) Wᵢⱼ
```

Summing over the *first* index of `W` is precisely what `Wᵀ` does. Hence the transpose.

**The intuition:** the forward pass sends information from inputs to outputs through `W`. The backward pass sends information the other way through the same connections — so you traverse the same matrix in the opposite direction, which is exactly `Wᵀ`.

There's also a shape argument that's fast and reliable when you're stuck: if `x` is `(n,)` and `y` is `(m,)`, then `W` is `(m, n)` and `∂L/∂y` is `(m,)`. The only way to produce something of shape `(n,)` is `(n, m) @ (m,)` — so it must be `Wᵀ`. **Shapes will often tell you the right formula.** Use this trick.

---

## 2.10 Batched layers: the shape conventions you'll live with

This section prevents a specific, extremely common frustration. Read it carefully now and you'll skip a whole category of bugs.

### The math convention

Textbooks write a layer as `y = Wx + b` with `x` a column vector:

```
x : (in,)          W : (out, in)          y : (out,)
```

### The code convention

Real code processes a whole **batch** at once, with examples as *rows*:

```
X : (batch, in)    W : (in, out)          Y : (batch, out)

Y = X @ W + b      where b is (out,)  and broadcasts over rows
```

Note that `W` is transposed relative to the math version. Both are correct; they're just different layouts of the same numbers.

### PyTorch, which does a third thing

`nn.Linear(in_features, out_features)` **stores** its weight with shape `(out, in)` — the math convention — but its input is `(batch, in)`. So internally it computes:

```
Y = X @ W.T + b
```

This is why `layer.weight.shape` looks "backwards" compared to how you'd multiply it. Nothing is wrong; it's a storage choice. Knowing this now will save you a confusing afternoon in Chapter 8.

### A batched layer, written out

```python
def linear_forward(X, W, b):
    """
    X : (batch, in)
    W : (in, out)
    b : (out,)
    returns (batch, out)
    """
    assert X.ndim == 2, f"X must be 2-D, got {X.shape}"
    assert X.shape[1] == W.shape[0], f"mismatch: {X.shape} @ {W.shape}"
    out = X @ W + b                    # b broadcasts across the batch
    assert out.shape == (X.shape[0], W.shape[1])
    return out


X = np.random.randn(32, 64)      # 32 examples, 64 features
W = np.random.randn(64, 128)     # layer: 64 in, 128 out
b = np.zeros(128)

linear_forward(X, W, b).shape    # (32, 128)
```

The bias broadcasting here is exactly the rule from §1.6: `(32, 128) + (128,)` aligns from the right and adds `b` to every row. One bias vector, thirty-two examples, no loop.

**Habit:** write the shape of every argument in the docstring of every function you write this year. It costs ten seconds and it's how professionals avoid shape bugs.

---

## 2.11 Span, linear independence, and rank

Now the concepts that explain what a layer *can't* do.

### Span

The **span** of a set of vectors is every point reachable by linear combinations of them.

- Span of one nonzero vector in 2-D: a line through the origin.
- Span of two vectors pointing in different directions: the whole plane.
- Span of two vectors pointing along the *same* line: still just that line. The second one added nothing.

### Linear independence

Vectors are **linearly independent** if none of them is a linear combination of the others — each contributes a genuinely new direction. If one is redundant, they're **dependent**.

```
[1, 0] and [0, 1]   independent  (span the plane)
[1, 2] and [2, 4]   dependent    (second = 2× first; span only a line)
```

### Rank

The **rank** of a matrix is the number of dimensions in its output space — equivalently, the number of linearly independent columns.

```python
np.linalg.matrix_rank(np.eye(3))                    # 3 — full rank
np.linalg.matrix_rank(np.array([[1., 2.],
                                [2., 4.]]))         # 1 — rank deficient
```

A `(3, 3)` matrix of rank 3 is **full rank**: invertible, loses nothing. Rank 2: it squashes 3-D space onto a plane, and that's irreversible.

### The fact that matters for modern AI

```
rank(AB) ≤ min(rank(A), rank(B))
```

Multiplying can only lose rank, never gain it.

Take `B` of shape `(d, r)` and `A` of shape `(r, k)` with `r` small. Then `BA` has shape `(d, k)` but rank at most `r`. You've built a big matrix out of few parameters.

**This is exactly LoRA** (Low-Rank Adaptation), the dominant method for fine-tuning large language models. Instead of updating a `(4096, 4096)` weight matrix — 16.8 million parameters — you freeze it and learn `ΔW = BA` with `r = 8`:

```
B : (4096, 8)  →    32,768 parameters
A : (8, 4096)  →    32,768 parameters
total          →    65,536 parameters — a 256× reduction
```

The bet is that the *update* needed for fine-tuning is intrinsically low-rank, even though the original weights aren't. Empirically, it usually is.

You now understand one of the most-used techniques in applied AI, and it fell straight out of a rank inequality.

---

## 2.12 Determinant and inverse

Lighter treatment — you need the concepts, not the hand computation.

### Determinant

The **determinant** is the factor by which a transformation scales areas (2-D) or volumes (3-D).

- `det = 2`: areas double.
- `det = 1`: areas preserved (rotations).
- `det = 0`: **everything collapses** to a lower dimension. Information destroyed. Not invertible.
- `det < 0`: space gets flipped over as well as scaled.

```python
np.linalg.det(np.array([[2., 0.], [0., 3.]]))   # 6.0
np.linalg.det(np.array([[1., 2.], [2., 4.]]))   # 0.0 — singular
```

For 2×2: `det([[a,b],[c,d]]) = ad − bc`. Beyond that, let the computer do it.

### Inverse

`A⁻¹` is the transformation that undoes `A`: `A⁻¹A = I`.

It exists **only if `det ≠ 0`** — you cannot undo a collapse.

```python
A = np.array([[2., 1.], [1., 1.]])
np.linalg.inv(A) @ A      # ≈ identity
```

**Practical warning that will matter later:** in numerical code, essentially never compute an explicit inverse. To solve `Ax = b`, use `np.linalg.solve(A, b)` rather than `np.linalg.inv(A) @ b`. It's faster and far more numerically stable. Explicit inverses are a common source of silent precision loss.

You will rarely invert anything in deep learning — networks are trained by gradient descent, not solved in closed form. But `det = 0` as "information destroyed" is intuition you'll use often.

---

## 2.13 Eigenvectors and eigenvalues

**Definition:** `v` is an eigenvector of `A` with eigenvalue `λ` if

```
Av = λv        (v ≠ 0)
```

In words: applying the transformation to `v` doesn't rotate it. `v` stays on its own line and just gets scaled by `λ`.

Most vectors get knocked off their line by a transformation. Eigenvectors are the special directions that don't move — the **axes of the transformation**.

```python
A = np.array([[2., 1.],
              [1., 2.]])

vals, vecs = np.linalg.eig(A)
print(vals)          # [3., 1.]
print(vecs[:, 0])    # eigenvector for λ=3, ≈ [0.707, 0.707]

v = vecs[:, 0]
np.allclose(A @ v, vals[0] * v)   # True — the definition holds
```

### Where you'll actually meet them

- **PCA.** The eigenvectors of the data's covariance matrix are the directions of greatest variance. Keeping the top few is dimensionality reduction.
- **Optimization.** The eigenvalues of the Hessian describe the curvature of the loss surface. A wide spread of eigenvalues — a high **condition number** — means the landscape is a long narrow valley, gradient descent zigzags, and training is slow. This is a large part of *why* Adam exists (Chapter 9).
- **Stability.** Whether repeated multiplication by a matrix explodes or vanishes is governed by its largest eigenvalue. This is precisely why RNNs suffer vanishing and exploding gradients (Chapter 11).
- **Spectral norm.** The largest singular value bounds how much a matrix can stretch anything — used in spectral normalization and in analysing training stability.

### Power iteration

The dominant eigenvector can be found with a loop so simple it's worth knowing by heart: repeatedly multiply by `A` and renormalize.

```python
def power_iteration(A, iters=100):
    v = np.random.randn(A.shape[0])
    v = v / np.linalg.norm(v)
    for _ in range(iters):
        v = A @ v
        v = v / np.linalg.norm(v)
    eigenvalue = v @ A @ v          # Rayleigh quotient
    return eigenvalue, v
```

Why it works: write your starting vector as a combination of eigenvectors. Each multiplication by `A` scales each component by its eigenvalue, so the largest one grows fastest in relative terms and eventually dominates everything else.

---

## 2.14 SVD and low-rank approximation

**Singular value decomposition.** Every matrix — any shape, any rank — factors as

```
A = U Σ Vᵀ
```

- `V` : orthogonal — a rotation in the input space
- `Σ` : diagonal, non-negative — pure stretching along axes
- `U` : orthogonal — a rotation in the output space

The claim is strong and true: **every linear transformation is a rotation, then a stretch, then another rotation.** Nothing else is possible.

The diagonal entries of `Σ` are the **singular values**, conventionally sorted largest first. They tell you how much the matrix stretches along each of its principal directions. The number of nonzero ones is the rank.

```python
A = np.random.randn(5, 3)
U, S, Vt = np.linalg.svd(A, full_matrices=False)

U.shape, S.shape, Vt.shape      # (5,3), (3,), (3,3)
np.allclose(U @ np.diag(S) @ Vt, A)     # True
```

### Low-rank approximation

Keep only the largest `k` singular values and discard the rest:

```python
def low_rank_approx(A, k):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

A theorem (Eckart–Young) says this is the *best possible* rank-`k` approximation of `A` under the Frobenius norm. Not a heuristic — provably optimal.

This single idea powers image compression, recommender systems, PCA, model compression, and — again — the intuition behind LoRA. If a matrix's singular values decay quickly, most of what it does lives in a few directions, and you can throw the rest away almost for free.

---

## 2.15 Projection and least squares

A geometric preview of Chapter 4, where you'll build your first learning algorithm.

**Projecting `a` onto `b`** means finding the point on `b`'s line closest to `a`:

```
proj_b(a) = ((a · b) / (b · b)) b
```

```python
def project(a, b):
    return (a @ b) / (b @ b) * b
```

The key geometric fact: the **error vector** `a − proj_b(a)` is perpendicular to `b`. Its dot product with `b` is exactly zero. Verify it in code — it's exercise 12.

### Why this is linear regression

Least squares asks: find `x` minimizing `|Ax − y|²`. Geometrically, `Ax` ranges over the span of `A`'s columns — some subspace — and `y` generally isn't in it. The closest reachable point is the **projection of `y` onto that subspace**, and the residual is perpendicular to it. That perpendicularity condition gives the normal equations:

```
AᵀAx = Aᵀy
```

You'll derive this again with calculus in Chapter 4 and get an identical answer. Two routes, same destination — which is a good sign your understanding is real.

---

## 2.16 Exercises

Attempt every one before looking at solutions. Put anything that beats you into `PARKED.md` and use the Unstuck Protocol.

**1.** Implement `cosine_similarity(a, b)`. Test it on: identical vectors (expect 1), opposite (−1), `[1,0]` and `[0,1]` (0).

**2.** Verify numerically that `a · b = |a||b|cos θ`: generate random `a`, `b`, compute the dot product directly, then compute the angle with `np.arccos` and reconstruct the product from the geometric formula. Assert they agree.

**3.** Implement L1, L2, and L∞ norms without `np.linalg.norm`. Check against it.

**4.** For each 2×2 matrix in §2.5, apply it to the four corners of the unit square `[0,0], [1,0], [1,1], [0,1]` and print the results. Plot them with matplotlib. Describe in words what each does.

**5.** Verify View A and View B of §2.6 give the same result, for random `M` and `v`, using only elementwise operations and sums — no `@`.

**6.** Verify `(AB)C = A(BC)` and `AB ≠ BA` numerically with random matrices.

**7.** **The important one.** Generate random `W1 (4,8), b1 (8,), W2 (8,3), b2 (3,)`. Compute `y = W2 @ (W1 @ x + b1) + b2` for a random `x`. Then find a single `W' (4,3)` and `b' (3,)` giving the same answer for *any* `x`, and verify on 100 random inputs. Then insert a ReLU between the layers and show no such single `W'` exists.

**8.** Verify `(AB)ᵀ = BᵀAᵀ`, and show `AᵀBᵀ` is generally different (or a shape error).

**9.** Write `linear_forward(X, W, b)` with shape assertions, then write `linear_backward(dY, X, W)` returning `dX`, `dW`, `db`. Use the shape trick from §2.9 to work out each formula — don't look them up. (`dW` should be `Xᵀ @ dY`, `db` should be `dY.sum(axis=0)`; derive why.)

**10.** Build `B (10,3)` and `A (3,10)` from random values. Compute `BA` and check its rank. Then count parameters in `BA` versus a full `(10,10)` matrix.

**11.** Implement `power_iteration`. Test on a symmetric matrix and check against `np.linalg.eig`.

**12.** Implement `project(a, b)`. Verify the residual `a − proj_b(a)` is perpendicular to `b` (dot product ≈ 0).

**13.** Take a grayscale image as a 2-D array (or `np.random.randn(200,200)` smoothed). Compute rank-`k` approximations for `k = 1, 5, 20, 50`. Plot them and plot the singular values on a log scale. At what `k` does it look acceptable?

---

## 2.17 Solutions

<details>
<summary>Open only after attempting</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


# --- 1 ---
def cosine_similarity(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

assert np.isclose(cosine_similarity(np.array([1., 2.]), np.array([1., 2.])), 1.0)
assert np.isclose(cosine_similarity(np.array([1., 0.]), np.array([-1., 0.])), -1.0)
assert np.isclose(cosine_similarity(np.array([1., 0.]), np.array([0., 1.])), 0.0)


# --- 2 ---
a, b = rng.standard_normal(5), rng.standard_normal(5)
algebraic = a @ b
theta = np.arccos(np.clip(cosine_similarity(a, b), -1, 1))
geometric = np.linalg.norm(a) * np.linalg.norm(b) * np.cos(theta)
assert np.isclose(algebraic, geometric)


# --- 3 ---
def l1(a):   return np.abs(a).sum()
def l2(a):   return np.sqrt((a ** 2).sum())
def linf(a): return np.abs(a).max()


# --- 4 ---
square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., 0.]])

mats = {
    "identity": np.array([[1., 0.], [0., 1.]]),
    "scale":    np.array([[2., 0.], [0., 3.]]),
    "rotate90": np.array([[0., -1.], [1., 0.]]),
    "shear":    np.array([[1., 1.], [0., 1.]]),
    "project":  np.array([[1., 0.], [0., 0.]]),
}

for name, M in mats.items():
    out = square @ M.T          # rows are points, so transpose M
    print(f"{name}: det={np.linalg.det(M):+.1f} rank={np.linalg.matrix_rank(M)}")
# "project" has det 0 and rank 1: the square collapses onto a segment.


# --- 5 ---
M, v = rng.standard_normal((4, 3)), rng.standard_normal(3)
view_a = np.array([(M[i] * v).sum() for i in range(M.shape[0])])
view_b = sum(v[j] * M[:, j] for j in range(M.shape[1]))
assert np.allclose(view_a, view_b) and np.allclose(view_a, M @ v)


# --- 6 ---
A, B, C = rng.standard_normal((3,3)), rng.standard_normal((3,3)), rng.standard_normal((3,3))
assert np.allclose((A @ B) @ C, A @ (B @ C))
assert not np.allclose(A @ B, B @ A)


# --- 7 ---  THE KEY ONE
W1, b1 = rng.standard_normal((4, 8)), rng.standard_normal(8)
W2, b2 = rng.standard_normal((8, 3)), rng.standard_normal(3)

def two_layer_linear(x): return (x @ W1 + b1) @ W2 + b2

W_eq = W1 @ W2            # (4,8)@(8,3) -> (4,3)
b_eq = b1 @ W2 + b2       # (8,)@(8,3)  -> (3,)

def one_layer(x): return x @ W_eq + b_eq

for _ in range(100):
    x = rng.standard_normal(4)
    assert np.allclose(two_layer_linear(x), one_layer(x))
print("two linear layers collapse to one: confirmed")

def two_layer_relu(x): return np.maximum(x @ W1 + b1, 0) @ W2 + b2

# No single (W', b') can match this. Proof by contradiction from data:
# a linear map is fully determined by its action on a basis plus the origin,
# so fit W',b' from those points, then test elsewhere.
b_fit = two_layer_relu(np.zeros(4))
W_fit = np.stack([two_layer_relu(np.eye(4)[i]) - b_fit for i in range(4)])
x_test = rng.standard_normal(4) * 3
print("relu net :", two_layer_relu(x_test))
print("best linear fit:", x_test @ W_fit + b_fit)      # different -> not linear


# --- 8 ---
A, B = rng.standard_normal((3, 4)), rng.standard_normal((4, 5))
assert np.allclose((A @ B).T, B.T @ A.T)
# A.T @ B.T would be (4,3)@(5,4) -> shape error


# --- 9 ---
def linear_forward(X, W, b):
    """X:(N,in) W:(in,out) b:(out,) -> (N,out)"""
    assert X.shape[1] == W.shape[0]
    return X @ W + b

def linear_backward(dY, X, W):
    """dY:(N,out) X:(N,in) W:(in,out) -> dX:(N,in), dW:(in,out), db:(out,)"""
    dX = dY @ W.T                 # (N,out)@(out,in) -> (N,in)
    dW = X.T @ dY                 # (in,N)@(N,out)   -> (in,out)
    db = dY.sum(axis=0)           # sum over batch    -> (out,)
    return dX, dW, db
# Each formula is the ONLY shape-legal combination. That is the shape trick.
# db sums over the batch because the same b is added to every example,
# so its gradient accumulates over all of them.


# --- 10 ---
B_lr, A_lr = rng.standard_normal((10, 3)), rng.standard_normal((3, 10))
BA = B_lr @ A_lr
print("shape", BA.shape, "rank", np.linalg.matrix_rank(BA))   # (10,10) rank 3
print("params:", B_lr.size + A_lr.size, "vs full:", 100)      # 60 vs 100


# --- 11 ---
def power_iteration(A, iters=200):
    v = rng.standard_normal(A.shape[0]); v /= np.linalg.norm(v)
    for _ in range(iters):
        v = A @ v; v /= np.linalg.norm(v)
    return v @ A @ v, v

S = rng.standard_normal((5, 5)); S = S + S.T        # symmetric
lam, vec = power_iteration(S)
assert np.isclose(abs(lam), np.abs(np.linalg.eigvals(S)).max(), atol=1e-6)


# --- 12 ---
def project(a, b): return (a @ b) / (b @ b) * b

a, b = rng.standard_normal(4), rng.standard_normal(4)
residual = a - project(a, b)
assert abs(residual @ b) < 1e-10          # perpendicular


# --- 13 ---
def low_rank_approx(A, k):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

M = rng.standard_normal((200, 200))
M = M @ M.T                                # correlated -> decaying spectrum
_, S, _ = np.linalg.svd(M)
plt.semilogy(S); plt.title("singular values"); plt.show()
for k in (1, 5, 20, 50):
    err = np.linalg.norm(M - low_rank_approx(M, k)) / np.linalg.norm(M)
    print(f"k={k:3d}  relative error {err:.4f}")
```

**Note on exercise 7.** The second half is the whole justification for activation functions. If you can produce that demonstration yourself, you understand something that a lot of people using PyTorch daily do not.

</details>

---

## 2.18 Chapter 2 checkpoint

Cold: blank file, no notes, no internet.

- [ ] State the geometric meaning of the dot product and what a zero dot product implies. **Written, in your own words.**
- [ ] Given a 2×2 matrix, state where `î` and `ĵ` land and describe the transformation. **Instantly, from the columns.**
- [ ] Explain matrix–vector multiplication both ways: rows-as-dot-products and columns-combined.
- [ ] **Prove that two stacked linear layers collapse into one.** On paper, three lines, from memory. Then state why this justifies activation functions.
- [ ] Derive `∂L/∂x = Wᵀ ∂L/∂y` using the shape argument, then explain the intuition.
- [ ] Implement `linear_forward` and `linear_backward` with correct shapes. **Target: 15 minutes.**
- [ ] Explain what rank means and how LoRA exploits it.
- [ ] Explain what `det = 0` means geometrically.

Fail any of the first five and you re-work the chapter. Those five are load-bearing for Chapter 5, which is the hardest chapter in the book.

### Anki cards

- Dot product: geometric formula and what it measures
- Zero dot product ⟹ ?
- Columns of a matrix are ___
- Shape rule: `(m,k) @ (k,n) → ?`
- `(AB)ᵀ = ?`
- Why do neural networks need nonlinearities? (state the collapse proof in one line)
- In backprop, `∂L/∂x = ?` given `y = Wx`
- `rank(AB) ≤ ?`
- What does LoRA exploit?
- SVD: `A = ?` and what each factor does geometrically
- `det = 0` means ___
- Eigenvector definition

### Commit and write up

```bash
git add .
git commit -m "Chapter 2: linear algebra, layer collapse proof, low-rank experiments"
git push
```

Blog post, 400 words: **"Why neural networks need activation functions."** Include your own version of the collapse proof and the numerical demonstration from exercise 7. This is a genuinely good post — most beginner explanations of activation functions never give the actual reason, and yours will.

---

*Next: Chapter 3 — Calculus for Gradients*
