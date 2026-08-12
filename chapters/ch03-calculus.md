# Chapter 3 — Calculus for Gradients

**Time: 10–12 days** (Weeks 3–4 of the plan)

**Prerequisite:** Chapter 2 checkpoint passed cold. In particular you must be able to prove the linear-layer collapse and derive `∂L/∂x = Wᵀ ∂L/∂y` from shapes.

**What you'll be able to do at the end:** differentiate every function that appears in a neural network, derive the chain rule through any composition, explain why gradient descent moves in the direction it does, and — most importantly — build the **numerical gradient checker** that will verify every derivative you write for the remaining eleven chapters.

That last item is the real deliverable of this chapter. You have no teacher to tell you your math is wrong. This tool tells you, mechanically, in milliseconds, forever.

---

## 3.0 What you actually need from calculus

You may have been taught calculus as a list of rules for finding derivatives of functions like `x³sin(x)`, with limits and epsilon-delta proofs at the start. Almost none of that matters here.

What matters is exactly this:

1. What a derivative **means** (three ways).
2. The **chain rule**, to the point of automaticity. This is 80% of the chapter.
3. Derivatives of about eight specific functions.
4. **Partial derivatives** and the **gradient**.
5. Why the gradient points in the direction of steepest increase.
6. How to compute derivatives **numerically**, and why that's too slow for real training.

That's the whole list. You do not need integration. You do not need limits as a formal apparatus. You do not need to differentiate `arctan(ln(x))`.

**Companion resource:** 3Blue1Brown's *Essence of Calculus*, chapters 1–4 and 11. Watch before starting.

---

## 3.1 The derivative: three views

Same object, three interpretations. As with vectors, fluency means switching freely.

### View 1 — the slope

`f'(x)` is the slope of the curve `f` at the point `x`. Steep upward → large positive. Flat → zero. Steep downward → large negative.

### View 2 — the rate of change

`f'(x)` answers: *if I nudge `x` a tiny bit, how much does `f(x)` move, per unit of nudge?*

```
f'(x) ≈ [f(x + h) − f(x)] / h        for very small h
```

If `f'(3) = 5`, then increasing `x` from 3 by 0.001 increases `f` by about 0.005.

### View 3 — the local linear approximation ★

This is the view that matters most for machine learning, and the one most people never get taught.

> **Zoom in far enough on any smooth curve and it looks like a straight line. The derivative is the slope of that line.**

Formally:

```
f(x + Δ) ≈ f(x) + f'(x)·Δ           for small Δ
```

Every curved function, examined closely enough, is linear. The derivative is the linear function that best approximates `f` near `x`.

**Why this is the framing that matters:** gradient descent works by pretending the loss surface is a plane, taking a small step downhill on that plane, then re-approximating. It's a curved landscape being navigated by repeated local linear approximations. If you hold View 3, every optimization concept in Chapter 9 — why the learning rate must be small, why curvature causes trouble, why momentum helps — is obvious. If you only hold View 1, they're arbitrary facts.

### Notation

```
f'(x)        Lagrange — compact
df/dx        Leibniz — shows what varies with respect to what
∂f/∂x        partial — several inputs, this one varies, others held fixed
∇f           gradient — the vector of all partials
```

Leibniz notation is worth preferring because it makes the chain rule look almost like arithmetic, and because in neural networks you always need to say *with respect to which of many variables*.

---

## 3.2 The rules

These seven cover everything you'll meet.

| Rule | Statement |
|---|---|
| Constant | `d/dx [c] = 0` |
| Power | `d/dx [xⁿ] = n·xⁿ⁻¹` |
| Constant multiple | `d/dx [c·f] = c·f'` |
| Sum | `d/dx [f + g] = f' + g'` |
| Product | `d/dx [f·g] = f'g + fg'` |
| Quotient | `d/dx [f/g] = (f'g − fg') / g²` |
| **Chain** | `d/dx [f(g(x))] = f'(g(x)) · g'(x)` |

**The chain rule is the one that matters.** Backpropagation is the chain rule applied systematically to a large composition. §3.4 covers it properly.

Worked examples — do these on paper before reading the answers:

```
f(x) = 3x² + 2x − 5      →   f' = 6x + 2
f(x) = x³ · eˣ           →   f' = 3x²eˣ + x³eˣ         (product)
f(x) = (2x + 1)⁵         →   f' = 5(2x+1)⁴ · 2         (chain)
f(x) = e^(x²)            →   f' = e^(x²) · 2x          (chain)
f(x) = ln(3x)            →   f' = (1/3x) · 3 = 1/x     (chain)
```

---

## 3.3 The eight functions you must know cold

These are the ones that actually appear in neural networks. **Derive each one yourself at least once**, then memorize the result.

| `f(x)` | `f'(x)` | Where you'll meet it |
|---|---|---|
| `c` | `0` | constants |
| `xⁿ` | `n·xⁿ⁻¹` | MSE loss (`n=2`) |
| `eˣ` | `eˣ` | softmax, exponentials |
| `ln(x)` | `1/x` | cross-entropy loss |
| `1/x` | `−1/x²` | normalization |
| `σ(x)` | `σ(x)(1 − σ(x))` | sigmoid activation |
| `tanh(x)` | `1 − tanh²(x)` | tanh activation |
| `max(0, x)` | `1` if `x>0`, else `0` | ReLU activation |

### Deriving the sigmoid derivative

Worth doing carefully — it's the prettiest result in elementary ML calculus, and it appears in every derivation involving logistic regression.

```
σ(x) = 1 / (1 + e⁻ˣ) = (1 + e⁻ˣ)⁻¹
```

Chain rule, with outer function `u⁻¹` and inner `u = 1 + e⁻ˣ`:

```
σ'(x) = −(1 + e⁻ˣ)⁻² · d/dx[1 + e⁻ˣ]
      = −(1 + e⁻ˣ)⁻² · (−e⁻ˣ)
      = e⁻ˣ / (1 + e⁻ˣ)²
```

Now the trick — rewrite it in terms of `σ` itself:

```
σ(x)(1 − σ(x)) = [1/(1+e⁻ˣ)] · [1 − 1/(1+e⁻ˣ)]
               = [1/(1+e⁻ˣ)] · [e⁻ˣ/(1+e⁻ˣ)]
               = e⁻ˣ/(1+e⁻ˣ)²        ✓  same thing
```

So `σ' = σ(1 − σ)`.

**Why this is beautiful and why it matters practically:** you already computed `σ(x)` in the forward pass. Its derivative costs one multiply and one subtract — no exponentials at all. Every framework exploits this. It's also the first hint of a general principle you'll see everywhere in Chapter 5: **the backward pass reuses values from the forward pass.**

### ReLU and the kink

```
relu(x) = max(0, x)
```

Slope 1 for positive `x`, slope 0 for negative. At exactly `x = 0` the derivative is undefined — there's a corner.

In practice everyone just picks a value (usually 0) and moves on. Hitting exactly `0.0` in floating point has probability essentially zero, and if it happens, either subgradient is valid. This is a case where the mathematically careful answer and the practically correct answer differ and the practical one is fine.

**Watch the saturation behaviour**, because it explains a lot of training pathology later:

- `σ'` peaks at 0.25 (at `x=0`) and approaches 0 for large `|x|`. Multiply many of those together across layers and the gradient vanishes. This is the historical reason deep sigmoid networks were untrainable.
- `tanh'` peaks at 1.0, which is better, but still saturates.
- `relu'` is exactly 1 for all positive inputs — no decay at all. This is the main reason ReLU made deep networks trainable.

---

## 3.4 The chain rule

Read this section three times. Everything in Chapter 5 depends on it.

### Statement

If `y = f(u)` and `u = g(x)`, then

```
dy/dx = (dy/du) · (du/dx)
```

### Intuition: gear ratios

Three connected gears. If `u` turns 3× as fast as `x`, and `y` turns 2× as fast as `u`, then `y` turns 6× as fast as `x`. **Rates multiply through a chain.**

That's the entire content of the chain rule. It's not a trick; it's what happens when you compose rates of change.

### Why Leibniz notation makes it easy

```
dy/dx = (dy/du) · (du/dx)
```

The `du` looks like it cancels. It isn't really a fraction, and treating it as one will occasionally mislead you in advanced settings — but for everything in this book, that mnemonic is safe and it works.

### Worked example, slowly

Differentiate `f(x) = sin(x²)`.

1. Identify the pieces: outer `sin(u)`, inner `u = x²`.
2. Outer derivative, evaluated at the inner: `cos(x²)`.
3. Inner derivative: `2x`.
4. Multiply: `f'(x) = cos(x²) · 2x`.

### Longer chains

The rule extends to any depth — just keep multiplying:

```
y = f(g(h(x)))

dy/dx = f'(g(h(x))) · g'(h(x)) · h'(x)
```

A neural network with 96 layers produces a chain 96 factors long. This is exactly why gradients vanish or explode in deep networks: multiply 96 numbers that are each slightly below 1 and you get approximately zero; each slightly above 1 and you get an enormous number. Chapters 9 and 11 are largely about managing this.

### A neural-network-shaped example

One neuron with sigmoid activation and squared loss:

```
z = wx + b
a = σ(z)
L = (a − y)²
```

Find `dL/dw`. Work outside-in:

```
dL/da = 2(a − y)
da/dz = σ(z)(1 − σ(z)) = a(1 − a)
dz/dw = x
```

Multiply:

```
dL/dw = 2(a − y) · a(1 − a) · x
```

**Look at what just happened.** You derived the gradient of a neural network. All of backpropagation is this, repeated systematically, with bookkeeping. The bookkeeping is the only new thing in Chapter 5.

Notice too that every factor uses values already computed going forward: `a`, `y`, `x`. Nothing new is calculated. Again: **the backward pass reuses the forward pass.**

---

## 3.5 Multiple paths — the rule that becomes backprop

Here is the part of the chain rule that beginners miss, and missing it makes backpropagation impossible to understand.

**If a variable influences the output through more than one route, add the contributions from each route.**

Example: `f(x) = x² · sin(x)`. Here `x` reaches `f` twice. By the product rule:

```
f' = 2x·sin(x) + x²·cos(x)
```

Notice that's a **sum of two terms** — one for each path. The product rule is a special case of multi-path chaining.

### Stated generally

If `L` depends on `y₁, y₂, ..., yₘ`, and every `yᵢ` depends on `x`:

```
∂L/∂x = Σᵢ (∂L/∂yᵢ) · (∂yᵢ/∂x)
```

**This formula is backpropagation.** Everything in Chapter 5 is an efficient implementation of it.

### The connection back to Chapter 2

Recall §2.9, where you derived `∂L/∂x = Wᵀ ∂L/∂y` for a layer `y = Wx`. That derivation was exactly this formula: `xⱼ` influences the loss through *every* output `yᵢ`, so you sum over `i`, and summing over the first index of `W` is what the transpose does.

Two chapters, two directions, same result. When separate lines of reasoning converge, your understanding is real.

### Why gradients add, intuitively

Think of a node in the network that feeds into two downstream places. Each downstream consumer sends back a message: *"if you increase by 1, I contribute this much to the loss."* The node's total effect is both messages combined, so it adds them.

If you ever forget whether to add or take one, remember: **the node changing by 1 affects both consumers simultaneously.** Both effects happen. Both count.

---

## 3.6 Partial derivatives

With more than one input, differentiate with respect to one variable while treating all others as constants.

```
f(x, y) = x²y + 3y

∂f/∂x = 2xy            (y is a constant here)
∂f/∂y = x² + 3         (x is a constant here)
```

That's the entire idea. Mechanically it's ordinary differentiation with the other letters frozen.

The `∂` symbol just signals "there are other variables, and they're being held fixed."

Neural network relevance: a model has millions of parameters. `∂L/∂w₄₇₃₂₁` asks *if I nudge this one weight and freeze everything else, how does the loss respond?* Answering that question for every parameter simultaneously is training.

---

## 3.7 The gradient

The **gradient** of a multi-input function is the vector of all its partial derivatives:

```
∇f = [ ∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ ]
```

Example:

```
f(x, y) = x² + 3y²
∇f = [2x, 6y]
∇f at (1, 2) = [2, 12]
```

**Two facts to memorize:**

1. `∇f` is a **vector** with the same shape as the input. If your model has 7 billion parameters, the gradient is a 7-billion-dimensional vector.
2. `∇f` **points in the direction of steepest increase** of `f`, and its magnitude `|∇f|` is how steep that increase is.

Fact 2 is why gradient descent works, and it deserves a proof rather than assertion.

---

## 3.8 Why the gradient points uphill

This proof takes four lines and uses the dot product from Chapter 2. Being able to reproduce it is one of the checkpoint items.

**Setup.** Pick a unit direction `u` (so `|u| = 1`). The **directional derivative** — the rate of change of `f` if you move along `u` — is:

```
D_u f = ∇f · u
```

**Apply the geometric form of the dot product** (§2.3):

```
D_u f = ∇f · u = |∇f| · |u| · cos θ = |∇f| · cos θ
```

since `|u| = 1`.

**Maximize.** `|∇f|` is fixed; the only free quantity is `cos θ`, which is largest when `θ = 0` — that is, when `u` points in the same direction as `∇f`. ∎

So:

- Moving **along** `∇f` increases `f` fastest.
- Moving **against** `∇f` decreases `f` fastest.
- Moving **perpendicular** to `∇f` (`cos θ = 0`) changes `f` not at all — you're walking along a contour line.

That last one is worth holding onto. Contour lines are always perpendicular to the gradient, which is why a loss surface's level curves and its gradient field look the way they do in every optimization diagram you'll ever see.

---

## 3.9 Gradient descent

The gradient points uphill. You want to go downhill. So step in the opposite direction:

```
θ ← θ − η · ∇L(θ)
```

- `θ` : all the parameters
- `η` : the **learning rate** — how big a step
- `∇L(θ)` : the gradient of the loss with respect to the parameters

That's the algorithm behind essentially every model in modern AI. Everything in Chapter 9 — momentum, Adam, schedules — is a refinement of this one line.

```python
def gradient_descent(f, grad_f, x0, lr=0.1, steps=100):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for _ in range(steps):
        x = x - lr * grad_f(x)
        history.append(x.copy())
    return x, np.array(history)
```

### Why the learning rate must be small

Back to View 3 from §3.1. The gradient is a **local linear approximation** — it tells you the slope *right here*. It says nothing about the terrain a long way off.

- `η` too small → correct direction, but thousands of steps needed. Slow.
- `η` too large → you step past the valley onto the opposite wall, possibly higher than where you started. The loss oscillates or diverges to `NaN`.
- `η` just right → steady descent.

**Do this today, before moving on:** run the gradient descent code above on `f(x) = x²` with learning rates `0.01`, `0.1`, `0.5`, `0.9`, `1.0`, and `1.1`. Plot the trajectories. Watching it converge, oscillate, and then explode teaches you more about learning rates in ten minutes than any amount of reading. (For `f(x) = x²` the exact threshold is `η = 1`; above that it diverges. Work out why — it's exercise 11.)

### What can go wrong

- **Local minima** — a valley that isn't the deepest valley. Less of a problem in high dimensions than people assume.
- **Saddle points** — flat in some directions, curved in others. The *actual* dominant obstacle in high-dimensional problems.
- **Plateaus** — near-zero gradient over a wide region, so progress crawls.
- **Ravines** — steep in one direction, shallow in another. Gradient descent zigzags across the narrow axis while barely advancing along the long one. This is exactly the high-condition-number situation from §2.13, and it's the specific problem momentum was invented to solve.

---

## 3.10 Jacobians and Hessians

You mostly need to recognize these, not compute them by hand.

### Jacobian — first derivatives of a vector-valued function

When `f` maps `n` inputs to `m` outputs, all the first partials form an `(m, n)` matrix:

```
J[i][j] = ∂fᵢ/∂xⱼ
```

For a linear layer `y = Wx`, the Jacobian is just `W` — which is another way to see why `Wᵀ` shows up in the backward pass.

**Frameworks never build these explicitly.** A layer mapping 4096 → 4096 would need a 16-million-entry Jacobian. Instead, autodiff computes **Jacobian-vector products** directly: given the incoming gradient vector, produce the outgoing one without ever forming the matrix. That efficiency trick is the core of what you'll build in Chapter 5.

### Hessian — second derivatives

The matrix of all second partials of a scalar function:

```
H[i][j] = ∂²f/∂xᵢ∂xⱼ
```

It describes **curvature** — how the slope itself is changing.

- All eigenvalues positive → bowl shape → local minimum
- All negative → dome → local maximum
- Mixed signs → **saddle point**

Second-order optimization methods (Newton's method) use the Hessian to take much smarter steps. Nobody does this for large neural networks: the Hessian for 7 billion parameters would have 49 × 10¹⁸ entries. Adam and friends are cheap approximations that capture a little curvature information without ever forming the matrix.

---

## 3.11 Putting it together: a network's gradient by hand

Do this derivation on paper right now. It's the bridge to Chapter 5.

**Network:**

```
z₁ = w₁x + b₁
a₁ = σ(z₁)
z₂ = w₂a₁ + b₂
a₂ = σ(z₂)
L  = (a₂ − y)²
```

**Backward, one step at a time:**

```
∂L/∂a₂ = 2(a₂ − y)
∂L/∂z₂ = ∂L/∂a₂ · a₂(1 − a₂)
∂L/∂w₂ = ∂L/∂z₂ · a₁
∂L/∂b₂ = ∂L/∂z₂ · 1
∂L/∂a₁ = ∂L/∂z₂ · w₂
∂L/∂z₁ = ∂L/∂a₁ · a₁(1 − a₁)
∂L/∂w₁ = ∂L/∂z₁ · x
∂L/∂b₁ = ∂L/∂z₁ · 1
```

**Three observations, each of which becomes a design principle in Chapter 5:**

1. **Every line reuses the line above it.** Compute once, pass backward. That reuse is the difference between backprop and brute force, and it's worth a factor of millions.
2. **Every quantity needed was already computed in the forward pass** (`a₁`, `a₂`, `x`, `y`, `w₂`). This is why frameworks cache forward activations — and why training uses so much more memory than inference.
3. **Each parameter's gradient is `(gradient arriving at its node) × (its local input)`.** Uniform pattern, no special cases.

---

## 3.12 Numerical differentiation — building your permanent grader

This is the chapter's deliverable. Build it well; you'll import it for the next eleven chapters.

### Forward difference — the naive version

```
f'(x) ≈ [f(x + h) − f(x)] / h
```

Straight from the definition. Its error shrinks like `O(h)` — slowly.

### Central difference — always use this instead

```
f'(x) ≈ [f(x + h) − f(x − h)] / (2h)
```

Error shrinks like `O(h²)`. For `h = 1e-5` that's the difference between roughly `1e-5` and `1e-10` accuracy — five orders of magnitude, for the cost of one extra function evaluation.

**Why it's better.** Taylor-expand both sides:

```
f(x+h) = f(x) + hf' + h²f''/2 + h³f'''/6 + ...
f(x−h) = f(x) − hf' + h²f''/2 − h³f'''/6 + ...
```

Subtract. The `f(x)` terms cancel, and so do the `h²f''/2` terms:

```
f(x+h) − f(x−h) = 2hf' + h³f'''/3 + ...
```

Divide by `2h`:

```
[f(x+h) − f(x−h)]/(2h) = f' + h²f'''/6 + ...
```

The leading error term is `O(h²)`. The symmetry killed the `O(h)` term. ∎

### Choosing h

There's a tension, and it's worth understanding rather than memorizing a number:

- **`h` too large** → truncation error. The approximation itself is bad.
- **`h` too small** → floating-point catastrophic cancellation. `f(x+h)` and `f(x−h)` become nearly identical, subtracting them destroys most significant digits, and dividing by a tiny `2h` amplifies the noise.

For `float64`, `h = 1e-5` sits near the sweet spot. For `float32`, use around `1e-3` — which is one reason gradient checking is normally done in double precision.

### The implementation

```python
import numpy as np


def numerical_gradient(f, x, h=1e-5):
    """
    Central-difference gradient of a scalar function f at point x.

    f : callable, takes an array of x's shape, returns a scalar
    x : np.ndarray of any shape
    returns : np.ndarray, same shape as x

    Cost: 2 * x.size evaluations of f.
    """
    x = x.astype(np.float64)
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]

        x[idx] = original + h
        f_plus = f(x)

        x[idx] = original - h
        f_minus = f(x)

        x[idx] = original            # always restore

        grad[idx] = (f_plus - f_minus) / (2 * h)
        it.iternext()

    return grad


def relative_error(a, b, eps=1e-12):
    """Scale-free comparison. Use this, not absolute difference."""
    return np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), eps)


def gradient_check(f, analytic_grad_fn, x, h=1e-5, tol=1e-6, verbose=True):
    """
    Compare an analytic gradient against the numerical one.
    Returns True if they agree everywhere within tol.
    """
    numeric = numerical_gradient(f, x.copy(), h)
    analytic = analytic_grad_fn(x.copy())

    err = relative_error(analytic, numeric)
    worst = err.max()
    ok = worst < tol

    if verbose:
        print(f"max relative error: {worst:.3e}   {'PASS' if ok else 'FAIL'}")
        if not ok:
            i = np.unravel_index(err.argmax(), err.shape)
            print(f"  worst at index {i}: analytic={analytic[i]:.8f} "
                  f"numeric={numeric[i]:.8f}")
    return ok
```

### How to read the relative error

| Relative error | Verdict |
|---|---|
| `< 1e-7` | Correct |
| `1e-7` to `1e-5` | Probably correct; suspicious if the function is simple |
| `1e-5` to `1e-3` | Likely a bug — investigate |
| `> 1e-3` | Broken |

Always use **relative** error, never absolute. A gradient of magnitude 10,000 being off by 0.01 is fine; a gradient of magnitude 0.0001 being off by 0.01 is completely wrong. Absolute difference cannot distinguish those cases.

### Two known gotchas

**ReLU at the kink.** Near `x = 0`, the two sides have genuinely different slopes and the numerical estimate is meaningless. This produces spurious failures. If a gradient check fails only at inputs very close to zero for a ReLU network, it's the checker that's wrong, not you.

**Randomness.** If `f` uses dropout or any random sampling, `f(x+h)` and `f(x−h)` use different random draws and the comparison is garbage. Fix the seed, or disable stochastic components while checking.

### Why not just use this for training?

Because it costs `2N` forward passes for `N` parameters. For a 7-billion-parameter model that's 14 billion forward passes per training step. At one step per century you would not finish.

Backpropagation gets all `N` gradients in roughly **one** extra pass — the same cost as the forward pass, regardless of parameter count. That's the ~billion-fold speedup that makes deep learning possible at all, and building it is Chapter 5.

**So: numerical gradients are your test suite, never your training algorithm.** Save this file as `utils/gradcheck.py` and import it forever.

---

## 3.13 Exercises

**1.** Differentiate by hand, then verify each with `numerical_gradient`:
```
f(x) = x³ − 4x² + 7x − 2
f(x) = e^(2x)
f(x) = ln(x² + 1)
f(x) = 1 / (1 + e^(−x))
f(x) = tanh(3x)
```

**2.** Derive `σ'(x) = σ(x)(1 − σ(x))` on paper without looking. Then verify numerically at `x = −2, 0, 2`.

**3.** Derive `tanh'(x) = 1 − tanh²(x)`. (Hint: `tanh = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ)`; use the quotient rule.)

**4.** Implement `relu` and `relu_grad`. Gradient-check at `x = −2, 2` and then at `x = 1e-9`. Explain what happens at the third point and why.

**5.** For `f(x, y) = x²y + sin(y)`, compute `∂f/∂x` and `∂f/∂y` by hand, then verify with `numerical_gradient` on a 2-element array.

**6.** For `f(x) = x₁² + 3x₂² + 2x₁x₂`, write `∇f` by hand. Verify numerically at three random points.

**7.** Implement `numerical_gradient` yourself from the definition, without looking at §3.12. Test it against the version there.

**8.** Compare forward difference and central difference on `f(x) = x³` at `x = 2` (true answer: 12) for `h = 1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11`. Plot the absolute error against `h` on log-log axes. Explain the U shape.

**9.** Implement `gradient_descent` and run it on `f(x) = x²` from `x₀ = 10` with `lr = 0.01, 0.1, 0.5, 0.9, 1.0, 1.1`. Plot each trajectory. Describe each behaviour.

**10.** Run gradient descent on the ravine `f(x, y) = x² + 100y²` from `(10, 10)`. Plot the path over contour lines. Explain the zigzag in terms of §2.13.

**11.** For `f(x) = x²`, prove that gradient descent diverges when `η > 1`. (Hint: the update is `x ← x − η·2x = (1 − 2η)x`. What must `|1 − 2η|` satisfy?)

**12.** Do the two-layer derivation of §3.11 by hand on paper. Then implement forward and backward for that network in NumPy and gradient-check every one of `w₁, b₁, w₂, b₂`.

**13.** Derive `∂L/∂w` for MSE with a linear model: `L = (1/n)|Xw − y|²`. Implement it and gradient-check against a random `X (20,5)`, `w (5,)`, `y (20,)`.

**14.** *(Harder — do it, it pays off.)* For sigmoid + binary cross-entropy:
```
p = σ(z),   L = −[y·ln(p) + (1−y)·ln(1−p)]
```
Show that `∂L/∂z = p − y`. Note how much simpler that is than either derivative alone, and gradient-check it.

---

## 3.14 Solutions

<details>
<summary>Open only after attempting</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from gradcheck import numerical_gradient, gradient_check, relative_error

rng = np.random.default_rng(0)


# --- 1 ---
# f' = 3x² − 8x + 7
# f' = 2e^(2x)
# f' = 2x/(x²+1)
# f' = σ(x)(1−σ(x))
# f' = 3(1 − tanh²(3x))
def check_scalar(f, fprime, x0):
    x = np.array([x0])
    num = numerical_gradient(lambda v: f(v[0]), x.copy())
    print(f"{fprime(x0):.8f} vs {num[0]:.8f}  rel={relative_error(fprime(x0), num[0]):.2e}")

check_scalar(lambda x: x**3 - 4*x**2 + 7*x - 2, lambda x: 3*x**2 - 8*x + 7, 1.7)
check_scalar(lambda x: np.exp(2*x), lambda x: 2*np.exp(2*x), 0.4)
check_scalar(lambda x: np.log(x**2 + 1), lambda x: 2*x/(x**2 + 1), 1.3)


# --- 2, 3 ---
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_grad(x): s = sigmoid(x); return s * (1 - s)
def tanh_grad(x): return 1 - np.tanh(x)**2

for x0 in (-2., 0., 2.):
    check_scalar(sigmoid, sigmoid_grad, x0)
    check_scalar(np.tanh, tanh_grad, x0)


# --- 4 ---
def relu(x):      return np.maximum(x, 0)
def relu_grad(x): return (x > 0).astype(float)

check_scalar(relu, relu_grad, -2.)     # fine
check_scalar(relu, relu_grad,  2.)     # fine
check_scalar(relu, relu_grad, 1e-9)    # FAILS
# At x=1e-9 with h=1e-5, x+h is positive and x−h is negative.
# The central difference straddles the kink and returns ~0.5,
# while the analytic answer is 1. The checker is wrong here, not the code.


# --- 5, 6 ---
# ∂f/∂x = 2xy ; ∂f/∂y = x² + cos(y)
f5 = lambda v: v[0]**2 * v[1] + np.sin(v[1])
g5 = lambda v: np.array([2*v[0]*v[1], v[0]**2 + np.cos(v[1])])
gradient_check(f5, g5, rng.standard_normal(2))

# ∇f = [2x₁ + 2x₂, 6x₂ + 2x₁]
f6 = lambda v: v[0]**2 + 3*v[1]**2 + 2*v[0]*v[1]
g6 = lambda v: np.array([2*v[0] + 2*v[1], 6*v[1] + 2*v[0]])
for _ in range(3):
    gradient_check(f6, g6, rng.standard_normal(2))


# --- 8 ---
f = lambda x: x**3
true = 12.0
hs = np.array([1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11])
fwd = [abs((f(2+h) - f(2)) / h - true) for h in hs]
ctr = [abs((f(2+h) - f(2-h)) / (2*h) - true) for h in hs]
plt.loglog(hs, fwd, "o-", label="forward")
plt.loglog(hs, ctr, "s-", label="central")
plt.xlabel("h"); plt.ylabel("abs error"); plt.legend(); plt.show()
# U shape: on the right, truncation error dominates and falls as h shrinks
# (slope 1 for forward, 2 for central). On the left, catastrophic cancellation
# dominates and error rises as h shrinks. The minimum is the sweet spot.


# --- 9, 11 ---
def gradient_descent(grad_f, x0, lr, steps=50):
    x = float(x0); hist = [x]
    for _ in range(steps):
        x = x - lr * grad_f(x); hist.append(x)
    return np.array(hist)

for lr in (0.01, 0.1, 0.5, 0.9, 1.0, 1.1):
    h = gradient_descent(lambda x: 2*x, 10.0, lr, 30)
    print(f"lr={lr:<5} final={h[-1]:+.4e}")
# 0.01 slow, 0.1 smooth, 0.5 one-step exact, 0.9 oscillating but converging,
# 1.0 oscillates forever between ±10, 1.1 diverges.
#
# Exercise 11: update is x ← x − η·2x = (1 − 2η)x, so after n steps
# xₙ = (1 − 2η)ⁿ x₀. This converges iff |1 − 2η| < 1, i.e. 0 < η < 1.
# η = 0.5 gives factor 0 — exact in one step. η = 1 gives factor −1 —
# perpetual oscillation. η > 1 gives |factor| > 1 — divergence.


# --- 10 ---
def gd2(grad_f, x0, lr, steps=60):
    x = np.array(x0, float); hist = [x.copy()]
    for _ in range(steps):
        x = x - lr * grad_f(x); hist.append(x.copy())
    return np.array(hist)

path = gd2(lambda v: np.array([2*v[0], 200*v[1]]), [10., 10.], 0.009)
xs = np.linspace(-11, 11, 200); ys = np.linspace(-11, 11, 200)
X, Y = np.meshgrid(xs, ys)
plt.contour(X, Y, X**2 + 100*Y**2, levels=40)
plt.plot(path[:, 0], path[:, 1], "r.-"); plt.show()
# Curvature is 2 in x and 200 in y — condition number 100. The learning rate
# is capped by the steep y direction, so progress along the shallow x
# direction crawls while y oscillates across the valley. This is exactly the
# high-condition-number problem of §2.13, and it is what momentum fixes.


# --- 12 ---
def two_layer_forward(params, x, y):
    w1, b1, w2, b2 = params
    z1 = w1*x + b1; a1 = sigmoid(z1)
    z2 = w2*a1 + b2; a2 = sigmoid(z2)
    return (a2 - y)**2, (z1, a1, z2, a2)

def two_layer_backward(params, x, y):
    w1, b1, w2, b2 = params
    _, (z1, a1, z2, a2) = two_layer_forward(params, x, y)
    dL_da2 = 2*(a2 - y)
    dL_dz2 = dL_da2 * a2*(1 - a2)
    dL_dw2 = dL_dz2 * a1
    dL_db2 = dL_dz2
    dL_da1 = dL_dz2 * w2
    dL_dz1 = dL_da1 * a1*(1 - a1)
    dL_dw1 = dL_dz1 * x
    dL_db1 = dL_dz1
    return np.array([dL_dw1, dL_db1, dL_dw2, dL_db2])

x_in, y_t = 0.7, 1.0
p0 = rng.standard_normal(4)
gradient_check(lambda p: two_layer_forward(p, x_in, y_t)[0],
               lambda p: two_layer_backward(p, x_in, y_t), p0)


# --- 13 ---
X = rng.standard_normal((20, 5)); y = rng.standard_normal(20)
def mse(w):      r = X @ w - y; return (r @ r) / len(y)
def mse_grad(w): return 2 * X.T @ (X @ w - y) / len(y)
gradient_check(mse, mse_grad, rng.standard_normal(5))
# Derivation: L = (1/n)rᵀr with r = Xw − y.
#   dL/dr = (2/n)r ; dr/dw = X ; chain with the transpose rule (§2.9):
#   dL/dw = Xᵀ (2/n) r = (2/n) Xᵀ(Xw − y).


# --- 14 ---
# L = −[y ln p + (1−y) ln(1−p)],  p = σ(z)
#   ∂L/∂p = −[ y/p − (1−y)/(1−p) ]
#   ∂p/∂z = p(1−p)
#   ∂L/∂z = −[ y/p − (1−y)/(1−p) ] · p(1−p)
#         = −[ y(1−p) − (1−y)p ]
#         = −[ y − yp − p + yp ] = p − y      ∎
def bce(z, y): p = sigmoid(z); return -(y*np.log(p) + (1-y)*np.log(1-p))
z0, y0 = np.array([0.6]), 1.0
gradient_check(lambda v: bce(v[0], y0),
               lambda v: np.array([sigmoid(v[0]) - y0]), z0)
# The messy 1/p and p(1−p) factors cancel exactly. This is not luck — sigmoid
# and cross-entropy are matched, and the same cancellation happens for
# softmax + cross-entropy (Chapter 6), giving ∂L/∂z = p − y there too.
# It is why frameworks fuse the two into one op.
```

</details>

---

## 3.15 Chapter 3 checkpoint

Cold — blank file, no notes, no internet.

- [ ] State all three views of the derivative and explain why View 3 is the one that makes gradient descent make sense.
- [ ] Derive `σ'(x) = σ(x)(1 − σ(x))` on paper from scratch. **Target: 5 minutes.**
- [ ] State the chain rule, and state the multi-path rule of §3.5. Explain why paths are **added**.
- [ ] **Prove the gradient points in the direction of steepest ascent**, using the dot product. Four lines.
- [ ] Write `numerical_gradient` using central differences, from scratch. **Target: 10 minutes.**
- [ ] Explain why central difference beats forward difference, and why `h` can't just be made tiny.
- [ ] Derive `∂L/∂w₁` for the two-layer network of §3.11, on paper.
- [ ] Explain why numerical gradients can't be used for training.

Items 3, 4 and 7 are mandatory before Chapter 5. Chapter 5 is the hardest in the book and it is pure chain rule.

### Anki cards

- Three views of the derivative
- `σ'(x) = ?`
- `tanh'(x) = ?`
- Chain rule statement
- Multi-path rule — and why add rather than choose?
- Why does the gradient point uphill? (one-line reason)
- Gradient descent update rule
- Central difference formula and its error order
- Good `h` for float64 gradient checking
- Why is `h` too small a problem?
- Relative error thresholds for a gradient check
- Why can't we train with numerical gradients?
- `∂L/∂z` for sigmoid + BCE

### Deliverable

`utils/gradcheck.py`, containing `numerical_gradient`, `relative_error`, and `gradient_check`, with tests. **You will import this file in every remaining chapter.** Make it clean.

```bash
git add .
git commit -m "Chapter 3: calculus, gradient checker, gradient descent experiments"
git push
```

### Write-up

500 words: **"How to know your derivative is right when nobody's checking."** Explain central differences, the `h` trade-off with your log-log plot from exercise 8, relative vs absolute error, and the ReLU kink gotcha. This is a genuinely useful post — most tutorials hand people a gradient checker without explaining a single one of these, and every one of them causes real confusion.

---

*Next: Chapter 4 — Your First Learning Algorithm*
