# Chapter 6 — Probability and Information Theory

**Time: 10–12 days** (Weeks 10–11 of the plan)

**Prerequisite:** Chapter 5 checkpoint passed — you can write the `Value` class cold.

**What you'll be able to do at the end:** derive MSE and cross-entropy from first principles instead of accepting them; explain what a model's loss number actually *means*; understand perplexity, temperature, top-k and top-p sampling before you meet them in a language model; and know why your geometric intuition breaks in high dimensions.

This chapter is lighter on code than Chapter 5 and heavier on ideas. That's deliberate — it's recovery time after the hardest chapter, and the ideas here are what separate someone who uses cross-entropy from someone who knows why it exists.

---

## 6.0 The promise from Chapter 4

Three times in Chapter 4 you derived a gradient and got `p − y`:

- Linear regression with MSE
- Logistic regression with binary cross-entropy
- Softmax with categorical cross-entropy

Three different models, three different losses, one answer. I said that wasn't coincidence. This chapter explains it.

The short version: **loss functions aren't invented, they're derived.** Each one falls out of an assumption about how your data was generated. Choose the assumption, and the loss function is forced on you. Once you see this, "which loss should I use?" stops being a matter of taste and becomes a question about your data.

---

## 6.1 The minimum probability

**Sample space `Ω`** — everything that could happen. For a die: `{1,2,3,4,5,6}`.

**Event** — a subset. "Rolled even" is `{2,4,6}`.

**Probability** — a number assigned to each event, satisfying three rules:

```
1.  P(A) ≥ 0                     never negative
2.  P(Ω) = 1                     something happens
3.  P(A ∪ B) = P(A) + P(B)       if A and B can't both occur
```

Everything in probability follows from those three lines. You don't need more foundations than this.

**Random variable** — a function from outcomes to numbers. "Number of heads in 10 flips" is a random variable. Conventionally capital `X`; a specific value is lowercase `x`.

**Discrete** random variables have a **probability mass function** `P(X = x)`. **Continuous** ones have a **probability density function** `p(x)`, where `p(x)` is *not* a probability — only integrals of it are. A density can exceed 1. This trips people up; it's worth pausing on.

---

## 6.2 The four distributions you need

### Bernoulli — one binary event

```
P(X = 1) = p,    P(X = 0) = 1 − p
mean = p,   variance = p(1 − p)
```

Used for: binary classification. Your logistic regression output *is* a Bernoulli parameter.

Note that the variance `p(1−p)` peaks at `p = 0.5` and vanishes at 0 and 1 — maximum uncertainty in the middle. That's the same expression as the sigmoid derivative, and that is not a coincidence either.

### Categorical — one event with k outcomes

```
P(X = i) = pᵢ,    Σpᵢ = 1
```

Used for: multiclass classification, and **every next-token prediction in every language model**. A GPT's output is a categorical distribution over ~50,000 tokens.

### Gaussian (normal) — the default for continuous quantities

```
p(x) = (1/(σ√(2π))) · exp(−(x − μ)²/(2σ²))
mean = μ,   variance = σ²
```

Used for: noise models, weight initialization, and as the assumption hiding inside squared-error loss.

**Why Gaussians appear everywhere** is the **central limit theorem**: sums of many independent random effects converge to a Gaussian, almost regardless of what the individual effects look like. Measurement noise is usually a sum of many small causes, so it's usually approximately Gaussian.

### Uniform

```
p(x) = 1/(b − a)  on [a, b]
```

Used for: random initialization, sampling, dropout masks.

```python
import numpy as np
rng = np.random.default_rng(0)

rng.binomial(1, 0.3, size=10)            # Bernoulli(0.3)
rng.choice(5, size=10, p=[.1,.2,.4,.2,.1])  # Categorical
rng.normal(0, 1, size=10)                # Gaussian
rng.uniform(-1, 1, size=10)              # Uniform
```

---

## 6.3 Expectation and variance

**Expectation** — the long-run average:

```
E[X] = Σₓ x·P(X = x)          (discrete)
E[X] = ∫ x·p(x) dx            (continuous)
```

**Linearity of expectation** — the most useful property in probability:

```
E[aX + bY] = a·E[X] + b·E[Y]
```

This holds **even when `X` and `Y` are dependent**. That's unusual and powerful.

**Variance** — average squared deviation from the mean:

```
Var[X] = E[(X − E[X])²] = E[X²] − E[X]²
```

Standard deviation is `√Var`, which has the same units as `X`.

**Key scaling facts:**

```
Var[aX] = a²·Var[X]
Var[X + Y] = Var[X] + Var[Y]        only if X, Y independent
```

### Why this matters for minibatches

The gradient over a minibatch of size `B` is an average of `B` per-example gradients. If each has variance `σ²`, the average has variance `σ²/B`, so the **standard error** shrinks like `1/√B`.

**Practical consequence:** quadrupling the batch size only halves the gradient noise. Diminishing returns are built into the mathematics. This is a large part of why batch sizes plateau around a few hundred to a few thousand rather than growing without limit, and it's the reason the "linear scaling rule" (double the batch, double the learning rate) eventually breaks down.

```python
# see it directly
for B in (1, 4, 16, 64, 256, 1024):
    means = [rng.standard_normal(B).mean() for _ in range(2000)]
    print(f"B={B:5d}  std of mean = {np.std(means):.4f}   1/√B = {1/np.sqrt(B):.4f}")
```

---

## 6.4 Joint, conditional, and Bayes

**Joint** — `P(X, Y)`: both happen.

**Marginal** — sum the joint over what you don't care about:

```
P(X) = Σ_y P(X, Y = y)
```

**Conditional** — given that `Y` happened:

```
P(X | Y) = P(X, Y) / P(Y)
```

**Independence** — knowing one tells you nothing about the other:

```
P(X, Y) = P(X)·P(Y)     ⟺     P(X | Y) = P(X)
```

**The chain rule of probability** (different from calculus's chain rule):

```
P(x₁, x₂, ..., xₙ) = P(x₁)·P(x₂|x₁)·P(x₃|x₁,x₂)···P(xₙ|x₁,...,xₙ₋₁)
```

**This is the entire justification for autoregressive language models.** The probability of a sentence factors exactly into a product of next-token probabilities, each conditioned on everything before it. A GPT learns each factor. Nothing is approximated in the factorization itself — it's an identity.

### Bayes' rule

```
P(A | B) = P(B | A)·P(A) / P(B)
```

Rearranged from the conditional definition, but conceptually enormous: it tells you how to update a belief given evidence.

- `P(A)` — **prior**: belief before evidence
- `P(B|A)` — **likelihood**: how well `A` explains the evidence
- `P(A|B)` — **posterior**: belief after evidence

**The classic worked example**, because the answer surprises nearly everyone. A test is 99% accurate both ways. The condition affects 1 in 1,000 people. You test positive. What's the probability you have it?

```
P(pos) = 0.99·0.001 + 0.01·0.999 = 0.00099 + 0.00999 = 0.01098
P(have | pos) = 0.99·0.001 / 0.01098 ≈ 0.090
```

**About 9%.** Not 99%. Because the condition is rare, false positives from the 999 healthy people vastly outnumber true positives from the 1 affected person.

The lesson generalizes far beyond medicine: **when the prior is small, even strong evidence leaves you uncertain.** Keep this in mind for the rest of your career — it's the correct response to almost every surprising experimental result, including your own.

---

## 6.5 Maximum likelihood: where loss functions come from

Here's the central idea of the chapter.

You have data. You have a model with parameters `θ`. **Maximum likelihood estimation** says: choose the `θ` that makes the observed data most probable.

```
θ* = argmax_θ  P(data | θ)
```

Assuming examples are independent, the joint probability is a product:

```
P(data | θ) = Πᵢ P(yᵢ | xᵢ, θ)
```

Products of many small numbers underflow to zero in floating point, and products are awkward to differentiate. So take the log — which is monotonic, so the argmax is unchanged — and products become sums:

```
log P(data | θ) = Σᵢ log P(yᵢ | xᵢ, θ)
```

By convention we minimize rather than maximize, so negate:

```
NLL(θ) = − Σᵢ log P(yᵢ | xᵢ, θ)          ← the negative log-likelihood
```

> **Every loss function you have used is a negative log-likelihood under some assumption about the data.**

Now watch it produce the two losses from Chapter 4.

### MLE with Gaussian noise ⟹ mean squared error

Assume the target equals the model's prediction plus Gaussian noise:

```
y = ŷ + ε,    ε ~ 𝒩(0, σ²)
```

Equivalently `p(y | x, θ) = 𝒩(y; ŷ, σ²)`. Substitute the Gaussian density:

```
log p(yᵢ) = log[ (1/(σ√(2π))) · exp(−(yᵢ − ŷᵢ)²/(2σ²)) ]
          = −log(σ√(2π)) − (yᵢ − ŷᵢ)²/(2σ²)
```

Sum over examples and negate:

```
NLL = n·log(σ√(2π)) + (1/(2σ²))·Σᵢ (yᵢ − ŷᵢ)²
```

The first term doesn't involve `θ`. The `1/(2σ²)` is a positive constant. So minimizing the NLL is **exactly** minimizing

```
Σᵢ (yᵢ − ŷᵢ)²
```

which is mean squared error. ∎

**So: squared error is not a design choice. It is what you get if you assume Gaussian noise.** And the converse matters practically — if your noise has heavy tails (occasional large outliers), the Gaussian assumption is wrong, and MSE is the wrong loss. That's precisely when people reach for L1 or Huber loss, and now you know why rather than just that.

### MLE with a Bernoulli ⟹ binary cross-entropy

Assume `y ∈ {0,1}` is Bernoulli with parameter `p = σ(z)`. The mass function written compactly:

```
P(y | p) = p^y · (1−p)^(1−y)
```

(Check it: `y=1` gives `p`; `y=0` gives `1−p`.) Take the log:

```
log P(y | p) = y·log(p) + (1−y)·log(1−p)
```

Negate and average:

```
NLL = −(1/n) Σᵢ [ yᵢ·log(pᵢ) + (1−yᵢ)·log(1−pᵢ) ]
```

That is binary cross-entropy, exactly as written in §4.9. ∎

The categorical case gives softmax cross-entropy by the same two lines.

### Why `p − y` keeps appearing

The Gaussian, Bernoulli, and categorical distributions all belong to the **exponential family**. There's a general theorem: for any exponential-family distribution, if you parametrize by its *natural parameter* (which is the logit for Bernoulli/categorical, and the mean for a Gaussian) and minimize negative log-likelihood, the gradient with respect to that parameter is always

```
predicted mean − observed value
```

Your three derivations weren't three results. They were three instances of one theorem. That's what mathematical structure looks like when you find it, and noticing such patterns is most of what "research taste" means in practice.

---

## 6.6 Entropy

Now the information-theoretic view of the same objects.

**Entropy** measures the average surprise of a distribution:

```
H(p) = −Σᵢ pᵢ·log(pᵢ)
```

The `−log(p)` term is the **surprise** of an outcome. Probability 1 → surprise 0 (you learned nothing). Probability 0.001 → large surprise. Entropy is the expected surprise.

With log base 2 the unit is **bits**; with natural log it's **nats**. ML uses nats by default because derivatives are cleaner.

**Intuitions worth holding:**

- A fair coin: `H = 1` bit. Maximum uncertainty for two outcomes.
- A coin that always lands heads: `H = 0`. No uncertainty, no information gained.
- A uniform distribution over `k` outcomes: `H = log(k)`. **The maximum possible** for `k` outcomes.

Entropy also has a concrete operational meaning: it is the theoretical minimum average number of bits needed to encode samples from that distribution. Predictable data compresses; random data doesn't.

```python
def entropy(p):
    p = np.asarray(p)
    p = p[p > 0]                       # 0·log0 := 0
    return -np.sum(p * np.log(p))
```

---

## 6.7 Cross-entropy and KL divergence

**Cross-entropy** — the average surprise when the truth is `p` but you *believe* `q`:

```
H(p, q) = −Σᵢ pᵢ·log(qᵢ)
```

Coding interpretation: the average bits needed if you built your code for `q` but the data actually comes from `p`. Using the wrong model costs you extra bits.

**KL divergence** — exactly that extra cost:

```
KL(p ‖ q) = Σᵢ pᵢ·log(pᵢ/qᵢ) = H(p, q) − H(p)
```

Two properties to memorize:

```
KL(p ‖ q) ≥ 0,   with equality iff p = q          (Gibbs' inequality)
KL(p ‖ q) ≠ KL(q ‖ p)                             NOT symmetric
```

The asymmetry is real and consequential. `KL(p‖q)` heavily punishes `q` for assigning near-zero probability where `p` has mass — so minimizing it produces a `q` that *covers* everything (mode-covering). `KL(q‖p)` punishes `q` for putting mass where `p` has none — producing a `q` that latches onto one mode (mode-seeking). Which direction you use changes what your model does, and it's a recurring design decision in generative modelling and in RLHF.

### The identity that justifies your entire training procedure

```
H(p, q) = H(p) + KL(p ‖ q)
```

In supervised learning, `p` is the true data distribution. **You can't change it** — `H(p)` is a constant determined by your dataset, not your model. Therefore:

> **Minimizing cross-entropy is exactly minimizing KL divergence from your model to the truth.**

That's what training a classifier actually does. Not "make the loss number go down" — *move your predicted distribution as close as possible to reality, measured in bits.*

And it tells you something useful: **cross-entropy cannot go to zero** unless the data is deterministic. The floor is `H(p)`, the irreducible noise in the data. If your loss stalls above zero, that may be the data speaking, not your model failing.

```python
def cross_entropy(p, q, eps=1e-12):
    return -np.sum(np.asarray(p) * np.log(np.asarray(q) + eps))

def kl_divergence(p, q, eps=1e-12):
    p, q = np.asarray(p), np.asarray(q)
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / (q[mask] + eps)))

p = np.array([0.7, 0.2, 0.1])
q = np.array([0.5, 0.3, 0.2])
assert np.isclose(cross_entropy(p, q), entropy(p) + kl_divergence(p, q))
assert kl_divergence(p, q) != kl_divergence(q, p)      # asymmetric
```

---

## 6.8 Perplexity

Language models report **perplexity** rather than loss:

```
perplexity = exp(cross-entropy in nats)
```

It converts an abstract loss into something interpretable: **the effective number of choices the model is deciding between at each step.**

- Perplexity 1 → perfectly certain, always right.
- Perplexity 50,000 on a 50,000-token vocabulary → no better than guessing uniformly.
- A good modern LLM sits somewhere in the low tens on typical text.

So a model with perplexity 20 is, on average, about as uncertain as if it were choosing uniformly among 20 options. That's a far more meaningful statement than "the loss is 3.0" — even though they're the same number, since `ln(20) ≈ 3.0`.

You'll use this constantly in Chapter 12.

---

## 6.9 Sampling from a distribution

Once a model outputs a distribution, you have to pick something from it. This is where the knobs on every text generation API come from.

**Greedy** — always take the highest-probability token. Deterministic, and reliably produces repetitive, flat text.

**Pure sampling** — draw from the distribution as-is. Diverse, but occasionally samples something absurd from the long tail.

**Temperature** — reshape the distribution before sampling:

```
pᵢ ∝ exp(zᵢ / T)
```

- `T → 0` : approaches greedy. Sharper, safer, duller.
- `T = 1` : the model's actual distribution.
- `T > 1` : flatter. More surprising, more incoherent.

**Top-k** — zero out everything except the `k` highest-probability tokens, renormalize, sample. Cuts the tail.

**Top-p (nucleus)** — take the smallest set of tokens whose cumulative probability reaches `p`, renormalize, sample. Adapts to context: when the model is confident the set is tiny; when it's uncertain the set is large. Usually better than top-k for that reason.

```python
def sample(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    rng = rng or np.random.default_rng()
    logits = np.asarray(logits, dtype=float) / max(temperature, 1e-8)
    logits -= logits.max()
    p = np.exp(logits); p /= p.sum()

    if top_k is not None:
        keep = np.argsort(p)[-top_k:]
        mask = np.zeros_like(p, dtype=bool); mask[keep] = True
        p = np.where(mask, p, 0.0); p /= p.sum()

    if top_p is not None:
        order = np.argsort(p)[::-1]
        cum = np.cumsum(p[order])
        cutoff = np.searchsorted(cum, top_p) + 1
        mask = np.zeros_like(p, dtype=bool); mask[order[:cutoff]] = True
        p = np.where(mask, p, 0.0); p /= p.sum()

    return rng.choice(len(p), p=p)
```

You've now implemented every sampling parameter exposed by every commercial LLM API. They are not complicated; they're four lines of numpy each.

---

## 6.10 High dimensions break your intuition

Chapter 2 told you to reason in 2-D and trust the algebra. That's right for *linear algebra*. For **probability and geometry**, high dimensions are genuinely strange, and knowing how prevents real confusion later.

**1. Random vectors are nearly orthogonal.** Two random directions in `d` dimensions have `cos θ ≈ 0`, with fluctuation about `1/√d`. In 1000 dimensions, essentially every pair of random vectors is perpendicular.

*Why it matters:* you can pack an enormous number of nearly-independent directions into a moderate-dimensional space. This is why embeddings work — a 768-dimensional space can hold far more than 768 distinguishable concepts.

**2. Gaussian mass lives on a thin shell.** A standard `d`-dimensional Gaussian has `E[|x|²] = d`, so samples cluster at radius ≈ `√d`. Almost none are near the origin — despite the origin having the highest density.

*Why it matters:* the highest-density point is not where the samples are. Density and typicality come apart in high dimensions, which confuses people reading about generative models.

**3. Volume concentrates at the boundary.** In a `d`-dimensional ball, the fraction of volume within radius `1−ε` is `(1−ε)ᵈ`, which collapses fast. At `d = 100` and `ε = 0.05`, over 99% of the volume is in the outer 5% shell.

**4. Distances become uninformative.** The ratio between nearest and farthest neighbour approaches 1. This is the "curse of dimensionality," and it's why nearest-neighbour methods degrade in high dimensions — and part of why cosine similarity (an angle) is often preferred to Euclidean distance for embeddings.

```python
# see #1 and #2 directly
for d in (2, 10, 100, 1000):
    A = rng.standard_normal((2000, d))
    B = rng.standard_normal((2000, d))
    cos = np.sum(A*B, axis=1) / (np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1))
    norms = np.linalg.norm(A, axis=1)
    print(f"d={d:5d}  mean|cos| {np.abs(cos).mean():.4f}   "
          f"‖x‖ {norms.mean():.2f} ± {norms.std():.2f}  (√d = {np.sqrt(d):.2f})")
```

Run it. Watching `|cos|` collapse toward zero and the norms concentrate at `√d` is worth more than reading about it.

---

## 6.11 Exercises

**1.** Implement `entropy(p)`. Compute it for a fair coin, a `p=0.9` coin, a fair 6-sided die, and a uniform distribution over 1000 outcomes. Plot entropy of a Bernoulli against `p ∈ (0,1)`. Where is the maximum, and why?

**2.** Implement `cross_entropy(p,q)` and `kl_divergence(p,q)`. Verify `H(p,q) = H(p) + KL(p‖q)` on random distributions.

**3.** Verify `KL(p‖q) ≥ 0` empirically over 10,000 random pairs, and that it equals zero only when `p = q`.

**4.** Demonstrate the asymmetry: find `p, q` where `KL(p‖q)` and `KL(q‖p)` differ by more than 10×. Explain which behaviour each direction encourages.

**5.** Work the medical-test example of §6.4 by hand. Then plot `P(have | positive)` as the base rate varies from `1e-5` to `0.5`. At what base rate does a positive test become more likely true than false?

**6.** Verify the standard error scaling: for `B ∈ {1,4,16,64,256,1024}`, estimate the standard deviation of the sample mean and confirm it tracks `1/√B`.

**7.** Demonstrate the central limit theorem: sum `n` uniform random variables for `n = 1, 2, 5, 30`, histogram each, and overlay a Gaussian with matching mean and variance.

**8.** **Derive MSE from MLE on paper**, from the Gaussian assumption, without looking. Then verify numerically: fit a linear model by minimizing MSE and by maximizing Gaussian log-likelihood directly, and confirm identical parameters.

**9.** **Derive BCE from MLE on paper**, from the Bernoulli assumption. Verify numerically the same way.

**10.** Show empirically that cross-entropy is minimized exactly when `q = p`: fix `p`, parametrize `q` by a logit vector, run gradient descent on cross-entropy, and confirm `q → p`.

**11.** Take a trained classifier from Chapter 4 and compute its perplexity. Interpret the number in words.

**12.** Implement `sample()` with temperature, top-k, and top-p. On a fixed logit vector, draw 10,000 samples at `T = 0.1, 0.5, 1.0, 2.0` and plot the resulting empirical distributions. Then compare `top_k=5` against `top_p=0.9`.

**13.** Run the high-dimensional experiment of §6.10. Plot mean `|cos θ|` against `d` on log-log axes and confirm the `1/√d` slope.

**14.** Show that a `d`-dimensional standard Gaussian has almost no mass near the origin: for `d = 1, 10, 100, 1000`, plot the histogram of `‖x‖` and mark `√d`.

**15.** **Chapter project.** Write a short empirical report, "What my loss number means." Take a trained multiclass classifier, and report: cross-entropy in nats and bits, perplexity, the entropy of the label distribution (your irreducible floor), and the KL divergence from your predictions to the true labels. Explain what would have to change for the loss to reach zero, and whether it can.

---

## 6.12 Solutions

<details>
<summary>Open only after attempting</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)


# --- 1-3 ---
def entropy(p):
    p = np.asarray(p, float); p = p[p > 0]
    return -np.sum(p * np.log(p))

def cross_entropy(p, q, eps=1e-12):
    return -np.sum(np.asarray(p) * np.log(np.asarray(q) + eps))

def kl_divergence(p, q, eps=1e-12):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return np.sum(p[m] * np.log(p[m] / (q[m] + eps)))

print(entropy([.5,.5]) / np.log(2))            # 1 bit
print(entropy([.9,.1]) / np.log(2))            # 0.469 bits
print(entropy(np.ones(1000)/1000))             # log(1000) = 6.908 nats

ps = np.linspace(.001, .999, 400)
plt.plot(ps, [entropy([p, 1-p]) for p in ps]); plt.axvline(.5, ls="--"); plt.show()
# Max at p=0.5: maximum uncertainty. At p=0 or 1 the outcome is known, H=0.

for _ in range(10_000):
    p = rng.dirichlet(np.ones(5)); q = rng.dirichlet(np.ones(5))
    assert kl_divergence(p, q) >= -1e-12
    assert np.isclose(cross_entropy(p, q), entropy(p) + kl_divergence(p, q))
assert np.isclose(kl_divergence(p, p), 0, atol=1e-9)


# --- 4 ---
p = np.array([.5, .5]); q = np.array([.999, .001])
print(kl_divergence(p, q), kl_divergence(q, p))    # 3.11 vs 0.69
# KL(p‖q) explodes because q assigns ~0 mass where p has mass: forward KL is
# MODE-COVERING (q must cover all of p). Reverse KL is MODE-SEEKING (q may
# ignore parts of p as long as it never puts mass where p has none).


# --- 5 ---
def posterior(base, sens=.99, spec=.99):
    return sens*base / (sens*base + (1-spec)*(1-base))
print(posterior(0.001))                            # 0.0902
bases = np.logspace(-5, np.log10(.5), 300)
plt.semilogx(bases, posterior(bases)); plt.axhline(.5, ls="--"); plt.show()
# Crosses 0.5 at base rate = 0.01 exactly (when sens = spec = 0.99).


# --- 6, 7 ---
for B in (1,4,16,64,256,1024):
    m = [rng.standard_normal(B).mean() for _ in range(3000)]
    print(f"B={B:5d}  {np.std(m):.4f}  vs 1/√B={1/np.sqrt(B):.4f}")

fig, ax = plt.subplots(1, 4, figsize=(14,3))
for a, n in zip(ax, (1,2,5,30)):
    s = rng.uniform(0,1,(20000,n)).sum(axis=1)
    a.hist(s, bins=60, density=True); a.set_title(f"n={n}")
plt.show()      # visibly Gaussian by n=5, indistinguishable by n=30


# --- 8 ---
X = rng.standard_normal((200, 3)); w_t = np.array([1.,-2.,.5])
y = X @ w_t + .3*rng.standard_normal(200)

def mse(w):   r = X@w - y; return (r@r)/len(y)
def neg_ll(w, sigma=.3):
    r = X@w - y
    return len(y)*np.log(sigma*np.sqrt(2*np.pi)) + (r@r)/(2*sigma**2)

from scipy.optimize import minimize          # only for the comparison
a = minimize(mse, np.zeros(3)).x
b = minimize(neg_ll, np.zeros(3)).x
print(a, b, np.allclose(a, b, atol=1e-4))    # identical
# The NLL differs from MSE by an additive constant and a positive scale
# factor, and neither changes the argmin.


# --- 10 ---
p_true = np.array([.6,.3,.1]); z = np.zeros(3)
for _ in range(4000):
    q = np.exp(z - z.max()); q /= q.sum()
    z -= 0.5 * (q - p_true)                  # ∂CE/∂z = q - p   (§4.11)
print(np.exp(z-z.max())/np.exp(z-z.max()).sum())   # → [.6,.3,.1]
# Cross-entropy is minimized exactly at q = p, which is Gibbs' inequality.


# --- 12 ---
def sample_probs(logits, T=1., top_k=None, top_p=None):
    l = np.asarray(logits,float)/max(T,1e-8); l -= l.max()
    p = np.exp(l); p /= p.sum()
    if top_k:
        keep = np.argsort(p)[-top_k:]
        m = np.zeros_like(p,bool); m[keep]=True; p = np.where(m,p,0); p/=p.sum()
    if top_p:
        o = np.argsort(p)[::-1]; c = np.cumsum(p[o])
        k = np.searchsorted(c, top_p)+1
        m = np.zeros_like(p,bool); m[o[:k]]=True; p = np.where(m,p,0); p/=p.sum()
    return p

logits = np.array([3.,2.,1.,0.,-1.,-2.,-3.,-4.])
for T in (.1,.5,1.,2.):
    plt.plot(sample_probs(logits, T=T), "o-", label=f"T={T}")
plt.legend(); plt.show()
# T→0 concentrates all mass on the argmax; T→∞ flattens toward uniform.
print("top_k=5:", sample_probs(logits, top_k=5).round(3))
print("top_p=.9:", sample_probs(logits, top_p=.9).round(3))


# --- 13, 14 ---
ds = [2,5,10,50,100,500,1000,5000]; cosines = []
for d in ds:
    A, B = rng.standard_normal((3000,d)), rng.standard_normal((3000,d))
    c = np.sum(A*B,1)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1))
    cosines.append(np.abs(c).mean())
    print(f"d={d:5d}  mean|cos| {cosines[-1]:.4f}  ‖x‖={np.linalg.norm(A,axis=1).mean():.2f} (√d={np.sqrt(d):.2f})")
plt.loglog(ds, cosines, "o-"); plt.loglog(ds, 0.8/np.sqrt(ds), "--"); plt.show()
# Slope -1/2: random directions become orthogonal like 1/√d.
# And ‖x‖ concentrates tightly at √d — the thin-shell phenomenon.
```

</details>

---

## 6.13 Chapter 6 checkpoint

Cold — blank file, no notes.

- [ ] State Bayes' rule and work the medical-test example to the right answer. **5 minutes.**
- [ ] **Derive MSE from maximum likelihood** under Gaussian noise. On paper. **10 minutes.**
- [ ] **Derive binary cross-entropy from maximum likelihood** under a Bernoulli. On paper.
- [ ] Define entropy, cross-entropy, and KL divergence, and state the identity connecting them.
- [ ] Explain why minimizing cross-entropy is minimizing KL divergence, and what the loss floor is.
- [ ] Define perplexity and interpret a perplexity of 20 in plain words.
- [ ] Explain why gradient noise falls as `1/√B` and what that implies for batch size.
- [ ] State three ways high-dimensional geometry defies intuition.

Items 2 and 3 are mandatory. They're what separates using a loss function from understanding one.

### Anki cards

- Bayes' rule, and why a positive test on a rare condition is usually a false positive
- NLL definition; "every loss is a negative log-likelihood under some assumption"
- Gaussian noise ⟹ which loss?
- Bernoulli ⟹ which loss?
- Entropy formula and what it measures
- `H(p,q) = ?` in terms of `H(p)` and KL
- Is KL symmetric? What does each direction encourage?
- Perplexity definition and interpretation
- Gradient noise scaling with batch size
- Variance of a Bernoulli — and where else you've seen `p(1−p)`
- Chain rule of probability, and why it justifies autoregressive LMs
- Three high-dimensional surprises

### Deliverables

```
utils/info.py          entropy, cross_entropy, kl_divergence, perplexity
utils/sampling.py      temperature / top-k / top-p
notebooks/highdim.py   the §6.10 experiments with plots
reports/loss_meaning.md  exercise 15
```

```bash
git add .
git commit -m "Chapter 6: probability, MLE derivations of MSE and cross-entropy, sampling"
git push
```

### Write-up

600 words: **"Nobody invented cross-entropy."** Show both MLE derivations, then the `H(p,q) = H(p) + KL(p‖q)` identity and what it means for the loss floor. Close with the exponential-family observation explaining why `p − y` appears in all three cases.

This post demonstrates something rarer than it should be: knowing that loss functions are *consequences of assumptions*, not arbitrary choices. It's a real signal of depth.

---

*Next: Chapter 7 — Multilayer Networks*
