# Chapter 5½ — Thinking Out Loud

**Read after Chapter 5. Reread after Chapters 9, 12, and 15.**

**Time: 2–3 days to read carefully. Then it's a reference.**

---

## 0. The lie in every other chapter

Every derivation in this book is presented finished. Three lines, correct, inevitable-looking.

**That's a pedagogical lie**, and it's the one that does the most damage to self-taught learners.

Nobody produces mathematics that way. What actually happens is: you write something, it's wrong, the shapes don't match, you back up, you try a different decomposition, you get a sign backwards, you check a special case and it fails, you find the missing dependency, you finally get it, and *then* you write the three clean lines.

The clean version is a summary of the result. **It is not a record of the process.**

Why this matters more than it sounds: when you sit down to derive something and immediately flounder, you compare your flailing against the book's polish and conclude you're not built for this. Everyone does this. The comparison is invalid — you're comparing your process to someone else's edited output.

So this chapter is three transcripts of unedited reasoning, wrong turns included. They're less flattering than the rest of the book and considerably more useful.

---

## 0.1 How to read these

**Don't read them straight through.** For each one:

1. Read the problem statement.
2. **Close the file and attempt it yourself for at least 30 minutes.**
3. Then read the transcript and compare — not answers, *processes*. Where did you go the same way? Where did you get stuck that I didn't, and vice versa?

The transcripts are marked like this:

```
→   a step forward
✗   a wrong turn (left in deliberately)
!   the moment something clicked
?   genuine uncertainty, unresolved at that point
```

I've compressed some tedium — you don't need to watch me re-copy an expression six times. But every wrong turn shown is one I actually made or would plausibly make, and none of the recoveries are smoother than they'd really be.

---

# Transcript A — Deriving the LayerNorm gradient

**The problem.** LayerNorm is

```
μ = (1/H)Σᵢxᵢ,   σ² = (1/H)Σᵢ(xᵢ − μ)²
x̂ = (x − μ)/√(σ² + ε)
y = γ ⊙ x̂ + β
```

Given `∂L/∂y`, find `∂L/∂x`. `H` is the feature dimension.

**Try it yourself first. 30 minutes minimum.**

---

## The transcript

```
→ OK. y = γx̂ + β, so ∂L/∂x̂ = ∂L/∂y ⊙ γ. That part's trivial.
  Write dx̂ for that from here.

→ Now x̂ = (x − μ)/s where s = √(σ²+ε).

✗ So ∂x̂/∂x = 1/s, and therefore dx = dx̂/s.
  Done? That was fast.

→ Let me gradient check it before I believe it.
  [writes numerical check, runs]
  max relative error: 2.7e-01

  FAIL. Badly. Not a small numerical thing — it's just wrong.

? But the shapes are right and 1/s is obviously the derivative of
  dividing by s. What am I missing?

→ Print both. Analytic and numerical, elementwise.

  analytic: [ 0.412  -0.203   0.887  -1.096]
  numeric:  [ 0.089  -0.526   0.564  -1.419]

  Difference: [0.323  0.323  0.323  0.323]

! They differ by a CONSTANT. Every element off by the same amount.
  That's not noise. A constant offset across all elements means
  I'm missing a term that's the same for every index — which means
  something that was summed over the features.

! μ. μ = mean(x). μ depends on x. I treated it as a constant.

→ Right, so x affects x̂ through more than one route:
    - directly, in the numerator
    - through μ, also in the numerator
    - through σ², in the denominator

  Multi-path rule (§3.5). I need to sum all three.
  I only did the first one.

→ Path 2: ∂μ/∂xⱼ = 1/H for every j.
  So the numerator (xᵢ − μ) contributes δᵢⱼ − 1/H.

  ∂x̂ᵢ/∂xⱼ = (δᵢⱼ − 1/H)/s     ... plus whatever σ² gives.

→ Check just this much numerically first, before doing the hard path.
  [runs]
  max relative error: 6.1e-02

  Better. Was 2.7e-01. So that term was real. Still wrong though.
  Good — I'm converging rather than guessing.

→ Now σ². This is the messy one.

  σ² = (1/H)Σᵢ(xᵢ − μ)²

  ∂σ²/∂xⱼ = (1/H)Σᵢ 2(xᵢ − μ)·∂(xᵢ − μ)/∂xⱼ
          = (2/H)Σᵢ (xᵢ − μ)(δᵢⱼ − 1/H)

✗ = (2/H)(xⱼ − μ) − (2/H²)Σᵢ(xᵢ − μ)

? The second term is ugly. Let me expand it before I keep carrying it.

! Σᵢ(xᵢ − μ) = Σxᵢ − Hμ = Hμ − Hμ = 0.

  The whole second term vanishes. Deviations from the mean sum
  to zero — of course they do. So:

    ∂σ²/∂xⱼ = (2/H)(xⱼ − μ)

  Much cleaner than I feared.

→ Now s = √(σ²+ε), so ∂s/∂xⱼ = (1/(2s))·∂σ²/∂xⱼ = (xⱼ − μ)/(H·s)

→ And x̂ᵢ = (xᵢ − μ)·s⁻¹, so the σ² path contributes
    −(xᵢ − μ)·s⁻² · ∂s/∂xⱼ
  = −(xᵢ − μ)(xⱼ − μ)/(H·s³)

→ Assemble:

  ∂x̂ᵢ/∂xⱼ = (δᵢⱼ − 1/H)/s − (xᵢ−μ)(xⱼ−μ)/(H·s³)

→ That third term is horrible. But (xᵢ−μ)/s = x̂ᵢ, so
  (xᵢ−μ)(xⱼ−μ)/s³ = x̂ᵢ·x̂ⱼ/s.

  ∂x̂ᵢ/∂xⱼ = (1/s)[δᵢⱼ − 1/H − x̂ᵢx̂ⱼ/H]

  Everything in terms of x̂ and s. Much better.

→ Chain it: dxⱼ = Σᵢ dx̂ᵢ · ∂x̂ᵢ/∂xⱼ

  dxⱼ = (1/s)[ dx̂ⱼ − (1/H)Σᵢdx̂ᵢ − (x̂ⱼ/H)Σᵢ dx̂ᵢx̂ᵢ ]

  In vector form:

  dx = (1/s)·( dx̂ − mean(dx̂) − x̂ ⊙ mean(dx̂ ⊙ x̂) )

→ Gradient check.
  max relative error: 3.1e-11

  Correct.

→ Sanity check before trusting it. If dx̂ is constant across features,
  then mean(dx̂) = dx̂ so the first two terms cancel; and mean(x̂) = 0
  by construction so the third term... needs mean(dx̂ ⊙ x̂) = c·mean(x̂) = 0.
  So dx = 0.

  Does that make sense? Yes — LayerNorm removes the mean, so shifting
  all outputs uniformly is invisible to the loss through this layer.
  The gradient should be zero. It is.

  Now I believe it.
```

---

## What to extract from Transcript A

**The wrong answer was fast and felt right.** `dx = dx̂/s` took ten seconds and looked obviously correct. If I hadn't gradient-checked, I'd have shipped it, and the network would have trained — worse, silently, forever.

> **The dangerous errors are the ones that produce plausible answers.**

**The error structure was the clue.** A *constant* offset across every element isn't random noise. It pointed directly at "something summed over the feature axis," which pointed at `μ`. Reading the shape of the error rather than just its magnitude is a skill, and it saved twenty minutes here.

**I checked partway.** After adding only the `μ` term I re-ran the check. Error went `2.7e-1 → 6.1e-2`. Still wrong, but *converging* — which told me the new term was real and I should keep going rather than start over. **Verify incrementally.** A single check at the end tells you only that something is wrong.

**The ugly term simplified.** `Σᵢ(xᵢ − μ) = 0` was sitting right there. When an expression looks disgusting, look for a term that vanishes before you commit to carrying it.

**Rewriting in terms of quantities you already have.** `(xᵢ−μ)/s` **is** `x̂ᵢ`. Recognizing that turned an unimplementable expression into three lines of code. This is a general move: express the answer using values the forward pass already computed (§3.4).

**The sanity check was separate from the gradient check.** Passing a numerical check tells you the derivative matches the function. It does not tell you the *function* is what you meant. The constant-shift argument was independent evidence.

---

# Transcript B — A bug that passed every test

**The situation.** I've implemented multi-head attention from scratch (§11.8) and built a small GPT with it. It trains. Loss goes down. But it plateaus at 3.9 where a reference implementation at the same config reaches 3.4.

Not broken. Just *worse*. Which is the hardest kind of bug.

---

## The transcript

```
✗ Probably hyperparameters. Let me sweep the learning rate.
  [40 minutes]
  3e-4: 3.91   1e-4: 3.94   1e-3: 4.12   6e-4: 3.89

  All the same. It's not the LR.

  [In hindsight: I should not have started here. LR sweeps are
   expensive and they were my first move because they're easy,
   not because they were likely. That's the wrong reason.]

→ Run the actual protocol (§7.9). Step 1: initial loss.

  10.83, expected ln(50257) = 10.82. Fine.
  So init and the loss function are OK.

→ Step 2: overfit 100 tokens.
  Loss 3e-5 after 400 steps.

  It CAN overfit. So gradients flow, the optimizer works,
  the data pipeline is aligned.

→ Step 3: gradient check the attention block against numerical.
  max rel error 4e-7. Passes.

? Now I'm stuck. Initial loss right, can overfit, gradients correct,
  LR doesn't matter. Every test I have says the model is fine.
  And it underperforms.

→ Think about what that combination actually means.

! All three of those tests check CONSISTENCY, not INTENT.

  - Gradient check: is my backward the derivative of my forward?
  - Overfit-tiny: can this function memorize?
  - Initial loss: is the output distribution uniform at init?

  A function that is wrong but smooth and differentiable passes
  all three. I've proven my code correctly computes SOMETHING.
  I have not proven it computes attention.

  That's a genuinely different class of bug and I need a different
  class of test.

→ The test for intent is parity against a reference (§8.13, ex 3).
  Same weights, same input, compare outputs.

  [loads reference MHA, copies weights across, runs]

  mine:      [ 0.213  -0.887   1.042 ...]
  reference: [ 0.641   0.208  -0.339 ...]

  Completely different. Not a precision issue. Structurally wrong.

  And my forward has been wrong this entire time.

→ Bisect the forward pass. Compare intermediates.

  after qkv projection:   match  ✓
  after head split:       DIFFER ✗

  It's in the reshape.

→ Mine:
    Q = (X @ Wq).reshape(B, n_heads, T, dk)

  Reference:
    Q = (X @ Wq).reshape(B, T, n_heads, dk).transpose(0, 2, 1, 3)

! Both produce shape (B, h, T, dk). Both are "valid". They are not
  the same operation.

  The projection output is laid out (B, T, d) with d contiguous
  in memory. Reshaping straight to (B, h, T, dk) reinterprets that
  memory — it slices the flat buffer as though heads varied slowest,
  when actually TIME varies slower than the feature index.

  So my "head 0" is not head 0's features across all positions.
  It's a scrambled mix of features from the first T/h positions.

  Reshape reinterprets memory. Transpose reorders it. I needed
  reorder and I wrote reinterpret.

→ Why did it still learn anything?

  Because it's a fixed, deterministic, differentiable permutation.
  The model can partially compensate — the projections can learn
  to pre-scramble so the post-scramble comes out usable. But it's
  spending capacity undoing my bug, and the causal mask is now
  applied over a scrambled position axis, so information leaks
  across positions in a way that's just noise.

  Hence: trains, but worse. Exactly the symptom.

→ Fix it. Rerun.
  Loss 3.38. Reference gets 3.4.

→ Add a permanent parity test to the test suite so this can never
  silently recur.
```

---

## What to extract from Transcript B

**I wasted 40 minutes on a learning rate sweep because it was easy, not because it was likely.** That's the most common failure of debugging discipline and I still do it. The protocol exists precisely to stop this — it's ordered by information-per-minute, not by comfort.

**The key realization is worth stating on its own:**

> **Gradient checks, overfit-tiny tests, and initial-loss checks all verify *consistency*. None verifies *intent*. A wrong-but-differentiable function passes all three.**

The book underemphasizes this and it's a real gap. Those tests catch broken code. They do not catch code that correctly implements the wrong thing. The test for that is **parity against a reference implementation** — and where no reference exists, the test is a *semantic* one: an assertion about what the operation should mean, not about whether its derivative is right.

For attention, the semantic test is the one from exercise 11.8: **change a token after position `i` and assert position `i`'s output is unchanged.** That's a claim about meaning. My scrambled version would have failed it instantly.

**The diagnostic move that broke it open was noticing what the passing tests had in common.** Not "which test can I run next" but "what does this *pattern* of results rule out?" Three tests passing and the model still underperforming was information — it narrowed the bug to a class, and the class determined the next test.

**Bisecting the forward pass** is the same binary search from §1.4, applied to intermediate values rather than lines of code. Compare intermediates against a reference until they diverge. Mechanical, boring, always works.

**`reshape` versus `transpose` is the single most common bug in hand-written attention.** Both give the right shape. Only one gives the right values. When a shape is correct but the numbers are wrong, suspect a reinterpretation where you wanted a reorder.

---

# Transcript C — From an observation to a research question

**The situation.** While making attention visualizations for the Chapter 12 write-up, I notice head 2 in layer 0 puts most of its weight on token 0, regardless of what token 0 is or what the rest of the sequence says.

This is the messiest of the three transcripts, because generating research questions genuinely is messy.

---

## The transcript

```
→ Head 2 layer 0 attends ~80% to position 0 on almost every input.
  Position 0 is a different token each time. It shouldn't be
  special.

✗ First thought: bug. Off-by-one in the mask, or positional
  encoding applied to V.

→ Check both. Mask is right (semantic test passes). PE applied to
  Q and K only. Not a bug.

→ Check whether it's just my model. Load GPT-2 small, plot the same.
  Same pattern. Multiple heads, multiple layers.

  So it's real and general, not mine.

→ Hypotheses. Generate freely, prune after.

  H1: Position 0 accumulates some "summary" of the sequence.
  H2: It's a no-op slot — heads that have nothing useful to do
      park their attention somewhere harmless.
  H3: An artifact of softmax: attention must sum to 1, so a head
      with nothing to attend to must still put its mass somewhere.
  H4: Position 0 has a unique positional encoding, making it
      easy to address.
  H5: It's related to the first token being unpredictable.

✗ H1 is weak — layer 0 hasn't computed anything to summarize yet.
  Drop it.

! H2 and H3 are the same hypothesis stated two ways. H3 gives the
  MECHANISM for H2. That's better — a mechanism is testable and a
  description isn't.

→ Search before going further. This is step 3 of §15.1 and I keep
  wanting to skip it.

  [searches]

  This is documented. "Attention sinks." Known, studied, and there's
  work on it in the context of streaming inference and quantization.

→ So the question is answered. Good — cost me 40 minutes instead
  of two months.

? Is there anything adjacent that ISN'T answered?

→ Re-read what I found. The known results are about large models
  and their consequences for inference. What I have not seen
  addressed:

  Q1: At what scale does the sink appear? Is it present in a
      10M-parameter model, or does it emerge?
  Q2: Does it appear at initialization, or is it learned? If
      learned, when during training?
  Q3: Does it appear if you remove the softmax normalization
      constraint (e.g. allow attention weights not to sum to 1)?

→ Filter (§15.1).

  Q1 — runnable? Yes, I can train models at 4 sizes on free compute.
       Interesting either way? Yes: if present at 10M, it's not an
       emergent property of scale, which contradicts the framing
       in some of what I read. If absent, the emergence point is
       itself a number worth knowing.
       ✓ Keep.

  Q2 — runnable? Yes, just log attention entropy during training.
       Cheap. Interesting either way? Yes.
       ✓ Keep. And it's nearly free to add to the Q1 runs.

  Q3 — runnable? Requires modifying attention and retraining.
       Doable but more work. Interesting either way? If removing
       the constraint kills the sink, that's strong evidence for
       H3's mechanism. If it doesn't, H3 is wrong. Genuinely
       both-ways interesting.
       ✓ Keep, but as the follow-up, not the first experiment.

? Am I sure Q1 and Q2 aren't done? Search again, differently.

  [searches with different terms, checks what cites the papers
   I found]

  Found one paper touching Q2 at large scale. Nothing at small
  scale, nothing tracking it across training in the 10M–100M range.

→ So: Q1+Q2 combined, as one experiment. Train 4 sizes, log
  attention entropy and sink strength every N steps. Q3 as the
  mechanism follow-up if the first result is interesting.

  That's a workshop paper's worth of work, runnable on Kaggle.

→ Write the proposal (§15.2) before touching any code. Including
  the falsification line: "if sink strength is roughly constant
  across all four scales and appears within the first 100 steps,
  there's no scale story and I report that."
```

---

## What to extract from Transcript C

**The first move was to assume it was a bug.** Correct instinct. Most anomalies are bugs, and checking is cheap. But notice the second move: **check whether it reproduces in someone else's model.** That one step separates "my code is broken" from "this is a property of the thing," and it takes ten minutes.

**H2 and H3 turned out to be the same hypothesis.** One described the behaviour, one proposed a mechanism. Realizing they were the same collapsed two ideas into one *better* one. When two hypotheses feel similar, check whether one is the mechanism for the other — that's usually an upgrade, because mechanisms are testable and descriptions aren't.

**I searched, and it had been done. That was a good outcome.** Forty minutes instead of two months. I keep wanting to skip this step because searching feels like admitting the idea might not be mine. Do it anyway, twice, exactly as §15.1 says.

**The real move was asking what's *adjacent* to the answered question.** "Someone answered this" almost never means "this whole area is closed." The known result was about large models and inference consequences. The scale question and the training-dynamics question were sitting right next to it, unasked, and cheap to run.

> **A question that's been answered is usually surrounded by questions that haven't.**

**The filter killed nothing but reordered everything.** All three survived "interesting either way," which is a good sign about the question quality. The ordering came from cost — do the cheap combined experiment first, keep the expensive mechanism test as a follow-up conditional on the first result.

**Notice how ordinary this is.** No brilliance anywhere in the transcript. Notice something odd → rule out the boring explanation → check it generalizes → generate several explanations → find the mechanism version → search → find the adjacent gap → filter by cost and both-ways-interest. That sequence is most of what research question generation actually is, and it's entirely learnable.

---

## The moves that recur

Across all three transcripts, the same handful of moves keep doing the work:

| Move | Where it appeared |
|---|---|
| **Check before believing** | A: gradient check killed a confident wrong answer immediately |
| **Read the *shape* of the error, not just its size** | A: a constant offset pointed straight at a summed quantity |
| **Verify incrementally** | A: error `2.7e-1 → 6.1e-2` said "keep going," not "start over" |
| **Look for the vanishing term** | A: `Σ(xᵢ−μ) = 0` removed half the mess |
| **Re-express using what you already have** | A: `(xᵢ−μ)/s` is just `x̂ᵢ` |
| **Ask what the passing tests have in common** | B: three passes + bad result = consistency verified, intent not |
| **Bisect against a reference** | B: compare intermediates until they diverge |
| **Rule out the boring explanation first** | C: is it a bug? does it reproduce elsewhere? |
| **Prefer the mechanism to the description** | C: H3 subsumed H2 and became testable |
| **Search before investing** | C: 40 minutes saved two months |
| **Sanity-check independently of the formal check** | A: the constant-shift argument |

None of these is clever. **All of them are habits**, and habits are trainable by deliberate practice, which is what the exercises below are.

---

## The one that isn't on the list

There's a move I used three times without naming it, and it's the most important one:

> **When stuck, ask what class of thing the evidence rules out — rather than what to try next.**

- Transcript A: "the error is constant, so it's something summed over features."
- Transcript B: "all my tests check consistency, so the bug is in intent."
- Transcript C: "it reproduces in GPT-2, so it's not my code."

The instinct when stuck is to generate more things to try. That's expensive and often random. The better move is to spend two minutes asking what you've already *eliminated* — because that usually determines the next test rather than leaving you to guess it.

This is Unstuck Protocol step 1 (§0.3), "name it precisely," doing more work than it looks like it does. Naming what you know narrows what you don't.

---

## Exercises

These are the point of the chapter. Reading transcripts teaches much less than producing them.

**1.** Before reading Transcript A, derive the LayerNorm backward yourself. Record your process as you go — every wrong turn, in the format above. Then compare processes with the transcript.

**2.** Do the same for **RMSNorm** (§7.6). It's simpler — no mean subtraction — so predict beforehand which terms disappear. Then derive it, and check whether your prediction was right.

**3.** Derive **softmax's backward pass** from scratch, recording your process. Then check it against §4.11's Jacobian.

**4.** Deliberately plant the reshape bug from Transcript B in your own attention implementation. Confirm it passes your gradient check and your overfit-tiny test. **Then write the semantic test that catches it.** Add that test permanently.

**5.** For each of these components, write down the **semantic test** — an assertion about what the operation should *mean*, not whether its derivative is right:
   - causal masking
   - a KV cache
   - LayerNorm
   - a residual connection
   - weight tying

   Add all five to your test suite.

**6.** Take a bug you've already fixed this year. Write the transcript retrospectively, honestly, including the time you wasted and why you wasted it. Retrospective transcripts are less useful than live ones but much better than nothing.

**7.** **The live one.** The next time you're stuck for more than 30 minutes, open a file and write the transcript *as it happens*. It slows you down maybe 15%, and it does two things: it forces the step-1 discipline of naming what you know, and it gives you a record to reread later.

**8.** Pick an anomaly you've noticed in your own models — a loss curve bump, a weird activation distribution, a head doing something strange. Run Transcript C's sequence on it: rule out the bug, check it generalizes, generate hypotheses, find the mechanism version, search, look for the adjacent gap, filter.

Whether or not you find a question, **you'll have practised the sequence** — and that's what's being trained.

**9.** Keep a `transcripts/` directory. One file per real episode. Reread it monthly. You'll start noticing your *own* recurring wrong turns, which are the ones worth fixing — and which nobody else can identify for you.

---

## Why this matters more at month six than now

Right now the transcripts probably read as reassuring: "even a careful derivation goes wrong, so my confusion is normal." That's true and worth having.

But the real value shows up later. Around month six you'll be doing work with no answer key, and the question that determines whether you continue is **"is my confusion productive or am I just lost?"**

The transcripts are a reference for what productive confusion looks like from the inside. It looks like: wrong answer, error with structure, hypothesis, partial fix, convergence, correct answer, independent check. Messy, non-linear, and *converging*.

Lost looks different: the same wrong thing repeatedly, no narrowing, no elimination, error not decreasing. **If your transcript shows no convergence over two hours, that's the signal to invoke Unstuck step 7** — park it, move on, and come back Sunday.

Being able to tell those apart is one of the things a mentor provides. Your own transcripts are the closest substitute, and unlike a mentor, they're calibrated to you.

---

*Next: Chapter 6 — Probability and Information Theory*
