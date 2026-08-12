# Chapter 13½ — Reading for Judgment

**Read after Chapter 13. Reread every three months for as long as you do this work.**

---

## 0. The thing this book can't give you

Everything else here teaches you to **answer** questions. Derive this, debug that, reproduce, ablate, write up.

None of it teaches you which questions are worth answering.

That judgment — call it taste — is the difference between a researcher who publishes steadily and one whose work the field builds on. And it's genuinely hard to transmit in text, because taste is normally acquired by *proximity*: you watch someone who has it make choices in real time. They reject an idea you thought was fine. They get interested in something that looked boring. They spot the flaw in your reasoning before you've finished the sentence.

You don't have that. So this chapter is the best available substitute: **taste is visible in artifacts, if you know how to look.**

Every paper is a record of hundreds of decisions. The paper reports the outcome. The decisions are mostly invisible — but they're *recoverable by inference*, and recovering them is a trainable skill.

This is the highest-leverage reading skill in the book, and Chapter 13 doesn't teach it.

---

## 1. Content versus choices

Chapter 13 taught you to read for **content**: what's the claim, does the evidence support it, is the baseline fair. That's necessary and it makes you a competent reader.

Reading for **judgment** asks a different set of questions:

> Why *this* question, out of every question they could have asked?
> Why this experiment, and not the obvious alternative?
> What did they deliberately not do?
> What would they have had to believe, before starting, for this project to seem worth a year?

That last one is the deepest. **Reconstructing the prior belief that made a project worth starting is the closest you can get to seeing inside a researcher's model of the field.**

The content of a paper tells you what's true. The choices tell you how a good researcher thinks. You need both, and only one of them is taught anywhere.

---

## 2. The choice inventory

Run this on any paper you've decided to take seriously. Ten questions, in writing. It takes about an hour after a pass-3 read.

**1. Why this question?** What made it worth a year of someone's life? What did they believe about the field that made this the important gap?

**2. Why this scale?** Why not smaller and cheaper? Why not larger and more convincing? The answer is usually budget — but sometimes it's that the effect doesn't survive at other scales, which is a much more interesting answer.

**3. Why these baselines?** More importantly: **which obvious baseline is missing?** For a strong group, "they didn't think of it" is unlikely. Absence is usually informative.

**4. Why this dataset?** What does it favour? What kind of method would look good on it that shouldn't?

**5. Why this metric?** What does it hide? Every metric hides something.

**6. What's the simplest version of this idea?** If they did something more complex than that, why? A method that's more complicated than necessary usually means the simple version didn't work — and the paper rarely says so.

**7. What ablation is missing?** If they claim mechanism X and never isolate X, the mechanism claim is weaker than the abstract suggests.

**8. What would they have had to believe to start?** Reconstruct the prior. This is the important one.

**9. What did they get lucky on?** Every project has a step that could have gone the other way. Finding it tells you how much of the result is method and how much is contingency.

**10. Where did they stop, and why there?** The last experiment in a paper is often the one that was still running at the deadline. Sometimes it's the one that would have complicated the story.

---

## 3. A worked example

**Important caveat before I start:** what follows is *reconstruction from evidence*, not knowledge of what the authors thought. I'm inferring. Being explicit about that uncertainty is part of the skill — you're building a hypothesis about someone's reasoning, and it can be wrong.

Take **Chinchilla** (§12.6). You've read it. Now read it for choices.

**Why this question?** The field was following Kaplan's scaling laws to allocate enormous compute budgets. The implicit prior: *everyone is doing something, and it might be wrong.*

Notice how valuable that prior is. If a widely-followed practice is wrong, correcting it changes the behaviour of an entire industry. **The expected value of checking consensus is high precisely because nobody is checking** — it's under-explored for social reasons, not technical ones.

That's a transferable move: look for things everybody does that nobody has verified. §13.9's fifth pattern, and it's the one that produces the biggest results.

**Why train hundreds of models?** Because the claim is about a *curve*, not a point. You cannot argue about scaling from two data points. The expense wasn't incidental — the shape of the claim dictated the shape of the experiment.

**Why three independent estimation methods?** This is the choice I'd most want you to notice. A single method's answer could be an artifact of that method. Three methods agreeing is a much stronger epistemic position than one method with tighter error bars.

**That's a deliberate decision about how much to be believed** — and it's the kind of choice that separates careful work from merely correct work. Most papers don't triangulate. Ask yourself why not, and whether yours should.

**Why the head-to-head against a much larger model?** Because a curve fit is an argument and a matched-compute comparison where the smaller model wins is a *demonstration*. They chose the experiment that would settle the question rather than the one that would merely support it.

**What did they not do?** They optimized *training* compute and not inference cost. That scoping decision is precisely the limitation that mattered most afterwards — most models since are trained well past the Chinchilla point because serving cost dominates over a model's lifetime.

**The lesson isn't that they were wrong.** It's that a well-executed result answers exactly the question it posed, and the question you *should* ask is often adjacent to the question that was asked. Reading for choices is how you find that adjacency.

---

## 4. The negative space

What a paper doesn't contain is frequently more informative than what it does.

| What's missing | What it probably means |
|---|---|
| An obvious baseline | It didn't favour them, or it wasn't tuned |
| Results at another scale | Couldn't afford it — or it didn't hold there |
| The ablation isolating the claimed mechanism | The mechanism claim is a hypothesis, not a finding |
| Variance / seed counts | The effect may be within noise |
| A second dataset | Generalization is narrower than the framing suggests |
| A negative result from the same project | It exists. It always exists |
| Compute used | The comparison may not be compute-matched |

**Be fair about this.** Page limits are real, and not every absence is evasion. But absence is *evidence*, and weighing it is part of reading well.

**Two uses.** First, calibration — how much should you actually believe this? Second, question generation — every gap in the negative space is a rung-1 contribution waiting (§13.9), and you now have working code from your reproduction.

---

## 5. Read programs, not papers

A single paper is a data point. **A researcher's last five papers is a trajectory, and the trajectory reveals taste in a way no single paper can.**

For any researcher whose work you respect, read their last five papers in order and ask:

- **What do they keep returning to?** The through-line is their actual research agenda, which is usually more coherent than any individual abstract suggests.
- **What did they abandon?** A line dropped after one paper usually means it didn't pan out. That's a negative result they'll never publish, and you can read it from the silence.
- **Did the questions get better, or just bigger?** Bigger is easy — more compute, more scale. Better is rare: sharper, more decisive, better-targeted.
- **What's their characteristic move?** Some people find the flaw in the standard setup. Some build the tool everyone then uses. Some ask the simple question nobody asked. Knowing your own characteristic move takes years, but recognizing others' is fast.

Do this for three or four researchers. The differences between them are more instructive than any single one — that's how you learn that there are multiple valid styles, and start noticing which is yours.

---

## 6. Watching judgment happen in public

There's one resource that is a literal, public record of experts exercising judgment on research, and almost nobody uses it for learning.

**OpenReview.** For ICLR, NeurIPS and others, the full review discussions are public — including for **rejected** papers.

You can read: what reviewers flagged, where they disagreed with each other, what the authors said in rebuttal, what changed a reviewer's mind, and what didn't. That's experts arguing about quality, in writing, at length, for free.

**How to use it:**

1. Take a paper you've read and thought was good. Read its reviews. **Find something a reviewer saw that you didn't.** There will be something.
2. Take a *rejected* paper on a topic you know. Read the reviews. Reconstruct why it was rejected before reading the meta-review, then check yourself.
3. Find a paper where reviewers strongly disagreed. That disagreement marks a genuine fault line in the field's standards — and knowing where those are is most of what "knowing a subfield" means.

**Do this ten times and your sense of what counts as a good result will shift measurably.** It's the closest available thing to sitting in a lab meeting.

Other proximity substitutes, ranked:

- **Getting your own work critiqued.** Highest bandwidth by far. Post reproduction reports and research proposals; ask specifically for objections (§15.2).
- **Recorded talks — watch the Q&A, not the talk.** The talk is rehearsed. The Q&A is someone thinking in real time, which is what you're trying to learn.
- **Reading groups.** Hearing three people disagree about a paper you've read teaches you more than reading it twice.

---

## 7. Calibrating what you believe

I stated things flatly throughout this book. **You should not hold them all at the same confidence**, and knowing which is which is part of taste.

| Confidence | Type of claim | Examples from this book |
|---|---|---|
| **Certain** | Mathematical results | Backprop; the layer-collapse theorem; `Var[q·k] = d_k`; the `√d` argument |
| **High** | Widely replicated empirical regularities | Residuals help at depth; Adam works broadly; dedup helps; He init for ReLU |
| **Moderate** | Frontier-scale results from few labs | Specific scaling exponents; optimal mixture ratios; most "emergent" claims |
| **Low** | *Explanations* for empirical findings | Why BatchNorm works; why flat minima generalize; why code data improves reasoning |
| **Expires fastest** | Current best practice | `β₂ = 0.95`; specific LR values; architecture defaults; tooling |

**The distinction that matters most is the second-to-last row.** A technique can be robust while its explanation is wrong.

The clean example: BatchNorm's original paper explained it via "internal covariate shift." That explanation is now widely doubted. **BatchNorm still works.** Technique and explanation are independent claims with independent evidence, and textbooks — including this one — routinely present them as one thing.

**Practical consequence:** when you read "X works because Y," treat those as two claims. The evidence for "X works" is usually strong. The evidence for "because Y" is usually much weaker and often just a plausible story attached after the fact.

**And apply this to your own work.** In Chapter 15, when you find an effect, resist the pull to also explain it. "We observe X" and "X happens because Y" require very different amounts of evidence, and conflating them is the single most common overreach in first papers.

---

## 8. How to notice this book has expired

Parts of it will. Here's how to tell which, and when.

**Age by chapter:**

| Ages | Chapters | Why |
|---|---|---|
| **Never** | 0, 2, 3, 5, 5½, 6 | Mathematics and method. The chain rule isn't going anywhere |
| **Slowly** | 1, 4, 7, 8, 10, App A, B | Fundamentals and framework basics |
| **Moderately** | 9, 13, 13½, 14, 15, App E, F | Practice and research craft |
| **Fast** | 11, 12, App C | Current architecture, tooling, defaults |

**Concrete signals that Chapters 11–12 have drifted:**

- A new normalization or positional encoding becomes standard
- Optimizer defaults shift, or something displaces AdamW
- Tokenization changes character — byte-level models, or no tokenizer at all
- Scaling law recommendations are revised again
- Something displaces attention at scale for real, not just in one paper

**The practice:** **every three months, take one fast-ageing chapter and audit it against the current literature.** An afternoon. Write the diff at the top of the file:

```markdown
> AUDIT 2027-02: §12.5's recipe still standard. §11.9 — RoPE variants now
> dominant for long context; see <papers>. §12.1 — byte-level approaches
> gaining ground, tokenization section may need rework.
```

A book that tells you how to detect its own expiry is strictly better than one that doesn't. **This is the mechanism by which the book stays useful past the date I wrote it**, and you're the only person who can run it.

---

## 9. Exercises

**1.** Run the full choice inventory (§2) on a paper you reproduced in Chapter 13. Ten questions, in writing.

**2.** Same paper: map the negative space (§4). What's missing, and what does each absence suggest?

**3.** Reconstruct the prior belief (question 8) for three papers in your track. Which of the three had the most valuable prior, and why?

**4.** **The OpenReview exercise.** Take a paper you thought was strong. Read its reviews. Write down one thing a reviewer saw that you missed.

**5.** Find a **rejected** paper on a topic you know well. Read it, predict the reviewers' objections, then read the reviews. How close were you?

**6.** Find a paper where reviewers strongly disagreed. Write a page on what the disagreement reveals about your subfield's standards.

**7.** Pick three researchers whose work you respect. Read their last five papers each. Write the through-line for each, and name their characteristic move.

**8.** Take 20 claims from this book. Assign each a confidence level from §7's table. Find at least two you think I overstated.

**9.** **Find something in this book that is now wrong or outdated.** There will be something. Write the audit note.

**10.** Take one fast-ageing chapter and audit it against the current literature. Add the note to the file. **Repeat quarterly, forever.**

**11.** **The payoff.** Run the choice inventory on **your own Chapter 15 proposal**, as though a stranger wrote it. Why this question? Why this scale? Which baseline is missing? What would a reader infer from your negative space?

Do this *before* you run the experiments, not after. It's the single most useful hour you can spend on a research project, and it's the whole point of learning to read this way.

---

## 10. What taste actually is

Having said all that, here's the honest compression.

Taste isn't mystical. It's a **compressed model of what has already been tried and what happened** — built from enough exposure that you can predict, before running an experiment, roughly how it will go. When someone rejects an idea in three seconds, they're not being brilliant. They're pattern-matching against a hundred similar things they've watched fail.

Which means two things.

**It's mostly volume.** Papers read, experiments run, results seen. There's no shortcut, and anyone offering one is selling something. The exposure has to accumulate.

**But the volume can be spent well or badly.** Fifty papers read for content build knowledge. Fifty papers read for *choices* build judgment. Same fifty papers, very different returns — and that difference is the only lever available to you.

That's the whole content of this chapter. You will read thousands of papers over the next decade. **Reading them for choices instead of only for content is the highest-leverage adjustment available**, and it costs you nothing but the habit.

---

## 11. The last thing

You started this book unable to write `matmul`.

If you finish it, you'll have an autograd engine, a transformer, a specialization, a reproduction record, a paper, and — if you run this chapter's habits — the beginnings of judgment about which questions deserve the next year.

**The book ends here. The method doesn't.**

The three habits that outlast every technical detail in these pages:

1. **Replace "do I feel like I understand?" with a test that can fail.** (§0.2)
2. **State what would prove you wrong, before you look.** (§15.2)
3. **Read for choices, not just content.** (this chapter)

Those transfer to every problem you will ever work on, in any field, for the rest of your life. The transformer will be obsolete. Those won't.

Good luck. Go do the work.

---

*This is the end of the book.*
