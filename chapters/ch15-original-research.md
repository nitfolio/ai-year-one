# Chapter 15 — Doing Original Research

**Time: 10 weeks** (Weeks 43–52 of the plan)

**Prerequisite:** Chapter 14 — a committed track, three reproductions, and three ranked candidate questions.

**What you'll produce:** one piece of original work, written up properly and put in front of the field.

---

## 15.0 What changes

Everything up to now had a known answer. Even Chapter 14's reproductions had a target number to hit.

From here there is no answer key. You will run experiments that don't work, for weeks, without knowing whether the problem is your code, your question, or the world. That's not a sign anything has gone wrong — **that's the job.**

Two things carry you through it:

1. **A question sharp enough that either answer teaches you something.**
2. **A process that produces information even when the result is negative.**

Both are learnable, and this chapter is both.

---

## 15.1 Choosing the question

Take your three candidates from §14.10 and apply these filters hard.

### Can you run the decisive experiment?

Not "could someone." *You*, on hardware you have, in under six weeks. If the answer is no, it's not your question this year — park it in a file and pick another.

### Is it interesting either way?

**The most important criterion, and the one most often failed.**

If only one outcome is worth reporting, you have set yourself up to fool yourself. You'll unconsciously tune toward the outcome you need, and you won't notice.

- ❌ "Does my new method beat the baseline?" — only one publishable answer.
- ✅ "Does method X's reported benefit survive at small scale?" — yes tells you it's robust; no tells you it's scale-dependent. Both are results.

Reframe until both outcomes are worth writing up.

### Has it been done?

Search properly, twice, a week apart. Google Scholar, Semantic Scholar, arXiv full-text, and the citation graph of the paper you're building on — check what cites *it*.

Finding it's been done is a good outcome, not a bad one. It costs you a day instead of two months.

### Can you get signal in two weeks?

If the smallest informative version of the experiment takes six weeks to run once, you get one shot and no iterations. Pick something with a faster pilot.

### Would you want to read the answer?

Honest test. If you wouldn't read this paper, don't write it.

### Scope: aim smaller than feels right

**Target a workshop paper, not a conference main track.** A tight, well-executed, clearly-scoped small result is worth far more than an ambitious half-finished one — and it's how essentially everyone's first paper looks.

Good first-paper shapes:

- "Method X, evaluated only on A, degrades on B — here's the pattern."
- "The standard practice of Y has never been ablated. It doesn't matter."
- "Papers P and Q disagree; the difference is explained by Z."
- "Effect E, reported at large scale, appears/doesn't appear at small scale."

Each of these is achievable in ten weeks on modest hardware.

---

## 15.2 The proposal — write it before any code

One page. **Before you write a line of code.**

```markdown
# Proposal — <date>

## Question
<one sentence, specific, falsifiable>

## Hypothesis
<what I expect, and the mechanism I think produces it>

## Decisive experiment
<the smallest experiment that distinguishes the outcomes>

## What would falsify my hypothesis
<be specific: "if X ≤ Y across 5 seeds, I'm wrong">

## What would make this uninteresting
<the outcome that means I should stop — decide it now>

## Baselines
<what I compare against, and how I'll tune them fairly>

## Compute and time budget
<GPU-hours, wall-clock weeks, and my ceiling>

## Closest related work
<5 papers and exactly how this differs from each>
```

**The falsification section is the most valuable part.** Writing down what would prove you wrong, *before* you have any data, is the single strongest defence against fooling yourself. It's preregistration in spirit, and it costs you an hour.

**Send it to two or three people.** A Discord channel in your track, a reading group, a researcher whose work you've reproduced. You'd be surprised how often someone replies "this was done in 2023, here's the paper" — and that hour saves you a month.

---

## 15.3 Designing experiments that could falsify you

**One variable at a time.** If you change the method and the learning rate together, you've learned nothing about either.

**Match compute.** Otherwise you may have discovered that more compute helps.

**Multiple seeds. Three minimum, five better.** From §6.3, run-to-run variance in deep learning is often larger than the effects people report. A single-seed result is not evidence.

**Tune the baseline as hard as your method.** §13.4's first critical question, now applied to yourself. This is where you are most likely to deceive yourself, because it's the step where nobody is checking.

**Predict before running.** Write down the expected number. Being surprised is informative; not having predicted means you can't be surprised.

### Negative controls — the technique nobody teaches

Before you claim "X has no effect," verify that **your setup can detect an effect you know is there.**

Concretely: plant a known effect of similar size — bump the learning rate by 2×, or remove a component you're confident matters — and confirm your measurement picks it up.

**If your setup can't detect a known effect, your null result means nothing.** It means your experiment lacks the power to see anything, not that there's nothing to see. This single practice separates a credible negative result from a worthless one, and almost no one does it.

---

## 15.4 The pilot: two weeks

**Goal of the pilot is not the result.** It's to answer two questions:

1. Is this experiment actually runnable, at this scale, in this time?
2. Is there any signal at all?

Build the smallest version. Fewest parameters, fewest steps, smallest dataset that could show the effect. Run it. Look at it.

### Decide your kill criteria now, in writing

> "If after two weeks I have no signal **and** no clear explanation for why, I move to question #2."

Deciding this in advance matters because in two weeks you'll be invested, and sunk cost will argue for continuing. Your two-weeks-ago self is the better judge.

**Killing a question is not failure.** It's the system working. Most researchers kill several questions for every one they pursue — you just don't see it because nobody publishes the killed ones.

---

## 15.5 Running the project

**The weekly loop:**

```
Monday      What is this week's decisive experiment?
Mon–Thu     Run it. Log everything, daily.
Friday      What did I learn? What does it change?
            What's next week's decisive experiment?
```

**The research log from §13.10, every day.** Five lines. This is what you'll write the paper from.

### The three ways projects die

**Infinite tuning.** "Maybe it works with a different learning rate." Set a budget before you start — say 20 configurations — and when it's spent, that's the answer. A method that only works at one hyperparameter setting is a finding, not a success.

**Scope creep.** "I should also try it on this other dataset." Write it in a `FUTURE_WORK.md` and don't do it. Ship the small thing.

**One more baseline.** There's always another comparison you could add. At some point the paper has to exist.

**Set a hard date.** "Week 51, I write up whatever I have." Then honour it.

---

## 15.6 When it fails

**Expect three of four ideas to fail.** That's the normal rate, for everyone, permanently. You are not doing it wrong.

But *localize* the failure — the four cases need different responses:

| What's happening | How you tell | What to do |
|---|---|---|
| **Bug** | Overfit-tiny test fails (§4.15); gradient check fails (§3.12) | Fix it. Always check this first. |
| **Effect doesn't exist** | Clean setup, negative controls pass, no effect across seeds | **This is a result.** Write it up. |
| **Effect exists, you can't detect it** | Negative controls also fail | Power problem. More seeds, bigger effect size, or larger scale. |
| **Question was ill-posed** | You can't state what result would settle it | Refine the question, or kill it. |

**The distinction between rows 2 and 3 is the whole game**, and negative controls are how you tell them apart. Without them, "no effect" and "no power" look identical — and one is publishable while the other is nothing.

### On negative results

A clean negative result — properly powered, fairly baselined, honestly reported — is a genuine contribution. The field is badly short of them, because they're hard to publish and nobody's incentives point that way.

They're also **exactly the kind of first paper a solo researcher can produce**, and workshops are notably more receptive to them than main tracks.

---

## 15.7 Knowing when you have something

Five tests. All five, or you're not done:

1. **The effect survives multiple seeds**, with variance reported.
2. **The baseline was tuned fairly** — as hard as your method.
3. **The ablation isolates the claimed mechanism**, not just the whole system.
4. **You can state the mechanism** — or you explicitly say you can't, which is honest and fine.
5. **Someone could reproduce it from your description alone.**

Test 5 is worth taking literally. Hand your methods section to someone in your track and ask whether they could run it. The gaps they find are the gaps a reviewer will find.

---

## 15.8 Statistical honesty

You are the person you're most likely to fool. Some specific defences:

**Always report variance.** Mean ± std across seeds, with the seed count stated. A number without a spread is not a result.

**Report the search.** If you tried 20 configurations and reported the best, say so. That's a result about your search, not about your method, and hiding it is the most common form of quiet dishonesty in the field.

**Distinguish preregistered from exploratory.** The analysis you planned in §15.2 is confirmatory. Everything you thought of afterwards is exploratory — still valuable, but label it. Exploratory findings need independent confirmation.

**Beware the garden of forking paths.** Every choice you made — which metric, which checkpoint, which subset — was a fork. Enough forks and you can find a significant result in noise. The defence is preregistration plus honest reporting of how many paths you walked.

**Report what didn't work.** Both because it's honest and because it's the most information-dense part of your project (§13.3).

---

## 15.9 Writing the paper

### Write it in this order

Not front to back:

```
1. Figures and tables      ← first
2. Method
3. Experiments
4. Related work
5. Limitations
6. Introduction
7. Abstract                ← last
```

**Figures first**, because making the plot forces you to know what your result actually *is*. If you can't make a figure that shows it, you may not have one.

**Abstract last**, because you can't summarize a paper you haven't written.

### The sections

**Abstract (150–250 words).** Context in one or two sentences → the gap → what you did → what you found, **with numbers** → why it matters. Vague abstracts are the most common flaw in first papers.

**Introduction.** The problem, why it matters, what's missing, what you contribute, in a bulleted list. Most reviewers form their opinion here.

**Related work.** Not a list. Organize by *idea*, and for each cluster say how yours differs. This is where you demonstrate you know the field.

**Method.** Precise enough to reimplement. Define every symbol. State every hyperparameter — or point to an appendix that does.

**Experiments.** Setup, then results. Report variance and seed counts. Include the ablation that isolates your mechanism.

**Limitations.** Write this honestly and at length. Name the scales you didn't test, the datasets you didn't try, the confounds you couldn't rule out. **Reviewers respect it, and it protects you** — a limitation you name yourself is context; one a reviewer finds is a weakness.

**Conclusion.** What you found, and what it opens up. Short.

### Practical

- **LaTeX via Overleaf.** Use the target venue's template from the start.
- **Every figure needs a caption that stands alone.** Many readers only look at figures.
- **Read it aloud.** Catches more bad writing than anything else.
- **Give it to two people** and ask them to tell you the main claim. If they get it wrong, that's your writing.

---

## 15.10 Submitting

**Workshops are the right target, and not as a consolation.** They exist precisely to surface early work, review is faster, acceptance rates are far higher, and they're genuinely welcoming to newcomers. NeurIPS, ICLR and ICML each run dozens; ACL and EMNLP likewise.

**Find them:** every major conference publishes a workshop list a few months ahead. Deadlines are usually 1–2 months after the main conference deadline. Match the workshop to your topic precisely — a well-matched workshop beats a prestigious mismatched one.

**Post to arXiv.** Timestamps your work, makes it citable and findable, and costs nothing. Do it when you submit, or on acceptance if the venue requires it.

**Write the blog version too.** Same result, accessible framing, more readers than the paper will get. You've had thirty weeks of practice.

**Open-source the code.** A clean repository with a README that reproduces your main figure is a significant part of what makes work credible — and it's what people actually use.

---

## 15.11 Reviews and rejection

**Expect rejection.** The median submitted paper is rejected, at every venue, including papers by established researchers. It is a statement about fit and polish far more often than about quality.

### How to read reviews

**Separate signal from noise.** Some reviews are careless. Some are excellent. The test:

> **If two reviewers misunderstood the same thing, that's your writing, not their reading.**

That's almost always true and almost always worth acting on.

**Address the strongest objection first** in any rebuttal. Concede what's fair — conceding a real limitation gains you more credibility than defending it does.

**Resubmit.** A rejected paper, revised with reviewer feedback, is a better paper. Most published work was rejected somewhere first.

---

## 15.12 After

- **Ship the code**, cleaned, with a working reproduction script.
- **Publish the blog version.**
- **Present it** — a local meetup, a reading group, a Discord talk. Explaining it out loud reveals what you don't understand.
- **Answer questions** from people who read it. That's how the next collaboration starts.
- **Note the next question.** Every result opens more than it closes. Your research log already has candidates in it.

---

## 15.13 The longer arc

Return to §0 of *The Year One Plan*, where I said one year gets you to the starting line rather than to "elite."

Look at what you now have:

- An autograd engine you wrote
- A transformer you built and trained
- Loss functions you derived from probability
- Optimizers you implemented from their motivating problems
- Three reproductions with published reports
- A specialization, a map of it, and a community in it
- One original result, written up and submitted
- Fifty-odd public write-ups and a year of commits

**That is a researcher's starting position.** Not a promise of anything, but a real one — people have been hired on less, and admitted to graduate programs on less.

What comes after, honestly:

**Years 2–3** are about volume and depth. More papers, better questions, first collaborations. Your second paper will be much better than your first, and your fifth better again. This is when your taste develops — the sense of which questions are worth asking, which is what actually distinguishes researchers and which cannot be taught directly.

**The paths from here** are a research role at a lab, a PhD, an engineering role with research components, or independent research funded by grants or work. All four are real. None requires the others.

**What "elite" actually takes** is what I said at the start: a decade of scar tissue and a body of work the field builds on. Nobody skips it. But the person who can start that decade is not the person who started this book, and the distance you've closed in one year is the hardest part of the whole path — going from outside the field to inside it.

**The habits are what matter now.** Build before you import. Verify instead of assuming. State what would prove you wrong. Write things down. Publish, including the failures. Those transfer to every problem you'll ever work on, and they'll still be working for you in year ten.

---

## 15.14 Exercises

**1.** Write the full proposal for your top question, using §15.2's template. Include the falsification criteria.

**2.** Send it to three people. Record the feedback.

**3.** Do the literature search properly, twice, a week apart. Document what you searched and what you found.

**4.** Design the decisive experiment. Write down the expected result *before* running anything.

**5.** **Build negative controls.** Plant a known effect of similar magnitude and confirm your setup detects it. Do this before your real experiment.

**6.** Run the two-week pilot. Write kill criteria first.

**7.** Make the go/kill decision explicitly, in writing, against the criteria you set.

**8.** Run the main experiments. Minimum 5 seeds. Log daily.

**9.** Run the ablation that isolates your claimed mechanism.

**10.** Tune your baseline as hard as your method. Document the search for both.

**11.** Make the main figure. Show it to someone and ask what it says. Revise until they get it right.

**12.** Write the paper in §15.9's order. Limitations section at full length.

**13.** Have two people read it. Ask each for the main claim in one sentence.

**14.** Submit to a workshop. Post to arXiv. Write the blog version.

**15.** Open-source the code with a script that reproduces your main figure from scratch.

**16.** **The final project.** Write a retrospective: what you set out to do, what happened, what you'd do differently, and what you learned about doing research that you couldn't have learned from reading about it.

Then look at your progress log from week one — the one §0.7 told you to keep. Read the first month of it.

---

## 15.15 Chapter 15 checkpoint

- [ ] Proposal written, with explicit falsification criteria, before any code.
- [ ] Literature search done twice, documented.
- [ ] Negative controls built and verified.
- [ ] Pilot run; go/kill decision made against pre-set criteria.
- [ ] Main experiments run with ≥5 seeds and variance reported.
- [ ] Baseline tuned as hard as the method, with both searches documented.
- [ ] Ablation isolating the claimed mechanism.
- [ ] Paper written, with a full and honest limitations section.
- [ ] Two independent readers correctly identified the main claim.
- [ ] Submitted to a workshop; preprint posted; code released.
- [ ] Retrospective written.

### Deliverables

```
research/proposal.md          dated, with falsification criteria
research/log.md               daily, the whole project
research/experiments/         configs, results, seeds
research/negative_controls/   the power check
paper/                        LaTeX source and figures
code/                         reproducible, with a one-command main figure
blog/                         the accessible version
retrospective.md              exercise 16
```

```bash
git add .
git commit -m "Chapter 15: original research — proposal, experiments, paper, submission"
git push
```

---

## The end of the main text

You started this book unable to write `matmul`.

You now have an autograd engine, a transformer, a specialization, a reproduction record, and a paper with your name on it.

The remaining appendices are reference material — a debugging playbook, a math card, a resource directory, and a starter Anki bank. Use them as needed.

**One last thing.** The single most valuable habit in this book is the one from §0.2: replace *"do I feel like I understand this?"* with a test that can fail. It's what got you through backprop, it's what makes your reproductions credible, and it's what will keep your own research honest for as long as you do this.

Everything else is details.

Good luck. Go do the work.

---

*Next: Appendix A — Debugging Playbook*
