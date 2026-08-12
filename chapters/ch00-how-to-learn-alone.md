# Chapter 0 — How to Learn This Alone

> Read this chapter completely before starting Chapter 1. Then re-read it at the start of every month. It is the operating system; the rest of the book is applications running on it.

---

## 0.1 The one real problem you now have

You do not have a teacher. That sounds like it means "you have less information," but information is not the scarce resource — every fact in this book is free on the internet already, and always was.

What a teacher actually provides is a **feedback loop**. Four specific things:

1. **Sequencing.** What to learn next, and what to ignore.
2. **Correction.** Telling you your understanding is wrong before you build six months on top of it.
3. **Calibration.** Telling you whether your confusion is normal or a sign you skipped something.
4. **Persistence.** Someone expecting work from you on Thursday.

This book handles #1 permanently — the sequence is fixed and correct, and you never have to wonder what's next.

The other three you have to build yourself. This chapter is how.

**The specific way self-taught learners fail is not laziness.** It's that they cannot tell the difference between *understanding something* and *having watched a good explanation of it*. Those two feel identical from the inside. A teacher would catch the gap. Alone, you can spend four months confidently learning nothing.

So the entire method below reduces to one principle:

> **Replace "do I feel like I understand this?" with a test that can fail.**

---

## 0.2 The falsifiable checkpoint

Feeling is not evidence. Here is what counts as evidence, ranked weakest to strongest:

| Level | Test | Worth |
|---|---|---|
| 0 | You watched the video and it made sense | Nothing |
| 1 | You can explain it out loud | Weak |
| 2 | You can explain it in writing, unprompted, in your own words | Moderate |
| 3 | You can derive it on paper from a blank page | Strong |
| 4 | You can implement it in code from a blank file, no references | Proof |
| 5 | You can implement it, then find and fix a bug you deliberately introduced | Mastery |

**Level 4 is the bar for advancing a chapter.** Not level 0, which is what "I studied for six hours today" usually means.

### The Cold Rebuild

The single most important practice in this book. Here it is:

> Take the central artifact of a chapter. Close every tab, every note, every file. Open a blank editor. Rebuild it from memory. Time yourself.

You will fail the first time you try this and it will feel humiliating. That feeling *is the correction a teacher would have given you*, arriving on schedule. It is not evidence you're not smart enough. It's evidence the method is working — it caught the gap.

Rules:
- Do a cold rebuild at the end of every chapter.
- If you can't do it, you don't advance. Re-work the chapter for 2–3 days and retry.
- Repeat the cold rebuild of *older* chapters once a month. Skills rot silently.

### Falsifiable, not vague

Bad checkpoint: *"Understand backpropagation."*
Good checkpoint: *"From a blank file, write a scalar autograd engine supporting `+`, `*`, and `tanh`, whose gradients match numerical finite differences to within 1e-6, in under 45 minutes."*

The second one can fail. That's what makes it useful. Every chapter in this book ends with one of these.

---

## 0.3 The Unstuck Protocol

This is the section you'll use most, so it's the one to internalize hardest. With no mentor, being stuck is the failure mode that ends runs. Being stuck for six hours feels like proof you're not cut out for this. It is not — it's proof you don't have a procedure.

Here is the procedure. **Follow it in order. Do not skip steps. Set a timer at each step.**

### Step 1 — Name it precisely (5 min)

Write, in a file, one sentence: *what exactly do I not understand?*

Not "I don't get backprop." That's not a question. Push until it's specific:

> "I don't understand why, when a node's output feeds into two different places, we *add* the two incoming gradients instead of taking one of them."

About 30% of the time, writing the precise sentence solves it outright. This is not a trick — vagueness was the blocker.

### Step 2 — Check the prerequisite (10 min)

Confusion at step *n* is very often an unfixed gap at step *n−3*. Ask: what does this build on? Go back one chapter and do that chapter's checkpoint cold.

If you can't pass the earlier checkpoint, **that's your real problem.** Go fix it. Everything downstream will suddenly become easy. This is the single most common cause of stuckness and almost nobody checks it.

### Step 3 — Shrink it (20 min)

Make the smallest possible version of the thing.

- Confused about a matrix operation? Do it with 2×2 matrices, by hand, on paper.
- Confused about a training loop? Train on 4 data points with 1 parameter and print every value each step.
- Confused about attention? Do it with a sequence of length 2 and 1 head.

Almost every confusion in machine learning survives only at scale. At size 2, with every intermediate value printed, it usually has nowhere to hide.

### Step 4 — Find a second explanation (30 min)

Different author, different medium. If you read it, watch it. If you watched it, read the math. Specifically useful when explanations conflict — the conflict shows you where the real subtlety is.

Search patterns that work:
- `"<concept>" intuition explained`
- `"<concept>" site:reddit.com/r/MachineLearning`
- `"<concept>" step by step derivation`
- `"<error message>"` — paste it verbatim, in quotes

### Step 5 — Print everything (30 min)

If it's code: print the shape and value of every intermediate. Not some — every one. Nine out of ten deep learning bugs are shape bugs, and they announce themselves the instant you print shapes.

```python
print(f"{x.shape=}  {W.shape=}  {out.shape=}")
```

### Step 6 — Sleep on it

Genuinely. Stop. Do something physical. A meaningful fraction of hard problems dissolve overnight and this is not mysticism — consolidation during sleep is well documented. Six more hours tonight is worth less than twenty minutes tomorrow morning.

### Step 7 — Park it, with a note (next day)

If it survives the night: write it in a file called `PARKED.md` with the precise sentence from step 1, and **move on to the next topic.**

This feels wrong. It isn't. Some concepts only become learnable after you've seen what they're *for*, which happens two chapters later. Review `PARKED.md` every Sunday. Items will resolve themselves without you touching them, and you'll be able to see it happening.

### Step 8 — Ask a human or a model

Save your rate-limited AI access for the things that survived steps 1–7. Those are the questions worth spending it on, and by then you'll have a precise question, a minimal reproduction, and printed intermediates — which is also the format that gets good answers from humans on forums.

**Time budget for the whole protocol: about two focused hours, spread over two days.** Not six hours in one night. Six-hour grinds are how runs end.

---

## 0.4 The build-first method

The rule from your syllabus, restated with the reason:

> **Never use a library for something you haven't implemented yourself at least once.**

Why this matters more than it sounds: the field's actual researchers are people who understand what happens *inside* the abstraction. `loss.backward()` is one line. Someone who has written that line ten thousand times and someone who has implemented reverse-mode autodiff are not the same person, and the difference shows up the moment something breaks in a novel way — which is the entire job of research.

The order for every new concept:

```
1. Understand the math on paper       (derive it by hand)
2. Implement it slowly and stupidly   (loops, no optimization, tiny data)
3. Verify it against a known-good     (library output, or numerical check)
4. Implement it fast and properly     (vectorized)
5. Now, and only now, use the library
```

Step 2 is the one people skip. Step 2 is the one that works.

### Verification without a teacher

You have no one to grade your code. So grade it mechanically:

- **Numerical gradient checking.** You'll build this in Chapter 3. Any derivative you implement can be checked against finite differences. This is a *complete* correctness test for the hardest part of the field.
- **Compare against a reference implementation.** Your NumPy version vs. `np.linalg`. Your attention vs. PyTorch's. Assert they match to `1e-6`.
- **Overfit a tiny dataset.** Any correctly implemented network can drive the loss to ~0 on 10 examples. If it can't, your implementation is broken — not your hyperparameters. This one test will save you weeks over the year.
- **Write tests.** Even three assertions per module. Future-you needs to know the old code still works.

---

## 0.5 Memory: what to keep in your head

You cannot look everything up. Certain things must be instant or you can't think fluidly — the same way you can't read if you're sounding out letters.

**Use spaced repetition (Anki), 15 minutes daily, every day, from day one.**

### What to make cards for

✅ Definitions (what is a Jacobian?)
✅ Derivative rules
✅ Distribution properties (variance of a Bernoulli?)
✅ Shapes (what shape does a multi-head attention Q projection produce?)
✅ Default hyperparameters (what's a typical Adam β₂?)
✅ "Why" questions (why does attention divide by √d?)

❌ Anything you could derive in 10 seconds
❌ Long code blocks — you learn code by writing it, not recalling it
❌ Things you don't already understand. **Never card something you haven't understood.** Cards are for retaining understanding, not manufacturing it.

### Card format that works

One fact per card. Question on front, minimal answer on back.

```
Front: Why does scaled dot-product attention divide by √d_k?
Back:  Dot products of d-dimensional vectors have variance ∝ d.
       Without scaling, large d pushes softmax into saturated
       regions → vanishing gradients.
```

Make cards *as you learn*, not in a batch later. Ten cards a day is plenty. Do the reviews every day including bad days — the whole value is in the schedule.

---

## 0.6 How to read technical material

You will read textbook chapters and, later, papers. Reading them like a novel does not work.

**Three passes:**

- **Pass 1 (5–10 min).** Title, abstract, section headers, figures, conclusion. Goal: what is this about, and do I need it? Nothing more.
- **Pass 2 (45–60 min).** Read properly, skipping proofs. Mark every equation you don't follow with a `?`. Do not stop to resolve them. Goal: understand the *structure* of the argument.
- **Pass 3 (2–4 hrs, only for important material).** Return to each `?`. Re-derive equations yourself. Implement the core idea in code, however small.

Most material only deserves pass 1. Some deserves pass 2. Very little deserves pass 3 — but the things that do are where all your growth comes from.

### Reading equations

When an equation is opaque, run this checklist:

1. **What are the shapes?** Write the dimension of every symbol in the margin. Half of all confusion dies here.
2. **What is learned vs. fixed?** Which symbols are parameters that change during training?
3. **What happens at the extremes?** Set a term to 0. Set it to infinity. What does the equation do?
4. **What's the smallest case?** Set d=1, n=2. Can you compute it by hand now?
5. **Why this and not the obvious alternative?** Why sum and not average? Why squared and not absolute? The answer to "why not the simpler thing" is usually the actual insight.

---

## 0.7 The operating system: your day and your week

### Daily template (5–8 hours)

| Block | Duration | Content | Skippable? |
|---|---|---|---|
| **A — Math** | 90 min | New concept + hand-worked problems. Paper, not screen. | No |
| **B — Build** | 120 min | Implement. The core block. | **Never** |
| **C — Input** | 90 min | Lecture / textbook / paper. Active notes only. | Yes |
| **D — Consolidate** | 45 min | Anki, write up the day, commit, review yesterday's code. | No |
| **E — Stretch** | 60–90 min | Only on good days. | Yes |

On a bad day, do **B and D only**. Ninety minutes of building plus fifteen minutes of Anki is a real day. Protecting this minimum is how you survive month four.

### Weekly

- **Mon–Fri:** the template.
- **Saturday:** Project Day. No new input at all. One long build session.
- **Sunday AM:** review the week, review `PARKED.md`, write the weekly post, plan the next week.
- **Sunday PM:** off. Fully off.

### Monthly

- One full day off.
- Cold rebuild of a chapter from two months ago.
- Reread this chapter.
- Update the progress log in the README and actually read the last month of it.

---

## 0.8 The crises, and what to do

These are predictable. They happen to everyone on this path. Knowing the schedule in advance removes most of their power.

**Week 3 — "I'm too stupid for math."**
You aren't. You're encountering the normal experience of learning math, which is 80% confusion. School math was easy because it was pre-digested. This isn't. *Action:* keep going, slow down 20%, use the Unstuck Protocol.

**Week 8 — "I'm just copying tutorials."**
Fair, and the fix is mechanical: raise your build-to-input ratio. If you're watching more than you're typing, invert it today.

**Month 4 — The Wall.** The novelty is gone, the finish line isn't visible, and it now feels like a job you don't get paid for. This is where most attempts die.
*Action, decided in advance:* cut to 4 hours/day for two weeks. Do not stop entirely. Reread your progress log from month 1 and note how much you couldn't do then. Then continue.

**Month 6 — "Someone already did this better."**
They did. That's true for every researcher alive, permanently. The work isn't to be first; it's to be able to do the work.

**Month 9 — "My research idea is stupid."**
Probably, and that's normal — most ideas are bad and the only way to find good ones is to generate many bad ones and test them fast. *Action:* run the experiment anyway. A cleanly-executed negative result is real research and is worth more than an unexecuted brilliant idea.

**Any time — comparison to people 5 years ahead.**
The only valid comparison is to yourself 30 days ago. This is exactly why the progress log exists. Use it.

---

## 0.9 Your replacement mentors

You are not actually alone. Cultivate these deliberately, starting week 1 — not month 6 when you're desperate.

**For correctness:** numerical gradient checking, reference implementations, the overfit-tiny-data test (§0.4). Mechanical, always available, never tired.

**For "why doesn't my code work":** Stack Overflow, the PyTorch forums, and GitHub issues on the library in question. Search the exact error string in quotes first — someone has hit it.

**For "I don't understand the concept":** r/MachineLearning and r/learnmachinelearning; the Fast.ai forums; Cross Validated (stats.stackexchange) for the math.

**For community:** EleutherAI's Discord and other open ML Discords are genuinely welcoming to beginners who ask specific, well-formed questions. Bengaluru has an active in-person ML meetup scene — go. Two people at your level and one person ahead of you changes the odds of finishing more than any resource in this book.

**For AI help:** free tiers exist across the major assistants and are rate-limited rather than absent. Because access is scarce, make each question count: give the precise sentence from Unstuck step 1, the minimal reproduction from step 3, the printed shapes from step 5, and what you already tried. That format gets a useful answer in one shot instead of five.

**For accountability:** post daily progress publicly — a build-log thread, a repo with visible commits, anything with witnesses. A public streak is a surprisingly strong forcing function.

---

## 0.10 Self-assessment: are you actually on track?

Run this at the end of every month. Score honestly — nobody sees it.

| Question | Yes | No |
|---|---|---|
| Did I pass every chapter checkpoint cold, without notes? | | |
| Is my build time ≥ 2× my watching time? | | |
| Did I do Anki at least 25 of the last 30 days? | | |
| Can I still pass last month's checkpoint? | | |
| Did I write something public every week? | | |
| Did I implement each thing before using the library version? | | |
| Did I take my rest days? | | |
| Do I have a `PARKED.md` with items that later resolved? | | |

**7–8 yes:** on track. Continue.
**5–6:** one specific thing is broken. Find which "no" it is and fix only that.
**≤4:** stop advancing. Spend a week repairing before adding anything new. Almost always the culprit is watching instead of building.

---

## 0.11 The mindset, stated once

Three things, then this chapter is done.

**Confusion is the work, not an obstacle to it.** The feeling of not understanding is what learning feels like from the inside. If you're comfortable, you've stopped growing. You are training tolerance for confusion as much as you're training math.

**Your rate of progress is not fixed.** "Bad at math" describes a history, not a capacity. Mathematical ability is overwhelmingly a function of hours spent struggling with problems, and you are about to spend a great many. The people who look naturally gifted mostly started earlier.

**Consistency beats intensity, and it isn't close.** Four hours a day for 365 days is 1,460 hours. Twelve hours a day for six weeks is 500 hours and then nothing, because you'll have stopped. The plan is built for the first pattern. Protect it.

---

## Checkpoint — Chapter 0

Before starting Chapter 1, you must have:

- [ ] Written the Unstuck Protocol's eight steps on an index card, by hand, and put it where you work
- [ ] Created `PARKED.md` in your repo, empty
- [ ] Installed Anki and made your first three cards from this chapter
- [ ] Started the progress log in `README.md` with today's date
- [ ] Written, in your own words, in a file, what a "falsifiable checkpoint" is and why feeling like you understand something isn't one

That last one is your first Level-2 test. Do it now, before Chapter 1.

---

*Next: Chapter 1 — Setup, Python, and NumPy*
