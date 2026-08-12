# Chapter 13 — Reading and Reproducing Papers

**Time: 12–14 days** (Weeks 31–32 of the plan)

**Prerequisite:** Chapter 12's GPT trained and written up.

**What changes here:** for twelve chapters you've been a student following a curriculum someone else designed. From now on nobody tells you what's next. The skill you need is reading the literature well enough to find your own problems — and that skill is specific, learnable, and almost never taught.

---

## 13.0 The uncomfortable part

Up to now, every problem you worked on had a known answer. Someone had verified it was solvable, sized it appropriately, and could tell you when you were done.

Research has none of that. Most ideas fail. Most experiments are inconclusive. Nobody can tell you whether you're stuck because the problem is hard or because you're confused.

**The one thing that reliably compounds is your ability to read the literature.** Not passively — critically, and with an eye for what's missing. Papers are where the field stores what it has tried, and reading them well is how you avoid re-deriving 2019.

Two skills, and this chapter is both:

1. **Reading**, so you know what exists and what's actually been shown.
2. **Reproducing**, so your understanding is verified rather than assumed — and because a reproduction is one small step from an original result.

---

## 13.1 The three-pass method

You cannot read every paper carefully. You shouldn't try. Triage instead.

### Pass 1 — 5 to 10 minutes

Title, abstract, section headers, all figures and their captions, conclusion.

**One question: what is the claim, and do I care?**

Most papers stop here, and that's correct. You're building a map of the field, not memorizing it.

### Pass 2 — 45 to 60 minutes

Read properly. **Skip proofs.** Mark every equation you don't follow with a `?` in the margin and keep going — do not stop to resolve them.

**Two questions: what is the method, and does the evidence support the claim?**

That second question is where most of your critical attention should go, and §13.4 is about how to answer it.

### Pass 3 — 2 to 4 hours, sometimes days

Only for papers you'll build on. Return to every `?`. Re-derive the equations yourself. Implement the core idea, however small the version.

**One question: could I have written this paper?**

Very few papers deserve pass 3. The ones that do are where all your growth comes from.

### The reading order inside a paper

Not front to back. This order gets you to the substance fastest:

```
Abstract → Figure 1 → Tables → Conclusion → Method → Experiments → Appendix
```

- **Figure 1** is almost always the method diagram. Often the single fastest route to understanding.
- **Tables** are the actual evidence. The prose describes them favourably; the numbers don't.
- **The appendix** is where the real details live — hyperparameters, the ablations that didn't fit, sometimes the honest failure notes. **Read appendices.** Most people don't, which is exactly why they're worth reading.
- **Related work** is your map of the subfield. Underrated when you're new: it tells you what to read next and who the players are.

---

## 13.2 Reading an equation you don't understand

Extending §0.6's checklist with what experience adds:

1. **Write the shape of every symbol in the margin.** Half of all confusion dies here.
2. **Which symbols are learned and which are fixed?**
3. **What happens at the extremes?** Set a term to 0, then to ∞.
4. **What's the smallest case?** `d=1`, `n=2`, one head, sequence length 2. Compute it by hand.
5. **What does it reduce to if you delete a term?** Often you'll find the paper's contribution is exactly one term, and everything else is a standard method.
6. **Why this and not the obvious alternative?** Why sum and not mean? Why squared and not absolute? The answer to "why not the simpler thing" is usually the actual insight.
7. **Is this something I already know, in disguise?** A surprising fraction of "new" formulations are a known method with different notation. You've already seen this once: LSTM cell state and residual connections are the same idea (§11.3).

**Implement the smallest version.** An equation you've run on `d=2, n=3` and printed intermediates for is understood. One you've only read is not.

---

## 13.3 What papers systematically leave out

Reproduction is hard for reasons that aren't anyone's fault — page limits are real. But knowing the standard omissions tells you where to look when your reproduction fails:

- **Learning rate schedule details** — warmup length, decay shape, final LR
- **Initialization specifics** — often decisive, rarely stated
- **Failed variants** — the twenty things that didn't work, which is the most information-dense part of any project and almost never published
- **Number of seeds and the variance** — a 0.3% improvement means nothing without it
- **Actual compute used**, and whether comparisons were compute-matched
- **Data preprocessing** — filtering, deduplication, tokenization details
- **Implementation tricks** discovered during development and folded in silently
- **Negative results from the same project**

**The systematic bias this creates:** published results are the top of a distribution over configurations. Your first reproduction attempt samples from the middle of it. Expect to underperform initially, and don't conclude the paper is wrong until you've checked the omissions above.

---

## 13.4 Evaluating a paper critically

Run these seven questions on every paper you take seriously. They are, in rough order, the most common ways ML results mislead.

**1. Was the baseline tuned as hard as the method?**

This is the single most common flaw in the literature. Authors spend months tuning their method and use default hyperparameters for the baseline. A large fraction of reported improvements evaporate under a fairly tuned baseline.

**Check:** does the paper report a hyperparameter search for the baseline? If not, be skeptical of anything under a few percent.

**2. How many seeds, and what's the variance?**

Deep learning results are noisy. A single run reporting +0.4% is not evidence when run-to-run standard deviation is 0.5%.

**Check:** are error bars reported? Over how many seeds? If neither is stated, assume the improvement is within noise until shown otherwise.

**3. Is the comparison compute-matched?**

A method that "improves results" while using 3× the compute hasn't necessarily improved anything — you could have trained the baseline longer.

**4. Does the ablation isolate the claimed mechanism?**

Showing the whole system works is not the same as showing *the proposed component* is why. A good ablation removes exactly one thing.

**Check:** if the paper claims mechanism X causes improvement Y, is there an experiment where X is removed and everything else held fixed?

**5. Is the evaluation contaminated?**

Benchmark data leaking into training is widespread and often unintentional, especially for models trained on web crawls. Results on contaminated benchmarks are meaningless.

**6. Does the abstract's claim match what was tested?**

Abstracts routinely generalize beyond the experiments. "Improves reasoning" when the tests were three math datasets at one model size is a much narrower result than it sounds.

**7. Would this survive at a different scale?**

Many methods that help at 100M parameters vanish at 10B, and vice versa. Single-scale results are weak evidence about the regime you care about.

**None of this is cynicism.** It's the normal reading posture of a working researcher, and it's also the checklist you'll apply to your *own* work in Chapter 15 — which is where it matters most, because you are the person you're most likely to fool.

---

## 13.5 Building a paper habit

**30 minutes a day, every day.** Skim abstracts. You will not understand most of them at first — read them anyway. You're building pattern recognition for what the field considers interesting, and that develops only through volume.

**Where to look:**

- **arXiv** cs.LG, cs.CL, cs.CV — the daily firehose. Skim titles, open a few.
- **Papers with Code** — links papers to implementations; excellent for reproduction.
- **Semantic Scholar / Connected Papers** — trace citation graphs backwards and forwards from a paper you like. The fastest way to map a subfield.
- **Conference proceedings** — NeurIPS, ICML, ICLR, ACL, EMNLP, CVPR. Skimming an accepted-papers list is a good quarterly habit.
- **A small number of researchers whose taste you trust.** Five to ten. Their reading lists are better filters than any algorithm.

**A caution about social media:** it's genuinely useful for noticing what the field is paying attention to, and genuinely bad at telling you what's important. Attention and importance correlate weakly. Use it as a signal, not a filter.

### The papers to have read regardless of track

These are the shared vocabulary. Work through them over Part IV:

| Area | Papers |
|---|---|
| Architecture | Attention Is All You Need; ResNet; Layer Normalization |
| Language models | GPT-2; GPT-3; BERT |
| Optimization | Adam; Decoupled Weight Decay (AdamW); Batch Normalization |
| Scaling | Kaplan et al. scaling laws; Chinchilla |
| Adaptation | LoRA; InstructGPT; Direct Preference Optimization |
| Regularization | Dropout |
| Interpretability | A Mathematical Framework for Transformer Circuits; In-context Learning and Induction Heads |

You have already implemented most of the ideas in these. **Reading a paper whose method you've built is a completely different experience** — you'll notice the choices they made and the ones they didn't justify.

---

## 13.6 The reproduction process

Reproduction is the bridge from reading to research. Here's the process that works.

### Step 1 — Write down the claim, precisely, before you code

One sentence. Falsifiable. With numbers.

> "Method X improves accuracy on dataset D from 71.2% to 74.8% at model scale S."

Not "Method X is better." If you can't state it precisely, you don't understand the paper well enough to reproduce it.

### Step 2 — Find the smallest version that could show the effect

You will not reproduce a 70B-parameter result. **You don't need to.** Find the smallest scale at which the claimed effect should still appear, and check whether the paper gives evidence it does.

If the effect only exists at scales you can't reach, pick a different paper. There are thousands.

### Step 3 — Reproduce the *baseline* first

**This is the step everyone skips and it is the most important one.**

Implement their baseline. Get it to match their reported baseline number. Only then implement the method.

**Why:** if your baseline doesn't match theirs, nothing downstream means anything. A method that "fails to reproduce" on a baseline that's 4% off is telling you about your data pipeline, not about the method.

### Step 4 — Implement the method

Smallest correct version. No optimizations. Gradient-check anything with a novel derivative (Chapter 3's tool, still earning its keep).

### Step 5 — Compare under matched conditions

Same data, same compute, same number of seeds, same evaluation protocol. Report variance.

### Step 6 — Log every discrepancy

Every place you had to guess a hyperparameter, every detail the paper omitted, every deviation you made. This log is what makes your reproduction *useful to other people* — and it's frequently the most valuable artifact you produce.

---

## 13.7 When reproduction fails

Failure is informative if you localize it. Work down this ladder:

| What you observe | What it means | What to do |
|---|---|---|
| Your baseline ≠ their baseline | Your setup differs | Check data, preprocessing, eval protocol, metric definition |
| Baseline matches; method gives no gain | Bug, missing detail, or a genuinely narrow result | Re-read the appendix; check the omissions in §13.3; email the authors |
| Method helps, but less | Scale or dataset dependence | Report the gap — this is a real finding |
| Method helps *more* than reported | Suspicious | Check for leakage or an unfair baseline in *your* setup |

**A failed reproduction is a result, not a failure.** Documented reproduction attempts — including negative ones — are genuinely valuable to the field and are exactly the kind of thing that gets cited and remembered. The field has a well-known shortage of them.

**Email the authors.** Researchers are generally responsive to a specific, well-formed question from someone who has clearly done the work. "I reproduced your baseline at 71.1% but get 71.4% with your method at this configuration — did you use warmup here?" gets answered. "Your paper doesn't work" does not.

---

## 13.8 The infrastructure

Tools you'll need from here. Learn them as you need them, not in advance.

**The Hugging Face ecosystem:**

- `transformers` — pretrained models and a common architecture interface
- `datasets` — efficient loading and streaming for large corpora
- `tokenizers` — fast implementations of what you built in §12.1
- `accelerate` — device and distributed handling with minimal code change
- `peft` — LoRA and other parameter-efficient fine-tuning (§2.11)

**Evaluation:** `lm-evaluation-harness` is the standard for language model benchmarks. Use a standard harness rather than writing your own eval — most reported-number discrepancies come from subtle eval differences, and using the common tool removes that variable.

**Distributed training** — conceptual understanding is enough for now:

- **DDP (data parallel)** — every GPU holds a full model copy, processes a different batch shard, and gradients are all-reduced. Simple, effective, requires the model to fit on one device.
- **FSDP / ZeRO** — shards parameters, gradients, and optimizer states across devices. Needed when the model doesn't fit. Recall §12.7: optimizer state is 16 bytes per parameter, so it's usually the state, not the model, that forces sharding.

**Compute discipline.** When renting GPUs: always test on the smallest instance first, always checkpoint, always set a spending cap, and never leave an instance running overnight without a reason. Everyone loses money to this once; it needn't be much.

---

## 13.9 From reproduction to contribution

Here's the ladder from "I copied a paper" to "I have an original result." Each rung is a small step, and **rung 1 is already novel.**

| Rung | What you do | Novel? |
|---|---|---|
| 0 | Reproduce the headline result | No, but valuable |
| 1 | Reproduce + run one ablation they didn't | **Yes** |
| 2 | Test at a different scale, dataset, or modality | Yes |
| 3 | Find where the method breaks | Yes — often interesting |
| 4 | Explain *why* it breaks | Strong |
| 5 | Fix it | Paper-worthy |

**Rung 1 is closer than it sounds.** After a reproduction you have working code and a validated baseline. Running one ablation the authors didn't costs a day and produces a result nobody has. That's not a trivial contribution — it's how most incremental research actually gets made.

**Rungs 3 and 4 are where the good workshop papers live.** "Method X, which the authors evaluated only on English, degrades sharply on morphologically rich languages, and here's the mechanism" is a genuine contribution that requires no frontier compute.

### Generating candidate questions

While reading, keep a running list of the following. Each is a research question in embryo:

- **"They only tested on ___."** Every scope limitation is an open question.
- **"They claim mechanism X but only showed outcome Y."** The mechanism is untested.
- **"This should break when ___."** Predict a failure and go check.
- **"These two papers disagree."** Reconciling them is a contribution.
- **"Everyone does ___ but nobody justified it."** Test the folk wisdom. Many defaults have never been ablated.

That last one is underrated. A lot of standard practice is inherited rather than validated.

---

## 13.10 The research log

Start this now, and keep it every day for the rest of the year.

```markdown
## 2026-11-14

**Goal:** check whether the §X gain survives at 10M params.

**Ran:** baseline (seed 0,1,2) and method (seed 0,1,2), 4k steps,
       config in configs/exp_014.yaml

**Result:** baseline 3.41 ± 0.03, method 3.39 ± 0.04. Within noise.

**Think:** either scale-dependent, or my warmup differs — paper says
       "standard warmup" without a number. Emailed authors.

**Next:** sweep warmup ∈ {0, 200, 500, 1000} at seed 0.
```

Five lines. Every day. Why it matters:

- You will otherwise re-run experiments you already ran. Everyone does.
- It's what you write the paper *from*. Reconstructing three months of work from memory is miserable and inaccurate.
- Writing "what I think this means" forces you to have a hypothesis, which is the difference between running experiments and doing research.
- On the days when nothing works, it's the only evidence that you did anything at all — and there will be many such days.

**Pair it with your config discipline from §8.11:** every run writes its config and metrics automatically. The log records what you *thought*; the artifacts record what you *did*.

---

## 13.11 Exercises

**1.** Pass-1 twenty papers from a recent arXiv listing in your area of interest. 10 minutes each, ~3 hours total. For each write two lines: the claim, and whether you'd read further. **Volume is the point.**

**2.** Pass-2 *Attention Is All You Need*. Mark every equation you can't follow. Then pass-3 it: re-derive scaled dot-product attention and multi-head attention from the paper alone, without your Chapter 11 notes. Compare to what you wrote there.

**3.** Pass-3 the AdamW paper. Re-derive the decoupling argument. Compare against §9.6 — did the paper convince you, and does its evidence support the claim?

**4.** Pass-3 the Chinchilla paper. Reproduce their compute-optimal calculation for a budget you could actually afford. Compare to what you did in §12.6.

**5.** Take one paper and apply all seven questions from §13.4 in writing. Be specific: quote the table, name the missing experiment. Then do the same for a paper you *like*.

**6.** Find a paper that reports no error bars and no seed count. Estimate, from your own experiments, whether its headline improvement is within plausible run-to-run variance.

**7.** Pick two papers that disagree about something. Write a page on what would settle it.

**8.** **Baseline reproduction.** Pick a small paper with public code. Reproduce only its *baseline* number. Log every discrepancy and every hyperparameter you had to guess.

**9.** **Full reproduction 1.** Reproduce LoRA on a small model: implement it yourself from §2.11's rank argument, fine-tune on a small task, and compare against full fine-tuning on parameters, memory, time, and final accuracy.

**10.** **Full reproduction 2.** Reproduce a training-recipe claim from any paper — a schedule, an initialization, an activation, a normalization placement. Run it against a properly tuned baseline with at least 3 seeds. Report whether it survives.

**11.** **Rung 1.** For one of your reproductions, run an ablation the authors did not. State beforehand what you expect and why. Report the result whether or not it's interesting.

**12.** **Rung 3.** Take a method you've reproduced and deliberately try to break it. Vary scale, data distribution, sequence length, language, or domain. Find a regime where the reported benefit disappears.

**13.** Set up `lm-evaluation-harness` and evaluate your Chapter 12 model on three standard tasks. Compare to a similarly-sized public model. Explain any gap.

**14.** Write a "reproduction report" for exercise 9 or 10: claim tested, setup, results with variance, discrepancies from the paper, and your conclusion about whether it holds. Publish it.

**15.** **Chapter project.** Choose one paper you'll build on for the rest of Part IV. Requirements: pass-3 it completely; reproduce its central result at a scale you can afford; run at least two ablations the authors didn't; write a full reproduction report; and produce a list of at least five open questions the paper raises, ranked by how tractable they are for you.

**That list is your Chapter 15 shortlist.** Take it seriously — you'll pick your research question from it.

---

## 13.12 Notes on the exercises

<details>
<summary>Guidance — read after attempting</summary>

**On exercise 1.** If twenty papers in three hours feels impossible, you're reading too carefully. Pass 1 is triage, not comprehension. The skill being trained is deciding fast, and you can only develop it at volume. Expect to understand perhaps three of the twenty. That's the correct hit rate.

**On exercise 2.** Most people find they cannot re-derive multi-head attention from the paper alone even after implementing it. The paper is terse and assumes a great deal. This is normal and worth noticing: **papers are not tutorials.** They're written for people who already know the surrounding literature. Reading them fluently is a skill separate from understanding the content.

**On exercise 5.** Applying the critical questions to a paper you *like* is the harder and more valuable half. It's easy to be rigorous about work you're skeptical of. The discipline that matters is being equally rigorous about work you want to be true — which is the exact posture you'll need for your own results in Chapter 15.

**On exercise 8.** Budget more time than you think. Baseline reproduction routinely takes days, and the discrepancies you find are the education. Common culprits, in order: different data preprocessing, different evaluation protocol, different metric definition (macro vs micro averaging catches people constantly), different train/test split, and undocumented warmup.

**On exercise 11.** State your expectation *before* running. Write it down. Then, whatever happens, you learn something: confirmation tells you your model of the method is right; a surprise tells you it isn't, and surprises are where research questions come from.

Resist the urge to only report interesting results. **A boring, correct ablation is a contribution. A cherry-picked interesting one is not.**

**On exercise 12.** Deliberately breaking a method feels adversarial and isn't. Every method has a domain of validity, and mapping it is legitimate, useful work — often more useful than another incremental improvement. Papers rarely state their limits because authors are not incentivized to look for them. That leaves the space wide open for you.

**On the chapter project.** Choosing well matters more than executing well at this stage. Criteria for a good paper to build on:

- Reproducible at a scale you can afford
- Public code exists (so you can check yourself against it)
- Recent enough to be live, old enough to have been examined
- In an area you'd be willing to spend six months on
- Leaves obvious things untested

Avoid: frontier-scale results, papers requiring proprietary data, and anything so new that nobody has tried to reproduce it yet.

</details>

---

## 13.13 Chapter 13 checkpoint

Not a coding checkpoint. A practice checkpoint.

- [ ] You have pass-1'd at least 40 papers and can triage one in under 10 minutes.
- [ ] You can state the three passes and the question each answers.
- [ ] You can recite the seven critical questions of §13.4 and have applied them in writing to at least three papers.
- [ ] You have reproduced at least one paper's baseline to within a stated tolerance.
- [ ] You have completed at least one full reproduction with a written report.
- [ ] You have run at least one ablation the original authors did not.
- [ ] You have a daily research log with at least 20 entries.
- [ ] You have a ranked list of five open questions from your chosen paper.
- [ ] You have a paper-reading habit you've sustained for at least three weeks.

Item 4 is the one that most reliably separates people who can do research from people who have read about it.

### Anki cards

- Three passes — time budget and question for each
- Reading order within a paper
- The seven critical questions
- The five things papers systematically omit
- Why reproduce the baseline first?
- The reproduction failure ladder
- The ablation ladder, rungs 0–5
- Five patterns that generate research questions
- DDP vs FSDP — when each is needed

### Deliverables

```
reading/log.md              one line per paper: claim + verdict
reading/notes/              pass-3 notes for papers you went deep on
reproductions/<paper>/      code, configs, reproduction report
research_log.md             daily, dated, from now until Chapter 15
questions.md                your ranked open-question shortlist
```

```bash
git add .
git commit -m "Chapter 13: reproductions, reading log, open questions"
git push
```

### Write-up

Publish your reproduction report. Not a blog post — a report: claim tested, method, setup, results with variance, discrepancies from the original, and an honest conclusion.

**Reproduction reports are undersupplied and disproportionately appreciated.** The field runs on results almost nobody independently checks. A careful reproduction — positive or negative — is a real contribution, it's citable, and it's the kind of artifact that makes people take a newcomer seriously.

**You have now done the thing that separates a student from a researcher: you took a claim from the literature and independently checked it.** Chapter 14 helps you pick a specialization. Chapter 15 is your own question.

---

*Next: Chapter 14 — Specialization Tracks*
