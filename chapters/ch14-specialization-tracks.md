# Chapter 14 — Specialization Tracks

**Time: 8–10 weeks** (Weeks 33–42 of the plan)

**Prerequisite:** Chapter 13 — at least one full reproduction with a written report, and your ranked list of open questions.

**What this chapter is:** not new technical content. It's the decision about where to spend the rest of the year, and then the deep work of actually going deep.

---

## 14.0 Why specialize now

Until now, breadth was correct. You needed the shared foundation, and every chapter was load-bearing for the next.

That stops being true here. The field is far too large for anyone to be current across all of it. Every researcher you admire is deep in one area and literate in the rest — the **T-shape**. You've spent thirty weeks building the horizontal bar. Now you build the vertical.

**Why depth beats breadth from here:**

- Original contributions require knowing a subfield well enough to see what's missing. That takes months in one place, not weeks in six.
- Reputation is built in a community, and communities are organized by subfield.
- Depth compounds. The fiftieth paper you read in an area teaches you more than the fifth, because you can finally see what's *not* being said.

**You are not choosing forever.** People switch subfields regularly, and the foundation transfers almost entirely. You're choosing where to spend the next six months.

---

## 14.1 How to choose

Four criteria, in order of how binding they are.

### 1. Compute you can actually access

The hard constraint, and the one people ignore until it stops them.

Some areas genuinely require GPUs you don't have. Others don't. **Choosing a track you can't afford to work in is the most common way a self-directed research plan fails** — not from lack of ability, but from being unable to run the experiment.

Each track below states its real compute floor honestly.

### 2. What you're actually curious about

The honest test isn't "which sounds most impressive." It's: **after reading a paper in this area, do you find yourself thinking about it the next day?**

Six months is long enough that interest is not a luxury. You will hit weeks where nothing works, and curiosity is the only thing that carries you through them.

### 3. Whether a solo person can produce a result there

Some areas are structurally dominated by large labs — training frontier models, for instance. A solo researcher with a free GPU cannot contribute there and shouldn't try.

Other areas are wide open to careful individual work. **Prefer those.** Not as a consolation — some of the most cited work of recent years came from small teams asking sharp questions about existing models.

### 4. Feedback loop length

Short loops teach faster. An area where an experiment takes an hour will make you competent faster than one where it takes a week, even if the second is more interesting.

Early on, optimize for loop length. You can move to slower problems once you know what you're doing.

### The two-week trial

**Don't choose by reading about the tracks. Choose by trying them.**

Pick your top two. Spend two weeks in each: read five papers, do one small reproduction, attempt one tiny original experiment. Then decide.

Four weeks spent choosing well beats six months spent in the wrong place, and the trial itself teaches you a great deal regardless of the outcome.

---

## 14.2 Track 1 — Mechanistic Interpretability

**What it is:** reverse-engineering the algorithms inside trained models. Not "which input features matter" but "what computation is this specific set of weights performing."

**Why it matters:** we build systems we don't understand. Interpretability is the attempt to fix that, and it's central to AI safety work.

| | |
|---|---|
| **Compute floor** | **Lowest of any track.** Real work on GPT-2 small runs on a laptop or free Colab. |
| **Feedback loop** | Hours |
| **Solo-friendly?** | Very. Individual researchers publish here regularly. |
| **Job market** | Growing fast, especially at safety-focused labs |

**Core papers:** *A Mathematical Framework for Transformer Circuits*; *In-context Learning and Induction Heads*; *Toy Models of Superposition*; *Towards Monosemanticity* (sparse autoencoders); activation patching and causal tracing work.

**Key skills:** TransformerLens; careful causal experimental design; visualization; the linear algebra from Chapter 2 used seriously.

**Starting projects:**
- Find and characterize induction heads in GPT-2 small (§12.13 hinted at these)
- Train a sparse autoencoder on one layer's activations and examine the features
- Use activation patching to localize where a specific behaviour lives
- Reproduce a toy-model superposition result

**Free resource:** the **ARENA curriculum** is the best structured material in any track — free, thorough, and built around exactly this kind of hands-on work. Neel Nanda's tutorials and reading lists are the other main entry point.

**Honest risks:** young field, evolving standards, and results can be hard to validate — it's easy to find a pattern that looks like a circuit and isn't. Rigor about causal claims matters more here than almost anywhere.

**Given a low compute budget, this is the track where the barrier is lowest and the ceiling is genuinely high.**

---

## 14.3 Track 2 — LLMs and Post-training

**What it is:** turning base models into useful ones. Instruction tuning, preference optimization, data curation, evaluation design.

| | |
|---|---|
| **Compute floor** | Moderate. LoRA on a 7B model fits on one 24GB GPU; smaller models work on free tiers. |
| **Feedback loop** | Hours to days |
| **Solo-friendly?** | Yes, at small scale |
| **Job market** | **Largest of any track by a wide margin** |

**Core papers:** *InstructGPT*; *LoRA*; *QLoRA*; *Direct Preference Optimization*; *Constitutional AI*; *Self-Instruct*; *LIMA*.

**Key skills:** the Hugging Face stack (`transformers`, `peft`, `trl`, `datasets`); data curation, which matters more than method choice more often than people expect; evaluation design.

**Starting projects:**
- Implement DPO from scratch and verify it against `trl`
- Build LoRA yourself from §2.11's rank argument, then compare to `peft`
- Fine-tune a small model on a specific domain and build an honest eval for it
- Study how much data instruction tuning actually needs — the LIMA question, at small scale

**Honest risks:** crowded, and moving fast enough that results date quickly. Differentiating yourself takes a sharper question than "I fine-tuned a model." The best solo work here tends to be about *data* and *evaluation* rather than method.

---

## 14.4 Track 3 — Efficiency and Systems

**What it is:** making models cheaper — quantization, distillation, pruning, custom kernels, serving throughput, speculative decoding.

| | |
|---|---|
| **Compute floor** | Moderate, and the field is about doing more with less — which suits constrained hardware |
| **Feedback loop** | Hours |
| **Solo-friendly?** | Yes, especially benchmarking and quantization work |
| **Job market** | Strong and less crowded than the LLM track |

**Core papers:** *FlashAttention*; *LLM.int8()*; *GPTQ*; *AWQ*; *Efficient Memory Management for LLM Serving* (PagedAttention/vLLM); speculative decoding; classic distillation.

**Key skills:** profiling, memory analysis, eventually Triton or CUDA; `vLLM` and `llama.cpp`; a real understanding of the memory hierarchy — §11.11's lesson that bandwidth, not FLOPs, is usually the bottleneck.

**Starting projects:**
- Implement int8 and int4 quantization from scratch; measure the quality/size trade-off honestly
- Write a Triton kernel for a fused operation and benchmark against PyTorch
- Implement speculative decoding and measure the actual speedup across acceptance rates
- Profile a serving stack and find where the time goes

**Honest risks:** more engineering than science, which suits some people and not others. The deepest work needs specific hardware.

---

## 14.5 Track 4 — Reinforcement Learning

**What it is:** learning from interaction rather than labels. Agents, control, game playing — and the foundations underneath RLHF.

| | |
|---|---|
| **Compute floor** | Low for classic control; moderate for Atari; high for anything modern at scale |
| **Feedback loop** | Days — RL runs are long and noisy |
| **Solo-friendly?** | At small scale, yes |
| **Job market** | Robotics, agents, and post-training |

**Core papers:** DQN; PPO; SAC; AlphaZero; Dreamer; Decision Transformer.

**Key skills:** Gymnasium; **debugging discipline above all** — RL is notoriously hard to debug, since a broken implementation and a hard problem look identical; variance reduction; patience with noisy results.

**Starting projects:**
- Implement PPO from scratch and solve CartPole, then LunarLander
- Reproduce a DQN Atari result at reduced scale
- Implement DPO and compare it directly against a PPO-based RLHF pipeline — this connects RL to the LLM track

**Free resource:** OpenAI's *Spinning Up in Deep RL* is excellent and free.

**Honest risks:** the highest-variance track. Reproduction is genuinely hard, results are seed-sensitive, and feedback loops are long. Rewarding if you like it; punishing if you're doing it for the job market.

---

## 14.6 Track 5 — Multimodal and Generative Models

**What it is:** images, video, audio. Diffusion models, vision transformers, vision-language models.

| | |
|---|---|
| **Compute floor** | **High** for training from scratch; moderate for fine-tuning small models |
| **Feedback loop** | Hours to days |
| **Solo-friendly?** | At small scale and for fine-tuning |
| **Job market** | Strong in creative tools and media |

**Core papers:** CLIP; ViT; DDPM; DDIM; Latent Diffusion; flow matching; LLaVA; Flamingo.

**Key skills:** the diffusion math (SDEs and ODEs — a real extension of your calculus); evaluating generative quality, which is genuinely hard; vision preprocessing.

**Starting projects:**
- Implement DDPM from scratch on MNIST, then CIFAR — very doable on free compute
- Derive and implement the DDIM sampler; compare sample quality against step count
- Build a small vision-language model by connecting a frozen vision encoder to a small LM
- Study classifier-free guidance: what does the guidance scale actually trade off?

**Honest risks:** compute-hungry, and evaluation of generative quality is genuinely unsolved — it's easy to produce results you can't defend.

---

## 14.7 Track 6 — Evaluation and the Science of Deep Learning

**What it is:** understanding what models actually do and can do. Benchmark design, emergent abilities, grokking, scaling behaviour, the empirical science of training.

| | |
|---|---|
| **Compute floor** | **Low to moderate** — much of it uses existing models |
| **Feedback loop** | Hours |
| **Solo-friendly?** | Very |
| **Job market** | Growing — every lab needs this and few people do it well |

**Core papers:** *Emergent Abilities of Large Language Models* **and** the *Mirage* critique of it — read both, in that order; *Grokking*; HELM; BIG-bench; Chinchilla (again, differently).

**Key skills:** statistics done properly — variance, significance, multiple comparisons; benchmark construction; a skeptical instinct about measurement.

**Starting projects:**
- Reproduce grokking on modular arithmetic; map when it happens and when it doesn't
- Take a published benchmark and audit it for contamination or ambiguity
- Test whether a reported "emergent" capability is emergent or an artifact of a discontinuous metric
- Measure seed variance across a class of published results and ask how many survive it

**Honest risks:** less glamorous, and negative results are harder to publish — though genuinely valued by people who know the field.

**This track is undersupplied relative to its importance**, and it needs very little compute. If §13.4's critical questions felt natural to you, this is worth a hard look.

---

## 14.8 A comparison, for the compute-constrained

If your hardware budget is a free Kaggle GPU and a laptop, the tracks are not equal:

| Track | Viable on free compute? |
|---|---|
| Mechanistic interpretability | **Fully.** Real, publishable work. |
| Evaluation / science of DL | **Fully.** Uses existing models. |
| Efficiency and systems | **Mostly.** Quantization and benchmarking, yes; custom kernels need specific hardware. |
| LLMs and post-training | **Partly.** Small models and LoRA, yes; anything at scale, no. |
| RL | **Partly.** Classic control, yes; modern scale, no. |
| Multimodal | **Least.** Small diffusion works; most else doesn't. |

**This is not a ranking by importance.** It's a ranking by whether you can do the experiment. The first two are where a determined individual with limited hardware can produce work that stands on its own.

---

## 14.9 What you build regardless of track

These matter more than track choice, and they're the reason your Chapter 15 work will be credible:

**Experimental design.** Isolate one variable. Fix seeds. Match compute. Predict the outcome before running. Most bad research is bad design, not bad code.

**Statistics.** Enough to know when a difference is real: variance across seeds, why 3 runs is a minimum, what multiple comparisons do to your p-values, why "we picked the best of 20 configurations" is a result about the search, not the method.

**Writing.** The bottleneck on most researchers' impact. A weekly post is the practice, and you've been doing it for thirty weeks already.

**Engineering hygiene.** Configs in files, seeds set, runs logged automatically, results reproducible from a commit hash. §8.11's habit, now non-negotiable.

**Visualization.** A plot that makes the result obvious is worth pages of prose. Learn to make good ones.

---

## 14.10 Going deep — the ten weeks

Here's the shape of the specialization period.

**Weeks 1–2: the trial.** Two candidate tracks, five papers and one small reproduction in each. Then commit.

**Weeks 3–4: read the canon.** Twenty to thirty papers in your track, pass-2 minimum. Build the citation map: who cites whom, which results everyone builds on, where the disagreements are.

**Weeks 5–8: three reproductions.** Increasing difficulty. For each: reproduce the baseline first (§13.6), reproduce the result, run at least one ablation the authors didn't, write a report.

**Weeks 9–10: converge on a question.** From your reproductions and reading, generate 10–15 candidate questions using the §13.9 patterns. Then filter hard:

- Can I run the decisive experiment on hardware I have?
- Would the answer be interesting *either way*? (If only one outcome is publishable, it's a weak question.)
- Has someone already done it? (Search properly. Twice.)
- Can I get a preliminary signal in under two weeks?

**Rank them and take the top three into Chapter 15.**

---

## 14.11 Community and visibility

Depth in isolation is much slower than depth in a community.

**Find where your track's people are.** Interpretability: the Alignment Forum, LessWrong, and Neel Nanda's community. LLMs and efficiency: EleutherAI's Discord, various open-source model communities. RL: research Discords and the Spinning Up community. Systems: the vLLM and Triton project channels.

**Contribute before you ask.** Answer a beginner's question. Fix a documentation bug. Reproduce someone's result and post the notes. Being known as helpful is the cheapest reputation there is, and it's genuine.

**Publish everything.** Reproduction reports, negative results, tutorials, code. You already have thirty weeks of write-ups. Keep going.

**Open source.** One merged PR to a serious repository (PyTorch, Hugging Face, vLLM, TransformerLens) is a stronger credential than most certifications, and maintainers are usually welcoming to a well-scoped first contribution.

**Local, too.** Bengaluru has an active ML meetup scene. In-person contact does something that Discord doesn't — go at least monthly.

---

## 14.12 Exercises

**1.** Write one page on each of the six tracks: what it is, why someone would care, and the single most interesting open question you can identify in it. Force yourself to find a real question in each, including the ones you don't like.

**2.** Score each track honestly on the four criteria in §14.1 — compute access, genuine curiosity, solo viability, loop length. Make a table. Note where curiosity and job market disagree.

**3.** **The trial.** Pick your top two. Two weeks in each: five papers, one small reproduction, one tiny original experiment. Write up both.

**4.** Commit. Write one page explaining the choice, including what you're giving up. Date it. You'll want this in three months.

**5.** Build the citation map for your track: 20–30 papers, who cites whom, the three or four results everyone builds on, and the live disagreements.

**6.** Identify the five researchers whose work in your track you most respect. Read their last three papers each. Notice what they have in common — that's the taste of the subfield.

**7.** Reproduction 1 — an easy, well-documented result with public code.

**8.** Reproduction 2 — harder, less documented. Log every guess you had to make.

**9.** Reproduction 3 — one where you extend it: at least two ablations the authors didn't run.

**10.** For each reproduction, publish a report.

**11.** Make one open-source contribution in your track. Start with documentation or a small bug if nothing else is obvious.

**12.** Join your track's main community. Answer three questions from people newer than you.

**13.** Generate 15 candidate research questions using the §13.9 patterns. For each, note the decisive experiment and whether you can run it.

**14.** Filter to three using §14.10's criteria. Search properly to confirm none has been done.

**15.** **Chapter project.** A written "state of my subfield" document: what the central questions are, what's been settled, what's contested, who's working on what, what's blocked on compute versus blocked on ideas, and where you think the open space is.

Writing this forces you to discover exactly how much of the map you still can't draw. That discovery is the point — and the document itself is a genuinely useful public artifact.

---

## 14.13 Chapter 14 checkpoint

- [ ] You have completed the two-week trial in two tracks and written both up.
- [ ] You have committed to one track, in writing, with the trade-offs named.
- [ ] You have read 25+ papers in that track and can draw its citation map from memory.
- [ ] You have completed three reproductions, each with a published report.
- [ ] At least one reproduction includes ablations the original authors did not run.
- [ ] You have made one open-source contribution.
- [ ] You are a participating member of at least one community in your track.
- [ ] You have three ranked, filtered candidate research questions, each with a decisive experiment you can actually run.
- [ ] You have written the "state of my subfield" document.

The last item is the real test. If you can write it well, you know the field. If you can't, you know exactly where the gaps are — which is nearly as useful.

### Deliverables

```
tracks/trial_<a>.md, tracks/trial_<b>.md   the two-week trials
tracks/decision.md                          the commitment, dated
reading/citation_map.md                     your subfield map
reproductions/1/, 2/, 3/                    code + reports
questions_shortlist.md                      the ranked three
state_of_field.md                           exercise 15
research_log.md                             still daily
```

```bash
git add .
git commit -m "Chapter 14: track selection, three reproductions, research questions"
git push
```

### Write-up

Publish the "state of my subfield" document. Written honestly — including what you don't yet understand — it's one of the most useful things a newcomer can produce, because the people who *could* write it usually don't bother, and the people who need it are all at your stage.

**You now have a specialization, a validated body of reproduced work, and three questions worth answering.** Chapter 15 is answering one of them.

---

*Next: Chapter 15 — Doing Original Research*
