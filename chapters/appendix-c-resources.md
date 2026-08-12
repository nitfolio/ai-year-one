# Appendix C — Free Resource Directory

Everything referenced in this book, plus what each thing is actually good for. **All of it is free.**

**A note on links:** courses move, sites reorganize, and this was compiled in August 2026. Names are given precisely enough to search for. If something has moved, search the exact title — these are all well-known enough to find.

---

## C.0 How to use this

**Do not work through these in parallel with the book.** The book is the sequence; these are the supplements it points to. Using three courses at once produces the illusion of progress and very little else.

The rule: **one primary resource at a time, plus this book.** Reach for a second only when the first has genuinely failed you on a specific topic (Unstuck Protocol step 4, §0.3).

---

## C.1 The core sequence

If you use nothing else, use these, in this order:

| Chapters | Resource | Why this one |
|---|---|---|
| 2–3 | **3Blue1Brown**, *Essence of Linear Algebra* and *Essence of Calculus* | The best mathematical exposition on the internet. Watch each series twice. |
| 5 | **Karpathy**, *Neural Networks: Zero to Hero*, video 1 (micrograd) | Watch once, then close it and build your own from memory |
| 7–8 | **Dive into Deep Learning** (d2l.ai) | Free, thorough, code alongside every concept |
| 10 | **Stanford CS231n** | Still the best convolution course. Do the assignments. |
| 11 | **The Annotated Transformer** + Karpathy videos 2–6 | Line-by-line implementation |
| 12 | **Stanford CS336**, *Language Modeling from Scratch* | Exactly this chapter's material, taught properly |
| 13–15 | **ARENA curriculum** | The best structured path into hands-on research |

---

## C.2 Video courses

**3Blue1Brown** — *Essence of Linear Algebra*, *Essence of Calculus*, *Neural Networks*. Unmatched for geometric intuition. **Watching is not doing math** — pair with problems.

**Andrej Karpathy, *Neural Networks: Zero to Hero*** — micrograd, makemore (several parts), nanoGPT, and a full GPT-2 reproduction. The single best video series for this book's approach: everything built from scratch. Watch a video, then rebuild it from memory before comparing.

**Stanford CS231n** — *Deep Learning for Computer Vision*. The assignments are the value; they're the from-scratch tradition this book follows.

**Stanford CS224n** — *NLP with Deep Learning*. Good for the RNN→attention transition.

**Stanford CS336** — *Language Modeling from Scratch*. Recent, and the closest thing to a Chapter 12 companion.

**fast.ai, *Practical Deep Learning for Coders*** — top-down: results first, theory later. The opposite of this book's order, which makes it a useful *complement* rather than a substitute. Good if you're demoralized and need a win.

**MIT 18.06 (Gilbert Strang), *Linear Algebra*** — if you want full linear algebra rather than the subset in Chapter 2. Optional; the book's subset is sufficient.

---

## C.3 Free books

**Mathematics for Machine Learning** — Deisenroth, Faisal, Ong. Free PDF. The right level for Chapters 2, 3 and 6. Ch. 2 (linear algebra), Ch. 5 (calculus), Ch. 6 (probability).

**Dive into Deep Learning** (d2l.ai) — free, interactive, with runnable code in multiple frameworks. Best general reference for Part II and III.

**Deep Learning** — Goodfellow, Bengio, Courville. Free online. Use as a **reference**, not a primary text; it predates transformers and is heavy going front-to-back.

**Probabilistic Machine Learning** — Kevin Murphy. Free drafts. Deep and rigorous. Reach for it when you want the probability treated seriously.

**Spinning Up in Deep RL** — OpenAI. Free. The canonical RL entry point if you take that track (§14.5).

---

## C.4 Hands-on curricula

**ARENA** (Alignment Research Engineer Accelerator) — free, structured, exercise-driven. Covers transformers from scratch, interpretability, and RL. **The best free resource for the transition into research**, and especially strong if you pick the interpretability track.

**Hugging Face courses** — the NLP course and the deep RL course. Practical, framework-focused. Good after you've built things yourself.

**Neel Nanda's interpretability tutorials** — the standard entry to mechanistic interpretability, alongside TransformerLens.

---

## C.5 The paper canon

Read these across Part IV. You'll have implemented most of the ideas already, which makes reading them a completely different experience.

| Area | Papers |
|---|---|
| **Architecture** | *Attention Is All You Need*; *Deep Residual Learning* (ResNet); *Layer Normalization* |
| **Language models** | GPT-2; GPT-3 (*Language Models are Few-Shot Learners*); BERT |
| **Optimization** | *Adam*; *Decoupled Weight Decay Regularization* (AdamW); *Batch Normalization* |
| **Scaling** | Kaplan et al., *Scaling Laws for Neural Language Models*; Hoffmann et al., *Training Compute-Optimal LLMs* (Chinchilla) |
| **Adaptation** | *LoRA*; *QLoRA*; *InstructGPT*; *Direct Preference Optimization* |
| **Regularization** | *Dropout* |
| **Efficiency** | *FlashAttention*; *LLM.int8()*; *Efficient Memory Management for LLM Serving* (vLLM) |
| **Interpretability** | *A Mathematical Framework for Transformer Circuits*; *In-context Learning and Induction Heads*; *Toy Models of Superposition*; *Towards Monosemanticity* |
| **Vision / generative** | *ViT*; *CLIP*; *DDPM*; *Latent Diffusion* |

---

## C.6 Blogs and writers

**Lilian Weng** — long, careful survey posts. Diffusion, RL, attention, agents. Often the best available explanation of a topic.

**Jay Alammar** — *The Illustrated Transformer* and companions. Visual, accessible, good first pass before the math.

**Chris Olah / Distill.pub** — archived but permanently valuable. The interpretability aesthetic starts here.

**The Annotated Transformer** (Harvard NLP) — the paper with code interleaved line by line.

**Karpathy's blog** — *A Recipe for Training Neural Networks* is the single most useful practical post in the field. Read it four times over the year: once now, once after Chapter 7, once after Chapter 9, once after Chapter 12.

**Anthropic, OpenAI, DeepMind, and Meta AI research blogs** — for what labs are doing and why.

---

## C.7 Tools

| Tool | For |
|---|---|
| **Anki** | Spaced repetition. Non-negotiable (§0.5, Appendix D) |
| **PyTorch** | Everything from Chapter 8 |
| **NumPy** | Everything before that |
| **Weights & Biases** | Experiment tracking; free for personal use |
| **TensorBoard** | Offline alternative |
| **Hugging Face** `transformers`, `datasets`, `tokenizers`, `accelerate`, `peft`, `trl` | The applied stack |
| **TransformerLens** | Mechanistic interpretability |
| **lm-evaluation-harness** | Standard LM benchmarks — use it rather than writing your own |
| **vLLM**, **llama.cpp** | Efficient serving and local inference |
| **Triton** | Custom GPU kernels, if you take the systems track |
| **Overleaf** | LaTeX for the Chapter 15 paper |
| **Papers with Code**, **Semantic Scholar**, **Connected Papers** | Finding and mapping literature |

---

## C.8 Compute

| Source | What you get | Notes |
|---|---|---|
| **Kaggle Notebooks** | ~30 GPU-hours/week, free | **The most generous free tier.** Badly underused. Start here. |
| **Google Colab** | Free GPU with session limits | Fine for experiments up to a few hours |
| **Your own CPU** | Everything through Chapter 11 | Genuinely sufficient if you keep models small |
| **Vast.ai / RunPod / Lambda** | Rented GPUs, cheap by the hour | Only needed for Chapter 12's larger runs |
| **Research credit programs** | Varies | Several cloud providers and labs offer credits to students and independent researchers — worth applying |

**Compute discipline when renting:** test on the smallest instance first, checkpoint always, set a spending cap, never leave an instance running overnight without a reason.

**Do not let hardware become a reason to stall.** Chapters 1–11 are laptop-feasible. Chapter 12 is achievable at 10M parameters on a free Kaggle GPU. §14.8 lists which research tracks stay viable on free compute — mechanistic interpretability and evaluation both fully do.

---

## C.9 Communities

**EleutherAI Discord** — open ML research, genuinely welcoming to beginners who ask specific questions. One of the best places on the internet for this.

**Alignment Forum / LessWrong** — where interpretability and safety research is discussed.

**r/MachineLearning** — research discussion. **r/learnmachinelearning** — beginner questions, less intimidating.

**Hugging Face forums and Discord** — applied, practical, fast answers.

**PyTorch forums** — for framework-specific problems.

**Cross Validated** (stats.stackexchange) — for the statistics.

**Local meetups** — Bengaluru has an active ML scene. Go monthly. In-person contact does something Discord doesn't.

**How to ask well** (§0.9): the precise sentence, the minimal reproduction, the printed shapes, and what you already tried. That format gets an answer in one shot.

---

## C.10 Datasets

| Dataset | Use |
|---|---|
| **MNIST**, **Fashion-MNIST** | Chapter 7's first real training |
| **CIFAR-10 / 100** | Chapter 10's CNNs |
| **TinyStories** | Small-scale language modelling — designed for it |
| **OpenWebText**, **The Pile**, **FineWeb** | Chapter 12 pretraining corpora |
| **WikiText-103** | Standard LM benchmark |
| **Hugging Face Datasets Hub** | Almost everything else |
| **`sklearn.datasets`** | Toy data for Chapters 4–5. **Data loading only — never their models.** |

**Always start smaller than you think.** A result you can iterate on in ten minutes teaches more than one you get overnight.

---

## C.11 What to skip, and the traps

**Skip:**

- **Paid courses.** Everything you need is free and better. The paid market is mostly repackaging.
- **Certificates.** Nobody in research cares. Your GitHub and your write-ups are the credential.
- **"Learn AI in 30 days" content.** It teaches API calls, which isn't this.
- **Framework-first tutorials** before Chapter 8. They teach you to use PyTorch without knowing what it does — the exact gap this book exists to close.

**The traps, ranked by how much time they cost people:**

1. **Collecting resources instead of using one.** A bookmarked course is not a studied course. Pick one. Finish it.
2. **Watching instead of building.** Keep the ratio at 1 hour of input to 2 hours of building, minimum.
3. **Starting over.** Restarting from linear algebra for the third time feels productive and is not. Push forward through the discomfort; §0.3 step 7 (park it, move on) exists for this.
4. **Waiting for better hardware.** See §C.8.
5. **Reading about research instead of doing it.** Chapter 13's reproduction is worth more than fifty papers skimmed.

---

*Next: Appendix D — Anki Card Bank*
