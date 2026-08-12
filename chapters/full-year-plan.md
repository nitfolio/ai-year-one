# The Year One Plan
### From weak math + shaky code → someone who can do real AI research

---

## 0. Read this part once, then never again

You asked for no ifs and buts, so here is the only honest calibration I'll give you, and then I'll stop.

**What one year at 5–8 hrs/day (~2,200 hours) genuinely buys you:** more focused technical time than most people get in a Master's degree. By month 12 you can realistically be someone who has built a transformer from scratch, reproduced published papers independently, run and debugged real training runs, and written one piece of original research. That is a real researcher's starting position. People get hired on less.

**What it does not buy you:** the word "elite." Elite means a decade of taste, scar tissue, and a body of work the field cites. Nobody skips that. What you *can* do in a year is arrive at the starting line already sprinting, instead of arriving lost.

So the goal is not "elite in 12 months." The goal is: **in 12 months, be the kind of person for whom becoming elite is now just a matter of continuing.** That is achievable. That is what this plan is built for.

Now stop thinking about it and start working.

---

## 1. Operating principles

These matter more than the syllabus. A mediocre syllabus followed this way beats a perfect one followed badly.

1. **Math is never studied for its own sake.** Every mathematical concept in this plan exists because something in AI breaks without it. You learn the chain rule the week you write backprop. You learn eigenvectors the week you need PCA. Motivation never dies because the payoff is always 3 days away.

2. **Build it before you import it.** You do not get to `import torch` until you have written the thing torch does. You do not get to call `.backward()` until you have written an autograd engine. This is the single highest-leverage rule in the document. It is also the one you will most want to break.

3. **48-hour rule.** Any concept you learn must appear in code you wrote within 48 hours. Unimplemented knowledge decays to zero.

4. **Everything is public.** One GitHub repo, daily commits, honest READMEs. One blog (even a plain markdown site) with a weekly post. Research careers are built on legible output, and you will be judged on artifacts, not hours. Start the trail on day 1 when it's embarrassing.

5. **Confusion is the signal, not the failure.** When you don't understand something, that's the job starting, not the job failing. The skill you are actually training this year is *tolerance for not understanding yet*.

6. **Rest is scheduled, not earned.** Sunday afternoon and one full day per month are off. Non-negotiable. Burnout at month 5 costs you months 5–12.

---

## 2. Daily and weekly structure

### The daily template (5–8 hours)

| Block | Time | What |
|---|---|---|
| **A. Math** | 90 min | New concept + worked problems. Pen and paper. No screen if possible. |
| **B. Build** | 120 min | Implement. This is the core block. Protect it. |
| **C. Input** | 90 min | Lecture, textbook chapter, or paper. Active: notes in your own words. |
| **D. Consolidate** | 45 min | Anki (15 min), write up what you learned, commit code, review yesterday's code. |
| **E. Stretch** *(optional)* | 60–90 min | Only on high-energy days. Side project, extra reading, or catch-up. |

Block B is sacred. If you only have 3 hours on a given day, do A + B and skip the rest.

### The weekly rhythm

- **Mon–Fri:** full daily template.
- **Saturday:** Project Day. No new material. One long uninterrupted build session on the week's deliverable.
- **Sunday morning:** review week's notes, write the weekly blog post, plan next week.
- **Sunday afternoon:** off. Actually off.

### Checkpoints

End of every phase, you do a **cold rebuild**: implement the phase's central artifact from a blank file, no notes, no reference. If you can't, you don't advance — you spend a week repairing. This is the only quality gate that matters.

---

## 3. Phase 0 — Foundations (Weeks 1–6)

**Goal:** By week 6 you write clean, vectorized Python and you understand linear algebra and derivatives well enough to derive and implement gradient descent yourself.

### Weeks 1–2: Tools + vectors and matrices

- Environment: Python 3.12, `uv` or venv, VS Code, git + GitHub, Linux command line basics, Jupyter.
- Python that's actually good: functions, classes, list/dict comprehensions, generators, decorators (lightly), type hints, `pdb` debugging, reading tracebacks properly.
- NumPy properly: arrays vs lists, dtypes, shapes, indexing, slicing, broadcasting, axis semantics. Broadcasting is the concept that will bite you for a year — over-invest here.
- Linear algebra: vectors, vector addition/scaling, dot product, geometric meaning of dot product, matrices as linear transformations, matrix–vector product, matrix–matrix product, transpose, identity, inverse (conceptually).

**Resources:** 3Blue1Brown *Essence of Linear Algebra* (whole series, twice). *Mathematics for Machine Learning* (Deisenroth, Faisal, Ong — free PDF) Ch. 2. NumPy official docs "absolute basics" + broadcasting guide.

**Deliverable:** `linalg_from_scratch.py` — dot product, matrix–vector, matrix–matrix, transpose, in pure Python with loops. Then the NumPy version. Then a benchmark showing the speed difference and a README explaining *why* NumPy is 100× faster.

### Weeks 3–4: Calculus for gradients

- Derivative as rate of change; derivative as local linear approximation. This second framing is the one that matters.
- Rules: power, product, quotient, chain. Chain rule until it is automatic — you will use it ten thousand times.
- Partial derivatives, gradients, directional derivatives, the gradient as direction of steepest ascent.
- Jacobians and the Hessian (conceptually — you need to recognize them, not compute them by hand).
- Numerical vs analytical derivatives; finite differences and why they're used for gradient checking.

**Resources:** 3Blue1Brown *Essence of Calculus*. Deisenroth Ch. 5. Khan Academy multivariable calculus for drilling problems.

**Deliverable:** `autodiff_numeric.py` — a numerical gradient checker. Given any Python function of a vector, estimate its gradient by finite differences. You will use this tool for the rest of the year to verify your own backprop.

### Weeks 5–6: Your first learning algorithm

- Linear regression: model, loss (MSE), the geometry of least squares.
- Gradient descent: derive the update rule for linear regression by hand, on paper, from scratch.
- Learning rate, convergence, divergence. Watch it explode. Understand why.
- Logistic regression, the sigmoid, cross-entropy loss, and why we use it instead of MSE for classification.
- Train/val/test splits, overfitting, underfitting, regularization (L2) as a first concept.

**Deliverable — Phase 0 capstone:** Pure NumPy implementation of linear + logistic regression trained by gradient descent on a real dataset. No scikit-learn. Include: loss curves, gradient check against your week-4 tool, and a blog post deriving the gradients with your own hand-written math.

**Cold rebuild gate:** blank file → working logistic regression with gradient descent in under 60 minutes.

---

## 4. Phase 1 — Neural Networks From Scratch (Weeks 7–16)

**Goal:** You have written your own automatic differentiation engine and trained real neural networks with it. Backprop holds no mystery.

### Weeks 7–9: Autograd

- Computational graphs. Forward pass, backward pass.
- Backpropagation derived by hand on a 2-layer network. Do this on paper three separate times across three weeks.
- Build a scalar-valued autograd engine: a `Value` class with `+`, `*`, `tanh`, `exp`, and `.backward()` doing topological-sort reverse-mode differentiation.
- Then extend it to tensors (arrays), which is where broadcasting comes back to hurt you.

**Resources:** Karpathy, *Neural Networks: Zero to Hero*, video 1 (micrograd). Watch once. Then close it and build your own from memory. Then compare.

**Deliverable:** `yourname-grad` — your own autograd library, on GitHub, with tests verifying gradients against your numerical gradient checker.

### Weeks 10–12: Probability and multilayer networks

- Probability: sample spaces, random variables, conditional probability, Bayes' rule, independence.
- Distributions: Bernoulli, categorical, Gaussian, uniform. Expectation and variance.
- Maximum likelihood estimation — and the derivation that **minimizing cross-entropy = maximizing likelihood**. This is a load-bearing insight.
- Entropy, cross-entropy, KL divergence.
- MLPs: layers, weights, biases, activation functions (sigmoid, tanh, ReLU, GELU), why nonlinearity is required at all.
- Initialization: why zeros fail, Xavier/Glorot, He initialization.

**Resources:** Deisenroth Ch. 6. Karpathy Zero to Hero videos 2–4 (makemore series). *Dive into Deep Learning* (d2l.ai) Ch. 3–5.

**Deliverable:** A character-level language model (name generator) built on *your own* autograd engine, not PyTorch.

### Weeks 13–16: PyTorch and real training

- Now, finally, PyTorch. Tensors, `nn.Module`, optimizers, `DataLoader`, GPU/device handling, `.backward()`.
- Reimplement everything from weeks 7–12 in PyTorch. Confirm identical results. This is the moment PyTorch stops being magic.
- Training loops done properly: batching, epochs, validation, early stopping, checkpointing.
- Debugging neural networks: shape errors, silent broadcasting bugs, NaN losses, dead ReLUs, vanishing/exploding gradients.
- Experiment tracking: Weights & Biases or TensorBoard. Start logging everything now.

**Resources:** Official PyTorch tutorials. Karpathy's *A Recipe for Training Neural Networks* (blog post — read it four times over the year).

**Deliverable — Phase 1 capstone:** MNIST and CIFAR-10 classifiers, first in your own framework, then PyTorch. Full writeup: architecture choices, hyperparameter sweeps, ablations, honest failure analysis.

**Cold rebuild gate:** blank file → working autograd engine with a trained 2-layer MLP, no references.

---

## 5. Phase 2 — Modern Deep Learning (Weeks 17–30)

**Goal:** You build a GPT from scratch and train it. You understand the architecture that the entire current field runs on.

### Weeks 17–19: Optimization and regularization

- SGD → momentum → Nesterov → RMSProp → Adam → AdamW. Implement each yourself before using the built-in.
- Learning rate schedules: step, cosine, warmup. Why warmup exists.
- BatchNorm, LayerNorm, RMSNorm — what they normalize over and why LayerNorm won for transformers.
- Dropout, weight decay, data augmentation, label smoothing.
- Gradient clipping, gradient accumulation.

### Weeks 20–22: Convolutional networks

- Convolution as an operation: kernels, stride, padding, dilation, receptive field.
- Implement conv2d from scratch in NumPy. Slow is fine. Then im2col.
- Pooling, classic architectures: LeNet → AlexNet → VGG → ResNet.
- Residual connections — the idea, and why it generalizes far beyond vision.
- Transfer learning and fine-tuning.

**Resources:** Stanford CS231n (lectures + assignments — do the assignments).

### Weeks 23–26: Sequences to attention

- RNNs, backprop through time, vanishing gradients in sequences.
- LSTMs and GRUs — understand the gating, then understand why the field abandoned them.
- Attention: query, key, value. Derive scaled dot-product attention. Understand *why* the √d scaling exists.
- Self-attention, multi-head attention, causal masking.
- Positional encodings: sinusoidal, learned, RoPE.
- Read *Attention Is All You Need* line by line. Then implement the full transformer block from the paper alone.

**Resources:** Stanford CS224n. *The Illustrated Transformer* (Jay Alammar). *The Annotated Transformer* (Harvard NLP).

### Weeks 27–30: Build a GPT

- Tokenization: characters → BPE. Implement BPE yourself.
- Full GPT architecture: embeddings, transformer blocks, weight tying, final projection.
- Train it on a real corpus. Watch the loss curve. Learn what a healthy one looks like.
- Sampling: greedy, temperature, top-k, top-p. Implement all four.
- Scaling laws (Kaplan; Chinchilla) — read both papers.
- Efficiency essentials: mixed precision, gradient checkpointing, `torch.compile`, what FlashAttention does and why.
- Practical: how to actually get compute (Colab → Kaggle → Lambda/Vast/RunPod). Budget for a few hundred dollars of GPU time across the year.

**Resources:** Karpathy's nanoGPT and his GPT-2 reproduction video. Stanford CS336 (*Language Modeling from Scratch*) — the single best course for this phase.

**Deliverable — Phase 2 capstone:** Your own GPT implementation, trained on a corpus you chose, with a full technical writeup: tokenizer, architecture, training curves, sampling comparisons, scaling experiments at 3 model sizes. This is the first artifact you'd actually show someone as evidence you're a researcher.

**Cold rebuild gate:** blank file → working transformer block with multi-head causal attention, no references, under 90 minutes.

---

## 6. Phase 3 — Specialization and Paper Reproduction (Weeks 31–42)

**Goal:** You stop being a student and start being a practitioner. You read papers and turn them into working code independently.

### Weeks 31–32: Learning to read papers

- The three-pass method (Keshav). Practice on 15 papers.
- How to read an equation you don't understand: identify shapes, identify what's learned vs fixed, implement the smallest version.
- Set up an arXiv habit: 30 min/day skimming abstracts in your area. Build taste for what matters.
- Learn the tools: Hugging Face ecosystem (transformers, datasets, accelerate), evaluation harnesses, distributed training basics (DDP, FSDP conceptually).

### Weeks 33–42: Pick a track and reproduce

Choose **one** track. Depth beats breadth from here.

| Track | Core papers to reproduce | Why pick it |
|---|---|---|
| **LLMs & post-training** | Instruction tuning, LoRA/PEFT, DPO, RLHF basics | Largest job market, most compute-friendly at small scale |
| **Mechanistic interpretability** | Induction heads, sparse autoencoders, circuit analysis | Lowest compute barrier, very research-y, hot field |
| **Reinforcement learning** | DQN, PPO, actor-critic, world models | Deepest math, strong for agents/robotics |
| **Efficiency & systems** | Quantization, distillation, speculative decoding, KV-cache tricks | Highly employable, engineering-heavy |
| **Multimodal** | CLIP, ViT, diffusion models (DDPM) | Visual results, generative work |

**Resources:** For interpretability: the ARENA curriculum and Neel Nanda's materials. For RL: OpenAI *Spinning Up in Deep RL*. For diffusion: the DDPM paper + Lilian Weng's blog posts.

**Deliverable:** Three papers reproduced from scratch. For each: a repo, a writeup of what the paper omitted, and at least one ablation the authors didn't run. That last part is where research actually begins — you are already generating novel results, just small ones.

**Milestone:** By week 42 you should have found at least three open questions in your track that you personally find annoying. One of them becomes Phase 4.

---

## 7. Phase 4 — Original Research (Weeks 43–52)

**Goal:** You produce one piece of original work and put it in front of the field.

- **Weeks 43–44:** Pick the question. Do a real literature review — 30+ papers, know what's been tried. Write a one-page proposal: hypothesis, method, experiments, what result would falsify it. Send it to me and to two people in the field.
- **Weeks 45–49:** Run experiments. Keep a dated research log every single day. Expect the first three ideas to fail — that is the normal rate, not a signal about you.
- **Weeks 50–51:** Write it up. Proper paper structure: abstract, intro, related work, method, experiments, limitations, conclusion. Learn LaTeX/Overleaf.
- **Week 52:** Ship. Post to arXiv. Submit to a workshop (NeurIPS/ICLR/ICML workshops are the realistic entry point and they are genuinely open to newcomers). Write the accessible blog version. Post it publicly.

**Note for you specifically:** your Phase 4 project can double as the technical core of something shippable. A research artifact that's also a working product is strictly more valuable than either alone — keep that option open when picking the question.

**Also in this phase:** contribute to a real open-source project (PyTorch, Hugging Face, vLLM, or a lab's codebase). One merged PR to a serious repo is a stronger signal than most credentials.

---

## 8. Running alongside, all year

- **Anki, 15 min/day, from week 1.** Cards for: definitions, derivative rules, distribution properties, architecture details. This is how you stop re-learning things.
- **A math drilling habit.** 30 min, 3×/week, of actual problems. Watching 3Blue1Brown is not doing math. Problems are doing math.
- **A weekly blog post.** Even 300 words. This is the single most underrated career move available to you.
- **One paper read per week minimum, all year**, even during Phase 0. You won't understand them early. Read them anyway — you're building pattern recognition.
- **Community.** Find 2–3 people at your level and 1–2 ahead of you. Discord servers, local ML meetups in Bengaluru, EleutherAI, open-source contributor channels. Solo for a year is the most common way this plan fails.

---

## 9. The failure modes that will actually get you

Ranked by how likely they are to be the thing that stops you:

1. **Tutorial hell.** Watching feels like progress. It is not progress. Ratio: 1 hour of input to 2 hours of building, minimum.
2. **Skipping the from-scratch builds** because a library already does it. This is the exact moment you become a user of AI instead of a researcher of it.
3. **Front-loading math forever.** Waiting to "finish" math before touching neural nets. You will never finish. Start Phase 1 on schedule even if calculus feels shaky.
4. **Perfectionism about the public repo.** Commit the ugly code. Nobody is watching yet, and later it will be evidence of growth.
5. **The month-4 wall.** Around month 4 the novelty dies and the finish line isn't visible. This is where most people quit. Plan for it now: when it hits, cut to 4 hours/day for two weeks rather than stopping.
6. **Comparing yourself to people 5 years ahead.** Compare to yourself 30 days ago. That's the only meaningful comparison.
7. **No rest.** 8 hours every day with no off days ends in week 20, not week 52.

---

## 10. How to use me

Bring me:

- **Your code, always.** Paste it. I'll review it like a supervisor would — correctness, then style, then the thing you didn't know you didn't know.
- **"I don't understand X."** Say it early and often. The faster you say it, the less time you waste.
- **Derivations you attempted.** Show me your handwritten math. Wrong derivations are more useful to me than questions.
- **Bugs and broken training runs.** Debugging together is where the most learning per minute happens.
- **Your weekly writeups**, for critique before you post them.
- **When you want to quit.** Bring me that too. It's a normal part of week 18.

Each week I'll give you the week's lesson plan, the concepts, the problems, and the build task. You come back with the work. That's the loop.

---

## 11. Right now, today

1. Install Python, git, VS Code. Create the GitHub repo. Call it `ai-year-one`.
2. Install Anki.
3. Commit an empty README with one line: your start date and your goal.
4. Do Day 1's task (in chat).

Start date: **5 August 2026.**
Target: **5 August 2027.**
