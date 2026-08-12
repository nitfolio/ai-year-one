# Appendix E — Data

**Read after Chapter 12. Reread before any project in Part IV.**

This appendix exists because the main text underweights its subject badly. Twelve chapters on architecture and optimization, two paragraphs on data — which is roughly the inverse of how much each actually determines your results.

---

## E.0 The claim

> **Data decisions beat architecture decisions more often than not, and the gap widens as models get larger.**

This isn't a fashionable opinion; it's what the evidence keeps saying:

- **Chinchilla** (§12.6): same compute, smaller model, more data — and it beat a model four times its size. That was a *data allocation* result, not an architecture result.
- **Deduplication alone** measurably improves language models: less memorization, better held-out loss, faster training. No architecture change.
- **LIMA** found that ~1,000 carefully curated instruction examples outperformed tens of thousands of noisy ones. Two orders of magnitude less data, better model.
- **The Phi line of models** got performance far above their parameter count from aggressively curated and synthetic training data.
- Meanwhile, a large share of published architecture improvements **evaporate under fairly tuned baselines** (§13.4). Data improvements tend not to.

There's a structural reason for the asymmetry. Architecture changes are cheap to try and heavily explored, so the remaining gains are small. Data work is tedious, unglamorous, and under-explored, so the remaining gains are large.

**For a solo researcher this is good news.** Careful data work needs judgement and patience far more than it needs GPUs — which makes it one of the few places where you can compete on something other than compute.

---

## E.1 The habit that matters most

Before any technique in this appendix:

> **Look at 100 examples by hand. Then look at 100 more after every processing step.**

Not summary statistics. The actual examples, rendered the way the model will see them.

This is the highest return-per-minute activity in applied machine learning and almost nobody does it. Every practitioner who does has a story about the hour it saved them a month. You will find:

- Boilerplate you didn't know was there (navigation menus, cookie banners, licence footers)
- Encoding damage (`â€™` instead of `'`)
- Truncation at exactly your max length, mid-word
- Labels that don't mean what the documentation says
- Near-duplicates you'd assumed were distinct
- Whole categories that are 90% one degenerate pattern

```python
import random

def eyeball(dataset, n=100, seed=0):
    """Print n random examples exactly as the model will see them."""
    rng = random.Random(seed)
    idx = rng.sample(range(len(dataset)), min(n, len(dataset)))
    for i in idx:
        x = dataset[i]
        print("=" * 70)
        print(repr(x))        # repr, not print — shows whitespace and escapes
```

Use `repr`. Invisible characters are exactly the ones that cause trouble.

---

## E.2 What "quality" actually means

"High-quality data" is used as though it names one thing. It names at least seven, and they trade off against each other.

| Axis | Question | Failure mode |
|---|---|---|
| **Correctness** | Is it true / correctly labelled? | Model learns wrong facts |
| **Diversity** | Does it cover the distribution? | Brittle outside a narrow band |
| **Informativeness** | Does each example teach something? | Compute spent on trivia |
| **Format consistency** | Same structure throughout? | Model learns format noise |
| **Non-duplication** | Repeated content removed? | Memorization, wasted compute |
| **Non-contamination** | Free of eval data? | Meaningless benchmark scores |
| **Provenance** | Do you know where it came from? | Legal and reproducibility risk |

**The important trade-off:** aggressive filtering raises correctness and lowers diversity. Filter hard enough and you get a clean, narrow dataset that produces a model good at one register of text and bad everywhere else.

There's no universal right setting. **What you can do is measure it** — hold out a diverse evaluation set that your filter never touched, and check that filtering harder still helps on it.

---

## E.3 Deduplication

The cheapest large win available.

**Why it matters:**

- Duplicated content is memorized rather than generalized
- Training compute is spent re-learning the same thing
- Duplicates across your train/test split silently inflate your results
- Web crawls are *heavily* duplicated — the same article appears on dozens of domains

### Exact deduplication

```python
import hashlib

def dedup_exact(docs):
    seen, out = set(), []
    for d in docs:
        h = hashlib.sha256(d.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(d)
    return out
```

Catches identical documents. Misses everything that differs by a timestamp or a whitespace change — which is most of it.

### Near-duplicate detection: MinHash + LSH

The standard approach. Represent each document by a set of shingles (overlapping n-grams), estimate Jaccard similarity between sets cheaply via MinHash, and use locality-sensitive hashing to avoid comparing every pair.

```python
from datasketch import MinHash, MinHashLSH

def shingles(text, k=5):
    words = text.split()
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def dedup_near(docs, threshold=0.8, num_perm=128):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep = []
    for i, d in enumerate(docs):
        m = MinHash(num_perm=num_perm)
        for s in shingles(d):
            m.update(s.encode("utf-8"))
        if not lsh.query(m):          # nothing similar already kept
            lsh.insert(str(i), m)
            keep.append(d)
    return keep
```

`threshold=0.8` is a reasonable default. Lower is more aggressive.

### Substring deduplication

MinHash works on whole documents. It misses a long passage repeated inside otherwise-different documents. Suffix-array based substring dedup removes repeated spans above some length (typically ~50 tokens). More expensive; worth it at scale.

### The rules

1. **Deduplicate before splitting**, always. Dedup after splitting leaves duplicates straddling the split, which is leakage.
2. **Deduplicate across splits too.** A document in train that near-matches one in test invalidates your test set.
3. **Log how much you removed.** If dedup removes 60% of your corpus, that's a fact about your data source you need to know.

---

## E.4 Filtering

### Heuristic filters

Cheap, interpretable, and they catch most of the garbage:

```python
def heuristic_ok(text, min_words=50, max_symbol_ratio=0.1,
                 min_mean_word_len=3, max_mean_word_len=10):
    words = text.split()
    if len(words) < min_words:
        return False

    mean_len = sum(len(w) for w in words) / len(words)
    if not (min_mean_word_len <= mean_len <= max_mean_word_len):
        return False        # catches code dumps, base64, character soup

    symbols = sum(c in "#{}[]<>|\\" for c in text)
    if symbols / max(len(text), 1) > max_symbol_ratio:
        return False

    lines = text.split("\n")
    if len(lines) > 5 and len(set(lines)) / len(lines) < 0.5:
        return False        # heavily repeated lines

    return True
```

Standard additions: language identification, a stopword-presence check (real prose contains common function words; generated spam often doesn't), and a check for excessive line-level repetition.

### Classifier-based filtering

Train a small classifier to distinguish "reference quality" text (Wikipedia, books, curated sources) from raw crawl, then keep crawl documents the classifier scores highly. A fast linear classifier on n-gram features is usually enough.

**The catch:** you're now filtering toward whatever your reference set looks like, and inheriting its biases about what counts as good writing. Be explicit about what you chose as "good."

### Perplexity filtering

Score documents with a small language model and drop the extremes. Very high perplexity is usually garbage; very low perplexity is often boilerplate or duplicated text. Keeping the middle band is a reasonable heuristic.

### The discipline

**Log the retention rate of every filter separately.** If one filter is removing 40% of your data, you need to know that, and you need to have looked at 100 of the things it removed. Filters routinely do something other than what you intended.

---

## E.5 Contamination

**Benchmark data leaking into training data**, which makes your evaluation numbers meaningless.

This is endemic, usually unintentional, and frequently undetected. Web crawls contain copies of benchmark datasets, papers quoting benchmark examples, GitHub repos with test sets, and blog posts working through benchmark problems.

### Detecting it

```python
def ngram_set(text, n=13):
    words = text.lower().split()
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}

def contamination_rate(train_docs, eval_docs, n=13):
    """Fraction of eval examples with an n-gram appearing in training."""
    train_ngrams = set()
    for d in train_docs:
        train_ngrams |= ngram_set(d, n)

    hits = sum(bool(ngram_set(e, n) & train_ngrams) for e in eval_docs)
    return hits / len(eval_docs)
```

13-gram overlap is a common convention. Longer `n` means fewer false positives, more misses.

### What to do

1. **Decontaminate before training** — remove training documents that overlap your evaluation sets.
2. **Report the contamination check** in any write-up. "We removed N documents with 13-gram overlap against our eval sets" is a sentence that makes your results credible.
3. **Be suspicious of others' numbers**, especially for models trained on undisclosed web data (§13.4, question 5).
4. **Hold out something private.** An evaluation set you built yourself and never published cannot be contaminated. For your own research this is the strongest guarantee available.

---

## E.6 Mixtures and ordering

Pretraining corpora are mixtures of sources — web, code, books, academic papers, maths. **The weights matter a great deal.**

Known effects worth knowing:

- **Code in the mixture improves reasoning** on non-code tasks. Robust finding, mechanism debated.
- **Upsampling high-quality sources** (books, Wikipedia) beyond their natural share generally helps.
- **Too much of any one domain narrows the model**, even a good domain.
- **Repeating high-quality data** a small number of times (roughly ≤4 epochs) is usually better than adding equivalent low-quality data. Beyond that, returns fall sharply and memorization rises.

**How to set weights:** the honest answer is mostly by experiment. Train small models on candidate mixtures, evaluate on a held-out diverse set, and pick. There are automated approaches (training a small proxy model to optimize mixture weights), but at your scale a small grid over 3–5 weightings is both cheaper and more interpretable.

**Ordering** matters less for pretraining than intuition suggests — shuffled is a strong default and curriculum ordering has a mixed record. It matters considerably more for fine-tuning, where the last data a model sees has outsized influence on its behaviour.

---

## E.7 Synthetic data

Generating training data with a model. Increasingly central, and genuinely double-edged.

**When it works well:**

- **You have a verifier.** Maths with a checker, code with tests, anything where correctness is machine-checkable. Generate many candidates, keep the ones that pass. This is the strongest case by a wide margin.
- **You need format coverage.** Turning existing content into instruction-response pairs.
- **You need rare cases.** Deliberately generating edge cases that are scarce in natural data.

**Where it goes wrong:**

- **Inherited errors.** Whatever the generator gets wrong, your data now asserts confidently.
- **Mode narrowing.** Generated data is less diverse than it looks. Train on it repeatedly and diversity collapses further — the "model collapse" concern.
- **Fluent wrongness.** Synthetic errors are grammatical and confident, which makes them harder to filter than natural noise.

**The rules:**

1. **Verify whatever can be verified.** A synthetic maths dataset without a checker is a synthetic wrong-maths dataset.
2. **Measure diversity explicitly** — n-gram entropy, embedding-space spread, deduplication rate. Don't assume it.
3. **Never train only on synthetic data** unless you have a hard verifier.
4. **Document it.** "This model was trained on data generated by model X" is material information.

---

## E.8 Labels and annotation

For supervised work, the labels are usually the bottleneck.

**Guidelines beat annotators.** Most label noise comes from ambiguous instructions, not careless people. Write the guidelines, label 50 examples yourself, find the cases the guidelines don't cover, rewrite. Iterate before scaling.

**Measure inter-annotator agreement.** Cohen's kappa or similar, on an overlapping subset. **Agreement is a ceiling on achievable accuracy** — if two humans agree only 80% of the time, a model scoring 85% is either superhuman or overfitting the annotation quirks. Usually the latter.

**Label noise is not benign.** Deep networks memorize noisy labels given enough capacity, and clean validation data is essential to detect it.

**Label the hard cases yourself.** The examples annotators disagree on are the ones that determine your model's behaviour at the boundary.

---

## E.9 Fine-tuning data

Different rules from pretraining, and the difference surprises people.

**Quality massively outweighs quantity.** A thousand excellent examples beat fifty thousand mediocre ones. This is the LIMA finding and it replicates.

**Why:** pretraining teaches capability; fine-tuning mostly teaches *format and style selection* from capabilities the model already has. That needs precision, not volume.

**What to optimize:**

- **Diversity of task types** over volume within a type
- **Format consistency** — the model will learn your template exactly, including mistakes in it
- **Response quality** — every example teaches the model what "good" looks like, including its bad habits
- **Difficulty spread** — all-easy examples teach the model to be shallow

**For preference data (DPO, RLHF):** the *margin* matters. Pairs where one response is clearly better teach more than pairs that are nearly tied. Filter for clear preferences before training.

---

## E.10 Data bugs

Structured like Appendix A. Symptom → cause → check.

| Symptom | Likely data cause | Check |
|---|---|---|
| Val accuracy suspiciously high | **Train/test leakage** | Hash examples, check the intersection across splits |
| Great val, terrible in deployment | Distribution shift | Compare basic statistics of both sets |
| Loss drops in steps at epoch boundaries | Data not shuffled | Print one batch's labels — all the same class? |
| Model predicts one class always | Class imbalance | `np.bincount(y)` |
| Loss decreases, generations are garbage | **Off-by-one in LM targets** | `assert (x[1:] == y[:-1]).all()` |
| Model learns nothing at all | Labels shuffled independently of inputs | Print 5 `(x, y)` pairs and verify by hand |
| Sudden `nan` at a specific step | Corrupt example | `assert torch.isfinite(x).all()` in the loader |
| Weirdly high memorization | Heavy duplication | Run dedup, see how much is removed |
| Benchmark scores don't survive a private eval | **Contamination** | n-gram overlap check (§E.5) |
| Tokenizer produces nonsense | Encoding mismatch | Check for `â€™` patterns; verify UTF-8 throughout |
| Truncated mid-sentence everywhere | Max-length truncation without warning | Plot the length distribution against your limit |
| Fine-tuned model won't stop generating | Missing EOS token in training data | Decode 10 training examples and look for it |

**The last one catches people constantly.** If your fine-tuning examples don't end with an end-of-sequence token, the model never learns to stop.

---

## E.11 The workflow

```
1. Acquire.        Record the source and licence for every piece.
2. EYEBALL 100.    Before anything else. repr(), not print().
3. Statistics.     Length distribution, language mix, source mix, duplicates.
4. Deduplicate.    Exact, then near. Log what fraction went.
5. Filter.         One filter at a time. Log retention per filter.
                   Eyeball 50 REJECTED examples per filter.
6. Decontaminate.  Against every eval set you'll use.
7. Split.          Only now. Train/val/test.
8. EYEBALL 100.    Again, post-processing. It will look different.
9. Version.        Hash the final dataset. Write a datasheet.
```

**Steps 2 and 8 are the ones people skip and the ones that pay.**

**Step 5's rejected-example check** is the most underrated line in this appendix. Filters routinely remove things you wanted. You only find out by looking.

### The datasheet

For every dataset you build, write down:

```markdown
- Source(s) and how obtained
- Licence and any usage restrictions
- Size: documents, tokens, bytes
- Processing steps applied, in order, with retention rate for each
- Deduplication method and removal rate
- Contamination checks: which eval sets, what overlap found
- Known limitations and biases
- Content hash of the final artifact
- Date built
```

Ten minutes. It makes your work reproducible, and in Chapter 15 it becomes your methods section almost verbatim.

---

## E.12 Datasets worth knowing

| Purpose | Datasets |
|---|---|
| **LM pretraining** | The Pile, C4, RedPajama, FineWeb, Dolma, OpenWebText |
| **Small-scale LM** | TinyStories (designed for it), WikiText-103 |
| **Instruction tuning** | FLAN, Alpaca, Dolly, OpenAssistant, UltraChat |
| **Preference** | Anthropic HH-RLHF, UltraFeedback |
| **Code** | The Stack, CodeParrot |
| **Evaluation** | MMLU, HellaSwag, GSM8K, HumanEval, TruthfulQA, BIG-bench |
| **Vision** | ImageNet, CIFAR, COCO, LAION |
| **Toy** | MNIST, Fashion-MNIST, `sklearn.datasets` |

**Read the datasheet before using any of them.** Most have known issues documented somewhere — contamination, licensing questions, quality problems in specific slices. Using a dataset without knowing its issues is how you inherit them silently.

---

## E.13 Licensing and ethics

Brief, honest, and genuinely unsettled.

**Copyright status of training data is contested and actively litigated.** As of this writing there is no clean settled answer in most jurisdictions. Anyone who tells you the question is simple, in either direction, is overstating.

**What you can do that's clearly defensible:**

- Use datasets with explicit permissive licences and record which
- Respect `robots.txt` and opt-out signals if you crawl
- Strip personally identifiable information — emails, phone numbers, IDs
- Don't redistribute data you don't have the right to redistribute, even if you can train on it
- **Document provenance.** If you can't say where your data came from, you can't answer any question about it later

**For a public artifact** (a released model, a paper), licence cleanliness matters more than for a private experiment. A model trained on unclear data is one you can't fully release, which limits everything you can do with it.

**On PII:** removal is imperfect but worth doing. Regex for the obvious patterns, then a named-entity pass if the data is sensitive. Models memorize and can regurgitate training data — that's demonstrated, not hypothetical.

---

## E.14 Exercises

**1.** Take any dataset you've used this year. **Eyeball 100 examples with `repr()`.** Write down every surprise. There will be surprises.

**2.** Compute basic statistics: length distribution, vocabulary size, duplicate rate, class balance. Plot the length distribution against your model's max length — how much is silently truncated?

**3.** Implement exact and near-duplicate deduplication. Run both on a web-derived corpus. Report the removal rate for each.

**4.** **Measure the effect.** Train identical small models on the deduplicated and non-deduplicated versions of the same corpus, matched on *token count seen*. Compare held-out loss. This is a real experiment with a real result.

**5.** Implement three heuristic filters. Run each separately, log retention, and **eyeball 50 rejected examples per filter.** Report anything a filter removed that it shouldn't have.

**6.** Implement the n-gram contamination check. Run it between a public pretraining corpus and a public benchmark. Report the overlap rate.

**7.** Build a private evaluation set — 100 examples you write yourself, never published. Evaluate a public model on it and on a public benchmark of the same task. Compare.

**8.** **Mixture experiment.** Take two corpora of different character (e.g. code and prose). Train small models at mixture ratios 0/25/50/75/100. Evaluate all five on both domains and on a third held-out domain. Plot the trade-off.

**9.** Generate synthetic instruction data with a model. Measure its diversity — n-gram entropy and near-duplicate rate — against a comparable human-written set. Report the gap.

**10.** Take a fine-tuning dataset and check whether every example ends with an EOS token. Then fine-tune with and without it and compare generation behaviour.

**11.** Write a datasheet for a dataset you built.

**12.** **The project.** Build a small pretraining corpus from scratch: acquire, eyeball, dedup, filter, decontaminate, split, version, datasheet. Then train identical models on your curated version and on the raw version, matched on tokens seen. Report the difference.

**This is a publishable-shaped experiment on free compute**, and it's the kind of result the field genuinely wants more of.

---

## E.15 What to take away

1. **Look at your data.** 100 examples, `repr()`, before and after every processing step. Nothing else in this appendix matters as much.
2. **Deduplicate before splitting.** Cheapest large win available.
3. **Check contamination**, and hold out something private.
4. **Quality beats quantity for fine-tuning; the reverse is closer to true for pretraining** — but only above a quality floor.
5. **Log the retention rate of every filter, and look at what it rejected.**
6. **Write the datasheet.** It becomes your methods section.
7. **Data work is under-explored relative to its impact**, which makes it one of the few areas where careful judgement beats compute.

That last point is the reason this appendix exists. If you're a solo researcher without a GPU cluster, **data is where you can compete.**

---

*Next: Appendix F — The Learner's Playbook*
