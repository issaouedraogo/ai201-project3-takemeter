# TakeMeter

**AI201 · Project 3 — Fine-tuning a text classifier**

TakeMeter sorts short technical-discussion comments by the *quality of the
opinion* they carry — substantive arguments vs. unsupported absolutism vs. pure
filler. It fine-tunes [DistilBERT](https://huggingface.co/distilbert-base-uncased)
on a labeled dataset and compares it against a zero-shot baseline (Groq
`llama-3.3-70b-versatile`).

See [`planning.md`](planning.md) for the pre-collection design doc (taxonomy,
data plan, evaluation plan, AI tool plan).

## Repository contents

| File | Description |
|------|-------------|
| [`planning.md`](planning.md) | Design doc: community, labels, edge cases, data + evaluation plan, AI tool plan. |
| [`takemeter_sample.csv`](takemeter_sample.csv) | The labeled dataset (`text`, `label`). |
| [`evaluation_results.json`](evaluation_results.json) | Metrics from the Colab run. |
| [`confusion_matrix.png`](confusion_matrix.png) | Fine-tuned model confusion matrix (image). |

> The notebook runs in Colab and is **not** committed here — this repo holds the
> planning, dataset, and downloaded outputs.

---

## Community choice and reasoning

**Community: Hacker News comment threads** (with r/programming and
r/ExperiencedDevs as secondary sources of the same discourse).

HN threads on technical posts attract a deliberately mixed crowd: senior
practitioners who write multi-paragraph tradeoff analyses, strong-opinion
regulars who fire off absolutist one-liners, and drive-by commenters who leave a
meme or a "+1". That mix is what makes the task interesting:

- **The full quality spectrum appears in one thread.** A single "monolith vs.
  microservices" thread contains a careful "we did this at 15 engineers and
  here's what happened" comment *and* "microservices are always the right call,
  monoliths are legacy garbage." The classifier must separate them on
  *reasoning*, not topic — the topic is identical.
- **Reasoning and assertion look superficially alike** (both confident,
  jargon-dense, opinionated), so the distinction is subtle and learnable rather
  than a keyword match.
- **High volume + natural class variety** make a balanced, real (non-synthetic)
  dataset feasible.

---

## Label taxonomy

Three mutually exclusive labels. **Core rule: classify only what is *stated in
the post itself*** — reasoning that is merely implied, borrowed from the parent
comment, or gestured at without specifics does **not** count.

### `reasoned`
A comment that states a technical opinion **and supports it within the post**
with a concrete tradeoff, a specific experience, evidence, or a mechanism
("X because Y").

- *"We moved off microservices back to a monolith — the network overhead and distributed-tracing cost outweighed the team-autonomy benefit at our ~15-engineer scale."*
- *"Use a hashmap here — O(1) lookup instead of the O(n) loop you've got, which is why it slows down past ~10k items."*

### `hot_take`
A comment that states a **strong, absolutist** technical opinion with **no
supporting reasoning inside the post** — sweeping "always/never" claims, or
judgment delivered as insult.

- *"ORMs are a cancer and anyone who uses them doesn't understand databases."*
- *"Microservices are always the right call. Monoliths are legacy garbage and that's just a fact."*

### `noise`
A comment that carries **no technical opinion at all**: agreement-filler, memes,
jokes, reactions, or "first"-style content.

- *"skill issue"*
- *"thanks for coming to my ted talk"*

---

## Dataset

### Source
Comments drawn from Hacker News (via the Algolia HN Search API and direct thread
reading) and r/programming / r/ExperiencedDevs, sampled from opinion-generating
threads (architecture debates, language/tooling choices, "we migrated X" posts)
so all three classes occur naturally.

### Labeling process
Each comment was labeled by hand against the taxonomy above, using the
"stated-not-implied" core rule and the tie-breakers documented in
[`planning.md` §3](planning.md). Label definitions were stress-tested with an LLM
**before** annotation began (see [AI usage](#ai-usage)), so the rules were
locked before any example was labeled.

> ⚠️ **TODO (annotation disclosure):** if you used an LLM to *pre-label* any
> batch before reviewing, state which tool and how many rows here, per the
> planning doc's §7.2 plan. If all labels were hand-applied, say so explicitly.

### Label distribution

| Label | Count |
|-------|-------|
| `reasoned` | 10 |
| `hot_take` | 10 |
| `noise` | 10 |
| **Total** | **30** |

> ⚠️ This sample is **balanced but small (30 rows)**. The project target is
> 200+; the small size is the direct cause of the weak fine-tuning result
> below. See [Reflection](#reflection).

### 3 difficult-to-label examples and decisions

| Post | Tension | Decision & reason |
|------|---------|-------------------|
| *"Tabs vs spaces matters less than just running a formatter in CI — we adopted Black and the bikeshedding thread disappeared overnight."* | `hot_take` vs `reasoned` | **`reasoned`** — it opens on a flame-war topic, but states a concrete action and observed outcome (adopted Black → bikeshedding stopped). A stated *why* outweighs the inflammatory framing. |
| *"lol Java moment"* | `noise` vs `hot_take` | **`noise`** — names a technology, so it looks topical, but expresses no actual position. No opinion present → `noise`, regardless of the dig. |
| *"Use a hashmap here — O(1) lookup instead of the O(n) loop you've got."* | `noise` vs `reasoned` | **`reasoned`** — terse and imperative, could read as a throwaway, but contains an explicit mechanism (O(1) vs O(n)). Length is not the signal; stated reasoning is. |

---

## Fine-tuning approach

- **Base model:** `distilbert-base-uncased` with a 3-class sequence-classification head.
- **Data split:** 70 / 15 / 15 train / val / test, **stratified**, `random_state=42`
  → 21 train / 4 val / 5 test.
- **Tokenization:** `truncation=True, max_length=256`.
- **Training setup** (Hugging Face `Trainer`, T4 GPU):

  | Hyperparameter | Value |
  |----------------|-------|
  | epochs | 3 |
  | learning rate | 2e-5 |
  | train batch size | 16 |
  | weight decay | 0.01 |
  | warmup steps | 50 |
  | eval / save strategy | per epoch, `load_best_model_at_end` on accuracy |

- **Hyperparameter decision — learning rate 2e-5.** I kept the standard
  BERT-family fine-tuning learning rate of `2e-5` rather than a larger value.
  With only 21 training examples, a higher LR risks the model lurching to a
  degenerate solution in very few steps; the conservative `2e-5` is the stable
  starting point. (In hindsight the *dataset size*, not the LR, was the binding
  constraint — see [Reflection](#reflection).)

---

## Baseline description

A zero-shot baseline using Groq `llama-3.3-70b-versatile` at `temperature=0`,
`max_tokens=20`, classifying each test post into one of the three labels. The
model's reply is lowercased and matched against the label strings (longest-first
to avoid substring collisions); unmatched replies count as unparseable. Results
were collected by running this over the **same 5-example locked test set** used
for the fine-tuned model, so the comparison is apples-to-apples.

**Prompt used:**

> ⚠️ **TODO (paste your actual prompt):** the prompt you wrote lives in your
> Colab notebook (the `SYSTEM_PROMPT` variable) and is not in the committed
> files. Paste the exact text here. It should name the community, define each
> label in one sentence, give one example per label, and instruct the model to
> reply with only the label name. (The baseline scored 1.00, so your filled-in
> prompt clearly worked.)

```text
<paste SYSTEM_PROMPT from your notebook here>
```

---

## Full evaluation report

Locked test set: **5 examples** (`reasoned` ×1, `hot_take` ×2, `noise` ×2).
Source: [`evaluation_results.json`](evaluation_results.json) + the Colab
`classification_report` / `confusion_matrix` output.

### Overall accuracy

| Model | Accuracy |
|-------|----------|
| Zero-shot baseline (Groq `llama-3.3-70b-versatile`) | **1.00** |
| Fine-tuned DistilBERT | **0.40** |
| Improvement | **−0.60** (regression) |

### Per-class metrics

**Fine-tuned DistilBERT** (`zero_division=0`):

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| `reasoned` | 0.00 | 0.00 | 0.00 | 1 |
| `hot_take` | 0.40 | 1.00 | 0.57 | 2 |
| `noise` | 0.00 | 0.00 | 0.00 | 2 |
| **macro avg** | 0.13 | 0.33 | 0.19 | 5 |
| **weighted avg** | 0.16 | 0.40 | 0.23 | 5 |

**Zero-shot baseline** (all 5 correct):

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| `reasoned` | 1.00 | 1.00 | 1.00 | 1 |
| `hot_take` | 1.00 | 1.00 | 1.00 | 2 |
| `noise` | 1.00 | 1.00 | 1.00 | 2 |
| **macro avg** | 1.00 | 1.00 | 1.00 | 5 |

### Confusion matrix (fine-tuned model)

Rows = true label, columns = predicted label.

| true ＼ predicted | `reasoned` | `hot_take` | `noise` |
|------------------|:----------:|:----------:|:-------:|
| **`reasoned`** | 0 | 1 | 0 |
| **`hot_take`** | 0 | 2 | 0 |
| **`noise`** | 0 | 2 | 0 |

Every test example was predicted `hot_take` — the model **collapsed to a single
class**. (Image version: [`confusion_matrix.png`](confusion_matrix.png).)

### 3 specific wrong predictions, with analysis

| Post | True | Predicted | Why it went wrong |
|------|------|-----------|-------------------|
| *"Tabs vs spaces matters less than just running a formatter in CI — we adopted Black and the bikeshedding thread disappeared overnight."* | `reasoned` | `hot_take` | The post opens on "tabs vs spaces," a topic that co-occurs with hot takes throughout training. With too few examples to learn that a *stated outcome* makes it `reasoned`, the model keyed on inflammatory-adjacent vocabulary. |
| *"+1"* | `noise` | `hot_take` | The model never learned a `noise` decision boundary at all (0 `noise` predictions). With no signal for short filler, it falls back to its default class. |
| *"same tbh"* | `noise` | `hot_take` | Same failure as "+1": agreement-filler with no opinion, but the collapsed model has no `noise` behavior to fall back on. |

These three errors *are* the entire gap: the 2 correctly-classified examples are
the two genuine `hot_take` posts, and they're "correct" only because the model
labels **everything** `hot_take`.

### Sample classifications

The fine-tuned model on the 5 test posts (predicted label is real; confidence is
the softmax probability the notebook computes as `ft_probs` — read it from your
Section-4 output and fill in):

| Post | True | Predicted | Confidence | Correct? |
|------|------|-----------|------------|:--------:|
| *"Agile is just a scam invented to sell certifications. Real engineers don't need standups."* | `hot_take` | `hot_take` | ⟨ from notebook ⟩ | ✅ |
| *"TypeScript is pointless. Types are training wheels for people who can't write JavaScript."* | `hot_take` | `hot_take` | ⟨ from notebook ⟩ | ✅ |
| *"Tabs vs spaces matters less than just running a formatter in CI…"* | `reasoned` | `hot_take` | ⟨ from notebook ⟩ | ❌ |
| *"+1"* | `noise` | `hot_take` | ⟨ from notebook ⟩ | ❌ |
| *"same tbh"* | `noise` | `hot_take` | ⟨ from notebook ⟩ | ❌ |

**Correct example, explained:** *"Agile is just a scam invented to sell
certifications. Real engineers don't need standups."* → predicted `hot_take`
(correct). This is a textbook `hot_take`: a sweeping, absolutist dismissal with
no supporting reasoning. The honest caveat is that the model gets it right for
the *wrong reason* — it predicts `hot_take` for every input, so it happens to
nail the genuine hot takes while missing everything else.

---

## Reflection

**What I intended the model to learn:** the distinction between a *supported*
opinion (`reasoned`), an *unsupported* opinion (`hot_take`), and *no opinion*
(`noise`) — i.e. to detect whether a stated justification is present.

**What it actually learned:** to always output `hot_take`. With 21 training
examples across 3 classes, DistilBERT never received enough signal to form
decision boundaries, so it minimized loss by collapsing onto one class. It
learned a constant, not a concept. The baseline LLM — which already understands
these distinctions from pretraining and only needed the definitions in a
prompt — got all 5 right.

The takeaway matches the planning hypothesis: **fine-tuning only pays off with
enough labeled data.** At 30 examples the zero-shot LLM is strictly better. The
fix is the dataset, not the architecture or the hyperparameters: reaching the
200+ target (~67 per class) and using a larger held-out test set (the
5-example set quantizes every metric into 0.20 steps) is the path to a
meaningful comparison.

## Spec reflection

- **Where the spec helped:** the notebook spec **locked the evaluation
  pipeline** — a fixed stratified split (`random_state=42`), a held-out test set
  the model never sees, and identical inputs for both models. That made the
  fine-tuned-vs-baseline comparison reproducible and trustworthy, and let me
  reconstruct the exact misclassifications after the fact.
- **Where implementation diverged, and why:** the spec calls for **200+ labeled
  examples**; this run used **30**. The divergence was a scoping/time choice to
  get an end-to-end run working first. The consequence was direct and
  predictable — the single-class collapse above — which is itself the clearest
  evidence for *why* the spec sets that minimum.

## AI usage

1. **Label stress-testing (definitions revised).** I directed an LLM to generate
   8 posts sitting on the boundary between two labels. It produced cases the
   original definitions couldn't resolve — e.g. *"Same, switched to Postgres for
   the same reason and never looked back"* (reasoning borrowed from the parent
   comment) and *"ORMs are fine until they aren't"* (gestures at a tradeoff
   without stating it). **I overrode the original definitions**, adding the
   "classify only what is stated in the post itself" core rule and four
   tie-breakers ([`planning.md` §3](planning.md)). This happened *before*
   annotation, exactly as the AI tool plan intended.

2. **Evaluation reconstruction (verified, not trusted).** I directed an AI tool
   to reproduce the deterministic test split (`random_state=42`) and recover the
   exact 5 test examples and their misclassifications, then to compute per-class
   metrics from the confusion matrix. I verified the output against the committed
   `evaluation_results.json` (accuracy 0.40) and `confusion_matrix.png` before
   using it — the reproduced split and the saved artifacts agree.

3. **Drafting.** AI assistance was used to draft this README and `planning.md`;
   I reviewed and edited the content and the reported numbers.

> ⚠️ **Annotation disclosure:** see the [Labeling process](#labeling-process)
> TODO — state whether an LLM pre-labeled any examples, per the planning doc's
> §7.2 plan.

---

## Demo video

A 3–5 minute walkthrough (link to be added on submission) covering: 3–5 posts
classified by the fine-tuned model with label + confidence; one correct
prediction narrated; one incorrect prediction narrated with the reason; and a
brief tour of this evaluation report. Script + demo cell in
[`demo/`](demo/).

---

## Stretch features

| Feature | Status | Where |
|---------|--------|-------|
| Error pattern analysis | ✅ completed | [`analysis/error_patterns.md`](analysis/error_patterns.md) |
| Deployed interface | ✅ completed | [`interface/`](interface/) |
| Inter-annotator reliability | 🟡 scaffolded (needs 2nd annotator) | [`annotation/`](annotation/) |
| Confidence calibration | 🟡 scaffolded (needs larger test set) | [`calibration/`](calibration/) |

### Error pattern analysis
[`analysis/error_patterns.md`](analysis/error_patterns.md) identifies the
*systematic* error — a **single-class collapse** (every input → `hot_take`,
driven by data starvation) — distinguished from a secondary, still-unconfirmed
topic-keyword hypothesis, with each claim verified against the confusion matrix.

### Deployed interface (Gradio)
[`interface/app.py`](interface/app.py) accepts a post and shows the predicted
label + confidence over all three classes. The model lives in the Colab runtime,
so run it there:

```python
# In a Colab cell, after training has produced ./takemeter-model:
!pip install -q gradio
!python interface/app.py        # prints a public *.gradio.live share URL
```

To run locally instead: download the `takemeter-model/` folder from Colab, then

```bash
pip install -r interface/requirements.txt
TAKEMETER_MODEL=/path/to/takemeter-model python interface/app.py
```

### Inter-annotator reliability (scaffolding)
1. `python annotation/make_sheet.py` → produces a shuffled, label-free
   `annotation_sheet.csv`.
2. A second person labels it independently using
   [`annotation/labeling_guide.md`](annotation/labeling_guide.md).
3. `python annotation/compute_agreement.py takemeter_sample.csv annotation/annotation_sheet.csv`
   → reports % agreement, **Cohen's kappa**, the disagreement list, and the
   annotator-vs-annotator confusion matrix.

### Confidence calibration (scaffolding)
1. Run [`calibration/collect_predictions.py`](calibration/collect_predictions.py)
   in Colab against a held-out set → `predictions_with_confidence.csv`.
2. `python calibration/calibration.py predictions_with_confidence.csv` → per-bin
   accuracy vs. confidence + **Expected Calibration Error**. Meaningful only once
   the test set is large (50+); the current 5-example set is too small.
