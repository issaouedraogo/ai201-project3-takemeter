# TakeMeter — Planning

> Planning doc for the dataset and classifier. Sections marked **_(fill in)_**
> depend on choices only you can confirm (data source, collection method).

## Problem

Classify short technical-discussion comments by the *quality of the opinion*
they express. The goal is to separate substantive, reasoned arguments from
low-effort absolutism and from pure filler — useful for surfacing the comments
worth reading in a long thread.

## Label taxonomy

Three mutually exclusive labels:

### `reasoned`
A substantive opinion that is **backed by something** — a stated tradeoff,
a concrete experience, evidence, or a "we did X and Y happened" anecdote.

- *"We moved off microservices back to a monolith — the network overhead and
  distributed-tracing cost outweighed the team-autonomy benefit at our
  ~15-engineer scale."*
- *"Use a hashmap here — O(1) lookup instead of the O(n) loop you've got, which
  is why it slows down past ~10k items."*

### `hot_take`
A **strong, absolutist** opinion stated with **no supporting reasoning** —
sweeping claims, "always/never", insults toward people who disagree.

- *"ORMs are a cancer and anyone who uses them doesn't understand databases."*
- *"Microservices are always the right call. Monoliths are legacy garbage and
  that's just a fact."*

### `noise`
Low-content reactions, memes, agreement-without-substance, or filler that
carries **no opinion** at all.

- *"skill issue"*
- *"+1"*
- *"thanks for coming to my ted talk"*

## Where `reasoned` and `hot_take` differ (the hard boundary)

Both express opinions; the distinction is **justification, not strength**.
A confidently-stated claim is still `reasoned` if it gives a *why*
("Postgres over Mongo for this: your data is clearly relational…"), and a mild
claim is still a `hot_take` if it's pure assertion. The annotation rule:

> If you removed the conclusion, would any supporting reason remain?
> Yes → `reasoned`. No → `hot_take`. No opinion at all → `noise`.

## Dataset

- **File:** [`takemeter_sample.csv`](takemeter_sample.csv) — columns `text`, `label`.
- **Size:** 30 examples, balanced 10 / 10 / 10 across the three labels.
- **Source / collection:** **_(fill in — where the comments came from and how
  you gathered/wrote them.)_**
- **Balance rationale:** equal counts per class so neither the fine-tuned model
  nor the evaluation metrics are skewed by class frequency.

> **Note:** the project brief calls for 200+ examples. This CSV is the balanced
> sample committed here; **_(fill in: confirm whether this is the full set you
> trained on, or expand to the full annotated set before training.)_**

## Edge cases & annotation decisions

- Sarcasm / jokes with no technical content → `noise`, even if topical
  (*"lol Java moment"*).
- A strong opinion that *also* includes a real reason → `reasoned` wins over
  `hot_take` (justification takes precedence).
- Short but substantive ("Use a hashmap — O(1) vs O(n)") → `reasoned`, not
  `noise`; length alone is not the signal.

## Modeling approach

- **Model:** fine-tune `distilbert-base-uncased` with a 3-class head.
- **Split:** 70 / 15 / 15 train / val / test, stratified, `random_state=42`.
- **Training:** 3 epochs on a T4 GPU (notebook defaults).
- **Baseline:** zero-shot Groq `llama-3.3-70b-versatile` on the same test set.
- **Metrics:** accuracy + per-class precision/recall, confusion matrix.

## Results

Recorded in [`README.md`](README.md) after running the Colab notebook
(`evaluation_results.json` + `confusion_matrix.png`).
