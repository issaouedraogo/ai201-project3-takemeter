# TakeMeter

**AI201 · Project 3 — Fine-tuning a text classifier**

TakeMeter classifies short technical-discussion comments (the kind you see on
developer forums, PR threads, and Hacker News) into three categories, then
compares a fine-tuned [DistilBERT](https://huggingface.co/distilbert-base-uncased)
model against a zero-shot large-language-model baseline (Groq
`llama-3.3-70b-versatile`).

## Labels

| Label | Meaning | Example |
|-------|---------|---------|
| `reasoned` | A substantive opinion backed by a tradeoff, an experience, or evidence. | *"We moved off microservices back to a monolith — the network overhead outweighed the team-autonomy benefit at our ~15-engineer scale."* |
| `hot_take` | A strong, absolutist opinion stated with no supporting reasoning. | *"ORMs are a cancer and anyone who uses them doesn't understand databases."* |
| `noise` | Low-content reactions, memes, or filler that carry no opinion. | *"skill issue"* |

See [`planning.md`](planning.md) for the full taxonomy, label definitions, and
dataset-design notes.

## Repository contents

| File | Description |
|------|-------------|
| [`planning.md`](planning.md) | Label taxonomy, dataset design, and annotation decisions. |
| [`takemeter_sample.csv`](takemeter_sample.csv) | The labeled dataset (`text`, `label` columns). |
| `evaluation_results.json` | Metrics produced by the Colab notebook (baseline vs. fine-tuned accuracy). |
| `confusion_matrix.png` | Confusion matrix for the fine-tuned model on the test set. |

> The notebook itself lives in Colab and is **not** committed here — this repo
> holds the planning, dataset, and downloaded outputs.

## Dataset

`takemeter_sample.csv` has two columns, `text` and `label`, with the label
distribution:

| Label | Count |
|-------|-------|
| `reasoned` | 10 |
| `hot_take` | 10 |
| `noise` | 10 |

The Colab notebook splits this 70 / 15 / 15 into train / validation / test
(stratified, `random_state=42`).

## How the model is built

1. **Fine-tune** `distilbert-base-uncased` with a 3-class classification head
   for 3 epochs on a T4 GPU.
2. **Baseline** — classify the same test set zero-shot with Groq
   `llama-3.3-70b-versatile` using a prompt built from the label definitions.
3. **Compare** accuracy and per-class metrics, and save:
   - `evaluation_results.json` — `baseline_accuracy`, `finetuned_accuracy`,
     `improvement`, `test_set_size`, `label_map`, `model`.
   - `confusion_matrix.png` — fine-tuned model confusion matrix.

## Evaluation report

> _Run the Colab notebook, download `evaluation_results.json` and
> `confusion_matrix.png`, add them to this repo, then fill in the numbers below._

- **Fine-tuned accuracy:** _TBD_
- **Zero-shot baseline accuracy:** _TBD_
- **Improvement:** _TBD_

![Confusion matrix](confusion_matrix.png)

_Add your analysis here: where does the fine-tuned model beat the baseline,
which classes get confused, and what the confusion matrix reveals._
