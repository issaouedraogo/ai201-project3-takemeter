# TakeMeter — Error Pattern Analysis

*Stretch feature — see [`planning.md` §8.1](../planning.md).*

## Method

Following the failure-analysis plan in [`planning.md` §7.3](../planning.md): the
full set of test-set misclassifications was examined for a **systematic**
pattern (not just a list of individual misses), and every claimed pattern was
verified against the committed [`confusion_matrix.png`](../confusion_matrix.png)
and the deterministically reproduced test split (`random_state=42`).

## The data

Locked test set (5 examples) and the fine-tuned model's predictions:

| Post | True | Predicted | ✓/✗ |
|------|------|-----------|:---:|
| "Agile is just a scam invented to sell certifications…" | `hot_take` | `hot_take` | ✓ |
| "TypeScript is pointless. Types are training wheels…" | `hot_take` | `hot_take` | ✓ |
| "Tabs vs spaces matters less than just running a formatter in CI…" | `reasoned` | `hot_take` | ✗ |
| "+1" | `noise` | `hot_take` | ✗ |
| "same tbh" | `noise` | `hot_take` | ✗ |

## Pattern 1 (primary, established): total single-class collapse

**The model predicts `hot_take` for 100% of inputs, independent of content.**
This is not a "confuses A with B" pattern — it is a *degenerate constant
classifier* that has learned to ignore the input.

**Verification:**
- Every row of the confusion matrix lands in the `hot_take` column; `reasoned`
  and `noise` each have **recall 0.00** and were **never predicted once**.
- This exactly explains the 0.40 accuracy: the only correct predictions are the
  2/5 examples that happen to be true `hot_take`.
- Holds across all three error cases — the failures are not topic-specific, they
  are *every* non-`hot_take` example.
- **The collapse is low-confidence.** Reading the softmax shows every prediction
  sits at **≈34%** — barely above the 33.3% chance level for 3 classes. The
  model's three class probabilities are nearly tied; `hot_take` wins by a hair.
  So this is less a *confident* collapse than a near-uniform output whose argmax
  is unstable: re-training reshuffles which examples land right (a fresh run gave
  different per-example predictions at the same ~0.40 accuracy). That instability
  is itself the signature of a model that learned almost nothing.

## Pattern 2 (contributing, hypothesis): topic-keyword bias over reasoning

The misclassified `reasoned` post opens on **"tabs vs spaces"** — a flame-war
topic that co-occurs with hot takes throughout the training data. A model with
too little data to learn that a *stated outcome* ("we adopted Black and the
bikeshedding disappeared") makes a post `reasoned` would instead key on the
inflammatory topic vocabulary.

**Honest limitation:** with only **one** `reasoned` and **two** `noise` test
points, Pattern 2 is a hypothesis *consistent with* the single data point, not a
measured effect. It cannot be separated from Pattern 1 on this test set — both
predict the same wrong answer here. Confirming it needs a larger, varied test
set (e.g. several `reasoned` posts on *neutral* topics: if those are classified
correctly while flame-war-topic `reasoned` posts fail, Pattern 2 is real).

## Root cause

**Data starvation.** Training had **21 examples across 3 classes** (~7 each).
Cross-entropy loss is minimized cheaply by collapsing onto a single class rather
than learning three decision boundaries, because there is not enough signal to
separate features that genuinely distinguish the classes. The model minimized
loss by *memorizing a constant*, not by *learning the concept*.

### Why `hot_take` specifically (not `reasoned` or `noise`)?

A plausible explanation is that `hot_take` examples carry the strongest, most
consistent lexical markers ("always", "never", "garbage", "pointless",
insults), giving that class the lowest-loss constant to settle on. **This is not
confirmed** — determining which class a collapse lands on would require
re-running with multiple random seeds and observing whether the collapsed class
is stable. With a single run it is reported as a hypothesis only.

## What would change the outcome

1. **More data** — the 200+ target (~67/class) is the primary lever.
2. **Class weighting / balanced sampling** to penalize the constant-prediction
   shortcut.
3. **A larger held-out test set** so per-class metrics aren't computed on 1–2
   examples (and so Pattern 2 becomes testable).
4. **Multiple seeds** to confirm the collapse and which class it targets.

## Bottom line

The systematic error is a **single-class collapse driven by data starvation**,
not a nuanced confusion between specific categories. Everything else — the
topic-keyword hypothesis, the choice of `hot_take` as the collapse target — is
secondary and currently under-determined by a 5-example test set. The fix is
data, not architecture.
