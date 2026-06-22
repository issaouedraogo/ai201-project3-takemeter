# TakeMeter — Planning

A classifier that sorts short technical-discussion comments by the *quality of
the opinion* they carry: substantive arguments vs. unsupported absolutism vs.
pure filler. This document covers the community, label scheme, data plan,
evaluation, success criteria, and how AI tools fit into the workflow.

---

## 1. Community

**Chosen community: Hacker News comment threads** (with r/programming and
r/ExperiencedDevs as secondary sources of the same discourse).

**Why this community.** HN comment sections on technical posts ("Show HN", a
database benchmark, a "we rewrote X in Y" blog post) attract a deliberately
mixed crowd: senior practitioners who write multi-paragraph tradeoff analyses,
strong-opinion regulars who fire off absolutist one-liners, and drive-by
commenters who leave a meme or a "+1". That mix is exactly what makes the
classification task interesting rather than trivial.

**Why the discourse is varied enough to be interesting.**

- **Full quality spectrum in one thread.** The same "monolith vs. microservices"
  thread contains a careful "we did this at 15 engineers and here's what
  happened" comment *and* "microservices are always the right call, monoliths
  are legacy garbage." The classifier has to separate them on *reasoning*, not
  topic — the topic is identical.
- **Reasoning and assertion look superficially alike.** Both reasoned comments
  and hot takes are confident, jargon-dense, and opinionated. The signal is
  whether a *why* is present, which is a subtle, learnable distinction — not a
  keyword match.
- **High volume and natural class variety.** Long threads reliably produce all
  three classes, so collecting a balanced, real (non-synthetic) dataset is
  feasible.

---

## 2. Labels

Three mutually exclusive labels. **Core rule for all three: classify only what
is *stated in the post itself*.** Reasoning that is merely implied, borrowed
from the parent comment, or gestured at without specifics does **not** count as
reasoning.

### `reasoned`
> A comment that states a technical opinion **and supports it within the post**
> with a concrete tradeoff, a specific personal experience, evidence, or a
> mechanism ("X because Y").

- *"We moved off microservices back to a monolith — the network overhead and
  distributed-tracing cost outweighed the team-autonomy benefit at our
  ~15-engineer scale."*
- *"Use a hashmap here — O(1) lookup instead of the O(n) loop you've got, which
  is why it slows down past ~10k items."*

### `hot_take`
> A comment that states a **strong, absolutist** technical opinion with **no
> supporting reasoning inside the post** — sweeping "always/never" claims, or
> judgment delivered as insult.

- *"ORMs are a cancer and anyone who uses them doesn't understand databases."*
- *"Microservices are always the right call. Monoliths are legacy garbage and
  that's just a fact."*

### `noise`
> A comment that carries **no technical opinion at all**: agreement-filler,
> memes, jokes, reactions, or "first"-style content.

- *"skill issue"*
- *"thanks for coming to my ted talk"*

---

## 3. Hard edge cases

The genuinely ambiguous posts all turn on one question: **is the reasoning
actually in the post, or only implied?** These were found by stress-testing the
definitions with an LLM (see §7); the rules below are the result of that
exercise.

| Ambiguous post | Torn between | Ruling |
|----------------|--------------|--------|
| *"Rewrote it in Rust over the weekend and now it's blazing fast. Skill issue if you can't."* | reasoned / hot_take | **reasoned** — a concrete experience + observed outcome is stated; the insult doesn't erase the *why*. |
| *"ORMs are fine until they aren't."* | reasoned / hot_take | **hot_take** — gestures at a tradeoff but never names it; no stated reasoning. |
| *"Same. Switched to Postgres for the same reason and never looked back."* | reasoned / noise | **noise** — the substance lives in the *parent* comment; this post borrows it. We judge the post alone. |
| *"Anyone who still uses tabs in 2026 lol"* | hot_take / noise | **hot_take** — meme cadence, but there's a clear absolutist judgment. Opinion present → not noise. |
| *"Cool tool but I'd never put this in production."* | reasoned / hot_take | **hot_take** — "I'd never" is an unsupported absolute; the implied tradeoff is never stated. |
| *"Tried it, didn't work for me, ymmv."* | reasoned / noise | **noise** — references a trial but gives zero detail and hedges to nothing; no usable reasoning. |

**Tie-breaking rules applied during annotation:**

1. **Stated-not-implied:** if the only reasoning is implied or borrowed from
   context, it is *not* `reasoned`.
2. **Opinion vs. no-opinion first:** decide whether *any* technical stance is
   present. If none → `noise`, regardless of tone.
3. **Reasoned beats hot_take:** if a real, stated *why* is present, the comment
   is `reasoned` even if it's also rude or absolute.
4. **The deletion test:** remove the conclusion — if a concrete supporting
   reason remains in the post, it's `reasoned`; if nothing substantive remains,
   it's `hot_take` (opinion present) or `noise` (no opinion).

---

## 4. Data collection plan

**Where.** Hacker News via the Algolia HN Search API and direct thread reading,
supplemented by r/programming and r/ExperiencedDevs comment threads. Source
threads will be opinion-generating discussions (architecture debates,
language/tooling choices, "we migrated X" posts) so all three classes appear
naturally.

**How many per label.** Target **200+ total, balanced** at ~67 per label
(reasoned / hot_take / noise). Balance is deliberate so neither training nor the
evaluation metrics are skewed by class frequency.

**Collection order.** Collect `reasoned` first — it is the rarest and hardest to
find, so it sets the ceiling. `hot_take` and `noise` are abundant and easy to
over-collect, so they're gathered to match the `reasoned` count rather than the
other way around.

**If a label is underrepresented after 200 examples** (in practice, `reasoned`):

1. **Targeted sourcing** — mine threads that skew substantive (e.g.
   r/ExperiencedDevs, post-mortems, long technical comments) rather than
   front-page flame wars.
2. **Never fabricate dataset rows.** Synthetic comments would teach the model an
   artificial style. If real `reasoned` examples can't reach parity, I will
   **document the imbalance** and handle it at training time with **class
   weighting / a stratified split**, and report per-class metrics so the
   imbalance is visible rather than hidden by accuracy.
3. **Cap the majority classes** to the `reasoned` count to keep the set balanced
   instead of inflating totals with easy `noise`.

---

## 5. Evaluation metrics

**Accuracy is not enough.** With three classes and a tool whose entire point is
finding the *rare, valuable* class (`reasoned`), a model can score decent
accuracy by leaning on the easy classes while failing at the one that matters —
exactly the failure mode the first run showed (it predicted `hot_take` for
everything and still got 40%). So:

| Metric | Why it's needed for *this* task |
|--------|--------------------------------|
| **Per-class precision & recall** | The classes have different costs. I need to see each class on its own, not an average that hides a collapsed class. |
| **`reasoned` recall** | The tool's job is surfacing good comments; **missing** a reasoned comment (burying it as noise) is the most expensive error. This is the headline metric. |
| **`reasoned` precision** | If the "best comments" feed is full of misclassified hot takes, users stop trusting it. Promoting junk is the second-most expensive error. |
| **Macro-averaged F1** | Averages F1 equally across classes, so a model can't win by acing the majority class. Robust to any residual class imbalance. |
| **Confusion matrix** | Shows *which* confusions happen. `reasoned`↔`hot_take` confusion (same topic, different quality) is the meaningful failure; `reasoned`↔`noise` is a different, worse failure. The matrix distinguishes them. |
| **Comparison vs. zero-shot baseline** | The Groq LLM baseline is the "do nothing / just call an API" option. Fine-tuning is only justified if it beats or cost-effectively matches the baseline on macro-F1. |

**Primary metric: macro-F1.** Secondary, tracked explicitly: `reasoned` recall
and `reasoned` precision. The confusion matrix is the diagnostic.

---

## 6. Definition of success

**Genuinely useful (the goal):**

- **Macro-F1 ≥ 0.80** on a held-out test set of **≥ 30 examples (≥ 10 per
  class)**.
- **`reasoned` recall ≥ 0.85** — it surfaces almost all the good comments.
- **`reasoned` precision ≥ 0.75** — the "best comments" feed is mostly real.
- **Beats the zero-shot baseline on macro-F1** (or matches it within 0.02 at
  meaningfully lower inference cost, which is fine-tuning's real advantage).

**"Good enough" to deploy in a real community tool (the floor):**

- **Macro-F1 ≥ 0.75** **and** **`reasoned` recall ≥ 0.80**, **and**
- **`reasoned`↔`hot_take` confusion < 15%** of `hot_take` test examples (the
  tool must not routinely mislabel unsupported takes as insightful).

Below the floor, the zero-shot LLM is the better classifier and TakeMeter
shouldn't ship as a fine-tuned model.

### Are these criteria specific enough to objectively check?

Yes. Each is a named metric with a numeric threshold and a defined test set
(≥10 per class), computable directly from `evaluation_results.json` and the
confusion matrix. At the end I can state, per criterion, hit or miss — no
judgment call required. The one dependency is a test set large enough that the
numbers aren't quantized into uselessness (the 5-example test set in the first
run made every metric move in 0.20 steps); the ≥30-example / ≥10-per-class
minimum is set specifically to make these thresholds measurable.

---

## 7. AI Tool Plan

This is a dataset-and-evaluation project, not an implementation project — there
is no application code to generate. AI tools help in three specific places.

### 7.1 Label stress-testing — **done during planning**

I gave an LLM the label definitions and the edge-case description and asked it to
generate 8 posts sitting on the boundary between two labels. The exercise
**broke the original definitions**: several posts ("Same, switched to Postgres
for the same reason", "ORMs are fine until they aren't", "Tried it, didn't work,
ymmv") couldn't be classified cleanly because the definitions never said whether
*implied* or *borrowed* reasoning counts.

**Fix applied before any annotation:** added the **"classify only what is stated
in the post itself"** core rule and the four tie-breaking rules in §3. The
boundary posts the LLM produced are now the worked examples in the edge-case
table. This is the intended use — tighten definitions *before* annotating 200
examples, not after.

### 7.2 Annotation assistance

**Decision: yes, with human review of every label.** I'll use an LLM (Groq
`llama-3.3-70b-versatile`, the same model as the baseline) to **pre-label
batches** of collected comments, then review and correct each one myself —
faster than labeling cold, without outsourcing the ground truth.

**Tracking for disclosure:** the dataset will carry a `source` column recording
whether each row was `pre-labeled-llm-then-reviewed` or `hand-labeled`, and the
AI-usage section will report how many rows were pre-labeled and the
human-correction rate (how often I overrode the LLM). To avoid circularity, the
pre-labeling LLM is treated as a typing aid only — it never sets a final label,
and its agreement with my labels is **not** reported as a result.

### 7.3 Failure analysis

After evaluation I'll give the LLM the **full list of misclassified test
examples** (text, true label, predicted label) and ask it to identify patterns —
e.g. "reasoned comments under ~15 words get read as noise" or
"`reasoned`↔`hot_take` errors cluster on rude-but-justified posts."

**What I'll look for:** systematic confusions tied to length, tone, or topic;
whether errors concentrate in one class; and whether any errors are actually
*mislabels in my own data* surfaced by the model.

**How I'll verify:** every proposed pattern is checked by hand against the actual
rows before it goes in the write-up — I confirm the cited examples really show
the pattern and count how many of the errors it explains. The LLM proposes; the
confusion matrix and a manual recount decide. Unverified patterns don't ship.
