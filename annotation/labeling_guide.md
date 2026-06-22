# TakeMeter — Labeling Guide (for a second annotator)

Thanks for helping measure inter-annotator reliability. Please label each row in
`annotation_sheet.csv` **independently** — fill the `label` column with exactly
one of `reasoned`, `hot_take`, or `noise`. Don't look at anyone else's labels
first.

## The three labels

**Core rule: judge only what is *written in the post itself*.** Reasoning that
is implied, borrowed from a parent comment, or merely gestured at does **not**
count.

- **`reasoned`** — states a technical opinion **and supports it in the post**
  with a concrete tradeoff, a specific experience, evidence, or a mechanism
  ("X because Y").
  - *"Use a hashmap here — O(1) lookup instead of the O(n) loop you've got."*

- **`hot_take`** — a **strong, absolutist** opinion with **no supporting
  reasoning** in the post (sweeping "always/never", or judgment as insult).
  - *"ORMs are a cancer and anyone who uses them doesn't understand databases."*

- **`noise`** — **no technical opinion at all**: agreement-filler, memes, jokes,
  reactions, "first".
  - *"skill issue"*

## Tie-breakers (apply in order)

1. **Stated, not implied** — if the only reasoning is implied or borrowed, it's
   *not* `reasoned`.
2. **Opinion vs. no-opinion first** — if there's no technical stance at all →
   `noise`, regardless of tone.
3. **Reasoned beats hot_take** — if a real, stated *why* is present, it's
   `reasoned` even if it's also rude or absolute.
4. **Deletion test** — remove the conclusion: if a concrete supporting reason
   remains → `reasoned`; if nothing substantive remains → `hot_take` (opinion
   present) or `noise` (no opinion).

## After labeling

Save the file and return it. We'll run:

```
python compute_agreement.py ../takemeter_sample.csv annotation_sheet.csv
```

to report percentage agreement and Cohen's kappa, and to list where we
disagreed.
