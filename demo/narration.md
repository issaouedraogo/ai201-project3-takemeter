# TakeMeter — Demo Video Narration Script (3–4 min)

**Setup before recording:** Have two things on screen — (1) your Colab notebook
with the `demo_inference.py` cell ready to run, and (2) this repo's `README.md`
evaluation report. Record your screen + voice.

Timings are guides; speak naturally. Total ≈ 3.5 min.

---

### [0:00–0:30] Intro — what TakeMeter is

> "Hi — this is TakeMeter, my project 3 classifier. The idea is to sort short
> technical-discussion comments — the kind you see on Hacker News — by the
> *quality of the opinion* they carry. There are three labels: **reasoned**, a
> substantive opinion that's actually backed by a reason or a tradeoff;
> **hot_take**, a strong absolutist opinion with no supporting reasoning; and
> **noise**, low-content filler like '+1' or memes. I fine-tuned DistilBERT on a
> labeled dataset and compared it against a zero-shot Groq baseline."

---

### [0:30–1:30] Live classification — run the demo cell

*Run the `demo_inference.py` cell. Let the table of predictions + confidence
appear on screen.*

> "Here's the fine-tuned model classifying five posts live. For each one you can
> see the predicted label and the model's confidence — that's the softmax
> probability — next to the true label.
>
> Right away you can see something important: the model predicts **hot_take for
> every single post**. The two genuine hot takes come out correct, but the
> reasoned comment and the two noise comments are all misclassified as hot_take
> too. So even before we look at the metrics, the behavior tells the story — the
> model collapsed onto one class."

---

### [1:30–2:00] One CORRECT prediction, narrated

*Point to the "Agile is just a scam…" row.*

> "Let's look at one it got right. This post — *'Agile is just a scam invented to
> sell certifications, real engineers don't need standups'* — is predicted
> **hot_take**, and that's correct. It's a textbook hot take: a sweeping,
> absolutist dismissal with zero supporting reasoning. But I want to be honest
> about *why* it's right — the model isn't recognizing this as a hot take
> specifically. It labels everything hot_take, so it happens to nail the real
> hot takes. It's right for the wrong reason."

---

### [2:00–2:45] One INCORRECT prediction, narrated

*Point to the "Tabs vs spaces…" row.*

> "Now one it got wrong. This post — *'Tabs vs spaces matters less than just
> running a formatter in CI; we adopted Black and the bikeshedding disappeared
> overnight'* — is actually **reasoned**: it states a concrete action and an
> observed outcome. But the model predicted **hot_take**.
>
> Two things are going on. First, the post opens on 'tabs vs spaces,' a classic
> flame-war topic that shows up with hot takes all over the training data — so
> the model keyed on the inflammatory *topic* instead of the reasoning. Second,
> and more fundamentally: with only about 21 training examples across three
> classes, the model never had enough signal to learn a real boundary, so it
> defaulted to predicting the same class every time."

---

### [2:45–3:30] Evaluation report walkthrough

*Switch to the README evaluation report.*

> "Here's the full evaluation. On the locked 5-example test set, the **zero-shot
> baseline scored 1.00** — it got all five right — while the **fine-tuned model
> scored 0.40**. So fine-tuning was actually a regression of 0.6 here.
>
> The confusion matrix shows exactly why: every row collapses into the hot_take
> column. And the per-class metrics confirm it — reasoned and noise both have
> zero recall, because the model never predicted them once.
>
> The takeaway is the real lesson of the project: **fine-tuning only pays off
> with enough labeled data.** At 30 examples, a pretrained LLM that already
> understands these distinctions just needs them defined in a prompt — and it
> wins easily. To make fine-tuning competitive I'd need to hit the 200+ example
> target, around 70 per class, plus a bigger test set so the metrics aren't
> quantized into 0.20 steps. That's the clear next step."

---

### [3:30–3:40] Close

> "So TakeMeter works end-to-end — dataset, fine-tuning, baseline, and a full
> evaluation — and it surfaced a genuine, well-understood failure mode rather
> than hiding it. Thanks for watching."

---

## Quick reference — numbers to have on screen

| | Baseline | Fine-tuned |
|--|--|--|
| Accuracy | 1.00 | 0.40 |
| Behavior | all 5 correct | predicted `hot_take` for all 5 |

- Correct example to narrate: **"Agile is just a scam…"** (true `hot_take` → pred `hot_take`)
- Incorrect example to narrate: **"Tabs vs spaces…"** (true `reasoned` → pred `hot_take`)
