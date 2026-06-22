"""Collect predictions + confidence for calibration (planning.md §8.4).

Paste this into a Colab cell AFTER training (so `model` and `tokenizer` exist),
or run it locally with ./takemeter-model downloaded. It writes
`predictions_with_confidence.csv` with columns: text, true, pred, confidence.

For an HONEST calibration result, point CSV_PATH at a HELD-OUT set the model was
not trained on (ideally a larger test set). Running it on the training data will
look over-confident.
"""
import torch
import torch.nn.functional as F
import pandas as pd

CSV_PATH = "takemeter_sample.csv"   # ← replace with your held-out/test CSV

try:
    model, tokenizer
except NameError:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained("./takemeter-model")
    tokenizer = AutoTokenizer.from_pretrained("./takemeter-model")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
id2label = model.config.id2label

df = pd.read_csv(CSV_PATH)
rows = []
for _, r in df.iterrows():
    inputs = tokenizer(
        r["text"], truncation=True, max_length=256, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)[0]
    conf, idx = probs.max(dim=-1)
    rows.append({
        "text": r["text"],
        "true": r["label"],
        "pred": id2label[idx.item()],
        "confidence": round(conf.item(), 4),
    })

pd.DataFrame(rows).to_csv("predictions_with_confidence.csv", index=False)
print("Wrote predictions_with_confidence.csv — download it and run:")
print("  python calibration.py predictions_with_confidence.csv")
