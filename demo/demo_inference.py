# =====================================================================
# TakeMeter — DEMO inference cell
# Paste this into a NEW cell at the BOTTOM of your Colab notebook,
# AFTER training has run (so `model` and `tokenizer` exist), then run it.
# It classifies the demo posts and prints predicted label + confidence,
# which is what the demo video needs to show on screen.
# =====================================================================
import torch
import torch.nn.functional as F

# Reuse the in-memory model/tokenizer from training.
# Falls back to the saved checkpoint if the variables aren't defined.
try:
    model, tokenizer
except NameError:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained("./takemeter-model")
    tokenizer = AutoTokenizer.from_pretrained("./takemeter-model")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
id2label = model.config.id2label

# 5 posts = the locked test set (true labels shown for narration).
demo_posts = [
    ("Agile is just a scam invented to sell certifications. Real engineers don't need standups.", "hot_take"),
    ("TypeScript is pointless. Types are training wheels for people who can't write JavaScript.", "hot_take"),
    ("Tabs vs spaces matters less than just running a formatter in CI — we adopted Black and the bikeshedding thread disappeared overnight.", "reasoned"),
    ("+1", "noise"),
    ("same tbh", "noise"),
]

print(f"{'PREDICTED':<10} {'CONF':>6}  {'TRUE':<9} {'✓/✗':<3}  POST")
print("-" * 100)
for text, true_label in demo_posts:
    inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)[0]
    conf, idx = probs.max(dim=-1)
    pred = id2label[idx.item()]
    mark = "✓" if pred == true_label else "✗"
    print(f"{pred:<10} {conf.item():>6.1%}  {true_label:<9} {mark:<3}  {text[:64]}")
