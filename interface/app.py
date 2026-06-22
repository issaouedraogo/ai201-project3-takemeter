"""TakeMeter — deployed interface (stretch feature, planning.md §8.2).

A Gradio app that accepts a technical-discussion comment and shows the predicted
label plus the model's confidence over all three classes.

The fine-tuned model lives in the Colab runtime at ./takemeter-model, so the
easiest way to run this is inside Colab (see the repo README for steps):

    !pip install -q gradio
    # ...after training has produced ./takemeter-model...
    !python interface/app.py        # or paste this file into a cell

To run locally instead, download the ./takemeter-model folder from Colab and
set MODEL_PATH below (or the TAKEMETER_MODEL env var) to its path.
"""
import os

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = os.environ.get("TAKEMETER_MODEL", "./takemeter-model")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
id2label = model.config.id2label


def classify(post: str):
    """Return {label: confidence} over all classes for a single post."""
    if not post or not post.strip():
        return {}
    inputs = tokenizer(
        post, truncation=True, max_length=256, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)[0]
    return {id2label[i]: float(probs[i]) for i in range(len(probs))}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        lines=4,
        label="Paste a technical-discussion comment",
        placeholder="e.g. 'ORMs are a cancer and anyone who uses them doesn't understand databases.'",
    ),
    outputs=gr.Label(num_top_classes=3, label="Predicted label + confidence"),
    title="TakeMeter",
    description=(
        "Classifies a tech-discussion comment as reasoned / hot_take / noise "
        "(fine-tuned DistilBERT). Confidence is the softmax probability."
    ),
    examples=[
        ["We moved off microservices back to a monolith — the network overhead outweighed the team-autonomy benefit at our ~15-engineer scale."],
        ["ORMs are a cancer and anyone who uses them doesn't understand databases."],
        ["skill issue"],
    ],
    allow_flagging="never",
)

if __name__ == "__main__":
    # share=True prints a public URL — needed when running inside Colab.
    demo.launch(share=True)
