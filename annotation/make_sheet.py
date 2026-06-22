"""Generate a shuffled, label-free annotation sheet for a second annotator.

Stretch feature (planning.md §8.3 — inter-annotator reliability).

Usage:
    python make_sheet.py [SOURCE_CSV] [OUT_CSV]

Defaults: SOURCE_CSV=../takemeter_sample.csv, OUT_CSV=annotation_sheet.csv

The output has columns `id, text, label` with `label` left blank. Hand it to a
second person along with labeling_guide.md; they fill `label` independently
(without seeing your labels). Shuffling avoids order bias (all `reasoned`
grouped together, etc.). `id` lets compute_agreement.py align the two label
sets regardless of row order.
"""
import sys

import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "../takemeter_sample.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "annotation_sheet.csv"

df = pd.read_csv(src).reset_index().rename(columns={"index": "id"})
shuffled = df.sample(frac=1, random_state=7).reset_index(drop=True)
shuffled[["id", "text"]].assign(label="").to_csv(out, index=False)

print(
    f"Wrote {out} with {len(shuffled)} rows (label column blank).\n"
    f"Give it to a second annotator with labeling_guide.md, then run:\n"
    f"  python compute_agreement.py {src} {out}"
)
