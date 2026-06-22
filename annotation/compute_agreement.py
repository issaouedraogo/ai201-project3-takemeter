"""Inter-annotator reliability for TakeMeter (stretch feature, planning.md §8.3).

Usage:
    python compute_agreement.py GOLD_CSV ANNOTATOR2_CSV

- GOLD_CSV: the original dataset (text,label) — annotator A (you).
- ANNOTATOR2_CSV: the filled annotation sheet (id,text,label) — annotator B.

Rows are aligned on `id` if both files have it, otherwise on exact `text`.
Reports: percentage agreement, Cohen's kappa, a disagreement list, and the
annotator-vs-annotator confusion matrix.
"""
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    return df


def main(gold_path, other_path):
    a_df, b_df = load(gold_path), load(other_path)
    key = "id" if ("id" in a_df.columns and "id" in b_df.columns) else "text"
    merged = a_df.merge(b_df, on=key, suffixes=("_a", "_b"))
    merged = merged.dropna(subset=["label_a", "label_b"])
    merged = merged[merged["label_b"].astype(str).str.strip() != ""]

    n = len(merged)
    if n == 0:
        print("No overlapping labeled rows found — is annotator B's `label` filled in?")
        return

    a = merged["label_a"].astype(str).str.strip()
    b = merged["label_b"].astype(str).str.strip()
    agree = int((a.values == b.values).sum())
    labels = sorted(set(a) | set(b))
    kappa = cohen_kappa_score(a, b, labels=labels)

    print(f"Compared {n} examples (aligned on '{key}').")
    print(f"Percentage agreement: {agree / n:.1%}  ({agree}/{n})")
    print(f"Cohen's kappa:        {kappa:.3f}")

    dis = merged[a.values != b.values]
    print(f"\nDisagreements ({len(dis)}):")
    if dis.empty:
        print("  (none)")
    else:
        text_col = "text_a" if "text_a" in merged.columns else "text"
        for _, r in dis.iterrows():
            txt = str(r.get(text_col, r[key]))[:64]
            print(f"  A={str(r['label_a']).strip():<9} B={str(r['label_b']).strip():<9} | {txt}")

    print("\nConfusion (rows = annotator A, cols = annotator B):")
    cm = confusion_matrix(a, b, labels=labels)
    print("         " + " ".join(f"{l[:8]:>9}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l[:8]:>8} " + " ".join(f"{cm[i][j]:>9}" for j in range(len(labels))))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compute_agreement.py <gold.csv> <annotator2.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
