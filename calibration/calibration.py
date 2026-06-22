"""Confidence calibration for TakeMeter (stretch feature, planning.md §8.4).

Usage:
    python calibration.py predictions_with_confidence.csv [n_bins]

Input CSV columns: text, true, pred, confidence (produced by
collect_predictions.py). Reports per-confidence-bin accuracy and the Expected
Calibration Error (ECE) — the gap between how confident the model is and how
often it's right.

NOTE: needs a reasonably large set (50+ predictions) to mean anything. On the
current 5-example test set this is illustrative only.
"""
import sys

import pandas as pd


def main(path, n_bins=5):
    df = pd.read_csv(path)
    df["correct"] = (
        df["pred"].astype(str).str.strip() == df["true"].astype(str).str.strip()
    ).astype(int)
    n = len(df)

    print(f"{n} predictions. Overall accuracy: {df['correct'].mean():.1%}")
    if n < 50:
        print("⚠️  Fewer than 50 predictions — the calibration below is NOT "
              "statistically meaningful; treat it as illustrative.")

    edges = [i / n_bins for i in range(n_bins + 1)]
    print(f"\n{'conf bin':<14}{'n':>5}{'avg conf':>11}{'accuracy':>11}{'gap':>9}")
    print("-" * 50)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (df["confidence"] >= lo) & (df["confidence"] <= hi)
        else:
            mask = (df["confidence"] > lo) & (df["confidence"] <= hi)
        b = df[mask]
        label = f"({lo:.1f}, {hi:.1f}]"
        if b.empty:
            print(f"{label:<14}{0:>5}{'—':>11}{'—':>11}{'—':>9}")
            continue
        avg_conf, acc = b["confidence"].mean(), b["correct"].mean()
        gap = abs(avg_conf - acc)
        ece += len(b) / n * gap
        print(f"{label:<14}{len(b):>5}{avg_conf:>11.1%}{acc:>11.1%}{gap:>9.1%}")

    print("-" * 50)
    print(f"Expected Calibration Error (ECE): {ece:.3f}")
    print("Lower ECE = better calibrated. A ~90%-confidence bin should be "
          "~90% accurate.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibration.py <predictions_with_confidence.csv> [n_bins]")
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
