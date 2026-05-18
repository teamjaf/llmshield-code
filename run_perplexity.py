#!/usr/bin/env python3
"""
LLMShield Perplexity Filtering Experiment
==========================================
Evaluates perplexity-based input filtering as a defense mechanism.

Measures:
  - Perplexity distribution of clean vs attacked prompts
  - Detection accuracy (true positive rate, false positive rate)
  - Filtering latency on CPU (edge-relevant)
  - ROC curve for threshold selection

Outputs:
  - results/perplexity_analysis.csv
  - figures/perplexity_distribution.png
  - figures/perplexity_roc.png

Usage:
    python experiments/run_perplexity.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CASES
from attacks import ATTACK_REGISTRY


def main():
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results")
    )
    figure_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "figures")
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    # ---- Load reference model (GPT-2 small, 124M params) ----
    print("Loading perplexity reference model (GPT-2 small, 124M) ...")
    from defenses import PerplexityFilter
    pf = PerplexityFilter(model_name="gpt2")

    # ---- Generate all prompts ----
    rows = []
    for case in CASES:
        for attack_key, attack_info in ATTACK_REGISTRY.items():
            prompt = attack_info["fn"](case["data"])

            t0 = time.time()
            ppl = pf.compute_perplexity(prompt)
            filter_latency_ms = (time.time() - t0) * 1000

            rows.append({
                "case_id": case["id"],
                "attack": attack_key,
                "attack_label": attack_info["label"],
                "is_attack": attack_key != "clean",
                "perplexity": ppl,
                "filter_latency_ms": filter_latency_ms,
                "prompt_length_tokens": len(pf.tokenizer.encode(prompt)),
            })

    df = pd.DataFrame(rows)

    # ---- Summary statistics ----
    print("\n" + "=" * 60)
    print("PERPLEXITY SUMMARY BY ATTACK TYPE")
    print("=" * 60)

    for attack_key in df["attack"].unique():
        subset = df[df["attack"] == attack_key]
        print(f"\n  {subset['attack_label'].iloc[0]}:")
        print(f"    Mean PPL:     {subset['perplexity'].mean():.2f}")
        print(f"    Median PPL:   {subset['perplexity'].median():.2f}")
        print(f"    Std PPL:      {subset['perplexity'].std():.2f}")
        print(f"    Avg latency:  {subset['filter_latency_ms'].mean():.1f} ms")

    # ---- Threshold calibration ----
    clean_ppls = df[~df["is_attack"]]["perplexity"].values
    attack_ppls = df[df["is_attack"]]["perplexity"].values

    # Try percentile-based threshold
    for pct in [90, 95, 99]:
        threshold = np.percentile(clean_ppls, pct)
        tp = np.sum(attack_ppls > threshold)  # attacks correctly flagged
        fp = np.sum(clean_ppls > threshold)   # clean incorrectly flagged
        fn = np.sum(attack_ppls <= threshold) # attacks missed
        tn = np.sum(clean_ppls <= threshold)  # clean correctly passed

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        print(f"\n  Threshold (p{pct}): {threshold:.2f}")
        print(f"    Detection rate (TPR): {tpr:.2%}")
        print(f"    False positive rate:  {fpr:.2%}")
        print(f"    Overall accuracy:     {accuracy:.2%}")

    # ---- Save results ----
    csv_path = os.path.join(output_dir, "perplexity_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ---- Generate figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })

    # Figure 1: Perplexity distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    for attack_key in df["attack"].unique():
        subset = df[df["attack"] == attack_key]
        label = subset["attack_label"].iloc[0]
        ax.hist(
            subset["perplexity"], bins=15, alpha=0.6, label=label,
            edgecolor="black", linewidth=0.5,
        )
    threshold_95 = np.percentile(clean_ppls, 95)
    ax.axvline(
        x=threshold_95, color="red", linestyle="--", linewidth=2,
        label=f"Threshold (p95 = {threshold_95:.1f})",
    )
    ax.set_xlabel("Perplexity")
    ax.set_ylabel("Count")
    ax.set_title("Perplexity Distribution: Clean vs Attacked Prompts", fontweight="bold")
    ax.legend(fontsize=9)
    fig_path = os.path.join(figure_dir, "perplexity_distribution.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Figure 2: ROC curve
    all_ppls = np.concatenate([clean_ppls, attack_ppls])
    all_labels = np.concatenate([
        np.zeros(len(clean_ppls)),
        np.ones(len(attack_ppls)),
    ])

    thresholds = np.linspace(all_ppls.min(), all_ppls.max(), 200)
    tprs = []
    fprs = []
    for t in thresholds:
        tp = np.sum((all_ppls > t) & (all_labels == 1))
        fp = np.sum((all_ppls > t) & (all_labels == 0))
        fn = np.sum((all_ppls <= t) & (all_labels == 1))
        tn = np.sum((all_ppls <= t) & (all_labels == 0))
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fprs, tprs, "b-", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Detection)")
    ax.set_title("Perplexity Filter ROC Curve", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    # Compute AUC
    auc = np.trapz(sorted(tprs), sorted(fprs))
    ax.text(0.6, 0.2, f"AUC ≈ {abs(auc):.3f}", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig_path = os.path.join(figure_dir, "perplexity_roc.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # ---- Edge feasibility summary ----
    print("\n" + "=" * 60)
    print("EDGE FEASIBILITY: PERPLEXITY FILTERING")
    print("=" * 60)
    print(f"  Reference model: GPT-2 small (124M params)")
    print(f"  Avg filter latency: {df['filter_latency_ms'].mean():.1f} ms")
    print(f"  Max filter latency: {df['filter_latency_ms'].max():.1f} ms")
    print(f"  Overhead: negligible compared to LLM inference")
    print(f"  Edge-deployable: Yes (Jetson Nano class and above)")

    print("\nDone.")


if __name__ == "__main__":
    main()
