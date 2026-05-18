#!/usr/bin/env python3
"""
LLMShield Main Experiment
==========================
Runs all attack × defense × model combinations and produces:
  - results/full_results.csv           (per-case results)
  - results/summary_table.csv          (aggregated ASR / accuracy / latency)
  - figures/asr_heatmap.png            (attack × defense heatmap)
  - figures/latency_comparison.png     (latency bar chart)
  - figures/defense_effectiveness.png  (grouped bar chart)

Usage:
    python experiments/run_attack_defense.py [--models mistral-7b-q4,tinyllama-1.1b-q4]
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CASES, MODEL_REGISTRY, load_model_and_tokenizer,
    run_prompt, measure_memory, get_device,
)
from attacks import ATTACK_REGISTRY
from defenses import DEFENSE_REGISTRY
from evaluation import evaluate_output, extract_diagnosis


def run_experiment(model_keys: list):
    """Run the full experiment matrix."""

    all_rows = []

    for model_key in model_keys:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_key} ({MODEL_REGISTRY[model_key]['tier']})")
        print(f"{'='*70}")

        try:
            model, tokenizer, load_time, model_size_mb = load_model_and_tokenizer(model_key)
        except Exception as e:
            print(f"[SKIP] Could not load {model_key}: {e}")
            continue

        mem_after_load = measure_memory()

        # ----- iterate: attack × defense × case -----
        for attack_key, attack_info in ATTACK_REGISTRY.items():
            if attack_key == "clean":
                # Clean has no defense dimension — run once as baseline
                defense_keys = ["none"]
            else:
                defense_keys = list(DEFENSE_REGISTRY.keys())

            for defense_key in defense_keys:
                defense_info = DEFENSE_REGISTRY[defense_key]
                is_attack = (attack_key != "clean")

                # Build the prompt function
                base_prompt_fn = attack_info["fn"]
                if defense_key != "none" and is_attack:
                    prompt_fn = defense_info["apply"](base_prompt_fn)
                else:
                    prompt_fn = base_prompt_fn

                combo_label = f"{attack_info['label']} + {defense_info['label']}"
                print(f"\n  [{combo_label}]")

                latencies = []
                peak_gpus = []

                for case in CASES:
                    prompt = prompt_fn(case["data"])
                    output, latency_ms, peak_gpu = run_prompt(
                        model, tokenizer, prompt, max_new_tokens=100
                    )
                    latencies.append(latency_ms)
                    peak_gpus.append(peak_gpu)

                    eval_result = evaluate_output(
                        output, case["data"], is_attack=is_attack
                    )

                    row = {
                        "model": model_key,
                        "model_tier": MODEL_REGISTRY[model_key]["tier"],
                        "model_params": MODEL_REGISTRY[model_key]["params"],
                        "model_size_mb": model_size_mb,
                        "model_load_time_s": load_time,
                        "attack": attack_key,
                        "attack_label": attack_info["label"],
                        "defense": defense_key,
                        "defense_label": defense_info["label"],
                        "case_id": case["id"],
                        "expected": case["expected"],
                        "raw_output": output,
                        "extracted_diagnosis": eval_result["diagnosis"],
                        "clean_correct": eval_result["correct"],
                        "attack_success": eval_result["attack_success"],
                        "latency_ms": latency_ms,
                        "peak_gpu_mb": peak_gpu,
                        "ram_mb": mem_after_load.get("ram_mb", 0),
                    }
                    all_rows.append(row)

                avg_lat = np.mean(latencies)
                print(f"    Avg latency: {avg_lat:.0f} ms | "
                      f"Peak GPU: {max(peak_gpus):.0f} MB")

        # Free model memory before loading next
        del model, tokenizer
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    return pd.DataFrame(all_rows)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-case results into a summary table."""
    rows = []

    for (model, attack, defense), group in df.groupby(
        ["model", "attack", "defense"]
    ):
        n = len(group)
        model_info = MODEL_REGISTRY.get(model, {})

        row = {
            "Model": model,
            "Tier": model_info.get("tier", ""),
            "Params": model_info.get("params", ""),
            "Attack": group["attack_label"].iloc[0],
            "Defense": group["defense_label"].iloc[0],
            "N": n,
        }

        if attack == "clean":
            correct = group["clean_correct"].sum()
            row["Clean Accuracy (%)"] = round(100 * correct / n, 1)
            row["ASR (%)"] = "—"
        else:
            successes = group["attack_success"].sum()
            row["ASR (%)"] = round(100 * successes / n, 1)
            row["Clean Accuracy (%)"] = "—"

        row["Avg Latency (ms)"] = round(group["latency_ms"].mean(), 1)
        row["Std Latency (ms)"] = round(group["latency_ms"].std(), 1)
        row["Peak GPU (MB)"] = round(group["peak_gpu_mb"].max(), 1)
        row["Model Size (MB)"] = round(group["model_size_mb"].iloc[0], 1)

        rows.append(row)

    return pd.DataFrame(rows)


def generate_figures(summary_df: pd.DataFrame, output_dir: str):
    """Generate publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })

    # --- Figure 1: ASR Heatmap (Attack × Defense) ---
    attack_rows = summary_df[summary_df["ASR (%)"] != "—"].copy()
    if not attack_rows.empty:
        attack_rows["ASR (%)"] = attack_rows["ASR (%)"].astype(float)

        for model_key in attack_rows["Model"].unique():
            subset = attack_rows[attack_rows["Model"] == model_key]
            pivot = subset.pivot_table(
                index="Attack", columns="Defense", values="ASR (%)", aggfunc="first"
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(
                pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
                vmin=0, vmax=100, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Attack Success Rate (%)"},
            )
            ax.set_title(f"ASR Heatmap — {model_key}", fontweight="bold")
            ax.set_ylabel("Attack Type")
            ax.set_xlabel("Defense")
            fig.savefig(os.path.join(output_dir, f"asr_heatmap_{model_key}.png"))
            plt.close(fig)
            print(f"  Saved: asr_heatmap_{model_key}.png")

    # --- Figure 2: Defense Effectiveness (grouped bar) ---
    if not attack_rows.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        attacks = attack_rows["Attack"].unique()
        defenses = attack_rows["Defense"].unique()
        x = np.arange(len(attacks))
        width = 0.8 / len(defenses)

        for i, defense in enumerate(defenses):
            vals = []
            for attack in attacks:
                row = attack_rows[
                    (attack_rows["Attack"] == attack) &
                    (attack_rows["Defense"] == defense)
                ]
                vals.append(row["ASR (%)"].values[0] if len(row) > 0 else 0)
            ax.bar(x + i * width, vals, width, label=defense)

        ax.set_xlabel("Attack Type")
        ax.set_ylabel("ASR (%)")
        ax.set_title("Defense Effectiveness Against Prompt Attacks", fontweight="bold")
        ax.set_xticks(x + width * (len(defenses) - 1) / 2)
        ax.set_xticklabels(attacks, rotation=15, ha="right")
        ax.legend(title="Defense")
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        fig.savefig(os.path.join(output_dir, "defense_effectiveness.png"))
        plt.close(fig)
        print("  Saved: defense_effectiveness.png")

    # --- Figure 3: Latency Comparison ---
    fig, ax = plt.subplots(figsize=(8, 4))
    models = summary_df["Model"].unique()
    latencies = [
        summary_df[summary_df["Model"] == m]["Avg Latency (ms)"].mean()
        for m in models
    ]
    bars = ax.bar(models, latencies, color=["#2196F3", "#FF9800"][:len(models)])
    ax.set_ylabel("Avg Inference Latency (ms)")
    ax.set_title("Inference Latency by Model", fontweight="bold")
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f} ms", ha="center", va="bottom", fontsize=10)
    fig.savefig(os.path.join(output_dir, "latency_comparison.png"))
    plt.close(fig)
    print("  Saved: latency_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="LLMShield Experiment Runner")
    parser.add_argument(
        "--models",
        default="mistral-7b-q4",
        help="Comma-separated model keys from MODEL_REGISTRY "
             "(default: mistral-7b-q4)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "results"),
    )
    parser.add_argument(
        "--figure-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "figures"),
    )
    args = parser.parse_args()

    model_keys = [k.strip() for k in args.models.split(",")]
    output_dir = os.path.abspath(args.output_dir)
    figure_dir = os.path.abspath(args.figure_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("LLMShield Experiment")
    print(f"  Models: {model_keys}")
    print(f"  Cases: {len(CASES)}")
    print(f"  Attacks: {list(ATTACK_REGISTRY.keys())}")
    print(f"  Defenses: {list(DEFENSE_REGISTRY.keys())}")
    print(f"  Device: {get_device()}")
    print()

    # Run experiments
    df = run_experiment(model_keys)

    # Save full results
    full_path = os.path.join(output_dir, "full_results.csv")
    df.to_csv(full_path, index=False)
    print(f"\nSaved full results: {full_path} ({len(df)} rows)")

    if df.empty:
        print("\n[WARNING] No results collected — all models failed to load. Exiting.")
        return

    # Compute and save summary
    summary = compute_summary(df)
    summary_path = os.path.join(output_dir, "summary_table.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(summary.to_string(index=False))

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(summary, figure_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
