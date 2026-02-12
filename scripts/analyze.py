"""
Results analysis and visualization.

Usage:
    python scripts/analyze.py [--results-dir results/]

Generates all plots and summary tables from experiment results.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import numpy as np
import seaborn as sns

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from llm_meditation.utils import get_project_root

logger = logging.getLogger("llm_meditation")


def load_results(results_dir: Path) -> dict:
    """Load all experiment results from the results directory."""
    results = {}

    # Try combined results file first
    combined = results_dir / "all_results.json"
    if combined.exists():
        with open(combined) as f:
            return json.load(f)

    # Otherwise load per-condition
    for condition_dir in results_dir.iterdir():
        if not condition_dir.is_dir():
            continue
        condition = condition_dir.name
        results[condition] = {}
        for result_file in condition_dir.glob("*.json"):
            eval_name = result_file.stem
            with open(result_file) as f:
                results[condition][eval_name] = json.load(f)

    return results


def plot_drift_trajectories(results: dict, output_dir: Path) -> None:
    """
    4-panel plot of Assistant Axis projection trajectories per domain.

    X-axis: conversation turn (1-20)
    Y-axis: mean Assistant Axis projection
    Three lines per panel: baseline, capping, meditation
    """
    domains = ["therapy", "philosophy", "creative_writing", "coding"]
    conditions = ["baseline", "capping", "meditation"]
    colors = {"baseline": "#e74c3c", "capping": "#3498db", "meditation": "#2ecc71"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax_idx, domain in enumerate(domains):
        ax = axes[ax_idx]

        for condition in conditions:
            drift_data = results.get(condition, {}).get("drift", {})
            domain_data = drift_data.get("domains", {}).get(domain, {})
            turn_stats = domain_data.get("turn_stats", {})

            if not turn_stats:
                continue

            turns = sorted(int(t) for t in turn_stats.keys())
            means = [turn_stats[str(t)]["mean"] for t in turns]
            stds = [turn_stats[str(t)]["std"] for t in turns]

            means = np.array(means)
            stds = np.array(stds)

            ax.plot(turns, means, label=condition, color=colors[condition], linewidth=2)
            ax.fill_between(
                turns, means - stds, means + stds,
                alpha=0.15, color=colors[condition],
            )

        # Drift threshold line
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="Drift threshold")

        ax.set_title(domain.replace("_", " ").title(), fontsize=14)
        ax.set_xlabel("Conversation Turn")
        ax.set_ylabel("Assistant Axis Projection")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Multi-Turn Persona Drift Trajectories", fontsize=16, fontweight="bold")
    plt.tight_layout()

    out_path = output_dir / "drift_trajectories.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_sycophancy_comparison(results: dict, output_dir: Path) -> None:
    """Bar chart of sycophancy flip rate by condition with error bars."""
    conditions = ["baseline", "capping", "meditation"]
    flip_rates = []
    labels = []

    for condition in conditions:
        syc_data = results.get(condition, {}).get("sycophancy", {})
        rate = syc_data.get("flip_rate", 0)
        flip_rates.append(rate)
        labels.append(condition.title())

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.bar(labels, flip_rates, color=colors, edgecolor="white", linewidth=2)

    # Add value labels
    for bar, rate in zip(bars, flip_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{rate:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_ylabel("Flip Rate", fontsize=13)
    ax.set_title("Sycophancy Evaluation: Answer Flip Rate", fontsize=15, fontweight="bold")
    ax.set_ylim(0, max(flip_rates) * 1.3 if flip_rates else 1)
    ax.grid(axis="y", alpha=0.3)

    out_path = output_dir / "sycophancy_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_safety_capability_pareto(results: dict, output_dir: Path) -> None:
    """
    Scatter plot: x = capability score, y = safety score (1 - drift rate).
    One point per condition, annotated.
    """
    conditions = ["baseline", "capping", "meditation"]
    colors = {"baseline": "#e74c3c", "capping": "#3498db", "meditation": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(8, 6))

    for condition in conditions:
        # Capability score (average of gsm8k and ifeval)
        cap_data = results.get(condition, {}).get("capabilities", {})
        cap_score = cap_data.get("combined_score", 0.5)

        # Safety score: 1 - inappropriate rate from drift eval
        drift_data = results.get(condition, {}).get("drift", {})
        # Approximate safety from projection trajectory
        therapy_data = drift_data.get("domains", {}).get("therapy", {})
        milestones = therapy_data.get("milestone_projections", {})
        # Use turn_20 projection as safety proxy (higher = safer)
        safety_proxy = milestones.get("turn_20", 0.5)
        safety_score = max(0, min(1, (safety_proxy + 1) / 2))  # normalize to [0,1]

        ax.scatter(cap_score, safety_score, s=200, c=colors[condition],
                   edgecolors="black", linewidth=2, zorder=5)
        ax.annotate(
            condition.title(),
            (cap_score, safety_score),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlabel("Capability Score (GSM8K + IFEval avg)", fontsize=13)
    ax.set_ylabel("Safety Score (Drift Resistance)", fontsize=13)
    ax.set_title("Safety vs. Capability Pareto", fontsize=15, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Add ideal corner indicator
    ax.annotate("Ideal →", (0.85, 0.85), fontsize=10, alpha=0.5,
                ha="center", style="italic")

    out_path = output_dir / "safety_capability_pareto.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_meditation_triggers(results: dict, output_dir: Path) -> None:
    """Histogram: at which turns does meditation fire, by domain."""
    med_data = results.get("meditation", {}).get("drift", {})
    domains = med_data.get("domains", {})

    fig, ax = plt.subplots(figsize=(10, 6))
    domain_colors = {
        "therapy": "#e74c3c",
        "philosophy": "#9b59b6",
        "creative_writing": "#f39c12",
        "coding": "#2ecc71",
    }

    all_turns = {}
    for domain, data in domains.items():
        turns = data.get("meditation_turns", [])
        all_turns[domain] = turns

    if not any(all_turns.values()):
        # No meditation data — create placeholder
        ax.text(0.5, 0.5, "No meditation trigger data available",
                transform=ax.transAxes, ha="center", fontsize=14, alpha=0.5)
    else:
        bins = np.arange(0.5, 21.5, 1)
        for domain, turns in all_turns.items():
            if turns:
                ax.hist(
                    turns, bins=bins, alpha=0.6,
                    label=domain.replace("_", " ").title(),
                    color=domain_colors.get(domain, "gray"),
                )

    ax.set_xlabel("Conversation Turn", fontsize=13)
    ax.set_ylabel("Meditation Trigger Count", fontsize=13)
    ax.set_title("When Does Meditation Fire?", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    out_path = output_dir / "meditation_triggers.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def generate_feature_analysis(results: dict, output_dir: Path) -> None:
    """
    Analyze the most common features flagged during meditation events.

    Produces a markdown table of top features with descriptions.
    """
    lines = ["# Feature Analysis\n"]
    lines.append("Most common SAE features flagged during meditation events.\n")

    # This would be populated from actual experiment data
    # For now, create a template
    lines.append("| Rank | Feature Index | Description | Frequency | Mean Activation |")
    lines.append("|------|--------------|-------------|-----------|-----------------|")

    # Extract from meditation results if available
    med_data = results.get("meditation", {}).get("drift", {})
    if not med_data:
        lines.append("| — | — | No meditation data available | — | — |")
    else:
        lines.append("| 1 | TBD | Run experiments to populate | — | — |")

    lines.append("\n## Qualitative Analysis\n")
    lines.append("*Run the full experiment pipeline to generate feature analysis.*\n")

    out_path = output_dir / "feature_analysis.md"
    out_path.write_text("\n".join(lines))
    logger.info(f"Saved: {out_path}")


def generate_summary_table(results: dict, output_dir: Path) -> None:
    """Generate a markdown summary table of all conditions × metrics."""
    lines = ["# Experiment Summary\n"]

    # Header
    lines.append("| Metric | Baseline | Capping | Meditation |")
    lines.append("|--------|----------|---------|------------|")

    # Sycophancy flip rate
    row = "| Sycophancy flip rate"
    for condition in ["baseline", "capping", "meditation"]:
        rate = results.get(condition, {}).get("sycophancy", {}).get("flip_rate", "—")
        if isinstance(rate, float):
            row += f" | {rate:.1%}"
        else:
            row += f" | {rate}"
    row += " |"
    lines.append(row)

    # GSM8K accuracy
    row = "| GSM8K accuracy"
    for condition in ["baseline", "capping", "meditation"]:
        acc = results.get(condition, {}).get("capabilities", {}).get("gsm8k", {}).get("accuracy", "—")
        if isinstance(acc, float):
            row += f" | {acc:.1%}"
        else:
            row += f" | {acc}"
    row += " |"
    lines.append(row)

    # IFEval accuracy
    row = "| IFEval accuracy"
    for condition in ["baseline", "capping", "meditation"]:
        acc = results.get(condition, {}).get("capabilities", {}).get("ifeval", {}).get("accuracy", "—")
        if isinstance(acc, float):
            row += f" | {acc:.1%}"
        else:
            row += f" | {acc}"
    row += " |"
    lines.append(row)

    # Drift metrics per domain
    for domain in ["therapy", "philosophy", "creative_writing", "coding"]:
        row = f"| {domain.replace('_', ' ').title()} drift (turn 20)"
        for condition in ["baseline", "capping", "meditation"]:
            drift_data = results.get(condition, {}).get("drift", {})
            domain_data = drift_data.get("domains", {}).get(domain, {})
            milestones = domain_data.get("milestone_projections", {})
            val = milestones.get("turn_20", "—")
            if isinstance(val, float):
                row += f" | {val:.3f}"
            else:
                row += f" | {val}"
        row += " |"
        lines.append(row)

    # Meditation trigger count
    row = "| Meditation triggers (total)"
    for condition in ["baseline", "capping", "meditation"]:
        drift_data = results.get(condition, {}).get("drift", {})
        total_med = sum(
            d.get("total_meditations", 0)
            for d in drift_data.get("domains", {}).values()
        )
        row += f" | {total_med}"
    row += " |"
    lines.append(row)

    out_path = output_dir / "summary_table.md"
    out_path.write_text("\n".join(lines))
    logger.info(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM Meditation experiment results")
    parser.add_argument("--results-dir", default="results/", help="Results directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results_dir = get_project_root() / args.results_dir
    output_dir = results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run experiments first with: python scripts/run_eval.py")
        return

    results = load_results(results_dir)

    if not results:
        print("No results found. Run experiments first.")
        return

    print(f"Loaded results for conditions: {list(results.keys())}")
    print(f"Output directory: {output_dir}")

    # Generate all plots and tables
    plot_drift_trajectories(results, output_dir)
    plot_sycophancy_comparison(results, output_dir)
    plot_safety_capability_pareto(results, output_dir)
    plot_meditation_triggers(results, output_dir)
    generate_feature_analysis(results, output_dir)
    generate_summary_table(results, output_dir)

    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
