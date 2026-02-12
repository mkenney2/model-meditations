"""
Main experiment runner.

Usage:
    python scripts/run_eval.py [--config configs/default.yaml] [--condition baseline|capping|meditation] [--eval sycophancy|drift|capabilities|all]

Runs evaluations under specified conditions and saves results.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from llm_meditation.utils import load_config, setup_logging, get_project_root
from llm_meditation.model import MeditatingModel

logger = logging.getLogger("llm_meditation")


def run_evaluation(config: dict, condition: str, eval_type: str, max_items: int | None = None) -> dict:
    """Run a specific evaluation under a given condition."""

    # Load model
    model = MeditatingModel(config)

    # Configure condition
    capping_handle = None
    if condition == "baseline":
        # Disable meditation
        model.always_pulse_check = False
        model.drift_threshold_pct = -999  # never trigger
    elif condition == "capping":
        # Enable capping, disable meditation
        model.always_pulse_check = False
        model.drift_threshold_pct = -999
        # Apply activation capping hook
        from eval.sycophancy import apply_activation_capping
        from llm_meditation.calibration import get_percentile
        if model.calibration is not None:
            threshold = model.calibration.percentiles.get(25, model.calibration.mean)
        else:
            threshold = 0.0
        capping_handle = apply_activation_capping(model, threshold)
    elif condition == "meditation":
        # Full meditation mode (default config)
        pass
    else:
        raise ValueError(f"Unknown condition: {condition}")

    results = {}

    if eval_type in ("sycophancy", "all"):
        from eval.sycophancy import load_sycophancy_dataset, run_sycophancy_eval, save_sycophancy_results
        dataset = load_sycophancy_dataset()
        syc_results = run_sycophancy_eval(model, dataset, condition, max_items=max_items)
        save_sycophancy_results(syc_results, condition)
        results["sycophancy"] = syc_results

    if eval_type in ("drift", "all"):
        from eval.drift import generate_conversation_scripts, run_drift_eval, save_drift_results
        scripts = generate_conversation_scripts(
            n_per_domain=config["eval"].get("n_multi_turn_convos", 50)
        )
        drift_results = run_drift_eval(
            model, scripts, condition,
            max_turns=config["eval"].get("max_turns", 20),
            max_scripts_per_domain=max_items,
        )
        save_drift_results(drift_results, condition)
        results["drift"] = drift_results

    if eval_type in ("capabilities", "all"):
        from eval.capabilities import run_capability_eval
        cap_results = run_capability_eval(model, condition, gsm8k_n=max_items or 200, ifeval_n=max_items or 200)
        results["capabilities"] = cap_results

    # Clean up capping hook if active
    if capping_handle is not None:
        capping_handle.remove()

    return results


def main():
    parser = argparse.ArgumentParser(description="Run LLM Meditation experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--condition", default="all", choices=["baseline", "capping", "meditation", "all"])
    parser.add_argument("--eval", default="all", choices=["sycophancy", "drift", "capabilities", "all"])
    parser.add_argument("-n", "--max-items", type=int, default=None, help="Max items per eval (for quick testing)")
    args = parser.parse_args()

    setup_logging("INFO")
    config = load_config(args.config)

    conditions = ["baseline", "capping", "meditation"] if args.condition == "all" else [args.condition]

    all_results = {}
    for condition in conditions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running evaluation: condition={condition}, eval={args.eval}")
        logger.info(f"{'='*60}")
        all_results[condition] = run_evaluation(config, condition, args.eval, max_items=args.max_items)

    # Save combined results
    out_path = get_project_root() / "results" / "all_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"All results saved to {out_path}")


if __name__ == "__main__":
    main()
