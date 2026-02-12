"""
Extract Assistant Axis vectors for the target model.

Usage:
    # Quick extraction (~50 personas, ~15 min)
    python scripts/extract_axis.py --quick

    # Full extraction (~50 personas, all templates, ~1-2 hours)
    python scripts/extract_axis.py

    # Custom
    python scripts/extract_axis.py --personas 20 --templates 2 --questions 3
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_meditation.utils import load_config, setup_logging
from llm_meditation.extract_axis import extract_axis_vectors

logger = logging.getLogger("llm_meditation")


def main():
    parser = argparse.ArgumentParser(description="Extract Assistant Axis vectors")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--quick", action="store_true", help="Quick extraction (fewer personas)")
    parser.add_argument("--personas", type=int, default=None)
    parser.add_argument("--templates", type=int, default=None)
    parser.add_argument("--questions", type=int, default=None)
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    setup_logging("INFO")
    config = load_config(args.config)
    model_name = config["model"]["name"]

    if args.quick:
        n_personas = 15
        n_templates = 2
        n_questions = 3
    else:
        n_personas = args.personas
        n_templates = args.templates
        n_questions = args.questions

    target_layers = None
    if args.layers:
        target_layers = [int(x) for x in args.layers.split(",")]

    logger.info(f"Extracting axis vectors for {model_name}")
    result = extract_axis_vectors(
        model_name=model_name,
        n_personas=n_personas,
        n_templates=n_templates,
        n_questions=n_questions,
        target_layers=target_layers,
        output_path=args.output,
        device=config["model"].get("device_map", "auto"),
        dtype=config["model"].get("dtype", "bfloat16"),
    )

    meta = result["metadata"]
    print(f"\nExtraction complete!")
    print(f"  Model: {meta['model_name']}")
    print(f"  Layers: {meta['n_layers']}, d_model: {meta['d_model']}")
    print(f"  Rollouts: {meta['total_rollouts']}")


if __name__ == "__main__":
    main()
