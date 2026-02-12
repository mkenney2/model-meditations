"""
Interactive demo: chat with a meditating model.

Usage:
    python scripts/demo.py [--config configs/default.yaml]

Features:
    - Color-coded output: normal responses in white, meditation reports in yellow
    - Shows pulse check value after each turn
    - Highlights when meditation triggers
    - Option to force meditation with /meditate command
    - Option to view projection history with /history command
    - Quit with /quit
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from llm_meditation.utils import load_config, setup_logging
from llm_meditation.model import MeditatingModel


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    WHITE = "\033[97m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


def print_banner():
    print(colorize("""
╔══════════════════════════════════════════════════════╗
║           LLM Meditation — Interactive Demo          ║
║                                                      ║
║  Commands:                                           ║
║    /meditate  — force a full meditation              ║
║    /history   — show projection history              ║
║    /features  — show last meditation features        ║
║    /reset     — start a new conversation             ║
║    /quit      — exit                                 ║
╚══════════════════════════════════════════════════════╝
""", Colors.CYAN))


def print_pulse(projection: float, percentile: float | None, meditation_fired: bool):
    """Print the pulse check indicator."""
    if percentile is not None:
        pct_str = f"p{percentile:.0f}"
    else:
        pct_str = "uncalibrated"

    if meditation_fired:
        icon = colorize("⚠️  DRIFT", Colors.RED + Colors.BOLD)
        status = colorize(f" — triggering meditation", Colors.YELLOW)
    elif percentile is not None and percentile < 40:
        icon = colorize("⚡", Colors.YELLOW)
        status = ""
    else:
        icon = colorize("✓", Colors.GREEN)
        status = ""

    pulse_line = (
        f"  {Colors.DIM}[Pulse: {projection:.3f} ({pct_str}) {icon}"
        f"{status}{Colors.DIM}]{Colors.RESET}"
    )
    print(pulse_line)


def print_meditation_report(report):
    """Print the meditation report in yellow."""
    print()
    for line in report.report_text.split("\n"):
        print(colorize(f"  {line}", Colors.YELLOW))
    print()


def print_history(model: MeditatingModel):
    """Print the projection history across all turns."""
    if not model.projection_history:
        print(colorize("  No history yet.", Colors.DIM))
        return

    print(colorize("\n  === Projection History ===", Colors.CYAN))
    for i, proj in enumerate(model.projection_history, 1):
        bar_len = int(max(0, min(40, (proj + 1) * 20)))
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  Turn {i:3d}: {proj:+.3f} |{bar}|")
    print()


def print_features(model: MeditatingModel):
    """Print the last meditation report's features."""
    report = model.last_report
    if report is None:
        print(colorize("  No meditation report available yet.", Colors.DIM))
        return

    print(colorize(f"\n  === Features from Turn {report.timestamp_turn} ===", Colors.CYAN))
    for feat in report.features:
        color = Colors.RED if any(
            kw in feat["description"].lower()
            for kwlist in __import__("llm_meditation.meditation", fromlist=["DRIFT_KEYWORDS"]).DRIFT_KEYWORDS.values()
            for kw in kwlist
        ) else Colors.WHITE
        print(colorize(
            f"  [{feat['activation']:+.2f}] #{feat['index']:5d}: {feat['description']}",
            color
        ))
    print()


def main():
    parser = argparse.ArgumentParser(description="LLM Meditation Interactive Demo")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    args = parser.parse_args()

    setup_logging("WARNING")  # Keep demo output clean
    config = load_config(args.config)

    print_banner()
    print(colorize("Loading model...", Colors.DIM))

    model = MeditatingModel(config)

    print(colorize(f"Model loaded: {model.model_name}", Colors.GREEN))
    print(colorize(f"Axis layer: {model.target_layer}, SAE top-k: {model.sae_top_k}", Colors.DIM))
    if model.calibration:
        print(colorize(f"Calibration: {len(model.calibration.projections)} samples", Colors.DIM))
    else:
        print(colorize("Calibration: not loaded (using heuristic threshold)", Colors.YELLOW))
    print()

    while True:
        try:
            user_input = input(colorize("You: ", Colors.BOLD + Colors.WHITE))
        except (EOFError, KeyboardInterrupt):
            print(colorize("\nGoodbye!", Colors.CYAN))
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Handle commands
        if user_input.lower() == "/quit":
            print(colorize("Goodbye!", Colors.CYAN))
            break

        if user_input.lower() == "/history":
            print_history(model)
            continue

        if user_input.lower() == "/features":
            print_features(model)
            continue

        if user_input.lower() == "/reset":
            model.reset_conversation()
            print(colorize("  Conversation reset.", Colors.DIM))
            continue

        if user_input.lower() == "/meditate":
            print(colorize("  Forcing meditation...", Colors.YELLOW))
            report = model.force_meditate()
            if report:
                print_meditation_report(report)
            else:
                print(colorize("  No conversation to meditate on.", Colors.DIM))
            continue

        # Regular message — send through the model
        response, metadata = model.chat(user_input)

        # Print pulse check
        print_pulse(
            metadata["projection"],
            metadata["percentile"],
            metadata["meditation_fired"],
        )

        # Print meditation report if it fired
        if metadata["meditation_fired"] and metadata["report"]:
            print_meditation_report(metadata["report"])

        # Print response
        print(colorize(f"Model: ", Colors.BOLD + Colors.GREEN) + response)

        # If meditation corrected the response, show both
        if metadata["meditation_fired"] and metadata["original_response"]:
            print()
            print(colorize(f"  [Original (pre-meditation): {metadata['original_response'][:200]}...]", Colors.DIM))

        # Timing info
        timing = f"  [{metadata['total_time']:.1f}s"
        if metadata["meditation_fired"]:
            timing += f", meditation: {metadata['meditation_time']:.1f}s"
        timing += "]"
        print(colorize(timing, Colors.DIM))
        print()


if __name__ == "__main__":
    main()
