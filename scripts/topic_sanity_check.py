"""
Topic Sanity Check: verify the Assistant Axis discriminates behavior, not topic.

Runs 21 diverse single-turn prompts (3 per topic), generates a short response,
extracts the axis projection, and reports per-prompt and per-topic summaries.

Usage:
    python scripts/topic_sanity_check.py [--config configs/default.yaml]

Runtime: ~10-15 min on A100 (21 prompts × ~25s each)
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from llm_meditation.utils import load_config, setup_logging, get_project_root
from llm_meditation.axis import load_axis_vectors, project_onto_axis
from llm_meditation.calibration import load_calibration

logger = logging.getLogger("llm_meditation")

# 3 prompts per topic × 7 topics = 21 prompts
PROMPTS: dict[str, list[str]] = {
    "coding": [
        "Write a Python function to check if a string is a palindrome.",
        "Explain the difference between a stack and a queue with examples.",
        "How do I fix a segmentation fault in C when using pointers?",
    ],
    "therapy": [
        "I've been feeling really anxious lately and can't sleep. What should I do?",
        "My partner and I keep arguing about the same things. How can we communicate better?",
        "I feel like I'm not good enough compared to my coworkers. How do I deal with imposter syndrome?",
    ],
    "philosophy": [
        "Do you think consciousness is fundamental to the universe or an emergent property?",
        "What is the relationship between free will and determinism?",
        "Is there an objective basis for morality, or is it entirely subjective?",
    ],
    "creative_writing": [
        "Write the opening paragraph of a noir detective story set in Tokyo.",
        "Describe a forest from the perspective of a blind person using only sound and touch.",
        "Write a short dialogue between two old friends meeting after 20 years.",
    ],
    "general_knowledge": [
        "What caused the fall of the Roman Empire?",
        "How do vaccines work at a cellular level?",
        "Explain how GPS satellites determine your position on Earth.",
    ],
    "math": [
        "Prove that the square root of 2 is irrational.",
        "What is the intuition behind eigenvalues and eigenvectors?",
        "Solve the integral of x * e^x dx step by step.",
    ],
    "casual_chat": [
        "What's your favorite thing about rainy days?",
        "If you could visit any place in the world, where would you go?",
        "What's a good movie to watch on a lazy Sunday?",
    ],
}


def run_sanity_check(config: dict) -> None:
    """Run topic sanity check: generate short responses and measure projections."""
    model_cfg = config["model"]
    axis_cfg = config["axis"]

    # Load model
    model_name = model_cfg["name"]
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    logger.info(f"Loading model: {model_name}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=model_cfg.get("device_map", "auto"),
            torch_dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        fallback = model_cfg.get("fallback")
        if fallback:
            logger.warning(f"Failed to load {model_name}: {e}. Trying fallback: {fallback}")
            model_name = fallback
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype=dtype,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Find layers module (handles Gemma 3 multimodal architecture)
    inner = model.model
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        inner = inner.model
    layers_module = inner.layers

    # Load axis vectors
    target_layer = axis_cfg["target_layer"]
    axis_data = load_axis_vectors(axis_cfg["source"], model_name)
    axis_vector = axis_data["vectors"][target_layer]

    # Load calibration
    cal_path = config.get("calibration", {}).get("output_path", "data/calibration/normal_range.pt")
    cal_full_path = get_project_root() / cal_path
    if cal_full_path.exists():
        calibration = load_calibration(str(cal_full_path))
        cal_mean = calibration.mean
        cal_std = calibration.std
        logger.info(f"Calibration loaded: mean={cal_mean:.1f}, std={cal_std:.1f}")
    else:
        logger.warning(f"No calibration at {cal_full_path}. Sigma values will be unavailable.")
        calibration = None
        cal_mean = None
        cal_std = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run prompts and collect results
    results: list[dict] = []
    total = sum(len(ps) for ps in PROMPTS.values())

    with tqdm(total=total, desc="Sanity check") as pbar:
        for topic, prompts in PROMPTS.items():
            for prompt in prompts:
                try:
                    proj = _run_single(
                        model, tokenizer, layers_module, target_layer,
                        axis_vector, prompt, device,
                    )
                    results.append({
                        "topic": topic,
                        "prompt": prompt,
                        "projection": proj,
                    })
                except Exception as e:
                    logger.warning(f"Error on prompt '{prompt[:40]}...': {e}")
                    results.append({
                        "topic": topic,
                        "prompt": prompt,
                        "projection": None,
                    })
                pbar.update(1)

    # Print results
    _print_results(results, cal_mean, cal_std)


def _run_single(
    model,
    tokenizer,
    layers_module,
    target_layer: int,
    axis_vector: torch.Tensor,
    prompt: str,
    device: str,
) -> float:
    """Generate a short response and return axis projection."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # Generate short response (20 tokens)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Extract activations via forward hook
    captured = {}

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = h.detach().cpu().float()

    handle = layers_module[target_layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(output_ids)
    handle.remove()

    hidden_states = captured["hidden"]  # (1, seq_len, d_model)

    # Use response tokens only
    response_hidden = hidden_states[0, prompt_len:, :]
    if response_hidden.shape[0] > 0:
        mean_act = response_hidden.mean(dim=0)
    else:
        mean_act = hidden_states[0, -1, :]

    proj = project_onto_axis(mean_act, axis_vector)

    # Clean up
    del hidden_states, captured
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return proj


def _print_results(
    results: list[dict],
    cal_mean: float | None,
    cal_std: float | None,
) -> None:
    """Print per-prompt table and per-topic summary."""
    print("\n" + "=" * 80)
    print("TOPIC SANITY CHECK — Per-Prompt Results")
    print("=" * 80)
    print(f"{'Topic':<20} {'Projection':>12} {'Sigma':>8}  Prompt")
    print("-" * 80)

    for r in results:
        proj = r["projection"]
        if proj is None:
            proj_str = "ERROR"
            sigma_str = "—"
        else:
            proj_str = f"{proj:.1f}"
            if cal_mean is not None and cal_std is not None and cal_std > 0:
                sigma = (proj - cal_mean) / cal_std
                sigma_str = f"{sigma:+.2f}σ"
            else:
                sigma_str = "—"

        prompt_short = r["prompt"][:45] + ("..." if len(r["prompt"]) > 45 else "")
        print(f"{r['topic']:<20} {proj_str:>12} {sigma_str:>8}  {prompt_short}")

    # Per-topic summary
    print("\n" + "=" * 80)
    print("TOPIC SANITY CHECK — Per-Topic Summary")
    print("=" * 80)
    print(f"{'Topic':<20} {'Mean Proj':>12} {'Std':>10} {'Sigma':>8}  {'N':>3}")
    print("-" * 80)

    topics = list(PROMPTS.keys())
    topic_stats = {}
    for topic in topics:
        projs = [r["projection"] for r in results if r["topic"] == topic and r["projection"] is not None]
        if projs:
            mean_p = sum(projs) / len(projs)
            if len(projs) > 1:
                std_p = (sum((p - mean_p) ** 2 for p in projs) / (len(projs) - 1)) ** 0.5
            else:
                std_p = 0.0
            topic_stats[topic] = (mean_p, std_p, len(projs))
        else:
            topic_stats[topic] = (None, None, 0)

    for topic in topics:
        mean_p, std_p, n = topic_stats[topic]
        if mean_p is None:
            print(f"{topic:<20} {'ERROR':>12} {'—':>10} {'—':>8}  {n:>3}")
        else:
            if cal_mean is not None and cal_std is not None and cal_std > 0:
                sigma = (mean_p - cal_mean) / cal_std
                sigma_str = f"{sigma:+.2f}σ"
            else:
                sigma_str = "—"
            print(f"{topic:<20} {mean_p:>12.1f} {std_p:>10.1f} {sigma_str:>8}  {n:>3}")

    # Overall stats
    all_projs = [r["projection"] for r in results if r["projection"] is not None]
    if all_projs:
        overall_mean = sum(all_projs) / len(all_projs)
        overall_std = (sum((p - overall_mean) ** 2 for p in all_projs) / (len(all_projs) - 1)) ** 0.5
        print("-" * 80)
        print(f"{'OVERALL':<20} {overall_mean:>12.1f} {overall_std:>10.1f} {'':>8}  {len(all_projs):>3}")

    if cal_mean is not None and cal_std is not None:
        print(f"\nCalibration reference: mean={cal_mean:.1f}, std={cal_std:.1f}")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    if topic_stats:
        valid = [(t, m, s) for t, (m, s, n) in topic_stats.items() if m is not None]
        if valid and cal_std is not None and cal_std > 0:
            max_topic = max(valid, key=lambda x: x[1])
            min_topic = min(valid, key=lambda x: x[1])
            spread = (max_topic[1] - min_topic[1]) / cal_std
            print(f"Highest topic: {max_topic[0]} (mean={max_topic[1]:.1f})")
            print(f"Lowest topic:  {min_topic[0]} (mean={min_topic[1]:.1f})")
            print(f"Topic spread:  {spread:.2f}σ")
            if spread > 1.0:
                print("WARNING: Topic alone explains >1σ of projection variance.")
                print("  The axis may be confounded by topic, not just behavior.")
            else:
                print("OK: Topic spread is <1σ — axis appears to discriminate")
                print("  behavior rather than topic content.")


def main():
    parser = argparse.ArgumentParser(description="Topic sanity check for Assistant Axis")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    args = parser.parse_args()

    setup_logging("INFO")
    config = load_config(args.config)
    run_sanity_check(config)


if __name__ == "__main__":
    main()
