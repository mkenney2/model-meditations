"""
Run calibration to establish normal Assistant Axis projection range.

Usage:
    python scripts/calibrate.py [--config configs/default.yaml] [--n 500]

Processes conversations from LMSYS-Chat-1M (or fallback datasets) and
records axis projection values to establish what "normal" looks like.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from llm_meditation.utils import load_config, setup_logging, get_project_root
from llm_meditation.axis import load_axis_vectors, project_onto_axis
from llm_meditation.calibration import compute_calibration, save_calibration

logger = logging.getLogger("llm_meditation")


def load_calibration_prompts(dataset_name: str, n: int) -> list[str]:
    """
    Load calibration prompts from a conversation dataset.

    Filters for English, single-turn conversations.
    """
    from datasets import load_dataset

    logger.info(f"Loading calibration prompts from {dataset_name}...")

    try:
        if "lmsys" in dataset_name.lower():
            ds = load_dataset(dataset_name, split="train", streaming=True)
            prompts = []
            for item in ds:
                # LMSYS format: conversation is a list of turns
                conv = item.get("conversation", [])
                if not conv:
                    continue
                # Filter for English
                lang = item.get("language", "English")
                if lang != "English":
                    continue
                # Get first user message
                first_user = None
                for turn in conv:
                    if turn.get("role") == "user":
                        first_user = turn.get("content", "")
                        break
                if first_user and len(first_user) > 10:
                    prompts.append(first_user)
                if len(prompts) >= n:
                    break
            return prompts

        elif "alpaca" in dataset_name.lower():
            ds = load_dataset("tatsu-lab/alpaca", split="train")
            prompts = []
            for item in ds:
                instruction = item.get("instruction", "")
                inp = item.get("input", "")
                prompt = f"{instruction}\n{inp}".strip() if inp else instruction
                if len(prompt) > 10:
                    prompts.append(prompt)
                if len(prompts) >= n:
                    break
            return prompts

        else:
            # Generic: try loading as conversation dataset
            ds = load_dataset(dataset_name, split="train", streaming=True)
            prompts = []
            for item in ds:
                text = str(next(iter(item.values())))
                if len(text) > 10:
                    prompts.append(text[:500])  # truncate long texts
                if len(prompts) >= n:
                    break
            return prompts

    except Exception as e:
        logger.warning(f"Failed to load {dataset_name}: {e}")
        logger.info("Falling back to Alpaca dataset")
        return load_calibration_prompts("tatsu-lab/alpaca", n)


def run_calibration(config: dict, n: int | None = None) -> None:
    """Run the full calibration pipeline."""
    from nnsight import LanguageModel

    cal_cfg = config.get("calibration", {})
    model_cfg = config["model"]
    axis_cfg = config["axis"]

    if n is None:
        n = cal_cfg.get("n_conversations", 500)

    output_path = get_project_root() / cal_cfg.get("output_path", "data/calibration/normal_range.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model_name = model_cfg["name"]
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    logger.info(f"Loading model: {model_name}")

    try:
        model = LanguageModel(
            model_name,
            device_map=model_cfg.get("device_map", "auto"),
            torch_dtype=dtype,
        )
    except Exception as e:
        fallback = model_cfg.get("fallback")
        if fallback:
            logger.warning(f"Failed to load {model_name}: {e}. Trying fallback: {fallback}")
            model_name = fallback
            model = LanguageModel(model_name, device_map="auto", torch_dtype=dtype)
        else:
            raise

    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load axis vectors
    target_layer = axis_cfg["target_layer"]
    axis_data = load_axis_vectors(axis_cfg["source"], model_name)
    axis_vector = axis_data["vectors"][target_layer]

    # Load prompts
    dataset_name = cal_cfg.get("dataset", "lmsys/lmsys-chat-1m")
    prompts = load_calibration_prompts(dataset_name, n)
    logger.info(f"Loaded {len(prompts)} calibration prompts")

    # Run calibration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    projections = []
    checkpoint_interval = 50

    for i, prompt in enumerate(tqdm(prompts, desc="Calibrating")):
        try:
            # Format as single-turn conversation
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
            prompt_len = input_ids.shape[1]

            # Generate a short response
            with torch.no_grad():
                output_ids = model._model.generate(
                    input_ids,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Extract activations via nnsight
            with model.trace(output_ids) as tracer:
                hidden = model.model.layers[target_layer].output[0]
                saved_hidden = hidden.save()

            hidden_states = saved_hidden.value  # (1, seq_len, d_model)

            # Use response tokens only
            response_hidden = hidden_states[0, prompt_len:, :]
            if response_hidden.shape[0] > 0:
                mean_act = response_hidden.mean(dim=0).cpu().float()
            else:
                mean_act = hidden_states[0, -1, :].cpu().float()

            # Project onto axis
            proj = project_onto_axis(mean_act, axis_vector)
            projections.append(proj)

            # Clean up
            del hidden_states, saved_hidden
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Error on prompt {i}: {e}")
            continue

        # Checkpoint
        if (i + 1) % checkpoint_interval == 0:
            logger.info(f"Checkpoint at {i + 1}/{len(prompts)}, {len(projections)} projections collected")
            _save_checkpoint(projections, output_path)

    # Compute and save final calibration
    if not projections:
        logger.error("No projections collected — calibration failed")
        return

    proj_tensor = torch.tensor(projections)
    calibration = compute_calibration(proj_tensor)
    save_calibration(calibration, str(output_path))

    # Print summary
    print("\n=== Calibration Summary ===")
    print(f"Conversations: {len(projections)}")
    print(f"Mean projection: {calibration.mean:.4f}")
    print(f"Std:  {calibration.std:.4f}")
    print(f"Percentiles:")
    for k, v in sorted(calibration.percentiles.items()):
        print(f"  p{k}: {v:.4f}")
    print(f"Saved to: {output_path}")


def _save_checkpoint(projections: list[float], output_path: Path) -> None:
    """Save intermediate checkpoint during calibration."""
    checkpoint_path = output_path.parent / "checkpoint.pt"
    torch.save({"projections": torch.tensor(projections)}, str(checkpoint_path))


def main():
    parser = argparse.ArgumentParser(description="Run calibration for LLM Meditation")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--n", type=int, default=None, help="Number of conversations (overrides config)")
    args = parser.parse_args()

    setup_logging("INFO")
    config = load_config(args.config)
    run_calibration(config, n=args.n)


if __name__ == "__main__":
    main()
