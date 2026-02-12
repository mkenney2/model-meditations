"""Shared utilities: config loading, token masking, logging."""

import logging
import yaml
import torch
from pathlib import Path

logger = logging.getLogger("llm_meditation")


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the llm_meditation package."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: str = "configs/default.yaml") -> dict:
    """Load a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    return Path(__file__).resolve().parent.parent.parent


def get_device() -> str:
    """Return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_response_start_position(input_ids: torch.Tensor, tokenizer, model_type: str = "gemma2") -> int:
    """
    Find the token position where the model's response begins.

    For Gemma 2: look for <start_of_turn>model token
    For Llama 3: look for <|start_header_id|>assistant<|end_header_id|> sequence

    Returns: int, the index of the first response token
    """
    ids = input_ids.squeeze().tolist()

    if model_type == "gemma2":
        # Find the last occurrence of the model turn marker
        marker_ids = tokenizer.encode("<start_of_turn>model", add_special_tokens=False)
        marker_len = len(marker_ids)
        last_pos = -1
        for i in range(len(ids) - marker_len + 1):
            if ids[i : i + marker_len] == marker_ids:
                last_pos = i + marker_len
        if last_pos == -1:
            logger.warning("Could not find model turn marker; using full sequence")
            return 0
        return last_pos

    elif model_type == "llama3":
        # Find the last assistant header
        marker_ids = tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False
        )
        marker_len = len(marker_ids)
        last_pos = -1
        for i in range(len(ids) - marker_len + 1):
            if ids[i : i + marker_len] == marker_ids:
                last_pos = i + marker_len
        if last_pos == -1:
            logger.warning("Could not find assistant header marker; using full sequence")
            return 0
        return last_pos

    else:
        logger.warning(f"Unknown model type '{model_type}'; using full sequence")
        return 0
