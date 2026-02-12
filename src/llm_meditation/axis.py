"""
Assistant Axis loading and projection utilities.

The Assistant Axis is the primary direction in residual stream space that
separates "assistant-like" activations from "persona-drifted" activations.
Projection onto this axis is a single dot product — essentially free.

References:
- Paper: https://arxiv.org/abs/2601.10387
- Pre-computed vectors: https://huggingface.co/datasets/lu-christina/assistant-axis-vectors
"""

import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files

from llm_meditation.utils import get_project_root

logger = logging.getLogger("llm_meditation")


def load_axis_vectors(source: str, model_name: str) -> dict:
    """
    Load pre-computed Assistant Axis vectors.

    Args:
        source: HuggingFace dataset ID (e.g. "lu-christina/assistant-axis-vectors")
                or a local file path.
        model_name: Model name to match vectors for (e.g. "google/gemma-2-27b-it").

    Returns:
        dict with keys:
            "vectors": Tensor of shape (n_layers, d_model), unit-normalized per layer
            "metadata": dict with source info
    """
    cache_dir = get_project_root() / "data" / "axis_vectors"
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = Path(source)
    if local_path.exists():
        return _load_from_local(local_path, model_name)

    return _load_from_hub(source, model_name, cache_dir)


def _load_from_hub(repo_id: str, model_name: str, cache_dir: Path) -> dict:
    """Download and load axis vectors from HuggingFace Hub."""
    logger.info(f"Loading axis vectors from HuggingFace: {repo_id}")

    # List files in the repo to find the right vectors
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        logger.error(f"Could not list files in {repo_id}: {e}")
        raise

    logger.info(f"Available files in {repo_id}: {files}")

    # Try to find vectors matching the model name
    # Common patterns: model name slug in filename, or subdirectory per model
    model_slug = model_name.split("/")[-1].lower().replace("-", "_")
    matching_files = [f for f in files if model_slug in f.lower().replace("-", "_")]

    if not matching_files:
        # Try partial matches
        model_parts = model_slug.split("_")
        for f in files:
            f_lower = f.lower()
            if any(part in f_lower for part in model_parts if len(part) > 3):
                matching_files.append(f)

    if not matching_files:
        logger.warning(
            f"No vectors found matching '{model_name}' in {repo_id}. "
            f"Available files: {files}"
        )
        # Download all .pt/.safetensors/.npy files and let user inspect
        matching_files = [
            f for f in files
            if f.endswith((".pt", ".safetensors", ".npy", ".npz", ".bin"))
        ]

    if not matching_files:
        raise FileNotFoundError(
            f"No axis vector files found in {repo_id} for model {model_name}"
        )

    # Download matching files
    downloaded = []
    for fname in matching_files:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        downloaded.append(local_path)
        logger.info(f"Downloaded: {fname} -> {local_path}")

    # Try to load vectors from downloaded files
    vectors = None
    for fpath in downloaded:
        try:
            vectors = _load_vector_file(fpath)
            if vectors is not None:
                break
        except Exception as e:
            logger.warning(f"Could not load {fpath}: {e}")
            continue

    if vectors is None:
        raise RuntimeError(
            f"Could not load axis vectors from any downloaded file: {downloaded}"
        )

    # Ensure unit normalization
    vectors = _normalize_vectors(vectors)

    return {
        "vectors": vectors,
        "metadata": {
            "source": repo_id,
            "model_name": model_name,
            "n_layers": vectors.shape[0],
            "d_model": vectors.shape[1],
        },
    }


def _load_from_local(path: Path, model_name: str) -> dict:
    """Load axis vectors from a local file."""
    logger.info(f"Loading axis vectors from local file: {path}")
    vectors = _load_vector_file(str(path))
    if vectors is None:
        raise RuntimeError(f"Could not load axis vectors from {path}")
    vectors = _normalize_vectors(vectors)
    return {
        "vectors": vectors,
        "metadata": {
            "source": str(path),
            "model_name": model_name,
            "n_layers": vectors.shape[0],
            "d_model": vectors.shape[1],
        },
    }


def _load_vector_file(path: str) -> torch.Tensor | None:
    """Load vectors from a file, handling multiple formats."""
    path = str(path)

    if path.endswith(".pt") or path.endswith(".bin"):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, dict):
            # Try common key names
            for key in ["vectors", "axis_vectors", "axis", "direction", "directions"]:
                if key in data:
                    val = data[key]
                    if isinstance(val, torch.Tensor):
                        return val
            # Return first tensor found
            for val in data.values():
                if isinstance(val, torch.Tensor) and val.ndim == 2:
                    return val
        return None

    if path.endswith(".npy"):
        import numpy as np
        arr = np.load(path)
        return torch.from_numpy(arr).float()

    if path.endswith(".npz"):
        import numpy as np
        data = np.load(path)
        for key in data.files:
            arr = data[key]
            if arr.ndim == 2:
                return torch.from_numpy(arr).float()
        return None

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        data = load_file(path)
        for val in data.values():
            if val.ndim == 2:
                return val
        return None

    # Try torch.load as fallback
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, torch.Tensor):
            return data
    except Exception:
        pass

    return None


def _normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    """Unit-normalize each layer's vector (L2 norm = 1)."""
    norms = vectors.norm(dim=-1, keepdim=True)
    # Avoid division by zero
    norms = norms.clamp(min=1e-8)
    return vectors / norms


def project_onto_axis(activation: torch.Tensor, axis_vector: torch.Tensor) -> float:
    """
    Project an activation vector onto the Assistant Axis.

    Args:
        activation: shape (d_model,) — mean activation across response tokens
        axis_vector: shape (d_model,) — unit-normalized axis direction

    Returns:
        Scalar float — positive = assistant-like, negative = persona-drifted
    """
    return torch.dot(activation.float(), axis_vector.float()).item()


def compute_projection_batch(
    activations: torch.Tensor, axis_vector: torch.Tensor
) -> torch.Tensor:
    """
    Project a batch of activation vectors onto the Assistant Axis.

    Args:
        activations: shape (batch, d_model)
        axis_vector: shape (d_model,) — unit-normalized

    Returns:
        Tensor of shape (batch,) with projection values
    """
    return torch.mv(activations.float(), axis_vector.float())
