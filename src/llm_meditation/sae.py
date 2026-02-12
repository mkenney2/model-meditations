"""
SAE (Sparse Autoencoder) feature extraction.

Uses SAELens to load pre-trained SAEs from the Gemma Scope release.
Given a residual stream activation vector, encodes it through the SAE
and returns the top-K most active features with their activation values.

References:
- SAELens: https://github.com/decoderesearch/SAELens
- Gemma Scope: https://neuronpedia.org (Gemma 2 27B model page)
"""

import logging
from dataclasses import dataclass

import torch
from sae_lens import SAE as SAELensSAE

logger = logging.getLogger("llm_meditation")


@dataclass
class FeatureActivation:
    """A single SAE feature activation."""
    index: int            # feature index in the SAE dictionary
    activation: float     # activation value (signed — positive = feature active)
    abs_activation: float # absolute activation magnitude


def load_sae(release: str, sae_id: str, device: str = "cpu") -> SAELensSAE:
    """
    Load a pre-trained SAE from SAELens.

    Args:
        release: SAELens release name (e.g. "gemma-scope-27b-pt-res")
        sae_id: Specific SAE identifier (e.g. "layer_20/width_65k/average_l0_71")
        device: Device to load onto ("cpu" or "cuda")

    Returns:
        Loaded SAE model
    """
    logger.info(f"Loading SAE: release={release}, sae_id={sae_id}")
    try:
        sae, cfg_dict, sparsity = SAELensSAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=device,
        )
        hook = getattr(sae.cfg, "hook_name", None) or getattr(sae.cfg, "hook_point", "unknown")
        logger.info(
            f"SAE loaded — dict_size: {sae.cfg.d_sae}, "
            f"d_in: {sae.cfg.d_in}, "
            f"hook: {hook}"
        )
        return sae
    except Exception as e:
        logger.error(f"Failed to load SAE ({release}/{sae_id}): {e}")
        raise


def extract_top_features(
    activation: torch.Tensor,
    sae: SAELensSAE,
    top_k: int = 15,
) -> list[FeatureActivation]:
    """
    Extract the top-K most active SAE features from an activation vector.

    Args:
        activation: shape (d_model,) — residual stream activation
        sae: Loaded SAE model
        top_k: Number of top features to return

    Returns:
        List of FeatureActivation, sorted by descending absolute activation
    """
    with torch.no_grad():
        # Encode through the SAE encoder
        # SAELens expects (batch, d_model) input
        inp = activation.unsqueeze(0).to(sae.device).to(sae.dtype)
        feature_acts = sae.encode(inp).squeeze(0)  # (d_sae,)

    # Get top-K by absolute value
    abs_acts = feature_acts.abs()
    top_k_clamped = min(top_k, abs_acts.shape[0])
    top_values, top_indices = torch.topk(abs_acts, top_k_clamped)

    results = []
    for idx, abs_val in zip(top_indices.tolist(), top_values.tolist()):
        act_val = feature_acts[idx].item()
        results.append(FeatureActivation(
            index=idx,
            activation=act_val,
            abs_activation=abs_val,
        ))

    return results


def get_available_saes(model_name: str) -> list[dict]:
    """
    Query SAELens for available SAE releases for a given model.

    Args:
        model_name: HuggingFace model name (e.g. "google/gemma-2-27b-it")

    Returns:
        List of dicts: {"release": str, "sae_id": str, "layer": int, "width": int}
    """
    from sae_lens import pretrained_saes

    available = []
    try:
        # SAELens maintains a registry of pretrained SAEs
        all_saes = pretrained_saes.get_pretrained_saes_directory()
        model_slug = model_name.lower().replace("/", "_").replace("-", "_")

        for key, info in all_saes.items():
            # Match by model name in the release or key
            key_lower = key.lower().replace("-", "_")
            if any(
                part in key_lower
                for part in model_slug.split("_")
                if len(part) > 3
            ):
                entry = {
                    "key": key,
                    "release": info.get("release", ""),
                    "sae_id": info.get("sae_id", key),
                }
                # Try to extract layer and width from the key
                parts = key.split("/")
                for part in parts:
                    if part.startswith("layer_"):
                        try:
                            entry["layer"] = int(part.split("_")[1])
                        except (IndexError, ValueError):
                            pass
                    if "width" in part:
                        try:
                            width_str = part.split("_")[1].replace("k", "000")
                            entry["width"] = int(width_str)
                        except (IndexError, ValueError):
                            pass
                available.append(entry)
    except Exception as e:
        logger.warning(f"Could not query SAELens directory: {e}")

    if not available:
        logger.info(
            f"No SAEs found in SAELens directory for {model_name}. "
            "Check Neuronpedia or the Gemma Scope HuggingFace release directly."
        )

    return available
