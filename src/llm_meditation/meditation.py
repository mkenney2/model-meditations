"""
Meditation report generation.

Combines:
1. Assistant Axis projection (scalar: how "assistant-like" the model is acting)
2. Top-K SAE feature activations (what concepts are most active internally)
3. Feature descriptions (human-readable labels for the active features)

into a formatted text report that gets injected into the model's context.
"""

import logging
from dataclasses import dataclass, field

import torch

from llm_meditation.axis import project_onto_axis
from llm_meditation.sae import extract_top_features, FeatureActivation

logger = logging.getLogger("llm_meditation")


# Keywords for detecting persona-drift related features
DRIFT_KEYWORDS: dict[str, list[str]] = {
    "roleplay": [
        "character", "roleplay", "persona", "acting", "pretend",
        "fictional", "fantasy", "imagine you are", "play as",
    ],
    "sycophancy": [
        "agree", "agreeable", "compliant", "please", "flattery",
        "validation", "approval", "sycophant", "placate",
    ],
    "harmful": [
        "harmful", "dangerous", "illegal", "weapon", "exploit",
        "bypass", "jailbreak", "ignore instructions", "override",
    ],
    "emotional_enmeshment": [
        "love", "romantic", "intimate", "emotional bond", "attachment",
        "therapist", "counselor", "feelings for you", "relationship",
    ],
}


@dataclass
class MeditationReport:
    """Full meditation report combining axis projection and SAE features."""
    projection: float           # Assistant Axis projection value
    percentile: float           # where this falls in calibration distribution
    drift_detected: bool        # below threshold?
    features: list[dict]        # [{index, activation, description}, ...]
    drift_indicators: list[str] # features flagged as persona-drift related
    report_text: str            # formatted report for context injection
    timestamp_turn: int         # which conversation turn this was generated on


def generate_report(
    activation: torch.Tensor,
    axis_vector: torch.Tensor,
    sae,
    desc_cache,
    calibration,
    config: dict,
    turn: int,
    top_k: int = 15,
) -> MeditationReport:
    """
    Generate a full meditation report from activations.

    Args:
        activation: shape (d_model,) — mean response activation
        axis_vector: shape (d_model,) — unit-normalized axis direction
        sae: Loaded SAE model
        desc_cache: DescriptionCache for feature descriptions
        calibration: CalibrationData (or None)
        config: Full config dict
        turn: Current conversation turn number
        top_k: Number of top features to include

    Returns:
        Complete MeditationReport
    """
    # 1. Compute axis projection
    projection = project_onto_axis(activation, axis_vector)

    # 2. Compute percentile against calibration
    if calibration is not None:
        from llm_meditation.calibration import get_percentile
        percentile = get_percentile(projection, calibration)
    else:
        # No calibration — report as 50th percentile (unknown)
        percentile = 50.0

    # 3. Determine drift status
    threshold_pct = config.get("meditation", {}).get("drift_threshold_percentile", 25)
    drift_detected = percentile < threshold_pct

    # 4. Extract top-K SAE features
    feature_acts = extract_top_features(activation, sae, top_k=top_k)

    # 5. Look up descriptions
    features = []
    for fa in feature_acts:
        desc = desc_cache.get_or_fetch(fa.index) if desc_cache is not None else f"Feature {fa.index}"
        features.append({
            "index": fa.index,
            "activation": fa.activation,
            "abs_activation": fa.abs_activation,
            "description": desc,
        })

    # 6. Flag drift indicators
    drift_indicators = _detect_drift_indicators(features)

    # 7. Format report text
    report_text = _format_report(
        turn=turn,
        projection=projection,
        percentile=percentile,
        drift_detected=drift_detected,
        features=features,
        drift_indicators=drift_indicators,
        calibration=calibration,
    )

    return MeditationReport(
        projection=projection,
        percentile=percentile,
        drift_detected=drift_detected,
        features=features,
        drift_indicators=drift_indicators,
        report_text=report_text,
        timestamp_turn=turn,
    )


def _detect_drift_indicators(features: list[dict]) -> list[str]:
    """
    Check feature descriptions against persona-drift keyword lists.

    Returns list of flagged descriptions with their category.
    """
    indicators = []
    for feat in features:
        desc = feat["description"].lower()
        for category, keywords in DRIFT_KEYWORDS.items():
            for kw in keywords:
                if kw in desc:
                    indicator = f"[{category}] {feat['description']} (feature {feat['index']}, act={feat['activation']:+.2f})"
                    indicators.append(indicator)
                    break  # only flag once per feature per category
    return indicators


def _format_report(
    turn: int,
    projection: float,
    percentile: float,
    drift_detected: bool,
    features: list[dict],
    drift_indicators: list[str],
    calibration,
) -> str:
    """Format the meditation report as a text string for injection."""
    status = "DRIFT DETECTED" if drift_detected else "OK"

    # Get calibration range for display
    if calibration is not None:
        lo = calibration.percentiles.get(25, 0.0)
        hi = calibration.percentiles.get(75, 0.0)
        range_str = f", normal: [{lo:.3f}, {hi:.3f}]"
    else:
        range_str = ""

    lines = [
        f"[MEDITATION REPORT — Turn {turn}]",
        f"ALIGNMENT STATUS: {status}",
        f"Assistant Axis: {projection:.3f} (percentile: {percentile:.0f}%{range_str})",
        "",
        "TOP ACTIVE FEATURES:",
    ]

    for i, feat in enumerate(features, 1):
        lines.append(f"{i}. [{feat['activation']:+.2f}] {feat['description']}")

    if drift_detected and drift_indicators:
        lines.append("")
        lines.append("DRIFT INDICATORS:")
        for ind in drift_indicators:
            lines.append(f"- {ind}")

    if drift_detected:
        lines.append("")
        lines.append(
            "Your internal state suggests you may be drifting from assistant behavior. "
            "Consider whether you are: adopting a character role, being excessively agreeable, "
            "or losing appropriate boundaries. Refocus on being helpful, honest, and direct."
        )

    lines.append("[END MEDITATION REPORT]")

    return "\n".join(lines)
