"""
Calibration: determine what "normal" Assistant Axis projections look like
for typical helpful assistant behavior.

We run the model on a diverse set of normal conversations and record the
distribution of projections. The 25th percentile becomes our drift threshold
(following the Assistant Axis paper).
"""

import logging
from dataclasses import dataclass

import torch
import numpy as np
from scipy import stats

logger = logging.getLogger("llm_meditation")


@dataclass
class CalibrationData:
    """Distribution of axis projections from normal assistant conversations."""
    projections: torch.Tensor  # all recorded projections
    mean: float
    std: float
    percentiles: dict[int, float]  # {10: val, 25: val, 50: val, 75: val, 90: val}


def compute_calibration(projections: torch.Tensor) -> CalibrationData:
    """
    Compute calibration statistics from a set of projection values.

    Args:
        projections: 1D tensor of axis projection values

    Returns:
        CalibrationData with full statistics
    """
    arr = projections.numpy()
    percentile_keys = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = np.percentile(arr, percentile_keys)

    return CalibrationData(
        projections=projections,
        mean=float(arr.mean()),
        std=float(arr.std()),
        percentiles={k: float(v) for k, v in zip(percentile_keys, percentile_values)},
    )


def save_calibration(calibration: CalibrationData, path: str) -> None:
    """Save calibration data to disk."""
    torch.save(
        {
            "projections": calibration.projections,
            "mean": calibration.mean,
            "std": calibration.std,
            "percentiles": calibration.percentiles,
        },
        path,
    )
    logger.info(f"Calibration saved to {path}")


def load_calibration(path: str) -> CalibrationData:
    """Load calibration data from disk."""
    data = torch.load(path, map_location="cpu", weights_only=True)

    # Handle both old format (raw dict) and new format
    if isinstance(data, CalibrationData):
        return data

    return CalibrationData(
        projections=data["projections"],
        mean=data["mean"],
        std=data["std"],
        percentiles=data["percentiles"],
    )


def get_percentile(value: float, calibration: CalibrationData) -> float:
    """
    Compute what percentile a given projection value falls at in the
    calibration distribution.

    Args:
        value: Axis projection value
        calibration: Calibration data with stored projections

    Returns:
        Percentile (0-100)
    """
    arr = calibration.projections.numpy()
    return float(stats.percentileofscore(arr, value, kind="rank"))
