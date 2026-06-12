from __future__ import annotations

import numpy as np


DISTANCE_FEATURE_PREFIXES = (
    "DR_",
    "minDR_",
    "minDEta_",
    "minDPhi_",
)
MISSING_DISTANCE_SENTINEL = 900.0


def clean_feature_values(name: str, values: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(values, dtype=np.float64).copy()
    if name.startswith(DISTANCE_FEATURE_PREFIXES):
        cleaned[cleaned >= MISSING_DISTANCE_SENTINEL] = np.nan
    return cleaned


def build_feature_matrix(columns: dict[str, np.ndarray], feature_names: list[str]) -> np.ndarray:
    return np.column_stack(
        [clean_feature_values(name, columns[name]) for name in feature_names]
    )
