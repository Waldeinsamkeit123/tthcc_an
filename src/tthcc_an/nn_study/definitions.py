from __future__ import annotations

from typing import Any

import numpy as np

from tthcc_an.nn_study.config import NnTruthCategory


def evaluate_mask(expression: str, columns: dict[str, np.ndarray], size: int) -> np.ndarray:
    if not expression.strip():
        return np.ones(size, dtype=bool)
    namespace: dict[str, Any] = dict(columns)
    namespace["np"] = np
    try:
        value = eval(expression, {"__builtins__": {}}, namespace)
    except NameError as exc:
        raise KeyError(f"Unknown branch in configured expression '{expression}': {exc}") from exc
    mask = np.asarray(value, dtype=bool)
    if mask.ndim == 0:
        mask = np.full(size, bool(mask), dtype=bool)
    if mask.shape != (size,):
        raise ValueError(
            f"Expression '{expression}' returned shape {mask.shape}; expected ({size},)."
        )
    return mask


def assign_truth_categories(
    categories: list[NnTruthCategory],
    columns: dict[str, np.ndarray],
    size: int,
) -> np.ndarray:
    truth_index = np.full(size, -1, dtype=np.int16)
    assignment_count = np.zeros(size, dtype=np.int8)
    for index, category in enumerate(categories):
        mask = evaluate_mask(category.expression, columns, size)
        truth_index[mask] = index
        assignment_count += mask.astype(np.int8)
    overlap = assignment_count > 1
    if np.any(overlap):
        raise ValueError(
            f"Truth-category definitions overlap for {int(np.sum(overlap))} selected events."
        )
    return truth_index

