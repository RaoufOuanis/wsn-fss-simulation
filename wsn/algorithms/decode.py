from __future__ import annotations

import numpy as np


def sigmoid01(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid mapping to (0,1).

    This matches the intent of the existing PSO/GWO/ABC decoders while avoiding
    overflow on large-magnitude values.
    """
    x = np.asarray(x, dtype=float)
    x_clip = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def decode_topk_bounded(
    scores: np.ndarray,
    *,
    alive_mask: np.ndarray,
    rng: np.random.Generator,
    k_min: int,
    k_max: int,
    use_sigmoid: bool = True,
) -> np.ndarray:
    """Decode a continuous score vector into a bounded top-k CH set.

    Invariants:
    - only alive nodes are eligible
    - k is sampled uniformly from [k_min, k_max] and clamped by #alive
    - returns sorted unique node indices
    """
    alive_mask = np.asarray(alive_mask, dtype=bool)
    alive_idx = np.where(alive_mask)[0]
    if alive_idx.size == 0:
        return np.array([], dtype=int)

    k_min_i = int(max(1, int(k_min)))
    k_max_i = int(min(int(k_max), int(alive_idx.size)))
    k = int(rng.integers(k_min_i, k_max_i + 1)) if k_max_i >= k_min_i else int(k_max_i)

    s = sigmoid01(scores) if bool(use_sigmoid) else np.asarray(scores, dtype=float)
    alive_scores = s[alive_idx]

    # Stable ordering not required; argsort is fine and matches existing code style.
    idx_sorted = alive_idx[np.argsort(-alive_scores)]
    ch = idx_sorted[:k]
    return np.sort(np.unique(ch.astype(int, copy=False)))
