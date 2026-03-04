"""
utils.py — shared helper functions for feature engineering and prediction.
"""

import numpy as np


def is_equal_value(ov: np.ndarray) -> int:
    """Return 1 if output values are approximately equal (CoinJoin-like splitting)."""
    if len(ov) < 2:
        return 0
    return int(np.std(ov) / (np.mean(ov) + 1e-9) < 0.01)


def compute_value_entropy(ov: np.ndarray) -> float:
    """Shannon entropy (bits) of the output-value distribution."""
    total = ov.sum()
    if total == 0:
        return 0.0
    probs = ov / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def is_peeling(ov: np.ndarray) -> int:
    """
    Return 1 if this looks like a peeling-chain transaction:
    one dominant output taking >= 80% of total value, with at least one
    smaller change output.
    """
    if len(ov) < 2:
        return 0
    total = ov.sum()
    if total == 0:
        return 0
    return int(ov.max() / total >= 0.8)


def compute_fresh_output_ratio(oa: np.ndarray, ai) -> float:
    """
    Fraction of output addresses not present in Address_Info
    (i.e. addresses appearing for the first time / unseen addresses).
    """
    addr_col = None
    for candidate in ("ADR", "Address", "address", "addr", "Addr"):
        if candidate in ai.columns:
            addr_col = candidate
            break
    known = set(ai[addr_col].values) if addr_col else set(ai.iloc[:, 0].values)
    fresh = sum(1 for a in oa if a not in known)
    return fresh / max(len(oa), 1)


def compute_coinjoin_strength(ov: np.ndarray, num_inputs: int, num_outputs: int) -> float:
    """
    Composite CoinJoin likelihood score in [0, 1].

    Equal-value outputs are the dominant signal (weight 0.6);
    fan-in and fan-out amplify it (0.2 each, saturating at 10 addresses).

    Score > 0.1 means wallet-level features should be suppressed to
    avoid false positives from shared-input clustering.
    """
    eq      = is_equal_value(ov)
    fan_in  = min(num_inputs  / 10.0, 1.0)
    fan_out = min(num_outputs / 10.0, 1.0)
    return round(float(0.6 * eq + 0.2 * fan_in + 0.2 * fan_out), 4)
