"""Training data format utilities for neural density estimator.

Converts SimulationConfig and summary_truth dicts to fixed-length
numeric vectors, and provides batch save/load for .npz files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from demantiq.config.simulation_config import SimulationConfig

MAX_CHANNELS = 20

# --- Saturation / adstock function encoding ---
_SAT_FN_MAP = {"hill": 0, "logistic": 1, "power": 2, "piecewise_linear": 3}
_ADS_FN_MAP = {"geometric": 0, "weibull_cdf": 1, "weibull_pdf": 2, "delayed_geometric": 3}
_NOISE_TYPE_MAP = {"gaussian": 0, "t_distributed": 1, "heteroscedastic": 2, "autocorrelated": 3}

# Per-channel vector length: beta(1) + sat_fn(1) + sat_params(4) + ads_fn(1) + ads_params(3) + spend_mean(1) + spend_std(1) = 12
_PER_CHANNEL_LEN = 12
# Global prefix: n_periods(1) + n_channels(1) + noise_scale(1) + noise_type(1)
#   + organic_level(1) + trend_slope(1) + seasonality_n_terms(1)
#   + has_pricing(1) + has_distribution(1) + has_endogeneity(1) + has_competition(1) + has_macro(1) + has_interactions(1)
_GLOBAL_LEN = 13
CONFIG_VECTOR_LEN = _GLOBAL_LEN + MAX_CHANNELS * _PER_CHANNEL_LEN

# Per-channel truth: true_beta(1) + true_roas(1) + true_total_contribution(1) = 3
_PER_CHANNEL_TRUTH_LEN = 3
# Global truth: total_media_contribution_pct(1) + price_elasticity(1) = 2
_GLOBAL_TRUTH_LEN = 2
TRUTH_VECTOR_LEN = _GLOBAL_TRUTH_LEN + MAX_CHANNELS * _PER_CHANNEL_TRUTH_LEN


def config_to_vector(config: SimulationConfig) -> np.ndarray:
    """Flatten SimulationConfig to a fixed-length numeric vector.

    Encodes global simulation parameters and per-channel configs,
    zero-padded to MAX_CHANNELS.

    Returns:
        np.ndarray of shape (CONFIG_VECTOR_LEN,).
    """
    vec = np.zeros(CONFIG_VECTOR_LEN, dtype=np.float64)

    # Global prefix
    vec[0] = config.n_periods
    vec[1] = len(config.channels)
    vec[2] = config.noise.noise_scale
    vec[3] = _NOISE_TYPE_MAP.get(config.noise.noise_type, 0)
    vec[4] = config.baseline.organic_level
    vec[5] = config.baseline.trend_params.get("slope", 0.0)
    vec[6] = config.baseline.seasonality_n_terms
    vec[7] = 1.0 if config.pricing is not None else 0.0
    vec[8] = 1.0 if config.distribution is not None else 0.0
    vec[9] = 1.0 if config.endogeneity is not None else 0.0
    vec[10] = 1.0 if config.competition is not None else 0.0
    vec[11] = 1.0 if config.macro is not None else 0.0
    vec[12] = 1.0 if config.interactions is not None else 0.0

    # Per-channel
    for i, ch in enumerate(config.channels):
        if i >= MAX_CHANNELS:
            break
        offset = _GLOBAL_LEN + i * _PER_CHANNEL_LEN
        vec[offset + 0] = ch.beta
        vec[offset + 1] = _SAT_FN_MAP.get(ch.saturation_fn, 0)

        # Saturation params — encode up to 4 values
        sat_vals = _encode_saturation_params(ch.saturation_fn, ch.saturation_params)
        vec[offset + 2: offset + 6] = sat_vals

        vec[offset + 6] = _ADS_FN_MAP.get(ch.adstock_fn, 0)

        # Adstock params — encode up to 3 values
        ads_vals = _encode_adstock_params(ch.adstock_fn, ch.adstock_params)
        vec[offset + 7: offset + 10] = ads_vals

        vec[offset + 10] = ch.spend_mean
        vec[offset + 11] = ch.spend_std

    return vec


def _encode_saturation_params(fn_name: str, params: dict) -> np.ndarray:
    """Encode saturation params into a fixed-length array of 4."""
    arr = np.zeros(4)
    if fn_name == "hill":
        arr[0] = params.get("K", 0.0)
        arr[1] = params.get("S", 0.0)
    elif fn_name == "logistic":
        arr[0] = params.get("k", 0.0)
        arr[1] = params.get("x0", 0.0)
    else:
        # Generic: store first 4 values
        for j, v in enumerate(list(params.values())[:4]):
            arr[j] = float(v)
    return arr


def _encode_adstock_params(fn_name: str, params: dict) -> np.ndarray:
    """Encode adstock params into a fixed-length array of 3."""
    arr = np.zeros(3)
    if fn_name == "geometric":
        arr[0] = params.get("alpha", 0.0)
        arr[1] = params.get("max_lag", 0.0)
    elif fn_name in ("weibull_cdf", "weibull_pdf"):
        arr[0] = params.get("shape", 0.0)
        arr[1] = params.get("scale", 0.0)
        arr[2] = params.get("max_lag", 0.0)
    elif fn_name == "delayed_geometric":
        arr[0] = params.get("alpha", 0.0)
        arr[1] = params.get("delay", 0.0)
        arr[2] = params.get("max_lag", 0.0)
    else:
        for j, v in enumerate(list(params.values())[:3]):
            arr[j] = float(v)
    return arr


def summary_to_vector(summary: dict) -> np.ndarray:
    """Flatten summary truth dict to a fixed-length label vector.

    Encodes per-channel true_beta, true_roas, true_total_contribution,
    plus global metrics.  Zero-padded to MAX_CHANNELS.

    Returns:
        np.ndarray of shape (TRUTH_VECTOR_LEN,).
    """
    vec = np.zeros(TRUTH_VECTOR_LEN, dtype=np.float64)

    # Global truth
    vec[0] = summary.get("true_total_media_contribution_pct", 0.0)
    vec[1] = summary.get("true_price_elasticity", 0.0)

    # Per-channel truth (ordered by config channel order via dict iteration)
    true_betas = summary.get("true_betas", {})
    true_roas = summary.get("true_roas", {})
    true_contrib = summary.get("true_total_contribution", {})

    channel_names = list(true_betas.keys())
    for i, name in enumerate(channel_names):
        if i >= MAX_CHANNELS:
            break
        offset = _GLOBAL_TRUTH_LEN + i * _PER_CHANNEL_TRUTH_LEN
        vec[offset + 0] = true_betas.get(name, 0.0)
        vec[offset + 1] = true_roas.get(name, 0.0)
        vec[offset + 2] = true_contrib.get(name, 0.0)

    return vec


def vector_to_summary(vector: np.ndarray, channel_names: list[str]) -> dict:
    """Inverse of summary_to_vector for verification.

    Args:
        vector: Truth vector of shape (TRUTH_VECTOR_LEN,).
        channel_names: Ordered list of channel names.

    Returns:
        Dict with reconstructed summary fields.
    """
    summary: dict = {
        "true_total_media_contribution_pct": float(vector[0]),
        "true_price_elasticity": float(vector[1]),
        "true_betas": {},
        "true_roas": {},
        "true_total_contribution": {},
    }

    for i, name in enumerate(channel_names):
        if i >= MAX_CHANNELS:
            break
        offset = _GLOBAL_TRUTH_LEN + i * _PER_CHANNEL_TRUTH_LEN
        summary["true_betas"][name] = float(vector[offset + 0])
        summary["true_roas"][name] = float(vector[offset + 1])
        summary["true_total_contribution"][name] = float(vector[offset + 2])

    return summary


def save_batch(tuples: list[dict], output_path: str, batch_id: int) -> None:
    """Save a batch of training tuples to a compressed .npz file.

    Each tuple dict contains:
        config_vector, y, spend_matrix, truth_vector, channel_names.

    Also writes a metadata JSON sidecar.

    Args:
        tuples: List of training tuple dicts.
        output_path: Directory to write files into.
        batch_id: Integer batch identifier.
    """
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    config_vectors = np.stack([t["config_vector"] for t in tuples])
    truth_vectors = np.stack([t["truth_vector"] for t in tuples])

    # y and spend_matrix may have different lengths per scenario (n_periods varies),
    # so we pad to the max length in the batch.
    max_periods = max(len(t["y"]) for t in tuples)
    n_samples = len(tuples)

    # Determine max channels in this batch
    max_ch = max(t["spend_matrix"].shape[1] if t["spend_matrix"].ndim == 2 else 0
                 for t in tuples)
    max_ch = max(max_ch, 1)  # at least 1 to avoid zero-dim

    y_batch = np.zeros((n_samples, max_periods), dtype=np.float64)
    spend_batch = np.zeros((n_samples, max_periods, max_ch), dtype=np.float64)

    for i, t in enumerate(tuples):
        n_t = len(t["y"])
        y_batch[i, :n_t] = t["y"]
        sm = t["spend_matrix"]
        if sm.ndim == 2 and sm.shape[1] > 0:
            spend_batch[i, :n_t, :sm.shape[1]] = sm

    # Save arrays
    npz_path = out / f"batch_{batch_id}.npz"
    np.savez_compressed(
        str(npz_path),
        config_vectors=config_vectors,
        truth_vectors=truth_vectors,
        y=y_batch,
        spend=spend_batch,
    )

    # Save metadata sidecar
    metadata = {
        "batch_id": batch_id,
        "n_samples": n_samples,
        "max_periods": int(max_periods),
        "max_channels": int(max_ch),
        "config_vector_len": int(config_vectors.shape[1]),
        "truth_vector_len": int(truth_vectors.shape[1]),
        "channel_names": [t["channel_names"] for t in tuples],
        "n_periods": [len(t["y"]) for t in tuples],
    }
    meta_path = out / f"batch_{batch_id}_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2))


def load_batch(path: str) -> dict:
    """Load a .npz batch file and return dict of arrays.

    Args:
        path: Path to the .npz file.

    Returns:
        Dict with keys: config_vectors, truth_vectors, y, spend,
        plus metadata if the sidecar JSON exists.
    """
    data = dict(np.load(path, allow_pickle=False))

    # Try to load metadata sidecar
    meta_path = Path(path).with_name(
        Path(path).stem + "_meta.json"
    )
    if meta_path.exists():
        data["metadata"] = json.loads(meta_path.read_text())

    return data
