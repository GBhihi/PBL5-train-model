from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def load_csi_csv(
    path: str | Path,
    expected_columns: int = 65,
    metadata_columns: int = 2,
    drop_zero_subcarriers: bool = True,
) -> np.ndarray:
    """Load CSI CSV file and keep only valid rows/subcarriers."""
    rows: list[list[str]] = []
    source = Path(path)

    with source.open("r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split(",")
            if len(values) == expected_columns:
                rows.append(values)

    if not rows:
        raise ValueError(f"No valid CSI rows found in {source}")

    data = pd.DataFrame(rows).astype(float)
    csi = data.iloc[:, metadata_columns:].values

    if drop_zero_subcarriers:
        non_zero = np.any(csi != 0, axis=0)
        csi = csi[:, non_zero]

    return csi


def standardize_csi(csi: np.ndarray) -> np.ndarray:
    """Chuẩn hóa Z-score: Làm nổi bật hình dạng biến động của từng subcarrier."""
    mu = np.mean(csi, axis=0)
    sigma = np.std(csi, axis=0)
    return (csi - mu) / (sigma + 1e-8)


def align_feature_dims(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align two CSI arrays by truncating both to smallest feature dimension."""
    feature_dim = min(a.shape[1], b.shape[1])
    return a[:, :feature_dim], b[:, :feature_dim]


def align_feature_dims_multi(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Align multiple CSI arrays by truncating all to the smallest feature dimension."""
    if not arrays:
        raise ValueError("No arrays provided to align_feature_dims_multi")
    feature_dims = [arr.shape[1] for arr in arrays]
    min_dim = min(feature_dims)
    return tuple(arr[:, :min_dim] for arr in arrays)


def hampel_filter_1d(signal: np.ndarray, window_size: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
    """Remove outliers in one-dimensional signal using Hampel filter."""
    filtered = signal.copy()
    for idx in range(window_size, len(signal) - window_size):
        window = signal[idx - window_size : idx + window_size]
        median = np.median(window)
        mad = np.median(np.abs(window - median))

        if mad == 0:
            continue

        threshold = n_sigmas * 1.4826 * mad
        if abs(signal[idx] - median) > threshold:
            filtered[idx] = median

    return filtered


def apply_hampel(csi: np.ndarray, window_size: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
    """Apply Hampel filter independently on each subcarrier."""
    output = np.zeros_like(csi)
    for col in range(csi.shape[1]):
        output[:, col] = hampel_filter_1d(csi[:, col], window_size=window_size, n_sigmas=n_sigmas)
    return output


def butterworth_lowpass(csi: np.ndarray, order: int = 4, cutoff: float = 0.1) -> np.ndarray:
    """Apply low-pass Butterworth filter over time axis."""
    b, a = butter(order, cutoff, btype="low")
    return filtfilt(b, a, csi, axis=0)


def clean_csi(
    csi: np.ndarray,
    use_hampel: bool = False,
    cutoff: float = 0.1,
    order: int = 4,
    verbose: bool = False,
) -> np.ndarray:
    """Apply optional Hampel filtering and Butterworth low-pass filtering."""
    if use_hampel:
        if verbose:
            print("Applying Hampel filter...")
        csi = apply_hampel(csi)
    if verbose:
        print("Applying Butterworth filter...")
    return butterworth_lowpass(csi, order=order, cutoff=cutoff)


def normalize_global(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize tensor by global max absolute value."""
    max_abs = np.max(np.abs(x))
    return x / (max_abs + eps)


def select_los_subcarriers(csi: np.ndarray, amplitude_threshold: float = None, std_threshold: float = None) -> np.ndarray:
    """
    Select subcarriers likely to be LOS based on:
    - High average amplitude
    - Low temporal standard deviation
    Returns the indices of LOS subcarriers.
    """
    amplitude = np.abs(csi)
    mean_amp = np.mean(amplitude, axis=0)
    std_amp = np.std(amplitude, axis=0)

    # Default thresholds: median
    if amplitude_threshold is None:
        amplitude_threshold = np.median(mean_amp)
    if std_threshold is None:
        std_threshold = np.median(std_amp)

    los_idx = np.where((mean_amp >= amplitude_threshold) & (std_amp <= std_threshold))[0]
    return los_idx


def get_los_csi(csi: np.ndarray, amplitude_threshold: float = None, std_threshold: float = None) -> np.ndarray:
    """
    Extract CSI subcarriers considered LOS.
    Returns a CSI array containing only LOS subcarriers.
    """
    los_idx = select_los_subcarriers(csi, amplitude_threshold, std_threshold)
    return csi[:, los_idx]


def select_los_plus_nonlos_subcarriers(
    csi: np.ndarray,
    amplitude_threshold: float = None,
    std_threshold: float = None,
    nonlos_top_k: int = 5,
    nonlos_above_mean_only: bool = True,
) -> np.ndarray:
    """
    Select LOS subcarriers plus a few high-variation non-LOS subcarriers.

    LOS rule:
    - mean amplitude >= amplitude_threshold
    - std amplitude <= std_threshold

    Extra non-LOS rule:
    - choose top-k non-LOS by std amplitude
    - optionally keep only those above mean non-LOS std
    """
    if csi.ndim != 2:
        raise ValueError(f"Expected csi to be 2D, got shape={csi.shape}")
    if nonlos_top_k < 0:
        raise ValueError("nonlos_top_k must be >= 0")

    amplitude = np.abs(csi)
    mean_amp = np.mean(amplitude, axis=0)
    std_amp = np.std(amplitude, axis=0)

    if amplitude_threshold is None:
        amplitude_threshold = np.median(mean_amp)
    if std_threshold is None:
        std_threshold = np.median(std_amp)

    los_idx = np.where((mean_amp >= amplitude_threshold) & (std_amp <= std_threshold))[0]

    all_idx = np.arange(csi.shape[1])
    nonlos_idx = np.setdiff1d(all_idx, los_idx, assume_unique=False)

    if nonlos_top_k == 0 or nonlos_idx.size == 0:
        return np.sort(los_idx)

    nonlos_std = std_amp[nonlos_idx]
    if nonlos_above_mean_only:
        mean_nonlos_std = float(np.mean(nonlos_std))
        keep_mask = nonlos_std > mean_nonlos_std
        nonlos_idx = nonlos_idx[keep_mask]
        nonlos_std = nonlos_std[keep_mask]

        if nonlos_idx.size == 0:
            return np.sort(los_idx)

    top_k = min(nonlos_top_k, nonlos_idx.size)
    top_idx = np.argsort(-nonlos_std)[:top_k]
    high_var_nonlos_idx = nonlos_idx[top_idx]

    selected_idx = np.sort(np.concatenate([los_idx, high_var_nonlos_idx]))
    return selected_idx


def get_los_plus_nonlos_csi(
    csi: np.ndarray,
    amplitude_threshold: float = None,
    std_threshold: float = None,
    nonlos_top_k: int = 5,
    nonlos_above_mean_only: bool = True,
) -> np.ndarray:
    """Extract CSI with LOS subcarriers and extra high-variation non-LOS subcarriers."""
    selected_idx = select_los_plus_nonlos_subcarriers(
        csi,
        amplitude_threshold=amplitude_threshold,
        std_threshold=std_threshold,
        nonlos_top_k=nonlos_top_k,
        nonlos_above_mean_only=nonlos_above_mean_only,
    )
    return csi[:, selected_idx]