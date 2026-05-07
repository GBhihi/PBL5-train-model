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


def add_gaussian_noise(x: np.ndarray, std: float) -> np.ndarray:
    if std is None or std <= 0:
        return x.copy()
    noise = np.random.normal(0, std, size=x.shape)
    return x + noise


def time_shift_segments(x: np.ndarray, max_shift_pct: float) -> np.ndarray:
    if max_shift_pct is None or max_shift_pct <= 0:
        return x.copy()
    out = []
    T = x.shape[1]
    max_shift = int(T * max_shift_pct)
    for seg in x:
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            out.append(seg.copy())
            continue
        if shift > 0:
            s = np.concatenate((seg[shift:], np.zeros((shift, seg.shape[1]))), axis=0)
        else:
            s = np.concatenate((np.zeros((-shift, seg.shape[1])), seg[:shift]), axis=0)
        out.append(s)
    return np.stack(out)


def scale_segments(x: np.ndarray, scale_min: float, scale_max: float) -> np.ndarray:
    if scale_min is None or scale_max is None:
        return x.copy()
    scales = np.random.uniform(scale_min, scale_max, size=(x.shape[0], 1, 1))
    return x * scales


def mixup(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if alpha is None or alpha <= 0:
        return x.copy(), y.copy()
    lam = np.random.beta(alpha, alpha, size=x.shape[0])
    idx = np.random.permutation(x.shape[0])
    x2 = x[idx]
    y2 = y[idx]
    lam_x = lam.reshape(-1, 1, 1)
    x_mix = x * lam_x + x2 * (1 - lam_x)
    # for labels, create soft labels as one-hot floats
    num_classes = int(np.max(y) + 1)
    y_one = np.eye(num_classes)[y]
    y2_one = np.eye(num_classes)[y2]
    lam_y = lam.reshape(-1, 1)
    y_mix = y_one * lam_y + y2_one * (1 - lam_y)
    # Convert mixed soft labels back to hard labels by argmax (keeps training loop simple)
    y_mix_hard = np.argmax(y_mix, axis=1).astype(np.int64)
    return x_mix, y_mix_hard


def augment_training_set(x: np.ndarray, y: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply simple augmentations and return augmented dataset appended to original.

    Config keys: noise_std, time_shift_pct, scale_min, scale_max, mixup_alpha
    """
    aug_x = []
    aug_y = []
    noise_std = config.get("noise_std", 0.0)
    time_shift_pct = config.get("time_shift_pct", 0.0)
    scale_min = config.get("scale_min", 1.0)
    scale_max = config.get("scale_max", 1.0)
    mixup_alpha = config.get("mixup_alpha", 0.0)

    # simple single-step augmentations
    if noise_std and noise_std > 0:
        aug_x.append(add_gaussian_noise(x, noise_std))
        aug_y.append(y.copy())
    if time_shift_pct and time_shift_pct > 0:
        aug_x.append(time_shift_segments(x, time_shift_pct))
        aug_y.append(y.copy())
    if (scale_min is not None and scale_max is not None) and (scale_min != 1.0 or scale_max != 1.0):
        aug_x.append(scale_segments(x, scale_min, scale_max))
        aug_y.append(y.copy())
    if mixup_alpha and mixup_alpha > 0:
        xm, ym = mixup(x, y, mixup_alpha)
        aug_x.append(xm)
        aug_y.append(ym)

    if not aug_x:
        return x, y

    aug_x = np.concatenate(aug_x, axis=0)
    aug_y = np.concatenate(aug_y, axis=0)

    x_all = np.concatenate((x, aug_x), axis=0)
    y_all = np.concatenate((y, aug_y), axis=0)
    return x_all, y_all


