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


def align_feature_dims(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Align two CSI arrays by truncating both to smallest feature dimension."""
	feature_dim = min(a.shape[1], b.shape[1])
	return a[:, :feature_dim], b[:, :feature_dim]


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

