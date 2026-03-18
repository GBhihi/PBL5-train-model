from __future__ import annotations

import numpy as np
from scipy.signal import stft


def segment_to_spectrogram(segment: np.ndarray, nperseg: int = 128) -> np.ndarray:
	"""Convert one CSI segment into a multi-channel spectrogram tensor."""
	specs: list[np.ndarray] = []

	for subcarrier_idx in range(segment.shape[1]):
		_, _, zxx = stft(segment[:, subcarrier_idx], nperseg=nperseg)
		specs.append(np.abs(zxx))

	spec = np.array(specs)
	return np.transpose(spec, (1, 2, 0))


def convert_segments_to_spectrogram(x: np.ndarray, nperseg: int = 128) -> np.ndarray:
	"""Convert all CSI segments to spectrogram representation."""
	return np.array([segment_to_spectrogram(seg, nperseg=nperseg) for seg in x])

