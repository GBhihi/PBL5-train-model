from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from src.preprocess import align_feature_dims, apply_hampel, butterworth_lowpass, load_csi_csv, normalize_global
from src.spectrogram import convert_segments_to_spectrogram
from src.window import create_segments


def prepare_input_from_csv(
	csv_path: str | Path,
	reference_path: str | Path,
	model_type: str,
	window_size: int,
	step: int,
	use_hampel: bool,
	cutoff: float,
) -> np.ndarray:
	"""Prepare model input tensor from one CSV file."""
	data = load_csi_csv(csv_path)
	ref = load_csi_csv(reference_path)
	data, _ = align_feature_dims(data, ref)

	if use_hampel:
		data = apply_hampel(data)
	data = butterworth_lowpass(data, cutoff=cutoff)

	x, _ = create_segments(data, label=0, window_size=window_size, step=step)
	if model_type.lower() == "cnn2d":
		x = convert_segments_to_spectrogram(x)

	x = normalize_global(x)
	return x


def predict_action(
	model_path: str | Path,
	x: np.ndarray,
	class_names: tuple[str, str] = ("sit", "stand"),
) -> dict[str, object]:
	model = load_model(model_path)
	probs = model.predict(x, verbose=0)

	avg_prob = probs.mean(axis=0)
	pred_idx = int(np.argmax(avg_prob))

	return {
		"predicted_class": class_names[pred_idx],
		"probabilities": {name: float(avg_prob[i]) for i, name in enumerate(class_names)},
	}

