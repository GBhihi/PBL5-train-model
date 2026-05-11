from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.model import build_model
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

	# Do not apply the training standardizer/global normalizer here —
	# keep this function returning finalized segments (spectrogram for cnn2d)
	# so that predict_action can apply the exact same standardizer saved in checkpoint.
	return x


def predict_action(
	model_path: str | Path,
	x: np.ndarray,
	class_names: tuple[str, ...] = ("sit", "stand", "walk"),
) -> dict[str, object]:
	checkpoint = torch.load(model_path, map_location="cpu")
	model_type = checkpoint.get("model_type", "cnn2d")
	input_shape = tuple(checkpoint["input_shape"])
	# prefer class names from checkpoint when available
	ck_classes = checkpoint.get("class_names")
	if isinstance(ck_classes, (list, tuple)) and len(ck_classes) > 0:
		class_names = tuple(ck_classes)

	# load standardizer/global stats if saved during training
	standardizer_mu = checkpoint.get("standardizer_mu", None)
	standardizer_sigma = checkpoint.get("standardizer_sigma", None)
	global_max_abs = checkpoint.get("global_max_abs", None)

	# Apply standardizer (if present) BEFORE transposing for cnn2d
	if standardizer_mu is not None and standardizer_sigma is not None:
		mu = np.asarray(standardizer_mu)
		sigma = np.asarray(standardizer_sigma)
		try:
			x = (x - mu) / (sigma + 1e-8)
		except Exception:
			# fallback: try broadcasting explicitly
			x = (x - mu.astype(x.dtype)) / (sigma.astype(x.dtype) + 1e-8)

	# Apply global normalizer if present, else fallback to local normalize
	if global_max_abs is not None:
		x = x / (float(global_max_abs) + 1e-8)
	else:
		x = normalize_global(x)

	# For cnn2d the model expects (N, C, H, W)
	if model_type == "cnn2d":
		x = np.transpose(x, (0, 3, 1, 2))

	# Build model using number of classes from class_names
	model = build_model(model_type=model_type, input_shape=input_shape, num_classes=len(class_names))

	# Load state dict (support both 'model' and 'state_dict' keys)
	state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
	# strip possible 'module.' prefixes
	new_sd = {}
	for k, v in state_dict.items():
		new_k = k[len("module."):] if k.startswith("module.") else k
		new_sd[new_k] = v
	model.load_state_dict(new_sd)
	model.eval()

	with torch.no_grad():
		tensor_x = torch.from_numpy(x.astype(np.float32))
		logits = model(tensor_x)
		probs = torch.softmax(logits, dim=1).cpu().numpy()

	avg_prob = probs.mean(axis=0)
	pred_idx = int(np.argmax(avg_prob))

	return {
		"predicted_class": class_names[pred_idx],
		"probabilities": {name: float(avg_prob[i]) for i, name in enumerate(class_names)},
	}

