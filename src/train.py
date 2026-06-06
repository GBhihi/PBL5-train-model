from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from src.model import build_model
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
	from torch.utils.tensorboard import SummaryWriter
except Exception:
	SummaryWriter = None

from src.evaluate import evaluate_classification
from src.model import build_model
from src.preprocess import (
	augment_training_set,
)
from src.spectrogram import convert_segments_to_spectrogram


def _load_json_config(config_path: str | None) -> dict[str, object]:
	if not config_path:
		return {}
	path = Path(config_path)
	if not path.exists():
		raise FileNotFoundError(f"Config file not found: {path}")
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def build_arg_parser() -> argparse.ArgumentParser:
	# First parse only --config, then inject defaults from JSON.
	config_parser = argparse.ArgumentParser(add_help=False)
	config_parser.add_argument("--config", default=None)
	known_args, _ = config_parser.parse_known_args()
	config_values = _load_json_config(known_args.config)

	parser = argparse.ArgumentParser(description="Train CSI activity recognition model")
	parser.add_argument("--config", default=None, help="Path to JSON config file")
	parser.add_argument("--model-type", choices=["cnn2d", "lstmcnn"], default="cnn2d")
	parser.add_argument(
		"--data-csv",
		dest="data_csv",
		default="data/clean.csv",
		help="Path to clean CSV file: features..., label, recording_id",
	)
	parser.add_argument("--window-size", type=int, default=512)
	parser.add_argument("--step", type=int, default=256)
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument("--nperseg", type=int, default=32)
	# optimizer / regularization / augmentation options (can be provided in JSON config)
	parser.add_argument("--optimizer", default="AdamW", help="Optimizer name (AdamW or SGD)")
	parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer")
	parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability to pass to model")
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--output-dir", default="experiments/results")
	parser.add_argument("--run-name", default=None)
	parser.add_argument("--save-model", default=None)

	if config_values:
		parser.set_defaults(**config_values)

	return parser


def default_model_output_path(model_type: str) -> Path:
	out_dir = Path("models") / "checkpoints"
	out_dir.mkdir(parents=True, exist_ok=True)
	return out_dir / f"{model_type}.pt"


def _create_run_dir(output_dir: str, run_name: str | None, model_type: str) -> Path:
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	name = run_name if run_name else f"{model_type}_{stamp}"
	run_dir = Path(output_dir) / name
	run_dir.mkdir(parents=True, exist_ok=True)
	return run_dir


def _save_history_csv(history: dict[str, list[float]], out_path: Path) -> None:
	rows = zip(
		range(1, len(history["train_loss"]) + 1),
		history["train_loss"],
		history["train_accuracy"],
		history["val_loss"],
		history["val_accuracy"],
	)
	with out_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
		for row in rows:
			writer.writerow(row)


def _to_torch_input(x: np.ndarray, model_type: str) -> torch.Tensor:
	if model_type == "cnn2d":
		# (N, H, W, C) -> (N, C, H, W)
		x = np.transpose(x, (0, 3, 1, 2))
	return torch.from_numpy(x.astype(np.float32))


def _finalize_segments_before_split(x: np.ndarray, model_type: str, nperseg: int) -> np.ndarray:
	if model_type == "cnn2d":
		print("Converting CSI segments to spectrogram...")
		x = convert_segments_to_spectrogram(x, nperseg=nperseg)
	return x


def _read_merged_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Read clean CSV and return (features, labels, recording_ids).

	New clean files use: feature columns..., label, recording_id.
	Older clean files with only feature columns..., label are still supported and
	get recording_id=0 for every row.
	"""
	rows: list[list[float]] = []
	labels: list[int] = []
	recording_ids: list[int] = []
	with open(path, 'r', newline='') as f:
		reader = csv.reader(f)
		for r in reader:
			if not r:
				continue
			# remove trailing empty cells at end (safety) and empty cells immediately
			# before the final label (handles patterns like "...,33,,0")
			while len(r) > 1 and r[-1] == "":
				r.pop()
			# remove any empty cells directly before the last metadata column
			while len(r) >= 2 and r[-2] == "":
				r.pop(-2)
			if len(r) < 2:
				continue

			has_recording_id = len(r) >= 194
			if has_recording_id:
				*feat, lab, recording_id = r
			else:
				*feat, lab = r
				recording_id = "0"

			parsed_feat = []
			for v in feat:
				try:
					parsed_feat.append(float(v))
				except Exception:
					parsed_feat.append(0.0)
			try:
				labels.append(int(float(lab)))
			except Exception:
				labels.append(-1)
			try:
				recording_ids.append(int(float(recording_id)))
			except Exception:
				recording_ids.append(-1)
			rows.append(parsed_feat)
	if not rows:
		raise ValueError(f"No rows found in {path}")

	# Keep only rows with the most common feature length.
	lengths = [len(r) for r in rows]
	from collections import Counter

	most_common_len = Counter(lengths).most_common(1)[0][0]
	filtered_rows = [r for r in rows if len(r) == most_common_len]
	filtered_labels = [lab for r, lab in zip(rows, labels) if len(r) == most_common_len]
	filtered_recording_ids = [rid for r, rid in zip(rows, recording_ids) if len(r) == most_common_len]
	if len(filtered_rows) != len(rows):
		print(
			f"Warning: dropped {len(rows) - len(filtered_rows)} rows with inconsistent length in {path}. "
			f"Using length={most_common_len}."
		)

	X = np.array(filtered_rows, dtype=np.float32)
	y = np.array(filtered_labels, dtype=np.int64)
	recording_ids_arr = np.array(filtered_recording_ids, dtype=np.int64)
	return X, y, recording_ids_arr


def _create_segments_from_labeled_rows(
	data: np.ndarray,
	labels: np.ndarray,
	recording_ids: np.ndarray,
	window_size: int,
	step: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Create sliding windows from time-series rows with per-row labels.

	A window is kept only if all label and recording_id values inside the window
	are identical.
	"""
	x_list: list[np.ndarray] = []
	y_list: list[int] = []
	length = len(data)
	for start in range(0, length - window_size + 1, step):
		end = start + window_size
		window_labels = labels[start:end]
		window_recording_ids = recording_ids[start:end]
		# if any -1 label/recording_id or mixed metadata -> skip
		if window_labels.size == 0:
			continue
		same_label = np.all(window_labels == window_labels[0]) and window_labels[0] >= 0
		same_recording = np.all(window_recording_ids == window_recording_ids[0]) and window_recording_ids[0] >= 0
		if same_label and same_recording:
			x_list.append(data[start:end])
			y_list.append(int(window_labels[0]))
		else:
			# mixed label/recording_id -> skip
			continue
	if not x_list:
		raise ValueError(f"No segments generated from labeled rows. Check window_size={window_size}, step={step}, data_length={length}")
	return np.array(x_list), np.array(y_list)


def _split_indices_by_label_recording_contiguous(
	labels: np.ndarray,
	recording_ids: np.ndarray,
	ratios: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	if not np.isclose(sum(ratios), 1.0):
		raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

	train_ratio, val_ratio, test_ratio = ratios

	train_idx: list[int] = []
	val_idx: list[int] = []
	test_idx: list[int] = []

	for label in np.unique(labels):
		if label < 0:
			continue
		label_recording_ids = np.unique(recording_ids[labels == label])
		for recording_id in label_recording_ids:
			if recording_id < 0:
				continue
			group_indices = np.where((labels == label) & (recording_ids == recording_id))[0]
			if group_indices.size == 0:
				continue
			group_indices = np.sort(group_indices)
			n_total = group_indices.size
			n_train = int(n_total * train_ratio)
			n_val = int(n_total * val_ratio)
			n_test = n_total - n_train - n_val

			train_idx.extend(group_indices[:n_train].tolist())
			val_idx.extend(group_indices[n_train : n_train + n_val].tolist())
			test_idx.extend(group_indices[n_train + n_val : n_train + n_val + n_test].tolist())

	return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def _fit_standardizer_3d(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	if x_train.ndim == 3:
		mu = np.mean(x_train, axis=(0, 1), keepdims=True)
		sigma = np.std(x_train, axis=(0, 1), keepdims=True)
	elif x_train.ndim == 4:
		mu = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
		sigma = np.std(x_train, axis=(0, 1, 2), keepdims=True)
	else:
		raise ValueError(f"Unsupported input ndim for standardization: {x_train.ndim}")
	return mu, sigma


def _apply_standardizer(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
	return (x - mu) / (sigma + 1e-8)


def _fit_global_normalizer(x_train: np.ndarray) -> float:
	return float(np.max(np.abs(x_train)))


def _apply_global_normalizer(x: np.ndarray, max_abs: float, eps: float = 1e-8) -> np.ndarray:
	return x / (max_abs + eps)


def _run_epoch(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer | None,
	device: torch.device,
) -> tuple[float, float]:
	training = optimizer is not None
	model.train(training)

	total_loss = 0.0
	total_correct = 0
	total_count = 0

	for xb, yb in loader:
		xb = xb.to(device)
		yb = yb.to(device)

		if training:
			optimizer.zero_grad()

		logits = model(xb)
		loss = criterion(logits, yb)

		if training:
			loss.backward()
			optimizer.step()

		pred = torch.argmax(logits, dim=1)
		total_loss += float(loss.item()) * yb.size(0)
		total_correct += int((pred == yb).sum().item())
		total_count += int(yb.size(0))

	avg_loss = total_loss / max(total_count, 1)
	avg_acc = total_correct / max(total_count, 1)
	return avg_loss, avg_acc


def _predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
	model.eval()
	all_probs: list[np.ndarray] = []
	with torch.no_grad():
		for xb, _ in loader:
			xb = xb.to(device)
			logits = model(xb)
			probs = torch.softmax(logits, dim=1)
			all_probs.append(probs.cpu().numpy())
	return np.vstack(all_probs)


def train(args: argparse.Namespace) -> dict[str, object]:
	print("Loading clean dataset for training...")
	data_csv_path = args.data_csv
	print(f"Loading clean CSV: {data_csv_path}")
	X_rows, row_labels, row_recording_ids = _read_merged_csv(data_csv_path)

	# Split rows by each (label, recording_id) group (80/10/10), then create segments inside each split.
	train_idx, val_idx, test_idx = _split_indices_by_label_recording_contiguous(
		row_labels,
		row_recording_ids,
		ratios=(0.8, 0.1, 0.1),
	)
	print(
		"Row split sizes:",
		{"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
	)

	x_train, y_train = _create_segments_from_labeled_rows(
		X_rows[train_idx],
		row_labels[train_idx],
		row_recording_ids[train_idx],
		window_size=args.window_size,
		step=args.step,
	)
	x_val, y_val = _create_segments_from_labeled_rows(
		X_rows[val_idx],
		row_labels[val_idx],
		row_recording_ids[val_idx],
		window_size=args.window_size,
		step=args.step,
	)
	x_test, y_test = _create_segments_from_labeled_rows(
		X_rows[test_idx],
		row_labels[test_idx],
		row_recording_ids[test_idx],
		window_size=args.window_size,
		step=args.step,
	)


	# Apply augmentation only to training set (configurable via JSON 'augmentation' object)
	aug_cfg = getattr(args, "augmentation", {}) or {}
	if aug_cfg:
		print("Applying augmentation to training set:", aug_cfg)
		x_train, y_train = augment_training_set(x_train, y_train, aug_cfg)

	# Finalize segments (e.g., convert to spectrogram for cnn2d) AFTER augmentation
	x_train = _finalize_segments_before_split(x_train, args.model_type, args.nperseg)
	x_val = _finalize_segments_before_split(x_val, args.model_type, args.nperseg)
	x_test = _finalize_segments_before_split(x_test, args.model_type, args.nperseg)

	print("Fitting standardizer on training set only...")
	mu, sigma = _fit_standardizer_3d(x_train)
	x_train = _apply_standardizer(x_train, mu, sigma)
	x_val = _apply_standardizer(x_val, mu, sigma)
	x_test = _apply_standardizer(x_test, mu, sigma)

	print("Applying global normalization from training set only...")
	max_abs = _fit_global_normalizer(x_train)
	x_train = _apply_global_normalizer(x_train, max_abs)
	x_val = _apply_global_normalizer(x_val, max_abs)
	x_test = _apply_global_normalizer(x_test, max_abs)

	labels, counts = np.unique(y_train, return_counts=True)
	print("Train label distribution:")
	for label, count in zip(labels, counts):
		print(f"Label {label}: {count} samples")

	labels_val, counts_val = np.unique(y_val, return_counts=True)
	print("\nValidation label distribution:")
	for label, count in zip(labels_val, counts_val):
		print(f"Label {label}: {count} samples")

	labels_test, counts_test = np.unique(y_test, return_counts=True)
	print("\nTest label distribution:")
	for label, count in zip(labels_test, counts_test):
		print(f"Label {label}: {count} samples")

	print(f"Train shape: {x_train.shape}, Validation shape: {x_val.shape}, Test shape: {x_test.shape}")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	run_dir = _create_run_dir(args.output_dir, args.run_name, args.model_type)
	print(f"Run directory: {run_dir}")

	x_train_tensor = _to_torch_input(x_train, args.model_type)
	x_val_tensor = _to_torch_input(x_val, args.model_type)
	x_test_tensor = _to_torch_input(x_test, args.model_type)
	y_train_tensor = torch.from_numpy(y_train.astype(np.int64))
	y_val_tensor = torch.from_numpy(y_val.astype(np.int64))
	y_test_tensor = torch.from_numpy(y_test.astype(np.int64))

	train_loader = DataLoader(
		TensorDataset(x_train_tensor, y_train_tensor),
		batch_size=args.batch_size,
		shuffle=True,
	)
	val_loader = DataLoader(
		TensorDataset(x_val_tensor, y_val_tensor),
		batch_size=args.batch_size,
		shuffle=False,
	)
	test_loader = DataLoader(
		TensorDataset(x_test_tensor, y_test_tensor),
		batch_size=args.batch_size,
		shuffle=False,
	)

	for _, y_batch in train_loader:
		batch_labels, batch_counts = np.unique(y_batch.numpy(), return_counts=True)
		print("\nFirst batch label distribution:")
		print(list(zip(batch_labels.tolist(), batch_counts.tolist())))
		break

	# Determine classes from labels present in training/validation sets
	unique_labels_all = np.unique(np.concatenate((y_train, y_val)))
	class_labels = sorted([int(x) for x in unique_labels_all.tolist()])
	# Allow optional class name override from config/args
	if getattr(args, "class_names", None):
		class_names = list(getattr(args, "class_names"))
	else:
		if set(class_labels) == {0, 1}:
			class_names = ["no_person", "person"]
		else:
			class_names = [f"class_{lbl}" for lbl in class_labels]

	num_classes = len(class_names)
	print(f"Detected classes: {class_labels} -> names={class_names}")

	model = build_model(model_type=args.model_type, input_shape=x_train.shape[1:], num_classes=num_classes, dropout=getattr(args, "dropout", 0.0))
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()

	# configure optimizer with weight decay
	opt_name = getattr(args, "optimizer", "adamw")
	wd = getattr(args, "weight_decay", 0.0)
	if isinstance(opt_name, str) and opt_name.lower().startswith("sgd"):
		optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=wd)
	else:
		optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=wd)

	# scheduler (support ReduceLROnPlateau from config scheduler dict)
	scheduler = None
	sch_cfg = getattr(args, "scheduler", None)
	if isinstance(sch_cfg, dict) and sch_cfg.get("type") == "ReduceLROnPlateau":
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, factor=sch_cfg.get("factor", 0.5), patience=sch_cfg.get("patience", 3)
		)

	history: dict[str, list[float]] = {
		"train_loss": [],
		"train_accuracy": [],
		"val_loss": [],
		"val_accuracy": [],
	}
	writer = SummaryWriter(log_dir=str(run_dir / "tb")) if SummaryWriter is not None else None
 
	if writer is None:
		print("TensorBoard writer unavailable. Install tensorboard to enable TB logs.")

	best_val_loss = float("inf")
	best_state: dict[str, torch.Tensor] | None = None

	for epoch in range(1, args.epochs + 1):
		train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_acc = _run_epoch(model, val_loader, criterion, None, device)

		history["train_loss"].append(train_loss)
		history["train_accuracy"].append(train_acc)
		history["val_loss"].append(val_loss)
		history["val_accuracy"].append(val_acc)

		print(
			f"Epoch {epoch}/{args.epochs} | "
			f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
			f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
		)

		if writer is not None:
			writer.add_scalar("loss/train", train_loss, epoch)
			writer.add_scalar("loss/val", val_loss, epoch)
			writer.add_scalar("accuracy/train", train_acc, epoch)
			writer.add_scalar("accuracy/val", val_acc, epoch)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

		# step scheduler if present
		if scheduler is not None:
			# ReduceLROnPlateau expects metric
			if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
				scheduler.step(val_loss)

	if best_state is not None:
		model.load_state_dict(best_state)

	if writer is not None:
		writer.close()

	y_pred_prob = _predict_proba(model, val_loader, device)
	result = evaluate_classification(y_val, y_pred_prob, labels=class_names)

	y_test_prob = _predict_proba(model, test_loader, device)
	test_result = evaluate_classification(y_test, y_test_prob, labels=class_names)

	print("Confusion Matrix:")
	print(result["confusion_matrix"])
	print("\nClassification Report:")
	print(result["classification_report"])

	print("\nTest Confusion Matrix:")
	print(test_result["confusion_matrix"])
	print("\nTest Classification Report:")
	print(test_result["classification_report"])

	model_path = Path(args.save_model) if args.save_model else default_model_output_path(args.model_type)
	best_epoch = int(np.argmin(history["val_loss"]) + 1)
	torch.save(
		{
			"model_type": args.model_type,
			"input_shape": list(x_train.shape[1:]),
			"num_classes": num_classes,
			"class_names": class_names,
			"standardizer_mu": mu,
			"standardizer_sigma": sigma,
			"global_max_abs": max_abs,
			"model": model.state_dict(),
			"state_dict": model.state_dict(),
			"optimizer": optimizer.state_dict(),
			"epoch": len(history["train_loss"]),
			"best_epoch": best_epoch,
			"args": vars(args),
			"history": history,
		},
		model_path,
	)
	print(f"Saved model to: {model_path}")

	history_json = run_dir / "history.json"
	history_csv = run_dir / "history.csv"
	report_txt = run_dir / "classification_report.txt"
	cm_npy = run_dir / "confusion_matrix.npy"
	cm_test_npy = run_dir / "confusion_matrix_test.npy"
	report_test_txt = run_dir / "classification_report_test.txt"

	with history_json.open("w", encoding="utf-8") as f:
		json.dump(history, f, indent=2)
	_save_history_csv(history, history_csv)
	report_txt.write_text(str(result["classification_report"]), encoding="utf-8")
	np.save(cm_npy, result["confusion_matrix"])
	report_test_txt.write_text(str(test_result["classification_report"]), encoding="utf-8")
	np.save(cm_test_npy, test_result["confusion_matrix"])

	print(f"Saved logs: {history_csv}, {history_json}, {report_txt}, {cm_npy}, {report_test_txt}, {cm_test_npy}")

	return {
		"model_path": str(model_path),
		"run_dir": str(run_dir),
		"history": history,
		"metrics": result,
		"test_metrics": test_result,
	}


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()
	train(args)


if __name__ == "__main__":
	main()

