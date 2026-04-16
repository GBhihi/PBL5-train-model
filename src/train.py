from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

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
from src.preprocess import align_feature_dims, align_feature_dims_multi, apply_hampel, butterworth_lowpass, load_csi_csv, normalize_global
from src.spectrogram import convert_segments_to_spectrogram
from src.window import create_segments


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
	parser.add_argument("--sit_path", default="data/raw/sit.csv")
	parser.add_argument("--stand_path", default="data/raw/stand.csv")
	parser.add_argument("--walk_path", default="data/raw/walk.csv")
	parser.add_argument("--window-size", type=int, default=256)
	parser.add_argument("--step", type=int, default=128)
	parser.add_argument("--cutoff", type=float, default=0.1)
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument("--nperseg", type=int, default=128)
	parser.add_argument("--use-hampel", action="store_true")
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
	print("Loading CSI files...")
	sit = load_csi_csv(args.sit_path)
	stand = load_csi_csv(args.stand_path)
	walk = load_csi_csv(args.walk_path)

	# Align feature dimensions across all classes by truncating to smallest feature dim
	sit, stand, walk = align_feature_dims_multi(sit, stand, walk)

	if args.use_hampel:
		print("Applying Hampel filter...")
		sit = apply_hampel(sit)
		stand = apply_hampel(stand)
		walk = apply_hampel(walk)

	print("Applying Butterworth filter...")
	sit = butterworth_lowpass(sit, cutoff=args.cutoff)
	stand = butterworth_lowpass(stand, cutoff=args.cutoff)
	walk = butterworth_lowpass(walk, cutoff=args.cutoff)

	x_sit, y_sit = create_segments(sit, 0, window_size=args.window_size, step=args.step)
	x_stand, y_stand = create_segments(stand, 1, window_size=args.window_size, step=args.step)
	# walk uses label index 2
	x_walk, y_walk = create_segments(walk, 2, window_size=args.window_size, step=args.step)

	x = np.vstack((x_sit, x_stand, x_walk))
	y = np.hstack((y_sit, y_stand, y_walk))

	if args.model_type == "cnn2d":
		print("Converting CSI segments to spectrogram...")
		x = convert_segments_to_spectrogram(x, nperseg=args.nperseg)

	x = normalize_global(x)

	x_train, x_test, y_train, y_test = train_test_split(
		x,
		y,
		test_size=args.test_size,
		random_state=args.random_state,
		stratify=y,
	)

	labels, counts = np.unique(y_train, return_counts=True)
	print("Train label distribution:")
	for label, count in zip(labels, counts):
		print(f"Label {label}: {count} samples")

	labels_val, counts_val = np.unique(y_test, return_counts=True)
	print("\nValidation label distribution:")
	for label, count in zip(labels_val, counts_val):
		print(f"Label {label}: {count} samples")

	print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	run_dir = _create_run_dir(args.output_dir, args.run_name, args.model_type)
	print(f"Run directory: {run_dir}")

	x_train_tensor = _to_torch_input(x_train, args.model_type)
	x_test_tensor = _to_torch_input(x_test, args.model_type)
	y_train_tensor = torch.from_numpy(y_train.astype(np.int64))
	y_test_tensor = torch.from_numpy(y_test.astype(np.int64))

	train_loader = DataLoader(
		TensorDataset(x_train_tensor, y_train_tensor),
		batch_size=args.batch_size,
		shuffle=True,
	)
	val_loader = DataLoader(
		TensorDataset(x_test_tensor, y_test_tensor),
		batch_size=args.batch_size,
		shuffle=False,
	)

	for _, y_batch in train_loader:
		batch_labels, batch_counts = np.unique(y_batch.numpy(), return_counts=True)
		print("\nFirst batch label distribution:")
		print(list(zip(batch_labels.tolist(), batch_counts.tolist())))
		break

	model = build_model(model_type=args.model_type, input_shape=x_train.shape[1:], num_classes=3)
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

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
	patience = 5 if args.model_type == "lstmcnn" else None
	wait = 0

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
			wait = 0
		else:
			if patience is not None:
				wait += 1
				if wait >= patience:
					print("Early stopping triggered.")
					break

	if best_state is not None:
		model.load_state_dict(best_state)

	if writer is not None:
		writer.close()

	y_pred_prob = _predict_proba(model, val_loader, device)
	result = evaluate_classification(y_test, y_pred_prob, labels=["sit", "stand", "walk"])

	print("Confusion Matrix:")
	print(result["confusion_matrix"])
	print("\nClassification Report:")
	print(result["classification_report"])

	model_path = Path(args.save_model) if args.save_model else default_model_output_path(args.model_type)
	best_epoch = int(np.argmin(history["val_loss"]) + 1)
	torch.save(
		{
			"model_type": args.model_type,
			"input_shape": list(x_train.shape[1:]),
			"num_classes": 3,
			"class_names": ["sit", "stand", "walk"],
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

	with history_json.open("w", encoding="utf-8") as f:
		json.dump(history, f, indent=2)
	_save_history_csv(history, history_csv)
	report_txt.write_text(str(result["classification_report"]), encoding="utf-8")
	np.save(cm_npy, result["confusion_matrix"])

	print(f"Saved logs: {history_csv}, {history_json}, {report_txt}, {cm_npy}")

	return {
		"model_path": str(model_path),
		"run_dir": str(run_dir),
		"history": history,
		"metrics": result,
	}


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()
	train(args)


if __name__ == "__main__":
	main()

