from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from src.evaluate import evaluate_classification
from src.model import build_model
from src.preprocess import align_feature_dims, apply_hampel, butterworth_lowpass, load_csi_csv, normalize_global
from src.spectrogram import convert_segments_to_spectrogram
from src.window import create_segments


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train CSI activity recognition model")
	parser.add_argument("--model-type", choices=["cnn2d", "lstmcnn"], default="cnn2d")
	parser.add_argument("--sit-path", default="data/raw/sit.csv")
	parser.add_argument("--stand-path", default="data/raw/stand.csv")
	parser.add_argument("--window-size", type=int, default=600)
	parser.add_argument("--step", type=int, default=200)
	parser.add_argument("--cutoff", type=float, default=0.1)
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument("--nperseg", type=int, default=128)
	parser.add_argument("--use-hampel", action="store_true")
	parser.add_argument("--save-model", default=None)
	return parser


def default_model_output_path(model_type: str) -> Path:
	out_dir = Path("models") / "checkpoints"
	out_dir.mkdir(parents=True, exist_ok=True)
	return out_dir / f"{model_type}.keras"


def train(args: argparse.Namespace) -> dict[str, object]:
	print("Loading CSI files...")
	sit = load_csi_csv(args.sit_path)
	stand = load_csi_csv(args.stand_path)
	sit, stand = align_feature_dims(sit, stand)

	if args.use_hampel:
		print("Applying Hampel filter...")
		sit = apply_hampel(sit)
		stand = apply_hampel(stand)

	print("Applying Butterworth filter...")
	sit = butterworth_lowpass(sit, cutoff=args.cutoff)
	stand = butterworth_lowpass(stand, cutoff=args.cutoff)

	x_sit, y_sit = create_segments(sit, 0, window_size=args.window_size, step=args.step)
	x_stand, y_stand = create_segments(stand, 1, window_size=args.window_size, step=args.step)

	x = np.vstack((x_sit, x_stand))
	y = np.hstack((y_sit, y_stand))

	if args.model_type == "cnn2d":
		print("Converting CSI segments to spectrogram...")
		x = convert_segments_to_spectrogram(x, nperseg=args.nperseg)

	x = normalize_global(x)
	y_onehot = to_categorical(y, 2)

	x_train, x_test, y_train, y_test = train_test_split(
		x,
		y_onehot,
		test_size=args.test_size,
		random_state=args.random_state,
		stratify=y,
	)

	print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

	model = build_model(model_type=args.model_type, input_shape=x_train.shape[1:], num_classes=2)
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
	model.summary()

	callbacks = []
	if args.model_type == "lstmcnn":
		callbacks.append(EarlyStopping(patience=5, restore_best_weights=True))

	history = model.fit(
		x_train,
		y_train,
		epochs=args.epochs,
		batch_size=args.batch_size,
		validation_data=(x_test, y_test),
		callbacks=callbacks,
		verbose=1,
	)

	y_pred_prob = model.predict(x_test, verbose=0)
	result = evaluate_classification(y_test, y_pred_prob, labels=["sit", "stand"])

	print("Confusion Matrix:")
	print(result["confusion_matrix"])
	print("\nClassification Report:")
	print(result["classification_report"])

	model_path = Path(args.save_model) if args.save_model else default_model_output_path(args.model_type)
	model.save(model_path)
	print(f"Saved model to: {model_path}")

	return {
		"model_path": str(model_path),
		"history": history.history,
		"metrics": result,
	}


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()
	train(args)


if __name__ == "__main__":
	main()

