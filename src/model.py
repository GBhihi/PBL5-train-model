from __future__ import annotations

from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, LSTM, MaxPooling1D, MaxPooling2D
from tensorflow.keras.models import Sequential


def create_cnn2d_model(input_shape: tuple[int, ...], num_classes: int = 2) -> Sequential:
	model = Sequential(
		[
			Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
			MaxPooling2D((2, 2)),
			Conv2D(64, (3, 3), activation="relu"),
			MaxPooling2D((2, 2)),
			Flatten(),
			Dense(128, activation="relu"),
			Dense(num_classes, activation="softmax"),
		]
	)
	return model


def create_lstmcnn_model(input_shape: tuple[int, ...], num_classes: int = 2) -> Sequential:
	model = Sequential(
		[
			Conv1D(64, kernel_size=5, activation="relu", input_shape=input_shape),
			MaxPooling1D(pool_size=2),
			Conv1D(128, kernel_size=3, activation="relu"),
			MaxPooling1D(pool_size=2),
			LSTM(64),
			Dropout(0.5),
			Dense(64, activation="relu"),
			Dense(num_classes, activation="softmax"),
		]
	)
	return model


def build_model(model_type: str, input_shape: tuple[int, ...], num_classes: int = 2) -> Sequential:
	key = model_type.lower()
	if key == "cnn2d":
		return create_cnn2d_model(input_shape=input_shape, num_classes=num_classes)
	if key == "lstmcnn":
		return create_lstmcnn_model(input_shape=input_shape, num_classes=num_classes)
	raise ValueError(f"Unsupported model_type: {model_type}")

