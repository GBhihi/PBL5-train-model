from __future__ import annotations

import torch
from torch import nn


class Cnn2dClassifier(nn.Module):
	def __init__(self, in_channels: int, num_classes: int = 2):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.AdaptiveAvgPool2d((4, 4)),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(64 * 4 * 4, 128),
			nn.ReLU(),
			nn.Linear(128, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		return self.classifier(x)


class LstmCnnClassifier(nn.Module):
	def __init__(self, feature_dim: int, num_classes: int = 2):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv1d(feature_dim, 64, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Conv1d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
		)
		self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
		self.classifier = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (batch, time, feature_dim)
		x = x.transpose(1, 2)
		x = self.conv(x)
		x = x.transpose(1, 2)
		output, _ = self.lstm(x)
		last_step = output[:, -1, :]
		return self.classifier(last_step)


def build_model(model_type: str, input_shape: tuple[int, ...], num_classes: int = 2) -> nn.Module:
	key = model_type.lower()
	if key == "cnn2d":
		in_channels = input_shape[-1]
		return Cnn2dClassifier(in_channels=in_channels, num_classes=num_classes)
	if key == "lstmcnn":
		feature_dim = input_shape[-1]
		return LstmCnnClassifier(feature_dim=feature_dim, num_classes=num_classes)
	raise ValueError(f"Unsupported model_type: {model_type}")

