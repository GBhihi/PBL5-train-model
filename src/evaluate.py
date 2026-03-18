from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_classification(
	y_true_onehot: np.ndarray,
	y_pred_prob: np.ndarray,
	labels: Sequence[str] | None = None,
) -> dict[str, object]:
	"""Evaluate classification result from one-hot labels and probability predictions."""
	y_true = np.argmax(y_true_onehot, axis=1)
	y_pred = np.argmax(y_pred_prob, axis=1)

	cm = confusion_matrix(y_true, y_pred)
	report = classification_report(y_true, y_pred, target_names=labels)

	return {
		"y_true": y_true,
		"y_pred": y_pred,
		"confusion_matrix": cm,
		"classification_report": report,
	}

