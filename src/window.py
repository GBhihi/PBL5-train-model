from __future__ import annotations

import numpy as np


def create_segments(
	data: np.ndarray,
	label: int,
	window_size: int = 256,
	step: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
	"""Create fixed-size sliding-window segments for a single label."""
	x: list[np.ndarray] = []
	y: list[int] = []

	for start in range(0, len(data) - window_size, step):
		x.append(data[start : start + window_size])
		y.append(label)

	if not x:
		raise ValueError(
			f"No segments generated. Check window_size={window_size}, step={step}, data_length={len(data)}"
		)

	return np.array(x), np.array(y)

