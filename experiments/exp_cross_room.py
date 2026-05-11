import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src2.train import main


if __name__ == "__main__":
	# Baseline cross-room script: train with home dataset files.
	# You can override file paths from CLI for cafe/home combinations.
	sys.argv = [
		sys.argv[0],
		"--model-type",
		"lstmcnn",
		"--sit-path",
		"data/raw/home/sit/sit_raw4.csv",
		"--stand-path",
		"data/raw/home/stand/stand_raw4.csv",
		*sys.argv[1:],
	]
	main()

