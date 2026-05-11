import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src2.train import main


if __name__ == "__main__":
	# Random split on standard sit/stand dataset.
	sys.argv = [
		sys.argv[0],
		"--model-type",
		"cnn2d",
		"--sit-path",
		"data/raw/sit.csv",
		"--stand-path",
		"data/raw/stand.csv",
		"--use-hampel",
		*sys.argv[1:],
	]
	main()

