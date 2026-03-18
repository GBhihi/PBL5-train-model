import sys
from pathlib import Path

# Ensure project root is importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--model-type",
        "lstmcnn",
        *sys.argv[1:],
    ]
    main()
