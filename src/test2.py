#!/usr/bin/env python3
"""Simple CLI wrapper that prepares input using `prepare_input_from_csv`
and runs `predict_action` from src.predict.

Usage examples:
  python src/test2.py --model checkpoints_amp3/lstmcnn.pt --csv data/test/test_stand1.csv
  python src/test2.py --model checkpoints_amp3/lstmcnn.pt --csv data/test/test_stand1.csv --reference data/raw/sit.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import torch

from src.predict import prepare_input_from_csv, predict_action


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare input and run prediction (uses src.predict)")
    p.add_argument("--model", required=True, help="Path to checkpoint (pt)")
    p.add_argument("--csv", required=True, help="Path to CSI CSV file to predict on")
    p.add_argument("--reference", default="data/raw/sit.csv", help="Reference CSV to align feature dims (default: data/raw/sit.csv)")
    p.add_argument("--window-size", type=int, default=None, help="Window size (override checkpoint args)")
    p.add_argument("--step", type=int, default=None, help="Step size (override checkpoint args)")
    p.add_argument("--use-hampel", action="store_true", help="Apply Hampel filter when preparing input")
    p.add_argument("--no-hampel", dest="use_hampel", action="store_false", help="Do not apply Hampel filter")
    p.set_defaults(use_hampel=None)
    p.add_argument("--cutoff", type=float, default=None, help="Butterworth cutoff (override checkpoint args)")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # load checkpoint to extract default preprocess params when available
    ckpt = torch.load(str(model_path), map_location="cpu")
    ck_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    model_type = ck_args.get("model_type") or ckpt.get("model_type") or "lstmcnn"
    window_size = args.window_size if args.window_size is not None else int(ck_args.get("window_size", 256))
    step = args.step if args.step is not None else int(ck_args.get("step", 128))
    use_hampel = args.use_hampel if args.use_hampel is not None else bool(ck_args.get("use_hampel", False))
    cutoff = args.cutoff if args.cutoff is not None else float(ck_args.get("cutoff", 0.1))

    print("Model:", model_path)
    print("CSV:", args.csv)
    print("Reference:", args.reference)
    print(f"Using model_type={model_type}, window_size={window_size}, step={step}, use_hampel={use_hampel}, cutoff={cutoff}")

    x = prepare_input_from_csv(args.csv, args.reference, model_type, window_size, step, use_hampel, cutoff)

    res = predict_action(model_path, x)

    print("\nPrediction:")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
