from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def load_numeric_csv(path: str | Path, metadata_columns: int = 1) -> np.ndarray:
    """Load numeric columns from CSV and drop metadata columns.

    This is a robust line-by-line parser that:
    - skips header or non-numeric lines
    - skips malformed rows with wrong field counts
    - drops the first `metadata_columns` numeric fields

    Returns a NumPy array of shape (num_frames, num_numeric_cols - metadata_columns).
    """
    rows: list[list[float]] = []
    path_p = Path(path)
    with path_p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if not parts:
                continue
            # try to parse all parts as float; if any fail, skip the line
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) <= metadata_columns:
                continue
            rows.append(vals)

    if not rows:
        raise ValueError(f"No numeric rows found in {path_p}")

    # keep only rows that match the most common number of fields (skip malformed lines)
    from collections import Counter

    lengths = [len(r) for r in rows]
    most_common_len = Counter(lengths).most_common(1)[0][0]
    filtered = [r for r in rows if len(r) == most_common_len]

    if not filtered:
        raise ValueError(f"No rows with consistent field count found in {path_p}")

    if len(filtered) != len(rows):
        # prefer not to import warnings at module level; emit simple print for visibility
        print(f"Warning: skipped {len(rows)-len(filtered)} malformed rows from {path_p}")

    data = np.array(filtered, dtype=float)
    return data[:, metadata_columns:]


def interleaved_to_complex(csi: np.ndarray) -> np.ndarray:
    """Convert interleaved real/imag columns into complex array.

    Input shape: (num_frames, 2 * num_subcarriers)
    Output shape: (num_frames, num_subcarriers) with complex dtype
    """
    if csi.ndim != 2:
        raise ValueError("CSI input must be 2D array")
    n_feats = csi.shape[1]
    if n_feats % 2 != 0:
        raise ValueError("Expected interleaved real/imag columns (even number of features)")

    real = csi[:, 0::2]
    imag = csi[:, 1::2]
    return real.astype(np.float64) + 1j * imag.astype(np.float64)


def compute_amplitude_phase(csi: np.ndarray, interleaved: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute amplitude and phase from CSI numeric array.

    - If `interleaved`=True, interprets columns as [re0, im0, re1, im1, ...].
    - Returns (amplitude, phase) where phase is in radians by default.
    """
    if interleaved:
        # extract real and imag parts from interleaved columns
        real = csi[:, 0::2].astype(np.float64)
        imag = csi[:, 1::2].astype(np.float64)
    else:
        # assume input is complex-valued
        real = np.real(csi).astype(np.float64)
        imag = np.imag(csi).astype(np.float64)

    amp = np.sqrt(real * real + imag * imag)
    phase = np.arctan2(imag, real)
    return amp, phase


def detect_interleaved(csi: np.ndarray) -> bool:
    """Heuristic to decide if numeric CSV is interleaved real/imag.

    Returns True when number of features is even and values look like signed bytes/ints.
    """
    if csi.ndim != 2:
        return False
    if csi.shape[1] % 2 != 0:
        return False

    # If values have large magnitude (>> 10) it's likely raw IQ integers
    mag = np.max(np.abs(csi))
    return mag > 20


def save_matrix_csv(mat: np.ndarray, path: str | Path, header_prefix: str = "") -> None:
    df = pd.DataFrame(mat)
    if header_prefix:
        df.columns = [f"{header_prefix}{i}" for i in range(df.shape[1])]
    df.to_csv(path, index=False)


def process_file(
    path: str | Path,
    metadata_columns: int = 1,
    mode: str = "auto",
    out_prefix: str | Path | None = None,
    phase_degrees: bool = False,
    save: bool = True,
) -> tuple[np.ndarray, np.ndarray, Path | None, Path | None]:
    """Load a CSV, compute amplitude and phase, optionally save results.

    Returns (amp, phase, amp_path, phase_path). If `save` is False, paths are None.
    """
    p = Path(path)
    if out_prefix:
        out_prefix_p = Path(out_prefix)
    else:
        out_prefix_p = p.with_suffix("")

    csi = load_numeric_csv(p, metadata_columns=metadata_columns)

    if mode == "auto":
        interleaved = detect_interleaved(csi)
    else:
        interleaved = mode == "interleaved"

    amp, phase = compute_amplitude_phase(csi, interleaved=interleaved)
    if phase_degrees:
        phase = np.degrees(phase)

    amp_path = None
    phase_path = None
    if save:
        amp_path = out_prefix_p.with_suffix(".amp.csv")
        phase_path = out_prefix_p.with_suffix(".phase.csv")
        save_matrix_csv(amp, amp_path, header_prefix="amp_")
        save_matrix_csv(phase, phase_path, header_prefix="phase_")

    return amp, phase, amp_path, phase_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute amplitude and phase from CSI CSV files")
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("--metadata-columns", type=int, default=1, help="Number of leading numeric metadata columns to drop (default: 1)")
    parser.add_argument("--mode", choices=("auto", "interleaved", "numeric_complex"), default="auto", help="How to interpret numeric columns")
    parser.add_argument("--out-prefix", default=None, help="Output file prefix (default: input basename without extension)")
    parser.add_argument("--phase-degrees", action="store_true", help="Save phase in degrees instead of radians")
    args = parser.parse_args()

    path = Path(args.input)
    if args.out_prefix:
        out_prefix = Path(args.out_prefix)
    else:
        out_prefix = path.with_suffix("")

    amp, phase, amp_path, phase_path = process_file(
        path,
        metadata_columns=args.metadata_columns,
        mode=args.mode,
        out_prefix=out_prefix,
        phase_degrees=args.phase_degrees,
        save=True,
    )
    print(f"Saved amplitude to {amp_path} and phase to {phase_path}")


if __name__ == "__main__":
    main()

