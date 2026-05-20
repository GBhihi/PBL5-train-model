# test_predict.py
import sys
import os
from pathlib import Path
import torch
import numpy as np

# --- 1. Thêm đường dẫn src ---
# --- 1. Thêm đường dẫn root của repo ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- 2. Import module ---
from src.model import build_model
from src.preprocess import butterworth_lowpass
from src.spectrogram import convert_segments_to_spectrogram
from src.window import create_segments

# --- 3. Đường dẫn file ---
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "../checkpoints_amp15/lstmcnn.pt")
csv_path = os.path.join(BASE_DIR, "../data/raw/no_3.5_15.csv")
# --- 4. Load checkpoint trước để lấy metadata ---
ckpt = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)

if isinstance(ckpt, dict):
    model_type = ckpt.get("model_type", "lstmcnn")
    num_classes = ckpt.get("num_classes", 3)
    train_args = ckpt.get("args", {})
    saved_input_shape = ckpt.get("input_shape", None)
    class_names = ckpt.get("class_names", None)
    standardizer_mu = ckpt.get("standardizer_mu", None)
    standardizer_sigma = ckpt.get("standardizer_sigma", None)
    global_max_abs = ckpt.get("global_max_abs", None)
else:
    model_type = "lstmcnn"
    num_classes = 3
    train_args = {}
    saved_input_shape = None
    class_names = ["sit", "stand", "walk"]
    standardizer_mu = None
    standardizer_sigma = None
    global_max_abs = None

# Ensure class_names aligns with num_classes; provide sensible defaults for 2-class person/no-person
try:
    n_cls = int(num_classes)
except Exception:
    n_cls = 3
if not class_names or len(class_names) != n_cls:
    if n_cls == 2:
        class_names = ["no_person", "person"]
    elif n_cls == 3:
        class_names = ["sit", "stand", "walk"]
    else:
        class_names = [f"class_{i}" for i in range(n_cls)]

# lấy state_dict
if isinstance(ckpt, dict):
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    else:
        sd = ckpt
else:
    sd = ckpt

# bỏ prefix module. nếu có
new_sd = {}
for k, v in sd.items():
    new_k = k[len("module."):] if k.startswith("module.") else k
    new_sd[new_k] = v
sd = new_sd

# --- 5. Lấy đúng tham số train ---
window_size = int(train_args.get("window_size", 4096))
step = int(train_args.get("step", 2048))
cutoff = float(train_args.get("cutoff", 0.1))
use_hampel = bool(train_args.get("use_hampel", False))
nperseg = train_args.get("nperseg", None)
# amp/phase options saved from training
amp_log = bool(train_args.get("amp_log", False))

if saved_input_shape is None:
    raise ValueError("Checkpoint missing input_shape; cannot align test data like train.")
if nperseg is None:
    raise ValueError("Checkpoint missing nperseg; cannot match train spectrogram settings.")
nperseg = int(nperseg)

if standardizer_mu is None or standardizer_sigma is None or global_max_abs is None:
    raise ValueError(
        "Checkpoint missing standardization stats. Retrain with updated train.py "
        "to save standardizer_mu, standardizer_sigma, and global_max_abs."
    )

standardizer_mu = np.asarray(standardizer_mu)
standardizer_sigma = np.asarray(standardizer_sigma)
global_max_abs = float(global_max_abs)

print(f"model_type   : {model_type}")
print(f"window_size  : {window_size}")
print(f"step         : {step}")
print(f"cutoff       : {cutoff}")
print(f"use_hampel   : {use_hampel}")
print(f"class_names  : {class_names}")


def _clean_raw_csi(csi: np.ndarray, use_hampel: bool, cutoff: float) -> np.ndarray:
    if use_hampel:
        from src.preprocess import apply_hampel
        csi = apply_hampel(csi)
    return butterworth_lowpass(csi, cutoff=cutoff)


def _finalize_segments_before_predict(segments: np.ndarray, model_type: str, nperseg: int) -> np.ndarray:
    if model_type == "cnn2d":
        print("Converting CSI segments to spectrogram...")
        segments = convert_segments_to_spectrogram(segments, nperseg=nperseg)
    return segments


def _apply_standardizer(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x - mu) / (sigma + 1e-8)


def _apply_global_normalizer(x: np.ndarray, max_abs: float, eps: float = 1e-8) -> np.ndarray:
    return x / (max_abs + eps)

# --- 6. Load dữ liệu test ---
from src.amp_phase import process_file_with_metadata as _process_amp_phase
from src.train import _to_torch_input


def _prepare_from_amp_phase(path: str) -> np.ndarray:
    amp, phase, _metadata = _process_amp_phase(path, metadata_columns=1, mode="auto")

    def _print_array_stats(name: str, arr: np.ndarray) -> None:
        arr_min = float(np.min(arr))
        arr_max = float(np.max(arr))
        arr_mean = float(np.mean(arr))
        arr_std = float(np.std(arr) + 1e-8)
        z_min = float(np.min((arr - arr_mean) / arr_std))
        z_max = float(np.max((arr - arr_mean) / arr_std))
        print(f"{name} min/max : {arr_min:.6f} / {arr_max:.6f}")
        print(f"{name} z-score: {z_min:.6f} .. {z_max:.6f}")

    _print_array_stats("amp(raw)", amp)
    _print_array_stats("phase(raw)", phase)

    # Step 2: amplitude processing
    if amp_log:
        amp = np.log(amp + 1e-8)
    amp_mu = np.mean(amp, axis=0, keepdims=True)
    amp_sigma = np.std(amp, axis=0, keepdims=True) + 1e-8
    amp_norm = (amp - amp_mu) / amp_sigma

    # Step 3: phase processing (always sin/cos)
    phase_cos = np.cos(phase)
    phase_sin = np.sin(phase)
    combined = np.concatenate((amp_norm, phase_cos, phase_sin), axis=1)

    return combined


data = _prepare_from_amp_phase(csv_path)
print("Raw data shape:", data.shape)

# --- 7. Khớp số chiều feature trước khi clean, giống train đang align theo feature dim ---
expected_feature_dim = int(saved_input_shape[-1])
current_feature_dim = int(data.shape[1])

if current_feature_dim < expected_feature_dim:
    raise ValueError(
        f"Feature dim của file test là {current_feature_dim}, "
        f"nhưng model được train với feature dim {expected_feature_dim}"
    )
if current_feature_dim > expected_feature_dim:
    data = data[:, :expected_feature_dim]
    print(f"Trim feature dim: {current_feature_dim} -> {expected_feature_dim}")
print(f"Expected input shape (train): {saved_input_shape}")
print(f"Current input shape (test) : {data.shape}")

# --- 8. Preprocess giống train ---
data = _clean_raw_csi(data, use_hampel, cutoff)

# --- 9. Tạo segment ---
segments, _ = create_segments(
    data,
    label=0,  # dummy thôi, không dùng để predict
    window_size=window_size,
    step=step,
)

print("Segments shape:", segments.shape)

# --- 10. Chuẩn hóa giống train ---
segments = _finalize_segments_before_predict(segments, model_type, nperseg)

segments = _apply_standardizer(segments, standardizer_mu, standardizer_sigma)
segments = _apply_global_normalizer(segments, global_max_abs)

# --- 11. Tensor input ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
inputs = _to_torch_input(segments, model_type).to(device)

# --- 12. Build model đúng kiểu ---
input_shape = segments.shape[1:]
dropout = float(train_args.get("dropout", 0.0))
model = build_model(model_type=model_type, input_shape=input_shape, num_classes=num_classes, dropout=dropout)
model.load_state_dict(sd)
model = model.to(device)
model.eval()

# --- 13. Predict ---
with torch.no_grad():
    outputs = model(inputs)
    print("\nSoftmax outputs (per segment):")
    print(torch.softmax(outputs, dim=1))
    probs = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

# --- 14. Thống kê ---
unique, counts = np.unique(predictions, return_counts=True)

print("\nSegment prediction counts:")
for u, c in zip(unique, counts):
    label_name = class_names[int(u)] if int(u) < len(class_names) else str(u)
    print(f"{label_name}: {c} segments")

pred_names = [class_names[int(p)] if int(p) < len(class_names) else str(p) for p in predictions]

print("\nPredictions per segment:")
print(pred_names)

# majority vote cho cả file
final_class = int(np.bincount(predictions).argmax())
final_name = class_names[final_class] if final_class < len(class_names) else str(final_class)

confidence = probs.mean(dim=0)[final_class].item()

print("\n==============================")
print(f"Final prediction for file: {final_name}")
print(f"Average confidence : {confidence:.4f}")
print("==============================")