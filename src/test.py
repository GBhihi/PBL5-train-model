# test_predict.py
import sys
import os
import torch
import numpy as np

# --- 1. Thêm đường dẫn src ---
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# --- 2. Import module ---
from model import build_model
from preprocess import (
    align_feature_dims,
    butterworth_lowpass,
    load_csi_csv,
)
from window import create_segments

# --- 3. Đường dẫn file ---
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "../checkpoints(10)/lstmcnn.pt")
csv_path = os.path.join(BASE_DIR, "../data/raw/sit.csv")

# --- 4. Load checkpoint trước để lấy metadata ---
ckpt = torch.load(model_path, map_location=torch.device("cpu"))

if isinstance(ckpt, dict):
    model_type = ckpt.get("model_type", "lstmcnn")
    num_classes = ckpt.get("num_classes", 3)
    train_args = ckpt.get("args", {})
    saved_input_shape = ckpt.get("input_shape", None)
    class_names = ckpt.get("class_names", ["sit", "stand", "walk"])
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
window_size = int(train_args.get("window_size", 256))
step = int(train_args.get("step", 128))
cutoff = float(train_args.get("cutoff", 0.1))
use_hampel = bool(train_args.get("use_hampel", False))
nperseg = train_args.get("nperseg", None)

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
        from preprocess import apply_hampel
        csi = apply_hampel(csi)
    return butterworth_lowpass(csi, cutoff=cutoff)



def _finalize_segments(segments: np.ndarray, model_type: str, nperseg: int) -> np.ndarray:
    if model_type == "cnn2d":
        from spectrogram import convert_segments_to_spectrogram

        print("Converting CSI segments to spectrogram...")
        segments = convert_segments_to_spectrogram(segments, nperseg=nperseg)
    return segments


def _apply_standardizer(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x - mu) / (sigma + 1e-8)


def _apply_global_normalizer(x: np.ndarray, max_abs: float, eps: float = 1e-8) -> np.ndarray:
    return x / (max_abs + eps)

# --- 6. Load dữ liệu test ---
data = load_csi_csv(csv_path)
print("Raw data shape:", data.shape)

# --- 7. Khớp số chiều feature nếu cần (giống bước align trước khi lọc ở train) ---
# saved_input_shape của lstmcnn thường là [window_size, feature_dim]
if model_type == "lstmcnn":
    expected_feature_dim = int(saved_input_shape[-1])
    current_feature_dim = int(data.shape[1])

    if current_feature_dim < expected_feature_dim:
        raise ValueError(
            f"Feature dim của file test là {current_feature_dim}, "
            f"nhưng model được train với feature dim {expected_feature_dim}"
        )
    if current_feature_dim > expected_feature_dim:
        dummy = np.zeros((data.shape[0], expected_feature_dim), dtype=data.dtype)
        data, _ = align_feature_dims(data, dummy)
        print(f"Trim feature dim: {current_feature_dim} -> {expected_feature_dim}")

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
segments = _finalize_segments(segments, model_type, nperseg)

segments = _apply_standardizer(segments, standardizer_mu, standardizer_sigma)
segments = _apply_global_normalizer(segments, global_max_abs)

# --- 11. Tensor input ---
inputs = torch.tensor(segments, dtype=torch.float32)

# --- 12. Build model đúng kiểu ---
input_shape = segments.shape[1:]
model = build_model(model_type=model_type, input_shape=input_shape, num_classes=num_classes)
model.load_state_dict(sd)
model.eval()

# --- 13. Predict ---
with torch.no_grad():
    outputs = model(inputs)
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