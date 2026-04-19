# test_predict.py
import sys
import os
import torch
import numpy as np
from sklearn.decomposition import PCA

# --- 1. Thêm đường dẫn src ---
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# --- 2. Import module ---
from model import build_model
from preprocess import (
    butterworth_lowpass,
    load_csi_csv,
    normalize_global,
    standardize_csi,
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
else:
    model_type = "lstmcnn"
    num_classes = 3
    train_args = {}
    saved_input_shape = None
    class_names = ["sit", "stand", "walk"]

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
pca_components = int(train_args.get("pca_components", 10))

if saved_input_shape is None:
    raise ValueError("Checkpoint missing input_shape; cannot align test data like train.")
if nperseg is None:
    raise ValueError("Checkpoint missing nperseg; cannot match train spectrogram settings.")
nperseg = int(nperseg)

print(f"model_type   : {model_type}")
print(f"window_size  : {window_size}")
print(f"step         : {step}")
print(f"cutoff       : {cutoff}")
print(f"use_hampel   : {use_hampel}")
print(f"pca_components: {pca_components}")
print(f"class_names  : {class_names}")


def _clean_raw_csi(
    csi: np.ndarray,
    use_hampel: bool,
    cutoff: float,
    n_components: int,
) -> np.ndarray:
    if use_hampel:
        from preprocess import apply_hampel
        print("Applying Hampel filter...")
        csi = apply_hampel(csi)

    print("Standardizing for PCA...")
    csi = standardize_csi(csi)

    print(f"Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    csi = pca.fit_transform(csi)

    print("Applying Butterworth filter...")
    return butterworth_lowpass(csi, cutoff=cutoff)


def _finalize_segments(segments: np.ndarray, model_type: str, nperseg: int) -> np.ndarray:
    print("Applying Z-score standardization...")
    segments = standardize_csi(segments)

    if model_type == "cnn2d":
        from spectrogram import convert_segments_to_spectrogram

        print("Converting CSI segments to spectrogram...")
        segments = convert_segments_to_spectrogram(segments, nperseg=nperseg)

    segments = normalize_global(segments)
    return segments

# --- 6. Load dữ liệu test ---
data = load_csi_csv(csv_path)
print("Raw data shape:", data.shape)

# --- 7. Kiểm tra PCA components khớp với model ---
# saved_input_shape của lstmcnn thường là [window_size, feature_dim] sau PCA
expected_feature_dim = int(saved_input_shape[-1])
if expected_feature_dim != pca_components:
    raise ValueError(
        f"Checkpoint expects {expected_feature_dim} PCA components, "
        f"but test is configured with {pca_components}."
    )

# --- 8. Preprocess giống train (Standardize -> PCA -> Butterworth) ---
data = _clean_raw_csi(data, use_hampel, cutoff, n_components=pca_components)
print("Data shape after PCA & filter:", data.shape)

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
print(f"Average confidence       : {confidence:.4f}")
print("==============================")