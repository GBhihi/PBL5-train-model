# Quy Trình Train Model Trên Kaggle (Ổn Định Lâu Dài)

## 1. Kiến Trúc Tổng Quan

```
GitHub (Code)
    ↓
    └─ main: ổn định, không train trực tiếp
    └─ develop: tích hợp các feature
    └─ feat/train_ai: branch train hiện tại
    └─ tags: v1.0, v1.1, ... (snapshot cố định)

Kaggle Dataset (Data + Snapshot)
    ↓
    ├─ Version 1: baseline data
    ├─ Version 2: data sau xử lý
    └─ ...

Kaggle Notebook (Execution)
    ↓
    └─ Clone từ GitHub tag cố định
    └─ Add Kaggle Dataset (data)
    └─ Train + log metadata
    └─ Output saved
```

## 2. Chu Kỳ Phát Triển Chi Tiết

### Giai Đoạn 1: Phát Triển (Development)
**Khi nào:** Đang code/debug locally
- Làm việc trên nhánh `feat/train_ai` hoặc `develop`.
- Chạy trên máy local với data nhỏ để test.
- Commit thường xuyên: `git commit -m "Fix preprocessing bug"`.
- Push lên GitHub định kỳ: `git push origin feat/train_ai`.

### Giai Đoạn 2: Tạo Release (Stable Version)
**Khi nào:** Code đang ổn định, sẵn sàng train chính thức
1. Hợp nhất nhánh vào `main`:
   ```bash
   git checkout main
   git pull origin main
   git merge feat/train_ai
   git push origin main
   ```

2. Tạo tag (snapshot cố định):
   ```bash
   git tag -a v1.0 -m "First training run: CNN2D baseline"
   git push origin v1.0
   ```

3. Ghi lại thông tin tag trong file (VCS):
   ```
   v1.0: CNN2D baseline, date=2026-03-20, epochs=20, batch_size=16
   v1.1: LSTM-CNN variant, date=2026-03-25, epochs=30, batch_size=16
   ```

### Giai Đoạn 3: Upload Data (Nếu Cần)
**Khi nào:** Data thay đổi, muốn snapshot cố định
- Local: Chuẩn bị data trong `data/raw/`.
- Tạo Kaggle Dataset:
  ```bash
  kaggle datasets create -p ./data -n "pbl5-v1-data" --public
  # hoặc update version cũ:
  kaggle datasets version -p ./data -m "Updated data with new samples"
  ```

### Giai Đoạn 4: Train Trên Kaggle
**Khi nào:** Sẵn sàng chạy thí nghiệm chính thức

#### 4.1 Tạo Notebook Kaggle
1. Vào Kaggle, tạo Notebook.
2. Bật GPU + Internet.
3. Add Kaggle Dataset (data).

#### 4.2 Cell 1: Clone & Checkout Tag
```python
!git clone -b main https://github.com/your-user/your-repo.git /kaggle/working/PBL5-train-model
%cd /kaggle/working/PBL5-train-model

# Checkout tag cố định (GHI RÕ VERSION)
!git checkout v1.0
!git rev-parse HEAD  # Lưu commit SHA

print("=" * 50)
print("Training from tag: v1.0")
print("=" * 50)
```

#### 4.3 Cell 2: Copy Data + Install
```python
import shutil
import os

# Copy data từ Kaggle Dataset
!cp /kaggle/input/pbl5-v1-data/sit.csv data/raw/sit.csv
!cp /kaggle/input/pbl5-v1-data/stand.csv data/raw/stand.csv

# Hoặc nếu data đã trong repo
# !ls data/raw/

# Cài package
!pip install -q -r requirements.txt

print("Data & dependencies OK")
```

#### 4.4 Cell 3: Train + Ghi Metadata
```python
import json
import subprocess
from datetime import datetime

# Metadata run
metadata = {
    "date": datetime.now().isoformat(),
    "kaggle_tag": "v1.0",
    "model_type": "cnn2d",
    "epochs": 20,
    "batch_size": 16,
    "commit_sha": subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
}

# Train
!python -m src.train \
  --config configs/train_default.json \
  --model-type cnn2d \
  --epochs 20 \
  --batch-size 16 \
  --run-name "kaggle_v1.0_cnn2d"

# Lưu metadata
with open("experiments/results/kaggle_v1.0_cnn2d/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Training done. Metadata saved.")
```

#### 4.5 Cell 4: Nén & Xuất Artifact
```python
import shutil
import os

os.makedirs("/kaggle/working/output", exist_ok=True)

# Nén checkpoint
shutil.make_archive(
    "/kaggle/working/output/checkpoints_v1.0",
    "zip",
    "models/checkpoints"
)

# Nén kết quả
shutil.make_archive(
    "/kaggle/working/output/results_v1.0",
    "zip",
    "experiments/results"
)

print("Artifacts ready in /kaggle/working/output/")
print(os.listdir("/kaggle/working/output/"))
```

## 3. Tracking & Reproducibility

### Thông Tin Bắt Buộc Lưu Lại Sau Mỗi Run
1. **GitHub Tag**: v1.0, v1.1, ...
2. **Commit SHA**: `git rev-parse HEAD`
3. **Data Version**: Kaggle Dataset version (nếu có)
4. **Hyperparameters**: epochs, batch_size, learning_rate, ...
5. **Random Seed**: trong config
6. **Ngày chạy**: trong metadata.json
7. **GPU Used**: T4, P100, ...
8. **Kết quả**: accuracy, loss, F1, ...

### Cách Ghi Lại
Tạo file `TRAINING_LOG.md` trong repo:
```markdown
## Training Log

### v1.0 - CNN2D Baseline
- **Date**: 2026-03-20
- **Commit**: abc1234...
- **Data Version**: pbl5-v1-data (V1)
- **Config**: epochs=20, batch_size=16
- **GPU**: Kaggle T4
- **Result**: Accuracy=92.3%, Loss=0.25
- **Output**: /kaggle/working/output/results_v1.0.zip
- **Notes**: First baseline run, no tuning

### v1.1 - LSTM-CNN Variant
- **Date**: 2026-03-25
- **Commit**: def5678...
- **Data Version**: pbl5-v1-data (V2)
- **Config**: epochs=30, batch_size=16
- **GPU**: Kaggle T4
- **Result**: Accuracy=94.1%, Loss=0.18
- **Output**: /kaggle/working/output/results_v1.1.zip
- **Notes**: Tuned hyperparams, improved preprocessing
```

## 4. Workflow Lặp Đi Lặp Lại

### Lần 1 (Baseline)
```
1. Code GitHub (dev)
   ↓
2. Git tag v1.0
   ↓
3. Kaggle notebook: clone v1.0 → train → output
   ↓
4. Ghi log: v1.0, kết quả, commit SHA
```

### Lần 2 (Tối ưu)
```
1. Sửa code trên feat/train_ai
   - git commit
   - git push
   ↓
2. Git tag v1.1 (từ main sau merge)
   ↓
3. Tạo Kaggle notebook mới (hoặc duplicate)
   - Cell 1: git checkout v1.1
   - Chạy từ đầu
   ↓
4. Compare v1.0 vs v1.1 trong TRAINING_LOG.md
```

## 5. Quy Tắc Quy Tắc An Toàn

### KHI NÀO ĐƯỢC PHÉP SỬA CODE
❌ **KHÔNG** sửa code khi đang train dài trên Kaggle
✅ **ĐƯỢC** sửa code local, commit, push, tạo tag mới

### KHI NÀO TẠO TAG MỚI
- Sẵn sàng train chính thức
- Code ổn định ≥ 2-3 ngày
- Compile & sanity check pass trên local

### KHI NÀO CẬP NHẬT DATA
- Thêm data mới → Version Kaggle Dataset
- Không bao giờ sửa data cũ, tạo version mới

### KHI NÀO CÓ THỂ TRAIN TRỰC TIẾP TỪ MAIN
**KHÔNG BAO GIỜ** cho train chính thức từ main/develop
- Chỉ train từ tag cố định

## 6. Lợi Ích Của Quy Trình Này

| Yêu Cầu | Cách Đạt Được |
|---------|---------------|
| Tái lập kết quả | Tag + commit SHA cố định |
| Theo dõi tiến độ | TRAINING_LOG.md + metadata.json |
| Rollback nhanh | Git tag, không cần sao lưu manual |
| Làm việc nhóm | Branch rõ ràng, tag stable cho team |
| Báo cáo chính xác | Liên kết kết quả ↔ code version |
| Không nhầm lẫn | Metadata ghi rõ GPU, date, seed |

## 7. Template Notebook Kaggle Final

```python
# Cell 1: Setup
!git clone -b main https://github.com/your-user/your-repo.git /kaggle/working/PBL5
%cd /kaggle/working/PBL5
!git checkout v1.0  # ← Ghi rõ tag

# Cell 2: Data + Dependencies
import shutil
shutil.copy("/kaggle/input/pbl5-v1-data/sit.csv", "data/raw/")
shutil.copy("/kaggle/input/pbl5-v1-data/stand.csv", "data/raw/")
!pip install -q -r requirements.txt

# Cell 3: Train
!python -m src.train --config configs/train_default.json --run-name "kaggle_v1.0"

# Cell 4: Metadata + Output
import json, shutil, os
metadata = {
    "tag": "v1.0",
    "date": "2026-03-20",
    "commit": "abc1234...",
    "gpu": "T4"
}
os.makedirs("/kaggle/working/output", exist_ok=True)
with open("/kaggle/working/output/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
shutil.make_archive("/kaggle/working/output/results", "zip", "experiments/results")
```

---

**Kết luận:** Workflow này đảm bảo:
- 🔒 Ổn định: Code snapshot rõ ràng
- 📊 Traceable: Mỗi run có metadata
- 👥 Team-friendly: Dễ hiểu, dễ chia công việc
- 🔄 Reproducible: Có thể chạy lại sau 1 tháng mà kết quả không đổi
