# Checklist: Setup & First Train on Kaggle

Thực hiện từng bước sau để đưa dự án lên Kaggle & train lần đầu ổn định.

---

## Phase 1: Local Setup (Máy của bạn)

### ✅ Bước 1: Chuẩn bị code local
```bash
# Đứng ở thư mục gốc project
cd f:\PBL5-train-model

# Kiểm tra git status
git status

# Commit mọi thay đổi
git add .
git commit -m "Prep for first Kaggle training"

# Push lên GitHub
git push origin feat/train_ai
```

### ✅ Bước 2: Tạo tag release
```bash
# Merge feat/train_ai vào main (nếu ready)
git checkout main
git pull origin main
git merge feat/train_ai
git push origin main

# Hoặc: nếu chưa ready merge, thì tag đúng nhánh feat/train_ai
git checkout feat/train_ai
git tag -a v1.0 -m "First training run: CNN2D baseline"
git push origin v1.0
```

**Lưu ý**: 
- Chỉ tạo tag khi code ổn định ≥ 2-3 ngày.
- Tag tên theo dạng v1.0, v1.1, v1.2 (semantic versioning).
- Mô tả tag rõ ràng: mô hình, ngày dự kiến, ghi chú.

### ✅ Bước 3: Chuẩn bị dữ liệu
Lựa chọn 1:
- **Dữ liệu đã trong repo** (sit.csv, stand.csv ở data/raw/):
  - Không cần làm gì, notebook sẽ tìm thấy.
  
Lựa chọn 2:
- **Dữ liệu riêng biệt, muốn snapshot trên Kaggle**:
  ```bash
  # Local: đặt data vào thư mục data/raw/
  
  # Tạo Kaggle Dataset
  kaggle datasets create -p ./data -n "pbl5-v1-data" --public
  
  # Hoặc update version cũ (dashboard Kaggle)
  kaggle datasets version -p ./data -m "Updated with new samples"
  ```

---

## Phase 2: Prepare Notebook Template

### ✅ Bước 4: Sửa REPO_URL trong notebook
Mở `notebooks/train_on_colab.ipynb` hoặc tạo Kaggle Notebook mới.

Ở Cell 2 (Clone Repository), thay:
```python
REPO_URL = 'https://github.com/your-username/PBL5-train-model.git'
TAG = 'v1.0'
DATA_INPUT = '/kaggle/input/pbl5-v1-data'  # nếu dùng Kaggle Dataset
```

Ghi rõ:
- URL GitHub của bạn (public repo).
- Tag mà bạn vừa tạo (v1.0, v1.1, ...).
- Tên Kaggle Dataset (nếu dùng).

---

## Phase 3: Setup Kaggle Notebook

### ✅ Bước 5: Tạo Kaggle Notebook
1. Đăng nhập Kaggle.
2. Tạo Notebook mới.
3. Đặt tên: "PBL5 Train Model - v1.0".

### ✅ Bước 6: Configure Settings
1. Vào **Settings** (nút gear).
2. Bật **Accelerator**: GPU (T4/P100/A100 tùy khả dụng).
3. Bật **Internet**: On (để git clone từ GitHub).
4. **(Nếu dùng Kaggle Dataset)** Vào **Add Data** → chọn dataset `pbl5-v1-data`.

### ✅ Bước 7: Copy Notebook Code
1. Copy Cell 1-7 từ [notebooks/train_on_colab.ipynb](./notebooks/train_on_colab.ipynb) hoặc từ file này.
2. Paste vào Kaggle Notebook.
3. **Sửa các giá trị cố định**:
   - `REPO_URL`
   - `TAG`
   - `DATA_INPUT` (nếu có)

---

## Phase 4: First Training Run

### ✅ Bước 8: Run Cell 1 - Check GPU
```python
# Cell 1: GPU Check
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
```
**Kỳ vọng**: In ra T4 hoặc P100 (không phải "None").
**Nếu lỗi**: Quay lại Settings, đảm bảo GPU bật.

### ✅ Bước 9: Run Cell 2 - Clone & Checkout
```python
# Cell 2: Clone từ GitHub
# Output: In ra TAG, Commit SHA, working directory
```
**Kỳ vọng**: Clone thành công, checkout tag đúng, Git log hiện commit.
**Nếu lỗi**: 
- Kiểm tra REPO_URL có đúng không.
- Kiểm tra tag v1.0 có tồn tại trên GitHub không.
- Internet phải bật.

### ✅ Bước 10: Run Cell 3 - Data & Dependencies
```python
# Cell 3: Copy Data + Install
# Output: "Dependencies & data OK"
```
**Kỳ vọng**: Tìm thấy sit.csv, stand.csv, cài package xong.
**Nếu lỗi data**:
- Nếu dùng Kaggle Dataset: kiểm tra tên data input.
- Nếu data local: kiểm tra path data/raw/ có đúng không.

### ✅ Bước 11: Run Cell 4 - Train CNN2D
```python
# Cell 4: Train
# Output: Epoch logs, ghi metadata
# Thời gian: ~10-30 phút (tùy data size)
```
**Kỳ vọng**: Epoch 1, 2, 3, ... in ra lần lượt, cuối cùng lưu checkpoint.
**Nếu lỗi**: 
- Kiểm tra terminal output có error gì.
- Kiểm tra config cấu hình có khớp không (window_size, nperseg, ...).

### ✅ Bước 12: Check Results (Cell 6)
```python
# Cell 6: Check results
!ls -lah experiments/results/
```
**Kỳ vọng**: Thấy `kaggle_v1.0_cnn2d/` folder, bên trong có:
- history.csv
- confusion_matrix.npy
- classification_report.txt
- metadata.json

### ✅ Bước 13: Pack Artifacts (Cell 7)
```python
# Cell 7: Pack
# Output: Tạo checkpoints_v1.0.zip, results_v1.0.zip trong /kaggle/working/output
```
**Kỳ vọng**: In ra danh sách file, kích thước.

### ✅ Bước 14: Download Output
1. Vào tab **Notebook Output** (bên phải).
2. Tải về:
   - `checkpoints_v1.0.zip`
   - `results_v1.0.zip`

---

## Phase 5: Record Results

### ✅ Bước 15: Update TRAINING_LOG.md
Mở [TRAINING_LOG.md](./TRAINING_LOG.md), điền vào section v1.0:

```markdown
### v1.0 - CNN2D Baseline
- **Date**: 2026-03-20
- **GitHub Tag**: v1.0
- **Commit SHA**: (copy từ Cell 2 output)
- **Accuracy**: XX.X%
- **Loss**: X.XXX
- **Notes**: First baseline run
```

Copy từ `experiments/results/kaggle_v1.0_cnn2d/classification_report.txt`.

### ✅ Bước 16: Commit Log Lên GitHub
```bash
cd f:\PBL5-train-model
git add TRAINING_LOG.md
git commit -m "Add v1.0 training results: CNN2D baseline, 92.3% accuracy"
git push origin main
```

---

## Phase 6: Next Training Runs

### Khi muốn train lại (v1.1, v1.2, ...):

1. **Local**: Sửa code/config → commit → push → **tạo tag v1.1**.
   ```bash
   git tag -a v1.1 -m "v1.1: LSTM-CNN, tuned hyperparams"
   git push origin v1.1
   ```

2. **Kaggle**: Tạo Notebook mới (hoặc duplicate):
   ```python
   TAG = 'v1.1'
   # Chạy từ Cell 1-7
   ```

3. **Local**: Update TRAINING_LOG.md, push lại GitHub.

---

## Troubleshooting

| Vấn đề | Nguyên Nhân | Cách Fix |
|--------|-----------|---------|
| "GPU not available" | GPU không bật trong Settings | Vào Settings > Accelerator > GPU |
| "fatal: could not read Username" | Internet tắt hoặc GitHub token lỗi | Vào Settings > Internet > On. Nếu private repo: dùng token SSH hoặc HTTPS. |
| "fatal: reference is not a tree" | Tag không tồn tại trên GitHub | Kiểm tra: `git tag` local có tag v1.0 không. Nếu không, tạo mới. |
| "FileNotFoundError: data/raw/sit.csv" | Data không ở đúng path | Kiểm tra Cell 3 output, Kaggle Dataset có chuẩn không, hoặc file data/raw/ cấu trúc. |
| "ImportError: No module named 'torch'" | Package chưa cài | Chạy lại Cell 3: `!pip install -r requirements.txt` |

---

## Summary

| Giai Đoạn | Thời Gian | Công Việc |
|---------|---------|---------|
| Phase 1 | 5 min | Commit code, tạo tag v1.0, push GitHub |
| Phase 2 | 2 min | Sửa REPO_URL, TAG trong notebook |
| Phase 3 | 5 min | Tạo Kaggle Notebook, bật GPU + Internet |
| Phase 4 | 30-60 min | Run Cell 1-7, train xong |
| Phase 5 | 5 min | Download output, update TRAINING_LOG.md |
| **Tổng** | **~1-2 giờ** | **First stable training run** |

---

## Kết Luận

Sau checklist này, bạn sẽ có:
✅ Code tag v1.0 trên GitHub (reproducible)
✅ Kết quả training lưu trên Kaggle Output
✅ Metadata ghi rõ commit SHA, config, kết quả
✅ TRAINING_LOG.md track version
✅ Workflow ổn định để tái lập hoặc update sau này
