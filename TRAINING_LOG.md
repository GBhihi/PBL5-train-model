# Training Log - Kaggle Runs

Ghi lại mỗi lần train chính thức để tracking kết quả, tái lập, và báo cáo.

---

## Template

### v{TAG} - {Model/Experiment Name}
- **Date**: YYYY-MM-DD HH:MM
- **GitHub Tag**: v1.0
- **Commit SHA**: abc1234def5678... (lấy từ Cell 2 output)
- **Data Version**: pbl5-v1-data (V1) / hoặc "local"
- **Model Type**: cnn2d / lstmcnn
- **Config**:
  - epochs: 20
  - batch_size: 16
  - learning_rate: 0.001
  - use_hampel: true
  - nperseg: 128
- **GPU Used**: Kaggle T4 / P100 / A100
- **Training Time**: HH:MM:SS
- **Results**:
  - Accuracy: XX.X%
  - Loss: X.XXX
  - F1-Score: X.XXX
  - Precision: X.XXX
  - Recall: X.XXX
- **Output**: 
  - Checkpoint: checkpoints_v1.0.zip
  - Results: results_v1.0.zip
- **Notes**: Mô tả ngắn (ví dụ: "First baseline", "Tuned hyperparams", ...)
- **Metadata File**: experiments/results/kaggle_v1.0_cnn2d/metadata.json

---

## Actual Runs

### v1.0 - CNN2D Baseline
*(Placeholder - bạn thay thế khi train xong)*

- **Date**: 2026-03-20 14:30
- **GitHub Tag**: v1.0
- **Commit SHA**: (sẽ có sau khi git checkout v1.0)
- **Data Version**: local data (sit.csv, stand.csv)
- **Model Type**: cnn2d
- **Config**:
  - epochs: 20
  - batch_size: 16
  - learning_rate: 0.001
  - use_hampel: true
  - nperseg: 128
- **GPU Used**: Kaggle T4
- **Training Time**: ~[XX] minutes
- **Results**:
  - Accuracy: [XX.X]%
  - Loss: [X.XXX]
  - F1-Score: [X.XXX]
  - Precision: [X.XXX]
  - Recall: [X.XXX]
- **Output**: 
  - Checkpoint: models/checkpoints/cnn2d.pt
  - Results: experiments/results/kaggle_v1.0_cnn2d/
    - history.csv
    - history.json
    - confusion_matrix.npy
    - classification_report.txt
    - metadata.json
- **Notes**: Baseline run, no tuning yet
- **Metadata File**: experiments/results/kaggle_v1.0_cnn2d/metadata.json

---

### v1.1 - LSTM-CNN Variant
*(Placeholder)*

- **Date**: 2026-03-25
- **GitHub Tag**: v1.1
- **Commit SHA**: (will be filled)
- **Data Version**: pbl5-v1-data (V2)
- **Model Type**: lstmcnn
- **Config**:
  - epochs: 30
  - batch_size: 16
  - learning_rate: 0.001
  - use_hampel: true
- **GPU Used**: Kaggle T4
- **Training Time**: ~[XX] minutes
- **Results**:
  - Accuracy: [XX.X]%
  - Loss: [X.XXX]
  - F1-Score: [X.XXX]
  - Precision: [X.XXX]
  - Recall: [X.XXX]
- **Output**: 
  - Checkpoint: models/checkpoints/lstmcnn.pt
  - Results: experiments/results/kaggle_v1.1_lstmcnn/
- **Notes**: Increased epochs, improved preprocessing
- **Metadata File**: experiments/results/kaggle_v1.1_lstmcnn/metadata.json

---

## Comparison Summary

| Version | Model | Accuracy | Loss | Date | GPU | Notes |
|---------|-------|----------|------|------|-----|-------|
| v1.0 | CNN2D | [XX.X]% | [X.XXX] | 2026-03-20 | T4 | Baseline |
| v1.1 | LSTM-CNN | [XX.X]% | [X.XXX] | 2026-03-25 | T4 | Tuned |

---

## How to Fill This Log

1. **Trước khi train**: Quyết định tag (v1.0, v1.1, ...) và checkout nó ở Kaggle.
2. **Lúc train**: Notebook tự ghi metadata.json (tag, commit, date, config).
3. **Sau khi train**:
   - Copy paste kết quả từ classification_report.txt vào đây.
   - Ghi GPU type, training time.
   - Ghi notes (ví dụ: tối ưu gì, thay đổi gì so với run trước).
4. **Commit lên GitHub** (optional): `git add TRAINING_LOG.md && git commit -m "Add v1.0 training results"`

---

## Lợi Ích

- 📖 **Truy vết**: Biết chính xác code/data version nào cho kết quả nào.
- 🔄 **Tái lập**: Có tag + commit SHA = có thể chạy lại y hệt sau 1 tháng.
- 👥 **Làm việc nhóm**: Team biết ai train lúc nào với config gì.
- 📊 **Báo cáo**: Dễ so sánh v1.0 vs v1.1 vs ...
- 🐛 **Debug**: Nếu kết quả lạ, check metadata để xem config có đúng không.
