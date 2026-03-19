# PBL5 - CSI Human Activity Training Project

Project nay duoc tai cau truc de de mo rong va bao tri trong giai doan train model.
Framework train chinh hien tai: PyTorch.

## 1) Cau truc thu muc

```text
data/
	raw/            # Du lieu goc CSV (sit/stand, home/cafe...)
	processed/      # Du lieu da xu ly (neu can luu)

src/
	preprocess.py   # Load CSV, Hampel, Butterworth, normalize
	window.py       # Sliding window segmentation
	spectrogram.py  # CSI -> spectrogram cho CNN2D
	model.py        # Kien truc CNN2D va LSTM-CNN (PyTorch)
	train.py        # Pipeline train chinh (CLI, PyTorch)
	evaluate.py     # Confusion matrix + classification report
	predict.py      # Ham predict tu file CSV

models/
	train_csi_cnn2d.py   # Wrapper goi src/train.py voi model cnn2d
	train_csi_lstmcnn.py # Wrapper goi src/train.py voi model lstmcnn
	checkpoints/         # File model sau khi train (.pt)

experiments/
	exp_random_split.py  # Train baseline random split
	exp_cross_room.py    # Train baseline cross-room (co the override path)
	results/             # Ket qua thuc nghiem

realtime/
	# Danh cho pipeline du doan real-time (se bo sung tiep)

Signal processing/
	save_csi_to_csv.py   # Script thu CSI tu serial va luu CSV
```

## 2) Cai dat moi truong

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2.1) Cau hinh train bang JSON (khuyen nghi cho teamwork)

File mau:

```text
configs/train_default.json
```

Chay theo config:

```bash
python -m src.train --config configs/train_default.json
```

## 3) Lenh train co ban

Train CNN2D:

```bash
python models/train_csi_cnn2d.py --epochs 20 --batch-size 16 --use-hampel
```

Train LSTM-CNN:

```bash
python models/train_csi_lstmcnn.py --epochs 30 --batch-size 16
```

Train truc tiep tu pipeline trung tam:

```bash
python -m src.train --model-type cnn2d --sit-path data/raw/sit.csv --stand-path data/raw/stand.csv
```

Them log/output dir:

```bash
python -m src.train --config configs/train_default.json --output-dir experiments/results --run-name run_01
```

Model duoc luu mac dinh tai:

```text
models/checkpoints/cnn2d.pt
models/checkpoints/lstmcnn.pt
```

## 4) Chay experiments

```bash
python experiments/exp_random_split.py
python experiments/exp_cross_room.py
```

Ban co the override cac tham so nhu `--sit-path`, `--stand-path`, `--epochs`, `--window-size` khi chay.

## 5) Huong phat trien tiep

- Them script luu metrics/plots vao `experiments/results/`.
- Hoan thien `realtime/` de doc stream va predict online.
- Them test cho cac ham xu ly trong `src/`.

## 6) Train bang Kaggle (khuyen nghi)

Ban co the train bang GPU tren Kaggle de tranh phu thuoc may local va de tai su dung notebook de hon.

Notebook tham khao hien co:

```text
notebooks/train_on_colab.ipynb
```

Ban co the dung lai notebook nay tren Kaggle, chi can doi path va cach nap du lieu.

Quy trinh nhanh tren Kaggle:

1. Tao Notebook moi tren Kaggle va bat `Accelerator = GPU` (Settings).
2. Upload source code len Kaggle Dataset (vi du: `pbl5-train-model`) hoac add tu GitHub.
3. Add Dataset vao Notebook.
4. Giai nen/chep project vao working dir, vi du:

```bash
!cp -r /kaggle/input/pbl5-train-model /kaggle/working/PBL5-train-model
%cd /kaggle/working/PBL5-train-model
```

5. Cai thu vien:

```bash
!pip install -r requirements.txt
```

6. Chay train:

```bash
!python -m src.train --config configs/train_default.json
```

7. Luu artifact/checkpoint tu `/kaggle/working/...` bang cach tao Kaggle Output hoac save sang Dataset moi.

Luu y ve du lieu:

- Duong dan trong config phai ton tai trong moi truong Kaggle.
- Neu can, override truc tiep tham so path khi chay:

```bash
!python -m src.train --config configs/train_default.json --sit-path data/raw/sit.csv --stand-path data/raw/stand.csv
```

## 7) Checkpoint va logging

Checkpoint `.pt` se luu day du:

- `model` va `state_dict`: trong so model
- `optimizer`: trang thai optimizer
- `epoch`, `best_epoch`
- `args`, `history`

Moi run se luu log trong `experiments/results/<run_name>/`:

- `history.csv` (epoch/loss/accuracy)
- `history.json`
- `classification_report.txt`
- `confusion_matrix.npy`
- `tb/` (TensorBoard, neu co)

