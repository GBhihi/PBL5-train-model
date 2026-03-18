# PBL5 - CSI Human Activity Training Project

Project nay duoc tai cau truc de de mo rong va bao tri trong giai doan train model.

## 1) Cau truc thu muc

```text
data/
	raw/            # Du lieu goc CSV (sit/stand, home/cafe...)
	processed/      # Du lieu da xu ly (neu can luu)

src/
	preprocess.py   # Load CSV, Hampel, Butterworth, normalize
	window.py       # Sliding window segmentation
	spectrogram.py  # CSI -> spectrogram cho CNN2D
	model.py        # Kien truc CNN2D va LSTM-CNN
	train.py        # Pipeline train chinh (CLI)
	evaluate.py     # Confusion matrix + classification report
	predict.py      # Ham predict tu file CSV

models/
	train_csi_cnn2d.py   # Wrapper goi src/train.py voi model cnn2d
	train_csi_lstmcnn.py # Wrapper goi src/train.py voi model lstmcnn
	checkpoints/         # File model sau khi train (.keras)

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

Model duoc luu mac dinh tai:

```text
models/checkpoints/cnn2d.keras
models/checkpoints/lstmcnn.keras
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

