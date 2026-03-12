#LSTM-CNN
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ============================================
# 1. LOAD CSI + CLEAN
# ============================================

def load_csi(path):
    rows = []

    with open(path, "r") as f:
        for line in f:
            values = line.strip().split(",")

            # giữ dòng đúng 65 cột
            if len(values) == 65:
                rows.append(values)

    data = pd.DataFrame(rows).astype(float)
    print(f"{path} -> Clean shape:", data.shape)

    # bỏ 2 cột metadata
    csi = data.iloc[:, 2:].values

    # bỏ subcarrier toàn 0
    non_zero = np.any(csi != 0, axis=0)
    csi = csi[:, non_zero]

    return csi


# ============================================
# 2. BUTTERWORTH FILTER
# ============================================

def butterworth_filter(csi, cutoff=0.1):
    b, a = butter(4, cutoff, btype='low')
    return filtfilt(b, a, csi, axis=0)


# ============================================
# 3. SLIDING WINDOW
# ============================================

def create_segments(data, label, window_size=600, step=200):
    X = []
    y = []

    for i in range(0, len(data) - window_size, step):
        segment = data[i:i+window_size]
        X.append(segment)
        y.append(label)

    return np.array(X), np.array(y)


# ============================================
# 4. LOAD DATA
# ============================================

print("Loading data...")

sit = load_csi("C:\\esp32\\csi\\data\\sit.csv")
stand = load_csi("C:\\esp32\\csi\\data\\stand.csv")

print("Applying Butterworth filter...")
sit = butterworth_filter(sit)
stand = butterworth_filter(stand)

X_sit, y_sit = create_segments(sit, 0)
X_stand, y_stand = create_segments(stand, 1)

X = np.vstack((X_sit, X_stand))
y = np.hstack((y_sit, y_stand))

print("Segment shape:", X.shape)

# Normalize
X = X / np.max(X)

# One-hot
y = to_categorical(y, 2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ============================================
# 5. CNN-LSTM MODEL
# ============================================

input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, subcarriers)

model = Sequential()

# CNN 1D trích feature theo subcarrier
model.add(Conv1D(64, kernel_size=5, activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# LSTM học quan hệ thời gian
model.add(LSTM(64))

model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# 6. TRAIN
# ============================================

early = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early]
)

# ============================================
# 7. EVALUATION
# ============================================

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Save model
model.save("csi_lstm_cnn_model.h5")

print("Training complete.")