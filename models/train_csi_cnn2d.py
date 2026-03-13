#CNN 2D
import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, stft
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ============================================
# 1. LOAD CSI + SUBCARRIER SELECTION
# ============================================

def load_csi(path):
    rows = []

    with open(path, "r") as f:
        for line in f:
            values = line.strip().split(",")

            # Chỉ giữ dòng có đúng 65 cột (2 metadata + 63 subcarrier)
            if len(values) == 65:
                rows.append(values)

    data = pd.DataFrame(rows).astype(float)

    print(f"{path} -> Clean shape:", data.shape)

    # bỏ 2 cột đầu
    csi = data.iloc[:, 2:].values

    # bỏ subcarrier toàn 0
    non_zero = np.any(csi != 0, axis=0)
    csi = csi[:, non_zero]

    return csi


# ============================================
# 2. HAMPEL FILTER
# ============================================

def hampel_filter(signal, window_size=5, n_sigmas=3):
    new_signal = signal.copy()

    for i in range(window_size, len(signal) - window_size):
        window = signal[i-window_size:i+window_size]
        median = np.median(window)
        mad = np.median(np.abs(window - median))

        if mad == 0:
            continue

        threshold = n_sigmas * 1.4826 * mad
        if np.abs(signal[i] - median) > threshold:
            new_signal[i] = median

    return new_signal


def apply_hampel(csi):
    filtered = np.zeros_like(csi)
    for i in range(csi.shape[1]):
        filtered[:, i] = hampel_filter(csi[:, i])
    return filtered


# ============================================
# 3. BUTTERWORTH FILTER
# ============================================

def butterworth_filter(csi, cutoff=0.1):
    b, a = butter(4, cutoff, btype='low')
    return filtfilt(b, a, csi, axis=0)


# ============================================
# 4. SLIDING WINDOW
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
# 5. CSI → SPECTROGRAM
# ============================================

def segment_to_spectrogram(segment):
    specs = []

    for i in range(segment.shape[1]):  # mỗi subcarrier
        f, t, Zxx = stft(segment[:, i], nperseg=128)
        specs.append(np.abs(Zxx))

    specs = np.array(specs)

    # reshape thành (height, width, channel=1)
    specs = np.transpose(specs, (1, 2, 0))
    return specs


def convert_all_to_spectrogram(X):
    spec_list = []
    for segment in X:
        spec = segment_to_spectrogram(segment)
        spec_list.append(spec)

    return np.array(spec_list)


# ============================================
# 6. LOAD DATA
# ============================================

print("Loading data...")

sit = load_csi("C:\\esp32\\csi\\data\\sit.csv")
stand = load_csi("C:\\esp32\\csi\\data\\stand.csv")

# ============================================
# 7. FILTERING
# ============================================

print("Applying Hampel filter...")
sit = apply_hampel(sit)
stand = apply_hampel(stand)

print("Applying Butterworth filter...")
sit = butterworth_filter(sit)
stand = butterworth_filter(stand)

# ============================================
# 8. SEGMENTATION
# ============================================

X_sit, y_sit = create_segments(sit, 0)
X_stand, y_stand = create_segments(stand, 1)

X = np.vstack((X_sit, X_stand))
y = np.hstack((y_sit, y_stand))

print("Segment shape:", X.shape)

# ============================================
# 9. SPECTROGRAM
# ============================================

print("Converting to spectrogram...")
X = convert_all_to_spectrogram(X)

print("Spectrogram shape:", X.shape)

# Normalize
X = X / np.max(X)

# One-hot
y = to_categorical(y, 2)

# ============================================
# 10. TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ============================================
# 11. CNN MODEL
# ============================================

input_shape = X_train.shape[1:]

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# 12. TRAIN
# ============================================

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# ============================================
# 13. EVALUATION
# ============================================

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# ============================================
# 14. SAVE MODEL
# ============================================

model.save("csi_action_model.h5")

print("Training complete.")