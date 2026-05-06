# estimate_cycle.py
import sys
import traceback

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from preprocess import load_csi_csv

def main() -> None:
	# --- 1. Tham số ---
	csv_path = "data/raw/walk.csv"  # file CSI của hành động muốn đo chu kỳ
	sample_rate = 100  # Hz, số frame / giây, điều chỉnh theo thiết bị của bạn

	# --- 2. Load dữ liệu ---
	data = load_csi_csv(csv_path)  # shape = (num_frames, num_features)
	print("Data shape:", data.shape)

	# --- 3. Chọn 1 cột đại diện (hoặc trung bình) ---
	signal = data.mean(axis=1)  # trung bình tất cả feature để dễ quan sát

	# --- 4. FFT / Welch để tìm tần số chính ---
	frequencies, power = welch(signal, fs=sample_rate, nperseg=1024)

	# --- 5. Tìm tần số có công suất lớn nhất (trừ f=0) ---
	frequencies = frequencies[1:]
	power = power[1:]
	peak_idx = np.argmax(power)
	peak_freq = frequencies[peak_idx]
	print(f"Peak frequency: {peak_freq:.2f} Hz")

	# --- 6. Tính chu kỳ ---
	cycle_sec = 1.0 / peak_freq
	cycle_frames = int(cycle_sec * sample_rate)
	print(f"Estimated cycle duration: {cycle_sec:.2f} s, {cycle_frames} frames")

	# --- 7. Đề xuất window_size và step ---
	window_size = cycle_frames  # ít nhất 1 chu kỳ
	step = window_size // 2     # overlap 50%
	print(f"Suggested window_size: {window_size}, step: {step}")

	# --- 8. Vẽ tín hiệu và FFT để kiểm tra ---
	plt.figure(figsize=(12,5))
	plt.subplot(1,2,1)
	plt.plot(signal)
	plt.title("CSI signal (mean of features)")
	plt.xlabel("Frame")
	plt.ylabel("Amplitude")

	plt.subplot(1,2,2)
	plt.plot(frequencies, power)
	plt.title("Power spectrum")
	plt.xlabel("Frequency (Hz)")
	plt.ylabel("Power")
	plt.axvline(peak_freq, color='r', linestyle='--', label=f"Peak: {peak_freq:.2f}Hz")
	plt.legend()
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	try:
		main()
	except Exception:
		# Print full traceback to show exact error location
		traceback.print_exc()
		sys.exit(1)