import serial
import csv

PORT = "COM9"
BAUD = 115200
OUTPUT = "walk1.csv"

ser = serial.Serial(PORT, BAUD, timeout=1)

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)

    print("Recording CSI only... Ctrl+C to stop")

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            parts = line.split(",")

            # chỉ nhận dòng CSI (bắt đầu bằng RSSI)
            try:
                float(parts[0])
            except:
                continue

            writer.writerow(parts)

    except KeyboardInterrupt:
        print("Stopped")

ser.close()
