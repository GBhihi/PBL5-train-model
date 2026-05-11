import os
import sys
import csv

base_dir = os.path.dirname(__file__)
raw_dir = os.path.join(base_dir, 'data', 'raw')
files = {'sit.csv': 0, 'stand.csv': 1, 'walk.csv': 2}
out_path = os.path.join(base_dir, 'data', 'data.csv')

def read_rows(path):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            rows.append(r)
    return rows

all_rows = []
per_file_counts = {}

for fname, label in files.items():
    path = os.path.join(raw_dir, fname)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    rows = read_rows(path)
    per_file_counts[fname] = len(rows)
    # attach label as sentinel for now; will normalize widths later
    labeled = [(r, label) for r in rows]
    all_rows.extend(labeled)

if not all_rows:
    print("No data rows found in provided files.")
    sys.exit(1)

# determine global max columns across all rows
global_max = max(len(r) for r, _ in all_rows)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
written = 0
with open(out_path, 'w', newline='') as fout:
    writer = csv.writer(fout)
    for r, label in all_rows:
        if len(r) < global_max:
            r = r + [''] * (global_max - len(r))
        writer.writerow(r + [label])
        written += 1

print(f"Wrote {written} rows to {out_path}. Per-file row counts: {per_file_counts}")
