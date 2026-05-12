import csv
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
no_path = os.path.join(base_dir, 'data', 'raw', 'no.csv')
yes_path = os.path.join(base_dir, 'data', 'raw', 'yes.csv')
out_path = os.path.join(base_dir, 'data', 'data1.csv')

pairs = [(no_path, '0'), (yes_path, '1')]

with open(out_path, 'w', newline='') as fout:
    writer = csv.writer(fout)
    for path, label in pairs:
        if not os.path.exists(path):
            print(f"Source file not found: {path}")
            continue
        with open(path, 'r', newline='') as fin:
            reader = csv.reader(fin)
            for row in reader:
                if not row:
                    continue
                row.append(label)
                writer.writerow(row)

print(f'Merged files into: {out_path}')
