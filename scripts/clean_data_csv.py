import csv
from pathlib import Path

infile = Path('data/data.csv')
outfile = Path('data/data_clean.csv')

if not infile.exists():
    print(f'Input not found: {infile}')
    raise SystemExit(1)

fixed = 0
total = 0
with infile.open('r', newline='', encoding='utf-8') as inf, outfile.open('w', newline='', encoding='utf-8') as outf:
    reader = csv.reader(inf)
    writer = csv.writer(outf)
    for row in reader:
        total += 1
        orig_len = len(row)
        # remove trailing empty cells but keep at least 2 columns (features + label)
        while len(row) > 1 and row[-1] == '':
            row.pop()
        # also remove empty cells immediately before the last column (label)
        while len(row) >= 2 and row[-2] == '':
            row.pop(-2)
        if len(row) == 0:
            continue
        if len(row) != orig_len:
            fixed += 1
        writer.writerow(row)

print(f'Wrote {total} rows to {outfile}. Fixed {fixed} rows.')
