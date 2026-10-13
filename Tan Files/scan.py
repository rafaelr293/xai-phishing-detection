import pandas as pd
import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'phish360-data')  # adjust if your data lives elsewhere

print("=" * 55)
print("SCANNING PHISH360 FOR CORRUPTED IMAGE FILES")
print("=" * 55)

frames = []
for fname, label_val in [('Phish360_legit.parquet', 0), ('Phish360_phish.parquet', 1)]:
    part = pd.read_parquet(os.path.join(DATA_DIR, fname))
    part['label_num'] = label_val
    frames.append(part)

df = pd.concat(frames, ignore_index=True)
df['full_image_path'] = df['image_path'].apply(lambda p: os.path.join(DATA_DIR, p))

print(f"\nTotal samples to check: {len(df):,}")
print("This may take a few minutes...\n")

bad_rows = []
for idx, row in df.iterrows():
    path = row['full_image_path']
    if not os.path.exists(path):
        bad_rows.append((idx, path, 'FILE NOT FOUND'))
        continue
    try:
        with Image.open(path) as img:
            img.verify()  # checks the file is a valid, complete image without fully decoding it
    except Exception as e:
        bad_rows.append((idx, path, str(e)))

    if idx % 2000 == 0 and idx > 0:
        print(f"  Checked {idx:,} / {len(df):,}...")

print(f"\n{'='*55}")
print(f"RESULT: {len(bad_rows):,} bad file(s) out of {len(df):,}")
print(f"{'='*55}")

if bad_rows:
    print("\nFirst 20 problem files:")
    for idx, path, reason in bad_rows[:20]:
        print(f"  [{idx}] {path}  →  {reason}")

    # Save the list of bad indices so the training script can exclude them
    bad_indices_path = os.path.join(BASE_DIR, 'phish360_bad_indices.txt')
    with open(bad_indices_path, 'w') as f:
        for idx, path, reason in bad_rows:
            f.write(f"{idx}\t{path}\t{reason}\n")
    print(f"\nFull list saved to → {bad_indices_path}")
else:
    print("\n✅ No corrupted files found.")