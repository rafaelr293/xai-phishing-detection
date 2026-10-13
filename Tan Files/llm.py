"""
CROSS-DATASET OVERLAP CHECK
Checks whether human-generated/legit.csv and human-generated/phishing.csv
(the "human" half of the Kaggle human-llm-generated dataset) share actual
email content with CEAS_08.csv -- since both use the exact same schema
(sender,receiver,date,subject,body,urls,label), there's a real risk this
"new" dataset is partly (or entirely) re-exported CEAS_08 data. If so,
using both together for training/augmentation would silently leak
train/test rows across "different" datasets.

Run this BEFORE deciding whether to merge human-generated/*.csv into the
main CEAS_08 + Enron pipeline.
"""

import pandas as pd
import os

BASE_DIR = "D:/Phishing Detection"
CEAS08_PATH = os.path.join(BASE_DIR, "CEAS_08.csv")  # adjust if needed
HUMAN_LEGIT_PATH = os.path.join(BASE_DIR, "human-generated", "legit(1).csv")
HUMAN_PHISH_PATH = os.path.join(BASE_DIR, "human-generated", "phishing(1).csv")


def load_ceas_like(path):
    df = pd.read_csv(path, encoding='utf-8', encoding_errors='replace')
    df.columns = [c.strip().lower() for c in df.columns]
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    df['message'] = (df['subject'] + ' ' + df['body']).str.strip()
    return df


print("Loading CEAS_08.csv ...")
ceas = load_ceas_like(CEAS08_PATH)
print(f"  {len(ceas):,} rows")

print("Loading human-generated legit + phishing ...")
human_legit = load_ceas_like(HUMAN_LEGIT_PATH)
human_phish = load_ceas_like(HUMAN_PHISH_PATH)
human = pd.concat([human_legit, human_phish], ignore_index=True)
print(f"  {len(human):,} rows combined")

# ── Exact match check ──
ceas_messages = set(ceas['message'])
human['exact_dupe_in_ceas'] = human['message'].isin(ceas_messages)
n_exact = human['exact_dupe_in_ceas'].sum()
print(f"\nExact message-text duplicates found in CEAS_08: "
      f"{n_exact:,} / {len(human):,} ({n_exact/len(human)*100:.1f}%)")

# ── Near-duplicate check via subject match (cheap proxy) ──
ceas_subjects = set(ceas['subject'].str.strip().str.lower())
human['subject_dupe_in_ceas'] = human['subject'].str.strip().str.lower().isin(ceas_subjects)
n_subj = human['subject_dupe_in_ceas'].sum()
print(f"Subject-line matches found in CEAS_08 (looser signal, may include "
      f"coincidental matches): {n_subj:,} / {len(human):,} "
      f"({n_subj/len(human)*100:.1f}%)")

if n_exact > 0:
    print("\nSample exact overlaps:")
    for _, row in human[human['exact_dupe_in_ceas']].head(5).iterrows():
        print(f"  [{row.get('label', '?')}] {row['message'][:90]}...")

print("\n" + "=" * 60)
if n_exact / len(human) > 0.3:
    print("VERDICT: substantial overlap -- human-generated/*.csv is very "
          "likely (partly) re-exported CEAS_08 data. Do NOT combine both "
          "in the same train/test pool without deduping across them first.")
elif n_exact > 0:
    print("VERDICT: some overlap exists -- dedupe across both files before "
          "combining them in any train/test split.")
else:
    print("VERDICT: no exact overlap detected -- safe to treat as "
          "independent datasets (still worth spot-checking a few samples "
          "manually, since paraphrased/near-duplicate content wouldn't be "
          "caught by this exact-match check).")