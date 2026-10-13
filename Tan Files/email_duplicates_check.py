import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_ceas08(filepath='CEAS_08.csv'):
    df = pd.read_csv(filepath, encoding='latin-1')
    df.columns = [c.strip().lower() for c in df.columns]
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    df['label_num'] = df['label'].astype(int)
    df['message'] = df['subject'] + ' ' + df['body']
    return df

print("=" * 55)
print("CHECKING CEAS_08 FOR DUPLICATE / NEAR-DUPLICATE EMAILS")
print("=" * 55)

df = load_ceas08()
print(f"\nTotal rows          : {len(df):,}")

# 1. Exact duplicate check (across the whole dataset, before splitting)
exact_dupes = df.duplicated(subset=['message']).sum()
print(f"Exact duplicate rows : {exact_dupes:,}  ({exact_dupes/len(df)*100:.1f}%)")

# 2. Same check, but specifically: how many test-set messages also
#    appear verbatim somewhere in the training set? This is the more
#    direct leakage check — if this number is high, the model may be
#    partly "memorising" rather than generalising.
X_train, X_test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label_num']
)

train_messages = set(X_train['message'])
leaked = X_test['message'].isin(train_messages).sum()
print(f"\nTest-set rows whose exact text also appears in training set:")
print(f"  {leaked:,} / {len(X_test):,}  ({leaked/len(X_test)*100:.1f}%)")

# 3. A softer near-duplicate check: same subject line reused many times
#    (common with templated phishing campaigns)
top_subjects = df['subject'].value_counts().head(10)
print(f"\nMost repeated subject lines (top 10):")
for subj, count in top_subjects.items():
    subj_display = subj[:60] + ('...' if len(subj) > 60 else '')
    print(f"  {count:>5}x  {subj_display}")

print("\n" + "=" * 55)
if leaked / len(X_test) > 0.05:
    print("⚠️  More than 5% of test emails have exact duplicates in the")
    print("    training set — this likely explains the near-perfect score.")
    print("    Consider deduplicating before splitting (see below).")
else:
    print("✅ Low direct duplication between train/test — the near-perfect")
    print("   score may genuinely reflect how separable this dataset is,")
    print("   though it's still worth treating with some caution.")
print("=" * 55)