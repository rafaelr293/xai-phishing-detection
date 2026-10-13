"""
TEST: Human/LLM-generated phishing-legitimate email dataset
(Kaggle: francescogreco97/human-llm-generated-phishing-legitimate-emails)

Verified structure (confirmed by directly inspecting the actual files):

  human-generated/legit(1).csv, human-generated/phishing(1).csv
    -> columns: sender,receiver,date,subject,body,urls,label
    -> proper RFC4180 quoting (multiline bodies quoted correctly)
    -> pd.read_csv() default (C engine) parses cleanly: 1000 rows each
    -> internal 'label' column IS trustworthy: legit(1).csv is all-0,
       phishing(1).csv is all-1 -- matches filename, use it directly

  llm-generated/legit.csv, llm-generated/phishing.csv
    -> columns: text,label
    -> legit.csv: properly quoted, parses cleanly -- BUT its 'label'
       column is a CONSTANT 1 for all 1000 rows (uninformative)
    -> phishing.csv: text field is NOT quoted, contains embedded commas,
       breaks standard CSV parsing -- but each record is exactly one
       line (no embedded newlines), so a per-line regex repair
       (split on the final ',0' or ',1') recovers all 1000/1000 rows
       cleanly. Its 'label' column is ALSO a constant 1.
    -> Conclusion: the llm-generated files' own 'label' column carries
       no legit/phishing signal at all (always 1, in both files) --
       it's leftover from the source dataset's construction, not a
       usable target. We ignore it and use the filename/folder as the
       ground-truth label instead.
"""

import pandas as pd
import numpy as np
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

RANDOM_STATE = 42

# ─────────────────────────────────────────
# BASE FOLDER -- adjust if your path differs
# ─────────────────────────────────────────
BASE_DIR = "D:/Phishing Detection"

FILES = {
    ("human", 0): os.path.join(BASE_DIR, "human-generated", "legit(1).csv"),
    ("human", 1): os.path.join(BASE_DIR, "human-generated", "phishing(1).csv"),
    ("llm",   0): os.path.join(BASE_DIR, "llm-generated",   "legit.csv"),
    ("llm",   1): os.path.join(BASE_DIR, "llm-generated",   "phishing.csv"),
}


def load_ceas_schema(path, expected_label):
    """human-generated/*.csv -- sender,receiver,date,subject,body,urls,label"""
    df = pd.read_csv(path, encoding='utf-8', encoding_errors='replace')
    required = {'subject', 'body', 'label'}
    if not required.issubset(df.columns):
        raise KeyError(f"'{path}' missing expected columns. Found: {list(df.columns)}")

    agree = (pd.to_numeric(df['label'], errors='coerce') == expected_label).mean()
    print(f"    label column agreement with expected={expected_label}: {agree*100:.1f}%")
    if agree < 0.95:
        print(f"    [WARNING] '{path}': label column doesn't match filename as "
              f"expected -- inspect manually before trusting this file.")

    out = pd.DataFrame()
    out['raw_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    out['label_num'] = expected_label
    return out


def load_text_label_schema(path, expected_label):
    """
    llm-generated/*.csv -- text,label. Tries standard parsing first; falls
    back to per-line regex repair for files like phishing.csv where the
    text field isn't quoted and contains commas that break normal CSV
    tokenizing. The file's own 'label' column is NOT used as ground truth
    (confirmed constant/uninformative in this dataset) -- expected_label
    from the folder/filename is used instead.
    """
    try:
        df = pd.read_csv(path, encoding='utf-8', encoding_errors='replace')
        if df.shape[1] != 2 or 'text' not in df.columns:
            raise ValueError("unexpected shape/columns from default parse")
        texts = df['text'].fillna('')
        print(f"    parsed with standard pd.read_csv -> {len(df):,} rows")
    except Exception as e:
        print(f"    standard parse failed ({e}); falling back to per-line "
              f"regex repair (unquoted text field, one record per line)")
        with open(path, encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()
        rows, bad = [], 0
        for line in lines[1:]:  # skip header
            if not line.strip():
                continue
            m = re.match(r'^(.*),([01])$', line)
            if not m:
                bad += 1
                continue
            rows.append(m.group(1))
        if bad:
            print(f"    [WARNING] {bad} line(s) in '{path}' didn't match the "
                  f"expected 'text,0_or_1' pattern and were dropped")
        texts = pd.Series(rows)
        print(f"    regex repair recovered {len(texts):,} / "
              f"{len(lines) - 1:,} data lines")

    out = pd.DataFrame()
    out['raw_text'] = texts
    out['label_num'] = expected_label
    return out


def load_one(path, origin, label):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Update BASE_DIR at the top of this "
            f"script to match where you placed the human-generated/ and "
            f"llm-generated/ folders."
        )
    print(f"  Loading {path} (origin={origin}, expected label={label})...")
    if origin == 'human':
        out = load_ceas_schema(path, label)
    else:
        out = load_text_label_schema(path, label)
    out['origin'] = origin
    out['source_file'] = os.path.basename(path)
    print(f"  -> {len(out):,} rows loaded\n")
    return out


# ─────────────────────────────────────────
# 1. LOAD ALL 4 FILES
# ─────────────────────────────────────────
print("=" * 60)
print("LOADING human-generated / llm-generated phishing+legit CSVs")
print("=" * 60)

parts = [load_one(path, origin, label) for (origin, label), path in FILES.items()]
combined = pd.concat(parts, ignore_index=True)

before = len(combined)
combined = combined.drop_duplicates(subset=['raw_text']).reset_index(drop=True)
print(f"Combined: {before:,} rows -> {len(combined):,} after dedupe "
      f"({before - len(combined)} duplicates dropped)")

print("\nBreakdown:")
print(combined.groupby(['origin', 'label_num']).size().rename('count'))

# ─────────────────────────────────────────
# 2. SHUFFLE + FEATURES
# ─────────────────────────────────────────
combined = combined.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

URL_PATTERN = re.compile(r'(https?://\S+|www\.\S+)')

def extract_url_features(text):
    urls = URL_PATTERN.findall(str(text))
    return len(urls), int(len(urls) > 0)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

url_feats = combined['raw_text'].apply(extract_url_features)
combined['url_count'] = url_feats.apply(lambda t: t[0])
combined['has_url'] = url_feats.apply(lambda t: t[1])
combined['clean_message'] = combined['raw_text'].apply(clean_text)

feature_cols = ['clean_message', 'url_count', 'has_url']
X = combined[feature_cols]
y = combined['label_num']

# ─────────────────────────────────────────
# 3. 80/20 SPLIT, stratified on label
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
origin_test = combined.loc[X_test.index, 'origin']

print(f"\nTrain: {len(X_train):,}  Test: {len(X_test):,}")

# ─────────────────────────────────────────
# 4. TRAIN
# ─────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ('tfidf', TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, max_df=0.95, analyzer='word', stop_words='english'
    ), 'clean_message'),
    ('url_features', StandardScaler(), ['url_count', 'has_url']),
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(C=1.0, max_iter=1000,
                                class_weight='balanced', solver='lbfgs')),
])

print("\nTraining...")
pipeline.fit(X_train, y_train)

# ─────────────────────────────────────────
# 5. EVALUATE -- overall + broken down by origin
# ─────────────────────────────────────────
y_pred = pipeline.predict(X_test)

print("\n" + "=" * 60)
print("OVERALL RESULTS (human + LLM mixed test set)")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, labels=[0, 1],
                             target_names=['Legitimate', 'Phishing'],
                             zero_division=0))

print("\n" + "=" * 60)
print("BREAKDOWN BY ORIGIN (human-written vs LLM-generated)")
print("=" * 60)
for origin_val in ['human', 'llm']:
    mask = (origin_test == origin_val).values
    if mask.sum() == 0:
        continue
    acc = accuracy_score(y_test[mask], y_pred[mask])
    print(f"\n--- origin = {origin_val} ({mask.sum()} test emails) ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test[mask], y_pred[mask], labels=[0, 1],
                                 target_names=['Legitimate', 'Phishing'],
                                 zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix (overall):")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

print("\nDone.")