"""
PHISHING EMAIL CLASSIFIER — v2
Fixes applied (per project debugging checklist):

Priority 1 (must-fix):
  A. Dedupe BEFORE split (not after)
  B. Proper train / validation / test split (validation used for threshold tuning)
  C. Threshold is tuned on validation data, not hard-coded (0.5/0.9/0.95/0.99)
  D. Enron fixed holdout — same holdout emails used for before/after comparison
  D2. Second cross-dataset holdout — human/llm-generated phishing+legit
      (Kaggle francescogreco97 dataset), deduped against CEAS_08 first
      (~2.9% exact overlap found and removed), evaluated with breakdown
      by origin (human-written vs LLM-generated) to check whether the
      model generalizes to AI-generated phishing specifically

Follow-up fixes after discovering 56.43% accuracy (near-chance) on the
human/llm-generated holdout, with an origin-split asymmetry (human:
FPR~0.6%/FNR~80% vs llm: FPR~72%/FNR~25%) -- evidence of concept drift
between 2000s-era CEAS_08/Enron phishing style and modern/LLM-written
email style:
  3b. The human/llm-generated train pool (other half of the same fixed
      split used for the holdout) is now also folded into train-set
      augmentation alongside Enron ham, with both its classes (unlike
      Enron which only contributes label=0).
  3c. Numeric/linguistic features (urgency, credential, financial, CTA,
      URL richness) are now explicitly up-weighted relative to the
      ~11,000-dim TF-IDF block via a tunable multiplier inside the
      ColumnTransformer, tuned by GridSearchCV rather than guessed --
      motivated by error-analysis cases where credential_count=26 still
      produced a near-zero phishing probability, implying TF-IDF
      dimensionality was drowning out the engineered cues.

Priority 2 (should-fix):
  E. PR-AUC reported alongside ROC-AUC (more informative under class imbalance)
  F. FPR / FNR reported explicitly, not just accuracy
  G. Hyperparameter tuning via GridSearchCV (C, max_features, ngram_range,
     stop_words, class_weight)
  H. Error analysis — structured false positive / false negative inspection

Priority 3 (model upgrades):
  I. Linguistic cue features (urgency, credential requests, financial terms,
     call-to-action phrases, uppercase ratio, exclamation density)
  J. Richer URL features (count, unique domain count, IP-literal URLs, URL length)
  K. Character n-gram TF-IDF alongside word TF-IDF (catches obfuscation:
     "cl1ck", "verify-your-acc0unt", etc.)

Priority 4 (not implemented here — see NOTE at bottom):
  Header-based features (From/Reply-To/Return-Path mismatch, HTML structure,
  attachment metadata) require raw .eml source, which CEAS_08's CSV form and
  the Enron CSV do not provide (they're pre-extracted subject/body only).
  This is the natural "v2 → v3" step once you're pulling from parse_eml_file()
  on real .eml samples rather than the CSV exports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    f1_score
)

RANDOM_STATE = 42

# ═══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════

def load_ceas08(filepath='CEAS_08.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find '{filepath}'")
    df = pd.read_csv(filepath, encoding='latin-1')
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={'urls': 'urls_flag'})
    for c in ['subject', 'body', 'label']:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}'. Found: {list(df.columns)}")
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    df['label_num'] = df['label'].astype(int)
    df['message'] = df['subject'] + ' ' + df['body']
    print(f"  Loaded {filepath} -> {len(df):,} emails")
    return df


def load_enron(filepath='enron_spam_data.csv'):
    """
    Enron corpus. IMPORTANT: 'spam' here is generic spam, NOT phishing.
    Only 'ham' rows are trustworthy as label_num=0 for this task.
    Spam rows are kept for reference/error-analysis only and are never
    treated as label_num=1 (that would mislabel legitimate-but-annoying
    marketing email as phishing).
    """
    df = pd.read_csv(filepath, encoding='latin-1')
    df.columns = [c.strip().lower().replace(' ', '_').replace('/', '_') for c in df.columns]
    df['subject'] = df['subject'].fillna('')
    df['message'] = df['message'].fillna('')
    df['full_text'] = df['subject'] + ' ' + df['message']
    before = len(df)
    df = df.drop_duplicates(subset=['subject', 'message']).reset_index(drop=True)
    print(f"  Enron: dropped {before - len(df)} exact duplicates")
    df['is_ham'] = (df['spam_ham'].str.lower() == 'ham').astype(int)
    print(f"  Loaded {filepath} -> {len(df):,} emails "
          f"({df['is_ham'].sum():,} ham / {(df['is_ham']==0).sum():,} spam)")
    return df


def load_human_llm_generated(base_dir='.'):
    """
    Kaggle 'human-llm-generated-phishing-legitimate-emails' dataset.
    Verified structure (confirmed by direct inspection, see project notes):

      human-generated/legit.csv, human-generated/phishing.csv
        -> sender,receiver,date,subject,body,urls,label  (CEAS_08-style
           schema, properly quoted). Internal 'label' column IS
           trustworthy here (100% agreement with filename).

      llm-generated/legit.csv, llm-generated/phishing.csv
        -> text,label. phishing.csv's text field is NOT quoted (embedded
           commas break standard CSV tokenizing) but each record is one
           line, so a per-line regex repair recovers it cleanly. Both
           llm-generated files have a CONSTANT, uninformative 'label'
           column (always 1) -- it carries no legit/phishing signal, so
           we ignore it and use the folder/filename as ground truth.

    Returns a single combined df with columns: message, label_num, origin
    ('human' or 'llm'), is_phishing_source ('legit'/'phishing' filename).
    """
    def _load_ceas_schema(path, expected_label):
        df = pd.read_csv(path, encoding='utf-8', encoding_errors='replace')
        agree = (pd.to_numeric(df['label'], errors='coerce') == expected_label).mean()
        if agree < 0.95:
            print(f"    [WARNING] '{path}': label column only {agree*100:.1f}% "
                  f"matches expected={expected_label} -- verify this file.")
        out = pd.DataFrame()
        out['message'] = (df['subject'].fillna('') + ' ' + df['body'].fillna('')).str.strip()
        out['label_num'] = expected_label
        return out

    def _load_text_label_schema(path, expected_label):
        try:
            df = pd.read_csv(path, encoding='utf-8', encoding_errors='replace')
            if df.shape[1] != 2 or 'text' not in df.columns:
                raise ValueError("unexpected shape/columns")
            texts = df['text'].fillna('')
        except Exception:
            with open(path, encoding='utf-8', errors='replace') as f:
                lines = f.read().splitlines()
            rows, bad = [], 0
            for line in lines[1:]:
                if not line.strip():
                    continue
                m = re.match(r'^(.*),([01])$', line)
                if not m:
                    bad += 1
                    continue
                rows.append(m.group(1))
            if bad:
                print(f"    [WARNING] {bad} unrecoverable line(s) in '{path}'")
            texts = pd.Series(rows)
        out = pd.DataFrame()
        out['message'] = texts.str.strip()
        out['label_num'] = expected_label
        return out

    # human-generated/ uses "(1)" suffixed filenames on disk; llm-generated/
    # does not. Adjust here if your folder uses different exact filenames.
    files = {
        ('human', 0): os.path.join(base_dir, 'human-generated', 'legit(1).csv'),
        ('human', 1): os.path.join(base_dir, 'human-generated', 'phishing(1).csv'),
        ('llm',   0): os.path.join(base_dir, 'llm-generated',   'legit.csv'),
        ('llm',   1): os.path.join(base_dir, 'llm-generated',   'phishing.csv'),
    }
    parts = []
    for (origin, label), path in files.items():
        if not os.path.exists(path):
            print(f"  [skip] '{path}' not found -- human/llm-generated "
                  f"cross-dataset holdout will be unavailable")
            return None
        loader = _load_ceas_schema if origin == 'human' else _load_text_label_schema
        part = loader(path, label)
        part['origin'] = origin
        parts.append(part)

    combined = pd.concat(parts, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=['message']).reset_index(drop=True)
    print(f"  human/llm-generated: dropped {before - len(combined)} internal "
          f"duplicates -> {len(combined):,} rows")
    return combined


# ═══════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

URL_PATTERN = re.compile(r'(https?://\S+|www\.\S+)')
IP_URL_PATTERN = re.compile(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')

URGENCY_WORDS = [
    'urgent', 'immediately', 'action required', 'act now', 'expire',
    'expiring', 'suspended', 'suspend', 'locked', 'deadline', 'within 24',
    'within 48', 'final notice', 'last chance', 'limited time'
]
CREDENTIAL_WORDS = [
    'password', 'verify your', 'confirm your', 'update your account',
    'login', 'log in', 'sign in', 'ssn', 'social security', 'pin number',
    'security question', 'account information', 'reset your'
]
FINANCIAL_WORDS = [
    'bank account', 'credit card', 'wire transfer', 'payment', 'invoice',
    'refund', 'billing', 'tax refund', 'irs', 'paypal', 'transaction'
]
CTA_WORDS = [
    'click here', 'click below', 'click the link', 'download now',
    'open the attachment', 'verify now', 'update now', 'confirm now'
]


def _count_terms(text, terms):
    text = text.lower()
    return sum(text.count(t) for t in terms)


def extract_url_features(text):
    """(url_count, has_url) -- kept for backward compatibility."""
    text = str(text)
    urls = URL_PATTERN.findall(text)
    return len(urls), int(len(urls) > 0)


def extract_url_features_rich(text):
    """Priority 3-J: richer URL signal set."""
    text = str(text)
    urls = URL_PATTERN.findall(text)
    url_count = len(urls)
    domains = set()
    lengths = []
    for u in urls:
        lengths.append(len(u))
        try:
            netloc = urlparse(u if u.startswith('http') else 'http://' + u).netloc
            if netloc:
                domains.add(netloc.lower())
        except Exception:
            pass
    ip_url_count = len(IP_URL_PATTERN.findall(text))
    avg_url_len = np.mean(lengths) if lengths else 0.0
    return pd.Series({
        'url_count': url_count,
        'has_url': int(url_count > 0),
        'unique_domain_count': len(domains),
        'ip_literal_url_count': ip_url_count,
        'avg_url_length': avg_url_len,
    })


def extract_linguistic_features(text):
    """Priority 3-I: urgency / credential / financial / CTA / style cues."""
    text = str(text)
    lower = text.lower()
    n_chars = max(len(text), 1)
    upper_chars = sum(1 for c in text if c.isupper())
    return pd.Series({
        'urgency_count': _count_terms(lower, URGENCY_WORDS),
        'credential_count': _count_terms(lower, CREDENTIAL_WORDS),
        'financial_count': _count_terms(lower, FINANCIAL_WORDS),
        'cta_count': _count_terms(lower, CTA_WORDS),
        'uppercase_ratio': upper_chars / n_chars,
        'exclamation_count': text.count('!'),
    })


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_feature_table(df, text_col='message'):
    """Apply all feature engineering (url + linguistic + cleaned text) to a df."""
    out = df.copy()
    url_feats = out[text_col].apply(extract_url_features_rich)
    ling_feats = out[text_col].apply(extract_linguistic_features)
    out = pd.concat([out, url_feats, ling_feats], axis=1)
    out['clean_message'] = out[text_col].apply(clean_text)
    return out


NUMERIC_FEATURES = [
    'url_count', 'has_url', 'unique_domain_count', 'ip_literal_url_count',
    'avg_url_length', 'urgency_count', 'credential_count', 'financial_count',
    'cta_count', 'uppercase_ratio', 'exclamation_count',
]


# ═══════════════════════════════════════════════════════════
# 3. LOAD + BUILD FEATURES
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("PHISHING EMAIL CLASSIFIER v2 -- loading data")
print("=" * 60)

df = load_ceas08('CEAS_08.csv') \
    if os.path.exists('CEAS_08.csv') else load_ceas08()

# ---- Priority 1-A: dedupe BEFORE split ----
before = len(df)
df = df.drop_duplicates(subset=['message']).reset_index(drop=True)
print(f"CEAS_08: dropped {before - len(df)} duplicate messages before split "
      f"({before} -> {len(df)})")

df = build_feature_table(df, text_col='message')

enron_df = load_enron('enron_spam_data.csv')
enron_df = build_feature_table(enron_df, text_col='full_text')
enron_ham = enron_df[enron_df['is_ham'] == 1].copy()

# ---- Priority 1-D: fixed Enron holdout, created ONCE, reused everywhere ----
enron_ham_train_pool, enron_ham_fixed_holdout = train_test_split(
    enron_ham, test_size=0.5, random_state=RANDOM_STATE
)
print(f"Enron ham: {len(enron_ham_train_pool)} available for train-augmentation pool, "
      f"{len(enron_ham_fixed_holdout)} frozen as fixed before/after holdout")

# ---- SECOND cross-dataset holdout: human/llm-generated phishing+legit ----
# This set shares CEAS_08's exact schema for its human-generated half, so
# cross-check for real overlap first (confirmed ~2.9% exact message-text
# duplicates in project testing) and drop anything that's also in CEAS_08
# before using it -- otherwise this "holdout" would partly be train data.
hl_df = load_human_llm_generated('.')
if hl_df is not None:
    ceas_messages = set(df['message'])
    before_hl = len(hl_df)
    hl_df = hl_df[~hl_df['message'].isin(ceas_messages)].reset_index(drop=True)
    print(f"human/llm-generated: dropped {before_hl - len(hl_df)} rows that "
          f"exactly match CEAS_08 messages (cross-dataset leakage guard) "
          f"-> {len(hl_df):,} rows remain")
    hl_df = build_feature_table(hl_df, text_col='message')
    # Fixed holdout, same spirit as Enron -- used only for evaluation,
    # never for training, so results are comparable across runs.
    hl_train_pool, hl_fixed_holdout = train_test_split(
        hl_df, test_size=0.5, random_state=RANDOM_STATE, stratify=hl_df['label_num']
    )
    print(f"human/llm-generated: {len(hl_train_pool):,} rows available for "
          f"train-augmentation pool, {len(hl_fixed_holdout):,} frozen as "
          f"fixed cross-dataset holdout (origin breakdown: "
          f"{hl_fixed_holdout['origin'].value_counts().to_dict()})")
else:
    hl_train_pool = None
    hl_fixed_holdout = None

feature_cols = ['clean_message'] + NUMERIC_FEATURES

# ═══════════════════════════════════════════════════════════
# 4. TRAIN / VALIDATION / TEST SPLIT (Priority 1-B)
# ═══════════════════════════════════════════════════════════

X = df[feature_cols]
y = df['label_num']

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765,  # -> ~15% of original
    random_state=RANDOM_STATE, stratify=y_trainval
)
# roughly 70% train / 15% val / 15% test
print(f"\nSplit sizes -> train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")

# ═══════════════════════════════════════════════════════════
# 5. PIPELINE (word TF-IDF + char TF-IDF + numeric features)  [Priority 3-K]
# ═══════════════════════════════════════════════════════════

# Priority-3c fix: the human/llm-generated error analysis showed cases
# like credential_count=26 still predicting near-0 phishing probability --
# evidence the ~11,000-dimensional TF-IDF block was drowning out the
# handful of engineered linguistic/URL numeric features in the linear
# model's coefficient budget. StandardScaler alone puts every numeric
# feature on a common unit-variance scale, but doesn't give them *more*
# relative weight than 11,000 TF-IDF dimensions can collectively exert.
# We multiply the scaled numeric block by a tunable constant so these
# features can't be ignored; the multiplier itself is tuned via
# GridSearchCV below rather than guessed.
#
# Implementation note: use ColumnTransformer's built-in `transformer_weights`
# instead of a custom function. A custom function defined in this script can't
# be pickled by joblib when GridSearchCV runs with n_jobs>1 AND Pipeline
# caching (memory=...) is on -- the worker processes can't find
# `__main__._scale_numeric` -> PicklingError. Built-in parameters have no such
# problem.
numeric_pipeline = StandardScaler()

preprocessor = ColumnTransformer(transformers=[
    ('word_tfidf', TfidfVectorizer(
        max_features=8000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, max_df=0.95, analyzer='word', stop_words='english'
    ), 'clean_message'),
    ('char_tfidf', TfidfVectorizer(
        max_features=3000, ngram_range=(3, 5), sublinear_tf=True,
        min_df=2, max_df=0.95, analyzer='char_wb'
    ), 'clean_message'),
    ('numeric', numeric_pipeline, NUMERIC_FEATURES),
], transformer_weights={'numeric': 3.0})  # default; overridden by the grid below

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(max_iter=2000, solver='liblinear')),
])

# ═══════════════════════════════════════════════════════════
# 6. HYPERPARAMETER TUNING (Priority 2-G)
# ═══════════════════════════════════════════════════════════

param_grid = {
    'preprocessor__word_tfidf__max_features': [5000, 8000],
    'preprocessor__word_tfidf__ngram_range': [(1, 1), (1, 2)],
    # Only 2 values: 1.0 vs 8.0 is enough to tell whether up-weighting the
    # numeric features helps at all. Each extra value here means another full
    # TF-IDF fit per fold (the expensive part), unlike C / class_weight.
    'preprocessor__transformer_weights': [
        {'numeric': 1.0}, {'numeric': 8.0}
    ],
    'clf__C': [0.1, 1.0, 3.0],
    'clf__class_weight': [None, 'balanced'],
}

# Why this was slow: fitting the preprocessor (word TF-IDF + char TF-IDF) on
# ~27k emails takes roughly a minute on its own. The grid used to refit it for
# every one of the 48 candidates x 5 folds = 240 times, even though C and
# class_weight don't change the TF-IDF output at all.
#
# Fix: (1) cache the fitted preprocessor on disk via Pipeline(memory=...), so
# candidates that differ only in C / class_weight reuse it -- only the 8
# distinct (max_features, ngram_range, numeric weight) combos are really
# fitted, once per fold; (2) 3 folds instead of 5; (3) run folds in parallel;
# (4) verbose=3 prints one line per finished fit so progress is visible
# (verbose=1 prints nothing between "Fitting..." and the final result).
import tempfile, shutil

GRID_N_JOBS = 4   # lower to 2 if RAM is tight, or raise (-1 = all cores)
GRID_CV_FOLDS = 3

_cache_dir = tempfile.mkdtemp(prefix='phish_grid_cache_')
pipeline.set_params(memory=_cache_dir)

print(f"\nRunning GridSearchCV: {len(ParameterGrid(param_grid))} candidates x "
      f"{GRID_CV_FOLDS} folds, n_jobs={GRID_N_JOBS}. Only 8 distinct preprocessor "
      f"fits per fold are real work, but every candidate still has to transform "
      f"its validation fold. Rough estimate: 15-30 minutes. One progress line "
      f"appears per finished fit -- if no line appears for 5+ minutes, "
      f"something is wrong.")
cv = StratifiedKFold(n_splits=GRID_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(
    pipeline, param_grid, scoring='average_precision',  # PR-AUC as tuning target
    cv=cv, n_jobs=GRID_N_JOBS, verbose=3
)
grid.fit(X_train, y_train)

# Stop caching for all later fits (augmentation refit etc.) and clean up.
best_pipeline = grid.best_estimator_
best_pipeline.set_params(memory=None)
pipeline.set_params(memory=None)
shutil.rmtree(_cache_dir, ignore_errors=True)

print(f"\nBest params: {grid.best_params_}")
print(f"Best CV PR-AUC: {grid.best_score_:.4f}")

# ═══════════════════════════════════════════════════════════
# 7. THRESHOLD TUNING ON VALIDATION SET (Priority 1-C)
# ═══════════════════════════════════════════════════════════

val_proba = best_pipeline.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)

# pick the threshold that maximises F1 on validation data
f1_scores = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
best_idx = np.argmax(f1_scores[:-1])  # last point has no corresponding threshold
best_threshold = thresholds[best_idx]
print(f"\nTuned decision threshold (max F1 on validation): {best_threshold:.4f}")
print(f"  -> Validation precision: {precisions[best_idx]:.4f}, "
      f"recall: {recalls[best_idx]:.4f}, F1: {f1_scores[best_idx]:.4f}")

# Also report a small fixed grid for comparison / documentation purposes
print("\nValidation performance at reference thresholds:")
for t in [0.3, 0.5, 0.7, best_threshold]:
    pred_t = (val_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred_t).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0
    print(f"  t={t:.3f}  F1={f1_score(y_val, pred_t):.4f}  FPR={fpr:.4f}  FNR={fnr:.4f}")


def predict_with_threshold(pipe, X, threshold):
    proba = pipe.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int), proba


# ═══════════════════════════════════════════════════════════
# 8. TEST SET EVALUATION (Priority 2-E, 2-F)
# ═══════════════════════════════════════════════════════════

y_pred_test, y_proba_test = predict_with_threshold(best_pipeline, X_test, best_threshold)

print("\n" + "=" * 60)
print("HELD-OUT TEST RESULTS -- CEAS_08 (deduped, tuned threshold)")
print("=" * 60)
print(f"Accuracy : {accuracy_score(y_test, y_pred_test)*100:.2f}%")
print(classification_report(y_test, y_pred_test, target_names=['Legitimate', 'Phishing']))

fpr_curve, tpr_curve, _ = roc_curve(y_test, y_proba_test)
roc_auc = auc(fpr_curve, tpr_curve)
pr_auc = average_precision_score(y_test, y_proba_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
fpr = fp / (fp + tn) if (fp + tn) else 0
fnr = fn / (fn + tp) if (fn + tp) else 0
print(f"ROC-AUC : {roc_auc:.4f}   PR-AUC : {pr_auc:.4f}")
print(f"FPR : {fpr:.4f}   FNR : {fnr:.4f}")

# ═══════════════════════════════════════════════════════════
# 9. ENRON CROSS-DATASET CHECK -- BEFORE augmentation (fixed holdout)
# ═══════════════════════════════════════════════════════════

X_holdout = enron_ham_fixed_holdout[feature_cols]
pred_holdout_before, proba_holdout_before = predict_with_threshold(
    best_pipeline, X_holdout, best_threshold
)
fpr_enron_before = pred_holdout_before.mean()
print("\n" + "=" * 60)
print("ENRON HAM -- FIXED HOLDOUT -- BEFORE augmentation")
print("=" * 60)
print(f"False positive rate (ham flagged as phishing): {fpr_enron_before*100:.2f}%")

# ═══════════════════════════════════════════════════════════
# 10. AUGMENT TRAIN WITH ENRON HAM POOL, REFIT, RE-CHECK SAME HOLDOUT
# ═══════════════════════════════════════════════════════════

aug_parts_X = [X_train, enron_ham_train_pool[feature_cols]]
aug_parts_y = [y_train, pd.Series([0] * len(enron_ham_train_pool))]
aug_note = f"+{len(enron_ham_train_pool):,} Enron ham"

# Priority-3b fix: also augment with the human/llm-generated train pool
# (labels 0/1 already correct per-file, both classes included -- unlike
# Enron which only contributes label=0 ham).
if hl_train_pool is not None:
    aug_parts_X.append(hl_train_pool[feature_cols])
    aug_parts_y.append(hl_train_pool['label_num'])
    aug_note += f", +{len(hl_train_pool):,} human/llm-generated (both classes)"

X_train_aug = pd.concat(aug_parts_X, ignore_index=True)
y_train_aug = pd.concat(aug_parts_y, ignore_index=True)
print(f"\nAugmented training set: {len(X_train_aug):,} ({aug_note})")

best_pipeline.fit(X_train_aug, y_train_aug)

# Re-tune threshold on validation set again (model changed)
val_proba_aug = best_pipeline.predict_proba(X_val)[:, 1]
precisions_a, recalls_a, thresholds_a = precision_recall_curve(y_val, val_proba_aug)
f1_a = 2 * precisions_a * recalls_a / np.clip(precisions_a + recalls_a, 1e-9, None)
best_threshold_aug = thresholds_a[np.argmax(f1_a[:-1])]
print(f"Re-tuned threshold after augmentation: {best_threshold_aug:.4f}")

pred_holdout_after, proba_holdout_after = predict_with_threshold(
    best_pipeline, X_holdout, best_threshold_aug
)
fpr_enron_after = pred_holdout_after.mean()

y_pred_test_aug, y_proba_test_aug = predict_with_threshold(
    best_pipeline, X_test, best_threshold_aug
)
print("\n" + "=" * 60)
print("BEFORE vs AFTER -- same fixed Enron-ham holdout, same CEAS_08 test set")
print("=" * 60)
print(f"Enron ham FPR   BEFORE: {fpr_enron_before*100:.2f}%   "
      f"AFTER: {fpr_enron_after*100:.2f}%")
print(f"CEAS_08 test F1 BEFORE: {f1_score(y_test, y_pred_test):.4f}   "
      f"AFTER: {f1_score(y_test, y_pred_test_aug):.4f}")
print(f"CEAS_08 test PR-AUC BEFORE: {pr_auc:.4f}   "
      f"AFTER: {average_precision_score(y_test, y_proba_test_aug):.4f}")

# ═══════════════════════════════════════════════════════════
# 10b. SECOND CROSS-DATASET CHECK -- human/llm-generated holdout
#      (evaluated with the final, Enron-augmented model + threshold;
#      never used for training, so this is a true generalization test)
# ═══════════════════════════════════════════════════════════

if hl_fixed_holdout is not None:
    X_hl = hl_fixed_holdout[feature_cols]
    y_hl = hl_fixed_holdout['label_num'].values
    pred_hl, proba_hl = predict_with_threshold(best_pipeline, X_hl, best_threshold_aug)

    print("\n" + "=" * 60)
    print("HUMAN/LLM-GENERATED -- FIXED HOLDOUT (second cross-dataset check)")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_hl, pred_hl)*100:.2f}%")
    print(classification_report(y_hl, pred_hl, labels=[0, 1],
                                 target_names=['Legitimate', 'Phishing'],
                                 zero_division=0))
    tn_hl, fp_hl, fn_hl, tp_hl = confusion_matrix(y_hl, pred_hl, labels=[0, 1]).ravel()
    fpr_hl = fp_hl / (fp_hl + tn_hl) if (fp_hl + tn_hl) else 0
    fnr_hl = fn_hl / (fn_hl + tp_hl) if (fn_hl + tp_hl) else 0
    print(f"FPR: {fpr_hl:.4f}   FNR: {fnr_hl:.4f}")

    # Breakdown by origin -- this is the interesting axis for this dataset:
    # does the model generalize equally to human-written vs LLM-generated
    # phishing/legit email?
    print("\nBreakdown by origin:")
    for origin_val in ['human', 'llm']:
        mask = (hl_fixed_holdout['origin'] == origin_val).values
        if mask.sum() == 0:
            continue
        acc_o = accuracy_score(y_hl[mask], pred_hl[mask])
        tn_o, fp_o, fn_o, tp_o = confusion_matrix(
            y_hl[mask], pred_hl[mask], labels=[0, 1]
        ).ravel()
        fpr_o = fp_o / (fp_o + tn_o) if (fp_o + tn_o) else 0
        fnr_o = fn_o / (fn_o + tp_o) if (fn_o + tp_o) else 0
        print(f"  origin={origin_val:5s} n={mask.sum():4d}  "
              f"accuracy={acc_o*100:5.2f}%  FPR={fpr_o:.4f}  FNR={fnr_o:.4f}")

# ═══════════════════════════════════════════════════════════
# 11. ERROR ANALYSIS (Priority 2-H)
# ═══════════════════════════════════════════════════════════

def error_analysis(df_source, feature_cols, y_true, y_pred, y_proba, label='CEAS_08 test'):
    results = df_source.copy().reset_index(drop=True)
    results['actual'] = np.asarray(y_true)
    results['predicted'] = y_pred
    results['proba'] = y_proba

    fp_rows = results[(results['actual'] == 0) & (results['predicted'] == 1)]
    fn_rows = results[(results['actual'] == 1) & (results['predicted'] == 0)]

    print(f"\n--- Error analysis: {label} ---")
    print(f"False positives: {len(fp_rows)}  |  False negatives: {len(fn_rows)}")

    if len(fp_rows):
        print("\nTop false positives by confidence (legit flagged as phishing):")
        for _, r in fp_rows.nlargest(5, 'proba').iterrows():
            snippet = str(r.get('message', r.get('full_text', '')))[:90]
            print(f"  [{r['proba']:.3f}] urgency={r.get('urgency_count',0)} "
                  f"credential={r.get('credential_count',0)} url={r.get('url_count',0)} "
                  f"-- {snippet}...")

    if len(fn_rows):
        print("\nTop false negatives by (low) confidence (phishing missed):")
        for _, r in fn_rows.nsmallest(5, 'proba').iterrows():
            snippet = str(r.get('message', r.get('full_text', '')))[:90]
            print(f"  [{r['proba']:.3f}] urgency={r.get('urgency_count',0)} "
                  f"credential={r.get('credential_count',0)} url={r.get('url_count',0)} "
                  f"-- {snippet}...")
    return fp_rows, fn_rows


test_source = df.loc[X_test.index]
fp_ceas, fn_ceas = error_analysis(
    test_source, feature_cols, y_test, y_pred_test_aug, y_proba_test_aug,
    label='CEAS_08 held-out test (after augmentation)'
)

enron_source = enron_ham_fixed_holdout
fp_enron, _ = error_analysis(
    enron_source, feature_cols,
    pd.Series([0] * len(enron_ham_fixed_holdout)),
    pred_holdout_after, proba_holdout_after,
    label='Enron ham fixed holdout (after augmentation)'
)

if hl_fixed_holdout is not None:
    error_analysis(
        hl_fixed_holdout, feature_cols, y_hl, pred_hl, proba_hl,
        label='human/llm-generated fixed holdout'
    )

# ═══════════════════════════════════════════════════════════
# 12. CHARTS
# ═══════════════════════════════════════════════════════════

def plot_confusion_matrix(cm_data, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved -> {filename}")


def plot_pr_roc(y_true, y_scores, title_prefix, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fpr_, tpr_, _ = roc_curve(y_true, y_scores)
    roc_auc_ = auc(fpr_, tpr_)
    ax1.plot(fpr_, tpr_, label=f'AUC={roc_auc_:.3f}')
    ax1.plot([0, 1], [0, 1], '--', color='gray')
    ax1.set_title(f'{title_prefix} -- ROC')
    ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR'); ax1.legend()

    prec_, rec_, _ = precision_recall_curve(y_true, y_scores)
    pr_auc_ = average_precision_score(y_true, y_scores)
    ax2.plot(rec_, prec_, label=f'PR-AUC={pr_auc_:.3f}')
    ax2.set_title(f'{title_prefix} -- Precision/Recall')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved -> {filename}")


plot_confusion_matrix(
    confusion_matrix(y_test, y_pred_test_aug),
    'Confusion Matrix -- CEAS_08 test (after augmentation)',
    'confusion_matrix_ceas08.png'
)
plot_confusion_matrix(
    confusion_matrix([0] * len(enron_ham_fixed_holdout), pred_holdout_after),
    'Confusion Matrix -- Enron ham holdout (after augmentation)',
    'confusion_matrix_enron.png'
)
plot_pr_roc(y_test, y_proba_test_aug, 'CEAS_08 test',
            'roc_pr_ceas08.png')

print("\nDone. Model, thresholds, and error-analysis tables are ready for the report.")