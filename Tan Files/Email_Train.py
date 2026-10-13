"""
train.py -- PHISHING EMAIL CLASSIFIER v2 (training)

Gồm: load dữ liệu, dedupe/split, TF-IDF + ColumnTransformer + LogisticRegression,
GridSearchCV, threshold tuning trên validation, metrics, error analysis, charts,
và lưu model + threshold ra disk cho inference.py.

Chạy:  python train.py
"""

import os
import re
import shutil
import tempfile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, ParameterGrid, StratifiedKFold, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Email_Features import FEATURE_COLS, NUMERIC_FEATURES, build_feature_table

RANDOM_STATE = 42
MODEL_PATH = 'phishing_model.joblib'

GRID_N_JOBS = 8        # máy 8 core; hạ xuống 4 nếu thiếu RAM
GRID_CV_FOLDS = 3


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

      human-generated/legit.csv, phishing.csv -> CEAS_08-style schema
      llm-generated/legit.csv, phishing.csv   -> text,label (phishing.csv
        không quote nên cần repair từng dòng; cột label hằng số -> bỏ qua,
        dùng tên file làm ground truth)

    Returns df với cột: message, label_num, origin ('human'/'llm').
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

    # human-generated/ dùng hậu tố "(1)" trên disk; llm-generated/ thì không.
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
# 2. PIPELINE + GRID
# ═══════════════════════════════════════════════════════════

def build_pipeline():
    """Word TF-IDF + char TF-IDF + numeric features -> LogisticRegression."""
    # transformer_weights (built-in) thay vì custom function để joblib pickle
    # được khi GridSearchCV chạy n_jobs>1 + Pipeline(memory=...).
    preprocessor = ColumnTransformer(transformers=[
        ('word_tfidf', TfidfVectorizer(
            max_features=8000, ngram_range=(1, 2), sublinear_tf=True,
            min_df=2, max_df=0.95, analyzer='word', stop_words='english'
        ), 'clean_message'),
        ('char_tfidf', TfidfVectorizer(
            max_features=3000, ngram_range=(3, 5), sublinear_tf=True,
            min_df=2, max_df=0.95, analyzer='char_wb'
        ), 'clean_message'),
        ('numeric', StandardScaler(), NUMERIC_FEATURES),
    ], transformer_weights={'numeric': 3.0})  # default; được grid ghi đè

    return Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter=2000, solver='liblinear')),
    ])


PARAM_GRID = {
    'preprocessor__word_tfidf__max_features': [5000, 8000],
    'preprocessor__word_tfidf__ngram_range': [(1, 1), (1, 2)],
    'preprocessor__transformer_weights': [{'numeric': 1.0}, {'numeric': 8.0}],
    'clf__C': [0.1, 1.0, 3.0],
    'clf__class_weight': [None, 'balanced'],
}


def run_grid_search(pipeline, X_train, y_train):
    # Cache preprocessor đã fit: các candidate chỉ khác C / class_weight dùng lại
    # kết quả TF-IDF -> chỉ 8 tổ hợp thực sự phải fit mỗi fold.
    cache_dir = tempfile.mkdtemp(prefix='phish_grid_cache_')
    pipeline.set_params(memory=cache_dir)

    print(f"\nRunning GridSearchCV: {len(ParameterGrid(PARAM_GRID))} candidates x "
          f"{GRID_CV_FOLDS} folds, n_jobs={GRID_N_JOBS}.")
    cv = StratifiedKFold(n_splits=GRID_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipeline, PARAM_GRID, scoring='average_precision',  # PR-AUC as tuning target
        cv=cv, n_jobs=GRID_N_JOBS, verbose=3
    )
    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_
    best_pipeline.set_params(memory=None)
    pipeline.set_params(memory=None)
    shutil.rmtree(cache_dir, ignore_errors=True)

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV PR-AUC: {grid.best_score_:.4f}")
    return best_pipeline


# ═══════════════════════════════════════════════════════════
# 3. THRESHOLD + METRICS HELPERS
# ═══════════════════════════════════════════════════════════

def tune_threshold(y_true, proba):
    """Chọn threshold tối đa F1 trên validation."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1_scores = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
    best_idx = np.argmax(f1_scores[:-1])  # điểm cuối không có threshold tương ứng
    return thresholds[best_idx], precisions[best_idx], recalls[best_idx], f1_scores[best_idx]


def predict_with_threshold(pipe, X, threshold):
    proba = pipe.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int), proba


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


# ═══════════════════════════════════════════════════════════
# 4. CHARTS
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


# ═══════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════

def main():
    feature_cols = FEATURE_COLS

    print("=" * 60)
    print("PHISHING EMAIL CLASSIFIER v2 -- loading data")
    print("=" * 60)

    # ---- CEAS_08: dedupe BEFORE split (1-A) ----
    df = load_ceas08('CEAS_08.csv')
    before = len(df)
    df = df.drop_duplicates(subset=['message']).reset_index(drop=True)
    print(f"CEAS_08: dropped {before - len(df)} duplicate messages before split "
          f"({before} -> {len(df)})")
    df = build_feature_table(df, text_col='message')

    # ---- Enron: fixed holdout (1-D) ----
    enron_df = load_enron('enron_spam_data.csv')
    enron_df = build_feature_table(enron_df, text_col='full_text')
    enron_ham = enron_df[enron_df['is_ham'] == 1].copy()
    enron_ham_train_pool, enron_ham_fixed_holdout = train_test_split(
        enron_ham, test_size=0.5, random_state=RANDOM_STATE
    )
    print(f"Enron ham: {len(enron_ham_train_pool)} available for train-augmentation pool, "
          f"{len(enron_ham_fixed_holdout)} frozen as fixed before/after holdout")

    # ---- human/llm-generated: cross-dataset holdout (D2) ----
    hl_df = load_human_llm_generated('.')
    if hl_df is not None:
        ceas_messages = set(df['message'])
        before_hl = len(hl_df)
        hl_df = hl_df[~hl_df['message'].isin(ceas_messages)].reset_index(drop=True)
        print(f"human/llm-generated: dropped {before_hl - len(hl_df)} rows that "
              f"exactly match CEAS_08 messages (cross-dataset leakage guard) "
              f"-> {len(hl_df):,} rows remain")
        hl_df = build_feature_table(hl_df, text_col='message')
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

    # ---- Train / validation / test split (1-B) ----
    X = df[feature_cols]
    y = df['label_num']
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1765,  # ~15% of original
        random_state=RANDOM_STATE, stratify=y_trainval
    )
    print(f"\nSplit sizes -> train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")

    # ---- GridSearchCV (2-G) ----
    pipeline = build_pipeline()
    best_pipeline = run_grid_search(pipeline, X_train, y_train)

    # ---- Threshold tuning on validation (1-C) ----
    val_proba = best_pipeline.predict_proba(X_val)[:, 1]
    best_threshold, p_val, r_val, f1_val = tune_threshold(y_val, val_proba)
    print(f"\nTuned decision threshold (max F1 on validation): {best_threshold:.4f}")
    print(f"  -> Validation precision: {p_val:.4f}, recall: {r_val:.4f}, F1: {f1_val:.4f}")

    print("\nValidation performance at reference thresholds:")
    for t in [0.3, 0.5, 0.7, best_threshold]:
        pred_t = (val_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred_t).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0
        fnr = fn / (fn + tp) if (fn + tp) else 0
        print(f"  t={t:.3f}  F1={f1_score(y_val, pred_t):.4f}  FPR={fpr:.4f}  FNR={fnr:.4f}")

    # ---- Test set evaluation (2-E, 2-F) ----
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

    # ---- Enron holdout BEFORE augmentation ----
    X_holdout = enron_ham_fixed_holdout[feature_cols]
    pred_holdout_before, _ = predict_with_threshold(best_pipeline, X_holdout, best_threshold)
    fpr_enron_before = pred_holdout_before.mean()
    print("\n" + "=" * 60)
    print("ENRON HAM -- FIXED HOLDOUT -- BEFORE augmentation")
    print("=" * 60)
    print(f"False positive rate (ham flagged as phishing): {fpr_enron_before*100:.2f}%")

    # ---- Augment train (Enron ham + human/llm pool), refit ----
    aug_parts_X = [X_train, enron_ham_train_pool[feature_cols]]
    aug_parts_y = [y_train, pd.Series([0] * len(enron_ham_train_pool))]
    aug_note = f"+{len(enron_ham_train_pool):,} Enron ham"

    if hl_train_pool is not None:
        aug_parts_X.append(hl_train_pool[feature_cols])
        aug_parts_y.append(hl_train_pool['label_num'])
        aug_note += f", +{len(hl_train_pool):,} human/llm-generated (both classes)"

    X_train_aug = pd.concat(aug_parts_X, ignore_index=True)
    y_train_aug = pd.concat(aug_parts_y, ignore_index=True)
    print(f"\nAugmented training set: {len(X_train_aug):,} ({aug_note})")

    best_pipeline.fit(X_train_aug, y_train_aug)

    # Re-tune threshold (model đã đổi)
    val_proba_aug = best_pipeline.predict_proba(X_val)[:, 1]
    best_threshold_aug, _, _, _ = tune_threshold(y_val, val_proba_aug)
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

    # ---- human/llm-generated holdout (D2) ----
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

    # ---- Error analysis (2-H) ----
    test_source = df.loc[X_test.index]
    error_analysis(
        test_source, feature_cols, y_test, y_pred_test_aug, y_proba_test_aug,
        label='CEAS_08 held-out test (after augmentation)'
    )
    error_analysis(
        enron_ham_fixed_holdout, feature_cols,
        pd.Series([0] * len(enron_ham_fixed_holdout)),
        pred_holdout_after, proba_holdout_after,
        label='Enron ham fixed holdout (after augmentation)'
    )
    if hl_fixed_holdout is not None:
        error_analysis(
            hl_fixed_holdout, feature_cols, y_hl, pred_hl, proba_hl,
            label='human/llm-generated fixed holdout'
        )

    # ---- Charts ----
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
    plot_pr_roc(y_test, y_proba_test_aug, 'CEAS_08 test', 'roc_pr_ceas08.png')

    # ---- Save model + threshold cho inference.py ----
    joblib.dump(
        {
            'pipeline': best_pipeline,
            'threshold': float(best_threshold_aug),
            'feature_cols': feature_cols,
        },
        MODEL_PATH,
    )
    print(f"\nSaved model + threshold -> {MODEL_PATH}")
    print("Done. Model, thresholds, and error-analysis tables are ready for the report.")


if __name__ == '__main__':
    # Guard này bắt buộc: GridSearchCV với n_jobs>1 spawn worker process
    # (đặc biệt trên Windows/macOS) và sẽ import lại file này.
    main()
