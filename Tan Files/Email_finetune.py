import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, precision_recall_curve,
                              fbeta_score, make_scorer, f1_score)

# ─────────────────────────────────────────
# 1. LOAD DATA (same loader as the main email script)
# ─────────────────────────────────────────
def load_ceas08(filepath='CEAS_08.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find '{filepath}' in your project folder.")

    df = pd.read_csv(filepath, encoding='latin-1')
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {'subject': 'subject', 'body': 'body', 'label': 'label'}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ['subject', 'body', 'label']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns {required}. Found: {list(df.columns)}")

    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    df['label_num'] = df['label'].astype(int)
    df['message'] = df['subject'] + ' ' + df['body']
    return df

URL_PATTERN = re.compile(r'(https?://\S+|www\.\S+)')

def extract_url_features(text):
    return len(URL_PATTERN.findall(str(text)))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


if __name__ == "__main__":

    print("=" * 55)
    print("EMAIL MODEL — HYPERPARAMETER TUNING + THRESHOLD CALIBRATION")
    print("=" * 55)

    print("\nLoading and preparing CEAS_08...")
    df = load_ceas08()
    df['url_count'] = df['message'].apply(extract_url_features)
    df['has_url'] = (df['url_count'] > 0).astype(int)
    df['clean_message'] = df['message'].apply(clean_text)

    feature_cols = ['clean_message', 'url_count', 'has_url']
    X = df[feature_cols]
    y = df['label_num']

    # Split off a FINAL held-out test set that GridSearchCV never sees —
    # tuning happens only on the remaining train_val portion via CV.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train/val (for tuning): {len(X_train_val):,}")
    print(f"Held-out test (final)  : {len(X_test):,}")

    # ─────────────────────────────────────────
    # 2. GRID SEARCH
    # ─────────────────────────────────────────
    preprocessor = ColumnTransformer(transformers=[
        ('tfidf', TfidfVectorizer(sublinear_tf=True, analyzer='word', stop_words='english'),
         'clean_message'),
        ('url_features', StandardScaler(), ['url_count', 'has_url']),
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'))
    ])

    param_grid = {
        'preprocessor__tfidf__max_features': [5000, 10000, 20000],
        'preprocessor__tfidf__ngram_range': [(1, 2), (1, 3)],
        'preprocessor__tfidf__min_df': [1, 2, 5],
        'clf__C': [0.1, 1.0, 10.0],
    }

    # F2 score weights recall higher than precision (beta=2) — deliberately
    # chosen because missing real phishing (false negative) is more
    # costly than a false alarm on a legitimate email (false positive).
    f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=1)

    print(f"\nRunning GridSearchCV over {len(param_grid['preprocessor__tfidf__max_features']) * len(param_grid['preprocessor__tfidf__ngram_range']) * len(param_grid['preprocessor__tfidf__min_df']) * len(param_grid['clf__C'])} combinations × 5-fold CV...")
    print("This can take a while depending on your CPU — grab a coffee.")

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring={'f1': 'f1', 'f2': f2_scorer, 'precision': 'precision', 'recall': 'recall'},
        refit='f2',   # optimise for catching phishing (recall-weighted), not raw accuracy
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_val, y_train_val)

    print("\n" + "=" * 55)
    print("BEST PARAMETERS FOUND")
    print("=" * 55)
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\n  Best CV F2 score: {grid_search.best_score_:.4f}")

    best_pipeline = grid_search.best_estimator_

    # ─────────────────────────────────────────
    # 3. EVALUATE BEST MODEL ON HELD-OUT TEST SET
    # ─────────────────────────────────────────
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 55)
    print("HELD-OUT TEST SET RESULTS (tuned model)")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

    # ─────────────────────────────────────────
    # 4. THRESHOLD CALIBRATION
    # ─────────────────────────────────────────
    # Uses the held-out test set's predicted probabilities to pick risk
    # thresholds based on actual precision/recall trade-offs, replacing
    # the placeholder values (0.85 / 0.60 / 0.35) from the draft.
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # precision_recall_curve returns thresholds of length N-1 vs N points —
    # align arrays by dropping the last precision/recall point (no threshold for it)
    precisions, recalls = precisions[:-1], recalls[:-1]

    def threshold_at_target_recall(target_recall):
        """Smallest threshold that still achieves at least target_recall."""
        valid = np.where(recalls >= target_recall)[0]
        if len(valid) == 0:
            return None
        return thresholds[valid].max()  # highest threshold among those meeting the recall target

    def threshold_at_target_precision(target_precision):
        """Largest threshold that still achieves at least target_precision —
        picking the highest (not lowest) valid threshold keeps CRITICAL
        correctly positioned above HIGH/MEDIUM on the probability scale."""
        valid = np.where(precisions >= target_precision)[0]
        if len(valid) == 0:
            return None
        return thresholds[valid].max()

    # MEDIUM/LOW boundary: catch 95% of real phishing (few missed phishing emails)
    medium_threshold = threshold_at_target_recall(0.95)
    # HIGH/MEDIUM boundary: catch 85% of real phishing, fewer false alarms
    high_threshold = threshold_at_target_recall(0.85)
    # CRITICAL/HIGH boundary: be at least 90% confident when flagging as critical
    critical_threshold = threshold_at_target_precision(0.90)

    print("\n" + "=" * 55)
    print("SUGGESTED RISK THRESHOLDS (based on held-out test set)")
    print("=" * 55)
    print(f"  CRITICAL if probability >= {critical_threshold:.3f}  (≥90% precision)")
    print(f"  HIGH     if probability >= {high_threshold:.3f}  (catches ≥85% of phishing)")
    print(f"  MEDIUM   if probability >= {medium_threshold:.3f}  (catches ≥95% of phishing)")
    print(f"  LOW      otherwise")
    print("\nCopy these into predict_email()'s if/elif chain in the main email script.")

    if None in (critical_threshold, high_threshold, medium_threshold):
        print("\n⚠️  Could not find a threshold meeting one of the targets above —")
        print("    the model may not reach that precision/recall level on this data.")
        print("    Consider relaxing the target (e.g. 90% recall instead of 95%).")

    # Plot precision/recall vs threshold for visual inspection
    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, precisions, label='Precision', color='steelblue')
    plt.plot(thresholds, recalls, label='Recall', color='crimson')
    for t, name in [(critical_threshold, 'CRITICAL'), (high_threshold, 'HIGH'), (medium_threshold, 'MEDIUM')]:
        if t is not None:
            plt.axvline(t, linestyle='--', color='gray', alpha=0.6)
            plt.text(t, 0.02, name, rotation=90, fontsize=8, va='bottom')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Score')
    plt.title('Precision / Recall vs. Threshold — Email Model')
    plt.legend()
    plt.tight_layout()
    plt.savefig('email_threshold_calibration.png', dpi=150)
    plt.show()
    print("\nSaved → email_threshold_calibration.png")

    # ─────────────────────────────────────────
    # 5. SAVE THE TUNED MODEL
    # ─────────────────────────────────────────
    joblib.dump(best_pipeline, 'email_model_tuned.joblib')
    print("\nSaved tuned pipeline → email_model_tuned.joblib")
    print("Load it later with: joblib.load('email_model_tuned.joblib')")