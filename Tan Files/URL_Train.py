"""
URL layer v3 — TRAIN
Huấn luyện, đánh giá validation (giữ nguyên các protocol của v3) và lưu model.

Các bước:
  1. Nạp PhiUSIIL + dataset external, canonicalise, khử trùng lặp.
  2. Trích feature (URL_layer_features.py).
  3. Đánh giá (có thể bỏ bằng --skip-eval):
       A) merged grouped 5-fold CV (group theo registered domain)
       B) leave-one-source-out, threshold chọn trên nguồn train (FPR=1%)
       C) adversarial check (--skip-adversarial để bỏ)
  4. Train model cuối trên dữ liệu gộp (grouped hold-out 15% để early stopping +
     chọn threshold FPR=1%), rồi lưu:
        models/url_pipeline.joblib   (model, cột feature, TLD categories, threshold)
        models/url_metadata.json     (cấu hình + metric validation)

Usage:
  python URL_layer_train.py --external "D:\\...\\dns_feature_94k.xlsx"
  python URL_layer_train.py --external ... --sample 40000 --skip-adversarial   # dry run
  python URL_layer_train.py --external ... --feature-set H --skip-eval         # chỉ train + lưu

Dependencies: pandas numpy scikit-learn lightgbm joblib openpyxl (xlsx)  optional: tldextract
"""

import argparse
import json
import os
import warnings
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import URL_Features as UF
from URL_Features import (FEATURE_SETS, FEATURE_SET_ALIASES, apply_tld, canonicalize,
                                fit_cats, build_feature_frame)

warnings.filterwarnings('ignore', message='.*eval_set.*')
warnings.filterwarnings('ignore', category=UserWarning)

SEED = 42
PHISHING_SRC, EXTERNAL_SRC = 'PhiUSIIL', 'external'

# ─────────────────────────────────────────
# EXTERNAL FILE LOADING (CSV / XLSX)
# ─────────────────────────────────────────
URL_COLS = ['url', 'urls', 'link', 'links', 'phish_url', 'website', 'address']
LABEL_COLS = ['label', 'class', 'type', 'category', 'is_phishing', 'target', 'status']
PHISH_WORDS = {'1', '1.0', 'true', 'phishing', 'phish', 'malicious', 'bad'}
LEGIT_WORDS = {'0', '0.0', 'false', 'legitimate', 'legit', 'benign', 'good', 'safe'}


def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xlsm', '.xls'):
        sheets = pd.read_excel(path, sheet_name=None)       # needs: pip install openpyxl
        return pd.concat(sheets.values(), ignore_index=True)
    return pd.read_csv(path)


def find_url_column(tbl):
    lower = {str(c).strip().lower(): c for c in tbl.columns}
    for name in URL_COLS:
        if name in lower:
            return lower[name]
    best, best_score = None, 0.0
    for c in tbl.columns:
        s = tbl[c].dropna().astype(str).head(500)
        if len(s):
            score = s.str.contains(r'^(?:https?://|www\.)|\.[a-z]{2,}', case=False,
                                   regex=True).mean()
            if score > best_score:
                best, best_score = c, score
    if best is None or best_score < 0.8:
        raise ValueError(f"Could not find a URL column. Columns: {list(tbl.columns)}")
    return best


def load_external(path):
    """Return DataFrame[url, label] with label 1 = phishing, 0 = legit."""
    tbl = read_table(path)
    ucol = find_url_column(tbl)
    lower = {str(c).strip().lower(): c for c in tbl.columns}
    lcol = next((lower[n] for n in LABEL_COLS if n in lower), None)
    if lcol is None:
        raise ValueError(f"{path}: no label column found among {list(tbl.columns)}")
    raw = tbl[lcol].astype(str).str.strip().str.lower()
    lab = raw.map(lambda v: 1 if v in PHISH_WORDS else 0 if v in LEGIT_WORDS else np.nan)
    if lab.isna().any():
        raise ValueError(f"{path}: unrecognised label values {list(raw[lab.isna()].unique())[:10]}")
    out = pd.DataFrame({'url': tbl[ucol].astype(str).str.strip(), 'label': lab.astype(int)})
    out = out[(out['url'].str.len() > 3) & ~out['url'].isin(['nan', 'None'])]
    print(f"  {os.path.basename(path)}: url='{ucol}', label='{lcol}', {len(out):,} rows, "
          f"labels={out['label'].value_counts().to_dict()}")
    return out


def load_phiusiil(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find '{path}'. Use --phiusiil to point to it.")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # raw PhiUSIIL: label 1 = LEGIT. Flip so 1 = phishing.
    return pd.DataFrame({'url': df['URL'].astype(str).str.strip(),
                         'label': 1 - df['label'].astype(int)})


def prepare(df, source, sample=None):
    """Canonicalise, drop within-source duplicates and label-conflicting URLs."""
    out = df.copy()
    out['canon'] = out['url'].map(canonicalize)
    n0 = len(out)
    conflict = out.groupby('canon')['label'].transform('nunique') > 1
    out = out[~conflict].drop_duplicates('canon')
    print(f"  {source}: {n0:,} rows -> {len(out):,} after canonical de-duplication "
          f"({int(conflict.sum()):,} label-conflicting dropped)")
    if sample and len(out) > sample:
        out = out.sample(sample, random_state=SEED)
    out['source'] = source
    return out.reset_index(drop=True)


# ─────────────────────────────────────────
# METRICS / MODEL HELPERS
# ─────────────────────────────────────────
def threshold_at_fpr(y, s, target=0.01):
    fpr, _, thr = roc_curve(y, s)
    ok = np.where(fpr <= target)[0]
    return float(thr[ok[-1]]) if len(ok) else float('inf')


def recall_at_fpr(y, s, target=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = np.where(fpr <= target)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


def slice_metrics(y, s, t=None):
    if len(y) == 0 or len(set(y)) < 2:
        return None
    r = {'n': len(y), 'phish%': round(100 * float(np.mean(y)), 1),
         'AUC': roc_auc_score(y, s), 'PR-AUC': average_precision_score(y, s),
         'recall@FPR1%': recall_at_fpr(y, s)}
    if t is not None:
        pred = s >= t
        r['TPR@t'] = float(pred[y == 1].mean())
        r['FPR@t'] = float(pred[y == 0].mean())
    return r


def fit_lgb(Xtr, ytr, Xva=None, yva=None, n_estimators=600):
    m = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
                           class_weight='balanced', random_state=SEED, verbose=-1)
    if Xva is not None:
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='auc',
              callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        m.fit(Xtr, ytr)
    return m


def top_gain(model, cols, k=3):
    g = pd.Series(model.booster_.feature_importance('gain'), index=cols)
    g = (g / g.sum()).sort_values(ascending=False).head(k)
    return ", ".join(f"{n} ({v:.0%})" for n, v in g.items())


# ─────────────────────────────────────────
# EVALUATION EXPERIMENTS (giữ nguyên từ v3)
# ─────────────────────────────────────────
def merged_cv(F, y, groups, source, bare, cols, name, rows):
    oof = np.zeros(len(y))
    X = F[cols]
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        cats = fit_cats(X.iloc[tr])
        m = fit_lgb(apply_tld(X.iloc[tr], cats), y[tr], n_estimators=300)
        oof[te] = m.predict_proba(apply_tld(X.iloc[te], cats))[:, 1]
    for src in (PHISHING_SRC, EXTERNAL_SRC):
        for sl, mask in (('all', source == src), ('bare', (source == src) & (bare == 1))):
            r = slice_metrics(y[mask], oof[mask])
            if r:
                rows.append({'protocol': 'merged grouped CV', 'direction': src,
                             'features': name, 'slice': sl, **r})


def loso(F, y, groups, source, bare, cols, name, train_src, test_src, rows):
    X = F[cols]
    tr_idx = np.where(source == train_src)[0]
    te_idx = np.where(source == test_src)[0]
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
                .split(tr_idx, y[tr_idx], groups[tr_idx]))
    fit_idx, val_idx = tr_idx[a], tr_idx[b]
    cats = fit_cats(X.iloc[fit_idx])
    m = fit_lgb(apply_tld(X.iloc[fit_idx], cats), y[fit_idx],
                apply_tld(X.iloc[val_idx], cats), y[val_idx])
    s_val = m.predict_proba(apply_tld(X.iloc[val_idx], cats))[:, 1]
    t = threshold_at_fpr(y[val_idx], s_val, 0.01)            # chosen on the TRAIN source only
    s_te = m.predict_proba(apply_tld(X.iloc[te_idx], cats))[:, 1]
    unseen = ~pd.Series(groups[te_idx]).isin(set(groups[tr_idx])).values
    direction = f"{train_src} -> {test_src}"
    print(f"  [{name}] {direction}: best_iter={m.best_iteration_}, threshold@FPR1%(train)={t:.3f}, "
          f"top gain: {top_gain(m, list(X.columns))}")
    r = slice_metrics(y[val_idx], s_val, t)                    # in-source reference
    if r:
        rows.append({'protocol': 'leave-one-source-out', 'direction': f"{train_src} (in-source val)",
                     'features': name, 'slice': 'all', **r})
    bt = bare[te_idx] == 1
    for sl, mask in (('all', np.ones(len(te_idx), bool)), ('unseen domains', unseen),
                     ('bare', bt), ('bare+unseen', bt & unseen)):
        r = slice_metrics(y[te_idx][mask], s_te[mask], t)
        if r:
            rows.append({'protocol': 'leave-one-source-out', 'direction': direction,
                         'features': name, 'slice': sl, **r})


def adversarial(F, y, groups, source, cols, name, rows, per_source=20000):
    for cls, cname in ((0, 'legit'), (1, 'phish')):
        idx = []
        for src in (PHISHING_SRC, EXTERNAL_SRC):
            i = np.where((y == cls) & (source == src))[0]
            if len(i) > per_source:
                i = np.random.default_rng(SEED).choice(i, per_source, replace=False)
            idx.append(i)
        idx = np.concatenate(idx)
        target = (source[idx] == EXTERNAL_SRC).astype(int)
        X = F.iloc[idx][cols].reset_index(drop=True)
        g = groups[idx]
        if len(set(target)) < 2:
            continue
        oof = np.zeros(len(idx))
        for tr, te in GroupKFold(n_splits=3).split(X, target, g):
            cats = fit_cats(X.iloc[tr])
            m = fit_lgb(apply_tld(X.iloc[tr], cats), target[tr], n_estimators=200)
            oof[te] = m.predict_proba(apply_tld(X.iloc[te], cats))[:, 1]
        cats = fit_cats(X)
        mfull = fit_lgb(apply_tld(X, cats), target, n_estimators=200)
        rows.append({'class': cname, 'features': name,
                     'AUC (source predictable?)': roc_auc_score(target, oof),
                     'top features': top_gain(mfull, list(X.columns))})


def run_evaluation(F, y, groups, source, bare, skip_adversarial):
    rows = []
    print("\n" + "=" * 72 + "\nA) MERGED, GROUPED 5-FOLD CV (grouped by registered domain)\n" + "=" * 72)
    for name, cols in FEATURE_SETS.items():
        merged_cv(F, y, groups, source, bare, cols, name, rows)
    print("\n" + "=" * 72 + "\nB) LEAVE-ONE-SOURCE-OUT (threshold = FPR 1% on the training source)\n" + "=" * 72)
    for name, cols in FEATURE_SETS.items():
        for tr_s, te_s in ((PHISHING_SRC, EXTERNAL_SRC), (EXTERNAL_SRC, PHISHING_SRC)):
            loso(F, y, groups, source, bare, cols, name, tr_s, te_s, rows)

    res = pd.DataFrame(rows)
    show = ['protocol', 'direction', 'features', 'slice', 'n', 'phish%', 'AUC', 'PR-AUC',
            'recall@FPR1%', 'TPR@t', 'FPR@t']
    out = res[show].copy()
    for c in ('AUC', 'PR-AUC', 'recall@FPR1%', 'TPR@t', 'FPR@t'):
        out[c] = out[c].astype(float).round(3)
    print("\n" + "=" * 72 + "\nRESULTS\n" + "=" * 72)
    print("TPR@t / FPR@t: phishing recall and legit false-positive rate when the threshold\n"
          "chosen on the training source (FPR=1%) is applied unchanged to the test data.\n")
    print(out.to_string(index=False))
    out.to_csv('url_layer_v3_results.csv', index=False)
    print("\nSaved -> url_layer_v3_results.csv")

    if not skip_adversarial:
        print("\n" + "=" * 72 + "\nC) ADVERSARIAL CHECK: can a model tell PhiUSIIL from external WITHIN a class?\n" + "=" * 72)
        arows = []
        for name, cols in FEATURE_SETS.items():
            adversarial(F, y, groups, source, cols, name, arows)
        adf = pd.DataFrame(arows)
        adf['AUC (source predictable?)'] = adf['AUC (source predictable?)'].round(3)
        print(adf.to_string(index=False))
        print("\n  ~0.5 = sources look alike; ~1.0 = remaining source artifacts the model could exploit.")


# ─────────────────────────────────────────
# FINAL TRAINING + SAVING
# ─────────────────────────────────────────
def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not serialisable: {type(o)}")


def train_and_save(F, y, groups, source, feature_set, target_fpr, model_dir):
    cols = FEATURE_SETS[feature_set]
    X = F[cols]
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
                .split(X, y, groups))
    cats = fit_cats(X.iloc[a])
    m = fit_lgb(apply_tld(X.iloc[a], cats), y[a], apply_tld(X.iloc[b], cats), y[b])
    s_val = m.predict_proba(apply_tld(X.iloc[b], cats))[:, 1]

    t = threshold_at_fpr(y[b], s_val, target_fpr)
    if not np.isfinite(t):
        print("  WARNING: could not reach the target FPR on validation; falling back to 0.5")
        t = 0.5

    val_metrics = {'all': slice_metrics(y[b], s_val, t)}
    for src in (PHISHING_SRC, EXTERNAL_SRC):
        mask = source[b] == src
        r = slice_metrics(y[b][mask], s_val[mask], t)
        if r:
            val_metrics[src] = r

    print(f"\n  feature set     : {feature_set}  ({len(cols)} features)")
    print(f"  train / val     : {len(a):,} / {len(b):,} (grouped by registered domain)")
    print(f"  best_iteration  : {m.best_iteration_}")
    print(f"  threshold       : {t:.4f}  (FPR={target_fpr:.0%} on validation)")
    print(f"  top gain        : {top_gain(m, cols)}")
    for k, r in val_metrics.items():
        print(f"  val[{k:<8}]    : AUC={r['AUC']:.4f}  PR-AUC={r['PR-AUC']:.4f}  "
              f"TPR@t={r['TPR@t']:.3f}  FPR@t={r['FPR@t']:.3f}  (n={r['n']:,})")

    os.makedirs(model_dir, exist_ok=True)
    pipe_path = os.path.join(model_dir, 'url_pipeline.joblib')
    meta_path = os.path.join(model_dir, 'url_metadata.json')
    joblib.dump({'model': m, 'feature_cols': cols, 'tld_cats': cats, 'threshold': float(t),
                 'feature_set': feature_set, 'target_fpr': target_fpr}, pipe_path)
    meta = {'version': 'url_layer_v3',
            'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'feature_set': feature_set, 'feature_cols': cols, 'tld_cats': cats,
            'threshold': float(t), 'threshold_rule': f'FPR={target_fpr} on grouped validation',
            'best_iteration': int(m.best_iteration_ or 0), 'n_train': int(len(a)),
            'n_val': int(len(b)), 'tldextract': UF._TLDX is not None,
            'validation': val_metrics}
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=_jsonable)
    print(f"\nSaved -> {pipe_path}\nSaved -> {meta_path}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="URL layer v3 — train / evaluate / save")
    ap.add_argument('--external', required=True, help="CSV/XLSX with URL + label columns")
    ap.add_argument('--phiusiil', default='PhiUSIIL_Phishing_URL_Dataset.csv')
    ap.add_argument('--sample', type=int, default=None,
                    help="use at most N rows per source (quick dry run)")
    ap.add_argument('--feature-set', choices=list(FEATURE_SET_ALIASES), default='H',
                    help="feature set of the saved model (default: H = host only)")
    ap.add_argument('--target-fpr', type=float, default=0.01,
                    help="FPR used to choose the decision threshold (default 0.01)")
    ap.add_argument('--model-dir', default='models')
    ap.add_argument('--skip-eval', action='store_true',
                    help="skip protocols A/B/C and only train + save the final model")
    ap.add_argument('--skip-adversarial', action='store_true')
    args = ap.parse_args()

    print("=" * 72 + "\nURL LAYER v3 — TRAIN\n" + "=" * 72)
    print("tldextract:", "yes" if UF._TLDX is not None else "NO (using simple domain split)")
    print("\nLoading and canonicalising...")
    ext = load_external(args.external)
    phi = load_phiusiil(args.phiusiil)
    data = pd.concat([prepare(phi, PHISHING_SRC, args.sample),
                      prepare(ext, EXTERNAL_SRC, args.sample)], ignore_index=True)

    print(f"\nExtracting features for {len(data):,} URLs (1-3 minutes)...")
    F = build_feature_frame(data['url'])
    y = data['label'].values.astype(int)
    source = data['source'].values
    groups = F['_reg'].values
    bare = F['_bare'].values

    if not args.skip_eval:
        run_evaluation(F, y, groups, source, bare, args.skip_adversarial)

    print("\n" + "=" * 72 + "\nFINAL MODEL\n" + "=" * 72)
    train_and_save(F, y, groups, source, FEATURE_SET_ALIASES[args.feature_set],
                   args.target_fpr, args.model_dir)


if __name__ == '__main__':
    main()
