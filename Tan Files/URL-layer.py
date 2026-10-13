"""
URL layer v3 — hostname-focused, source-aware phishing URL classifier

Why v3 exists (findings from v2 on real data):
  * Every legit URL in PhiUSIIL is exactly 'https://www.<domain>'. Models learned
    "https + www + no path = legit": legit URLs of another dataset WITHOUT 'www'
    were flagged as phishing 100% of the time (FPR 80-85% cross-dataset).
  * Dropping the has_www column was not enough: 'www.' leaks through n_subdomains,
    n_dots, n_tokens, host_len and url_len.

What v3 changes:
  1. CANONICALISE every URL first (lowercase, strip scheme, strip leading 'www.',
     strip trailing '/'). Scheme and www are never features, and cannot leak.
  2. Two feature sets, both computed on the canonical URL:
       H  = hostname-only (lexical host features + TLD)        <- the main URL layer
       H-noTLD = H without the TLD (TLD mix often differs by data source)
       HP = H + path/query features                            <- shows the path shortcut
  3. Both sources are merged and evaluated three ways:
       a) merged grouped CV (grouped by registered domain), reported per source
       b) leave-one-source-out (train on A, test on B, and the reverse), with the decision
          threshold chosen on the TRAINING source (FPR=1%) and applied unchanged to the
          test source -> shows whether the threshold transfers
       c) adversarial check: how easily a model can tell the two sources apart
          *within the same class* (AUC 0.5 = indistinguishable, 1.0 = trivially separable)

Usage:
  python phishing_url_layer_v3.py --external "D:\\...\\dns_feature_94k.xlsx"
  python phishing_url_layer_v3.py --external ... --sample 40000     # quick dry run

Dependencies: pandas numpy scikit-learn lightgbm openpyxl (xlsx)   optional: tldextract
"""

import argparse
import math
import os
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

warnings.filterwarnings('ignore', message='.*eval_set.*')
warnings.filterwarnings('ignore', category=UserWarning)

try:
    import tldextract
    _TLDX = tldextract.TLDExtract(suffix_list_urls=())      # bundled snapshot, no network
except Exception:
    _TLDX = None

SEED = 42
PHISHING_SRC, EXTERNAL_SRC = 'PhiUSIIL', 'external'
VOWELS = set('aeiou')
IP_RE = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')
SCHEME_RE = re.compile(r'^[a-z][a-z0-9+.\-]*://')
WWW_RE = re.compile(r'^www\d*\.')
FILE_EXT_RE = re.compile(r'\.(php|html?|asp|aspx|jsp|cgi)$')
SUSPICIOUS_WORDS = ['login', 'signin', 'secure', 'account', 'update', 'verify',
                    'banking', 'confirm', 'paypal', 'password', 'webscr',
                    'wallet', 'support', 'invoice', 'free', 'bonus']

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


# ─────────────────────────────────────────
# CANONICALISATION + FEATURES
# ─────────────────────────────────────────
def canonicalize(url):
    """lowercase, strip scheme, strip leading www./www2., strip trailing '/'."""
    s = str(url).strip().lower()
    s = SCHEME_RE.sub('', s).lstrip('/')
    s = WWW_RE.sub('', s)
    return s.rstrip('/')


def split_canon(c):
    m = re.match(r'^([^/?#]*)(.*)$', c)
    hostport, rest = m.group(1), m.group(2)
    host = re.sub(r':\d+$', '', hostport.rsplit('@', 1)[-1])
    rest = rest.split('#')[0]
    path, _, query = rest.partition('?')
    return host, path, query


def split_domain(host):
    """-> (subdomain, domain_label, registered_domain, tld)"""
    if _TLDX is not None:
        e = _TLDX(host)
        reg = f"{e.domain}.{e.suffix}" if e.suffix else e.domain
        return e.subdomain, e.domain, reg, e.suffix
    parts = host.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[:-2]), parts[-2], '.'.join(parts[-2:]), parts[-1]
    return '', host, host, ''


def shannon_entropy(s):
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def max_consonant_run(s):
    best = cur = 0
    for ch in s:
        if ch.isalpha() and ch not in VOWELS:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def url_features(url):
    c = canonicalize(url)
    host, path, query = split_canon(c)
    sub, label, reg, suffix = split_domain(host)
    labels = [l for l in host.split('.') if l]
    host_digits = sum(ch.isdigit() for ch in host)
    pq = path + '?' + query
    pq_digits = sum(ch.isdigit() for ch in pq)
    return {
        # ---- hostname features (H) ----
        'host_len': len(host),
        'n_labels': len(labels),
        'n_subdomains': len([s for s in sub.split('.') if s]) if sub else 0,
        'domain_label_len': len(label),
        'tld_len': len(suffix),
        'max_label_len': max((len(l) for l in labels), default=0),
        'n_hyphens_host': host.count('-'),
        'n_digits_host': host_digits,
        'digit_ratio_host': host_digits / max(len(host), 1),
        'longest_digit_run_host': max((len(d) for d in re.findall(r'\d+', host)), default=0),
        'host_entropy': shannon_entropy(host),
        'domain_entropy': shannon_entropy(label),
        'vowel_ratio_domain': sum(ch in VOWELS for ch in label) / max(len(label), 1),
        'max_consonant_run_domain': max_consonant_run(label),
        'n_host_tokens': len([t for t in re.split(r'[^a-z0-9]+', host) if t]),
        'is_ip_host': int(bool(IP_RE.match(host))),
        'has_punycode': int('xn--' in host),
        'n_suspicious_words_host': sum(w in host for w in SUSPICIOUS_WORDS),
        'tld': suffix,
        # ---- path / query features (P) ----
        'canon_len': len(c),
        'path_len': len(path),
        'path_depth': len([s for s in path.split('/') if s]),
        'query_len': len(query),
        'n_equals': query.count('='),
        'n_amp': query.count('&'),
        'n_percent': pq.count('%'),
        'digit_ratio_path': pq_digits / max(len(pq), 1),
        'path_entropy': shannon_entropy(pq),
        'n_suspicious_words_path': sum(w in pq for w in SUSPICIOUS_WORDS),
        'has_file_ext': int(bool(FILE_EXT_RE.search(path))),
        'longest_digit_run_path': max((len(d) for d in re.findall(r'\d+', pq)), default=0),
        # ---- bookkeeping (never model inputs) ----
        '_bare': int(len(path) == 0 and len(query) == 0),
        '_reg': reg or host,
    }


H_COLS = ['host_len', 'n_labels', 'n_subdomains', 'domain_label_len', 'tld_len',
          'max_label_len', 'n_hyphens_host', 'n_digits_host', 'digit_ratio_host',
          'longest_digit_run_host', 'host_entropy', 'domain_entropy', 'vowel_ratio_domain',
          'max_consonant_run_domain', 'n_host_tokens', 'is_ip_host', 'has_punycode',
          'n_suspicious_words_host', 'tld']
P_COLS = ['canon_len', 'path_len', 'path_depth', 'query_len', 'n_equals', 'n_amp',
          'n_percent', 'digit_ratio_path', 'path_entropy', 'n_suspicious_words_path',
          'has_file_ext', 'longest_digit_run_path']
FEATURE_SETS = {'H (host only)': H_COLS,
                'H-noTLD (host, no TLD)': [c for c in H_COLS if c != 'tld'],
                'HP (host+path)': H_COLS + P_COLS}


# ─────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────
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


def tld_cats(tld_train, top_k=50):
    return list(tld_train.value_counts().head(top_k).index)


def fit_cats(X):
    return tld_cats(X['tld']) if 'tld' in X.columns else []


def apply_tld(X, cats):
    X = X.copy()
    if 'tld' in X.columns:
        X['tld'] = pd.Categorical(X['tld'].where(X['tld'].isin(cats), 'other'),
                                  categories=cats + ['other'])
    return X


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
# EXPERIMENTS
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
    a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED)
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


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--external', required=True, help="CSV/XLSX with URL + label columns")
    ap.add_argument('--phiusiil', default='PhiUSIIL_Phishing_URL_Dataset.csv')
    ap.add_argument('--sample', type=int, default=None,
                    help="use at most N rows per source (quick dry run)")
    ap.add_argument('--skip-adversarial', action='store_true')
    args = ap.parse_args()

    print("=" * 72 + "\nURL LAYER v3 — canonicalised, hostname-focused, source-aware\n" + "=" * 72)
    print("tldextract:", "yes" if _TLDX is not None else "NO (using simple domain split)")
    print("\nLoading and canonicalising...")
    ext = load_external(args.external)
    phi = load_phiusiil(args.phiusiil)
    data = pd.concat([prepare(phi, PHISHING_SRC, args.sample),
                      prepare(ext, EXTERNAL_SRC, args.sample)], ignore_index=True)

    print("\nExamples (original -> canonical):")
    for u in data.sample(5, random_state=1)['url']:
        print(f"  {u[:70]:<72} -> {canonicalize(u)[:60]}")

    print(f"\nExtracting features for {len(data):,} URLs (1-3 minutes)...")
    F = pd.DataFrame([url_features(u) for u in data['url']])
    y = data['label'].values.astype(int)
    source = data['source'].values
    groups = F['_reg'].values
    bare = F['_bare'].values

    print("\n" + "=" * 72 + "\nCANONICALISATION CHECK: form of URLs after cleaning\n" + "=" * 72)
    chk = pd.DataFrame({'group': [f"{s} {'phish' if l else 'legit'}" for s, l in zip(source, y)],
                        'share_bare': bare, 'mean_n_subdomains': F['n_subdomains'],
                        'mean_host_len': F['host_len'], 'mean_canon_len': F['canon_len']})
    print(chk.groupby('group').agg(n=('share_bare', 'size'), share_bare=('share_bare', 'mean'),
                                   mean_n_subdomains=('mean_n_subdomains', 'mean'),
                                   mean_host_len=('mean_host_len', 'mean'),
                                   mean_canon_len=('mean_canon_len', 'mean')).round(3).to_string())
    print("  (v2 problem: PhiUSIIL legit had www=100% vs 16% elsewhere; both are now stripped)")

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

    if not args.skip_adversarial:
        print("\n" + "=" * 72 + "\nC) ADVERSARIAL CHECK: can a model tell PhiUSIIL from external WITHIN a class?\n" + "=" * 72)
        arows = []
        for name, cols in FEATURE_SETS.items():
            adversarial(F, y, groups, source, cols, name, arows)
        adf = pd.DataFrame(arows)
        adf['AUC (source predictable?)'] = adf['AUC (source predictable?)'].round(3)
        print(adf.to_string(index=False))
        print("\n  ~0.5 = sources look alike; ~1.0 = remaining source artifacts the model could exploit.")


if __name__ == '__main__':
    main()