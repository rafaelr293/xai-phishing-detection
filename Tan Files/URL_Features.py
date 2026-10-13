"""
URL layer v3 — FEATURES
Canonicalisation + hostname-focused feature extraction (shared by train and inference).

Quy tắc v3:
  * Canonicalise mọi URL trước: lowercase, bỏ scheme, bỏ 'www.' đầu, bỏ '/' cuối.
    Scheme và www không bao giờ là feature nên không thể leak.
  * Hai nhóm feature, đều tính trên URL canonical:
        H        = chỉ hostname (lexical host + TLD)       <- URL layer chính
        H-noTLD  = H bỏ TLD
        HP       = H + path/query
Không có logic train/đánh giá ở đây: file này chỉ chứa phần biến URL -> feature,
nên train và inference luôn dùng CHÍNH XÁC cùng một code.
"""

import math
import re
from collections import Counter

import pandas as pd

try:
    import tldextract
    _TLDX = tldextract.TLDExtract(suffix_list_urls=())      # bundled snapshot, no network
except Exception:
    _TLDX = None

VOWELS = set('aeiou')
IP_RE = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')
SCHEME_RE = re.compile(r'^[a-z][a-z0-9+.\-]*://')
WWW_RE = re.compile(r'^www\d*\.')
FILE_EXT_RE = re.compile(r'\.(php|html?|asp|aspx|jsp|cgi)$')
SUSPICIOUS_WORDS = ['login', 'signin', 'secure', 'account', 'update', 'verify',
                    'banking', 'confirm', 'paypal', 'password', 'webscr',
                    'wallet', 'support', 'invoice', 'free', 'bonus']


# ─────────────────────────────────────────
# CANONICALISATION
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


# ─────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────
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


def build_feature_frame(urls):
    """list/Series of raw URLs -> DataFrame of features (incl. bookkeeping cols)."""
    return pd.DataFrame([url_features(u) for u in urls])


def matched_suspicious_words(url):
    """Which suspicious words appear in the host / in the path+query (for evidence)."""
    host, path, query = split_canon(canonicalize(url))
    pq = path + '?' + query
    return ([w for w in SUSPICIOUS_WORDS if w in host],
            [w for w in SUSPICIOUS_WORDS if w in pq])


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
# tên ngắn dùng cho CLI
FEATURE_SET_ALIASES = {'H': 'H (host only)',
                       'H-noTLD': 'H-noTLD (host, no TLD)',
                       'HP': 'HP (host+path)'}


# ─────────────────────────────────────────
# TLD CATEGORICAL HANDLING
# ─────────────────────────────────────────
def tld_cats(tld_train, top_k=50):
    return list(tld_train.value_counts().head(top_k).index)


def fit_cats(X):
    return tld_cats(X['tld']) if 'tld' in X.columns else []


def apply_tld(X, cats):
    """Map TLD to the fixed training categories (+ 'other'). No-op if no 'tld' column."""
    X = X.copy()
    if 'tld' in X.columns:
        X['tld'] = pd.Categorical(X['tld'].where(X['tld'].isin(cats), 'other'),
                                  categories=list(cats) + ['other'])
    return X
