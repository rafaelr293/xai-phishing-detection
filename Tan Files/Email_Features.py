"""
features.py -- clean_text + toàn bộ feature engineering.

Dùng chung cho train.py và inference.py để đảm bảo email mới được xử lý
GIỐNG HỆT lúc train (tránh train/serve skew).
"""

import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

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

NUMERIC_FEATURES = [
    'url_count', 'has_url', 'unique_domain_count', 'ip_literal_url_count',
    'avg_url_length', 'urgency_count', 'credential_count', 'financial_count',
    'cta_count', 'uppercase_ratio', 'exclamation_count',
]

# Các cột mà pipeline mong đợi làm input
FEATURE_COLS = ['clean_message'] + NUMERIC_FEATURES


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
