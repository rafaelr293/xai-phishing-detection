"""
url_shift_diagnostic.py — NO model training (runs in ~1-2 minutes).

Compares the *form* of URLs between PhiUSIIL and an external dataset, and (optionally)
shows which URL traits the trained model false-alarms on, using the predictions CSV
written by URL-layer.py.

Usage (from the folder containing URL-layer.py and the PhiUSIIL csv):
  python url_shift_diagnostic.py --external "D:\\...\\dns_feature_94k.xlsx" ^
         --pred url_layer_v2_external_predictions.csv

Options: --phiusiil PhiUSIIL_Phishing_URL_Dataset.csv   --module URL-layer.py
"""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

GROUPS = ['PhiUSIIL legit', 'PhiUSIIL phish', 'external legit', 'external phish']


def load_module(path):
    spec = importlib.util.spec_from_file_location("urllayer", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # the __main__ guard keeps main() from running
    return m


def traits(f, urls):
    return pd.DataFrame({
        'share_https':          f['is_https'].values,
        'share_www':            f['has_www'].values,
        'share_bare_url':       ((f['path_len'] <= 1) & (f['query_len'] == 0)).astype(int).values,
        'share_trailing_slash': urls.astype(str).str.strip().str.endswith('/').astype(int).values,
        'share_ip_host':        f['is_ip_host'].values,
        'mean_slashes':         f['n_slash'].values,
        'mean_url_len':         f['url_len'].values,
        'mean_host_len':        f['host_len'].values,
        'mean_subdomains':      f['n_subdomains'].values,
        'mean_hyphens_in_host': f['n_hyphens_host'].values,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--external', required=True)
    ap.add_argument('--pred', default=None)
    ap.add_argument('--phiusiil', default='PhiUSIIL_Phishing_URL_Dataset.csv')
    ap.add_argument('--module', default='URL-layer.py')
    a = ap.parse_args()

    m = load_module(a.module)
    print("Loading data...")
    df = m.load_phiusiil(a.phiusiil)
    ext = m.load_external(a.external, None).drop_duplicates('url').reset_index(drop=True)

    print("Extracting features (1-2 minutes)...")
    own = m.extract_frame(df['URL'])
    feats = m.extract_frame(ext['url'])

    # ---- 1. trait table per source x class -------------------------------------------
    t1 = traits(own, df['URL'])
    t1['group'] = np.where(df['y'].values == 1, 'PhiUSIIL phish', 'PhiUSIIL legit')
    t2 = traits(feats, ext['url'])
    t2['group'] = np.where(ext['label'].values == 1, 'external phish', 'external legit')
    allt = pd.concat([t1, t2], ignore_index=True)
    tbl = allt.groupby('group').mean().T.reindex(columns=GROUPS)
    print("\n" + "=" * 70 + "\n1) URL FORM BY SOURCE AND CLASS\n" + "=" * 70)
    print(tbl.round(3).to_string())

    # ---- 2. random samples -------------------------------------------------------------
    print("\n" + "=" * 70 + "\n2) RANDOM URL SAMPLES (eyeball the difference)\n" + "=" * 70)
    samples = {
        'PhiUSIIL legit':  df.loc[df['y'] == 0, 'URL'],
        'PhiUSIIL phish':  df.loc[df['y'] == 1, 'URL'],
        'external legit':  ext.loc[ext['label'] == 0, 'url'],
        'external phish':  ext.loc[ext['label'] == 1, 'url'],
    }
    for g in GROUPS:
        print(f"\n  {g}:")
        for u in samples[g].sample(6, random_state=0):
            print("    " + str(u)[:100])

    # ---- 3. standardised mean difference, same class -----------------------------------
    num = [c for c in feats.columns if c not in ('tld', '_reg_domain')]
    print("\n" + "=" * 70 + "\n3) BIGGEST FEATURE SHIFTS (external vs PhiUSIIL, same class)\n" + "=" * 70)
    for lab, nm in ((0, 'legit'), (1, 'phish')):
        x = own.loc[df['y'].values == lab, num]
        y = feats.loc[ext['label'].values == lab, num]
        smd = (y.mean() - x.mean()) / np.sqrt((x.var() + y.var()) / 2 + 1e-9)
        top = smd.reindex(smd.abs().sort_values(ascending=False).index).head(8)
        print(f"\n  {nm}:")
        for f, v in top.items():
            print(f"    {f:<20} smd={v:+.2f}   PhiUSIIL={x[f].mean():8.3f}   external={y[f].mean():8.3f}")

    # ---- 4. where does the model false-alarm? ------------------------------------------
    if a.pred and os.path.exists(a.pred):
        p = pd.read_csv(a.pred)
        pf = m.extract_frame(p['url'])
        leg = (p['label'] == 0).values
        top_tld = pf['tld'].value_counts().head(8).index
        keys = {
            'has_www':          pf['has_www'],
            'is_https':         pf['is_https'],
            'n_subdomains (3 = 3+)': pf['n_subdomains'].clip(upper=3),
            'hyphen in host':   (pf['n_hyphens_host'] > 0).astype(int),
            'url_len quartile': pd.qcut(pf['url_len'], 4, duplicates='drop').astype(str),
            'tld (top 8)':      pf['tld'].where(pf['tld'].isin(top_tld), 'other'),
        }
        print("\n" + "=" * 70 + "\n4) LEGIT FALSE-POSITIVE RATE (proba >= 0.5) BY URL TRAIT\n" + "=" * 70)
        for col in [c for c in p.columns if c.startswith('proba_')]:
            fp = (p[col] >= 0.5).values
            for kname, k in keys.items():
                d = pd.DataFrame({'k': k.values[leg], 'fp': fp[leg]})
                g = d.groupby('k')['fp'].agg(FPR='mean', n='size')
                print(f"\n  [{col}]  by {kname}")
                print("    " + g.round(3).to_string().replace("\n", "\n    "))
    else:
        print("\n(4) skipped: pass --pred url_layer_v2_external_predictions.csv to see "
              "which URL traits the model false-alarms on)")


if __name__ == '__main__':
    main()