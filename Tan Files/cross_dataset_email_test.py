import pandas as pd
import re
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────
# IMPORTANT CONTEXT — read before interpreting results
# ─────────────────────────────────────────
# CEAS_08's "legitimate" class was found to be mostly newsletters/mailing
# lists (CNN alerts, opensuse threads), not typical personal/business
# email. Enron-Spam's "ham" class is real workplace email instead —
# useful for checking whether the model over-fit to newsletter-style
# legitimate text and now misfires on more realistic legitimate email.
#
# CAVEAT: Enron-Spam's "spam" class is general spam (ads, scams), not
# specifically phishing. So this test mainly checks the LEGITIMATE side
# (false positive rate on realistic email) — it's a weaker signal for
# whether phishing detection itself generalises.

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

def load_enron(filepath='enron_spam_data.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find '{filepath}'. Download and extract it from "
            f"https://github.com/MWiechmann/enron_spam_data first."
        )
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    df['Subject'] = df['Subject'].fillna('')
    df['Message'] = df['Message'].fillna('')
    df['message'] = df['Subject'] + ' ' + df['Message']
    # Map to the same convention used by the email model: 1 = phishing/spam, 0 = legitimate
    df['label_num'] = df['Spam/Ham'].str.lower().map({'spam': 1, 'ham': 0})
    return df


print("=" * 55)
print("CROSS-DATASET TEST — CEAS_08-tuned model on Enron-Spam")
print("=" * 55)

print("\nLoading tuned model (email_model_tuned.joblib)...")
if not os.path.exists('email_model_tuned.joblib'):
    raise FileNotFoundError(
        "email_model_tuned.joblib not found — run email_finetune.py first."
    )
pipeline = joblib.load('email_model_tuned.joblib')

print("Loading Enron-Spam dataset...")
df = load_enron()
print(f"  Total Enron emails : {len(df):,}")
print(f"  Spam                : {(df['label_num']==1).sum():,}")
print(f"  Ham (legitimate)    : {(df['label_num']==0).sum():,}")

print("\nPreprocessing (same steps as CEAS_08 training)...")
df['url_count'] = df['message'].apply(extract_url_features)
df['has_url'] = (df['url_count'] > 0).astype(int)
df['clean_message'] = df['message'].apply(clean_text)

X_enron = df[['clean_message', 'url_count', 'has_url']]
y_enron = df['label_num']

print("\nRunning predictions...")
y_pred = pipeline.predict(X_enron)

print("\n" + "=" * 55)
print("RESULTS ON ENRON-SPAM (out-of-distribution test)")
print("=" * 55)
print(classification_report(y_enron, y_pred, target_names=['Legitimate', 'Spam']))

cm = confusion_matrix(y_enron, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"True  Negatives (Legit → Legit) : {tn:,}")
print(f"True  Positives (Spam → Spam)   : {tp:,}")
print(f"False Positives (Legit → Spam)  : {fp:,}  ⚠️  ← legitimate work email wrongly flagged")
print(f"False Negatives (Spam → Legit)  : {fn:,}")

fp_rate = fp / (fp + tn) * 100
print(f"\nFalse positive rate on realistic legitimate email: {fp_rate:.1f}%")

print("\n" + "=" * 55)
if fp_rate > 10:
    print("⚠️  Meaningfully higher false-positive rate than on CEAS_08's own")
    print("    test set — this suggests the model DID partly over-fit to")
    print("    CEAS_08's newsletter-style 'legitimate' class, as suspected.")
    print("    Worth considering: adding some Enron ham into training data")
    print("    (or another realistic-email source) to improve generalisation.")
else:
    print("✅ False-positive rate stayed reasonably low on more realistic")
    print("   legitimate email — a good sign the model isn't purely relying")
    print("   on CEAS_08's newsletter-specific patterns.")
print("=" * 55)
print("\nRemember: this dataset's 'spam' isn't specifically phishing, so")
print("treat the Spam-class numbers above as a secondary signal only.")