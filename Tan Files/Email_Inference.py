"""
inference.py -- tải pipeline đã lưu bởi train.py và dự đoán email mới.

Dùng như thư viện:
    from inference import PhishingPredictor
    predictor = PhishingPredictor()
    predictor.predict("Urgent: verify your account", "Click here to log in ...")

Hoặc chạy nhanh:
    python inference.py --subject "..." --body "..."
"""

import argparse

import joblib
import pandas as pd

from Email_Features import build_feature_table

MODEL_PATH = 'phishing_model.joblib'


class PhishingPredictor:
    def __init__(self, model_path=MODEL_PATH):
        artifact = joblib.load(model_path)
        self.pipeline = artifact['pipeline']
        self.threshold = artifact['threshold']
        self.feature_cols = artifact['feature_cols']

    def predict_batch(self, subjects, bodies):
        """Trả về list dict, mỗi dict ứng với một email."""
        df = pd.DataFrame({'subject': subjects, 'body': bodies})
        df['subject'] = df['subject'].fillna('')
        df['body'] = df['body'].fillna('')
        # Giống lúc train: message = subject + ' ' + body
        df['message'] = df['subject'] + ' ' + df['body']
        feats = build_feature_table(df, text_col='message')

        proba = self.pipeline.predict_proba(feats[self.feature_cols])[:, 1]
        return [
            {
                'label': 'phishing' if p >= self.threshold else 'legitimate',
                'is_phishing': bool(p >= self.threshold),
                'probability': float(p),
                'threshold': self.threshold,
            }
            for p in proba
        ]

    def predict(self, subject, body):
        return self.predict_batch([subject], [body])[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phishing email prediction')
    parser.add_argument('--subject', default='')
    parser.add_argument('--body', default='')
    parser.add_argument('--model', default=MODEL_PATH)
    args = parser.parse_args()

    result = PhishingPredictor(args.model).predict(args.subject, args.body)
    print(result)
