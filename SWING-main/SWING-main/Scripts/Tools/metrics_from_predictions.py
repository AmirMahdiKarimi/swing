import os
import argparse
import pandas as pd

from ..common.metrics import compute_metrics, save_metrics_json


def main():
    ap = argparse.ArgumentParser(description='استخراج معیارها از فایل خروجی تست')
    ap.add_argument('--preds_csv', required=True, help='مسیر فایل CSV شامل ستون‌های prob و label')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--metrics_path', default='metrics_from_preds.json', help='مسیر ذخیرهٔ فایل JSON معیارها')
    args = ap.parse_args()

    df = pd.read_csv(args.preds_csv)
    if 'prob' not in df.columns or 'label' not in df.columns:
        raise ValueError('فایل باید شامل ستون‌های prob و label باشد.')

    metrics = compute_metrics(df['label'].values, df['prob'].values, threshold=args.threshold)
    save_metrics_json(metrics, args.metrics_path)
    print('Saved metrics to', args.metrics_path)


if __name__ == '__main__':
    main()