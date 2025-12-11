import os
import argparse
import sys

import numpy as np
import pandas as pd
from joblib import load

CURRENT_DIR = os.path.dirname(__file__)
SCRIPTS_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if SCRIPTS_ROOT not in sys.path:
    sys.path.append(SCRIPTS_ROOT)

from common.io import load_dataset, detect_columns
from common.metrics import compute_metrics, save_metrics_json, find_best_threshold
from common.features import build_features, FeatureConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_set', required=True)
    ap.add_argument('--seq_col', default=None)
    ap.add_argument('--label_col', default=None)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--output_dir', default=os.path.join('Results', 'LightGBM'))
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--kmer_k', type=int, default=2, help='k برای k-mer (0 یعنی غیرفعال)')
    ap.add_argument('--no_physchem', action='store_true', help='غیرفعال کردن ویژگی‌های فیزیکوشیمیایی')
    ap.add_argument('--no_embed_aaindex', action='store_true', help='غیرفعال کردن embedding ساده AAindex')
    ap.add_argument('--add_mhc', action='store_true', help='افزودن ویژگی وان‌هات برای ستون MHC در صورت وجود')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = 'LightGBM'
    dataset_base = os.path.splitext(os.path.basename(args.data_set))[0]
    df = load_dataset(args.data_set)
    if args.seq_col is None or args.label_col is None or args.seq_col not in df.columns or args.label_col not in df.columns:
        auto_label, auto_seq = detect_columns(df)
        args.label_col = args.label_col or auto_label
        args.seq_col = args.seq_col or auto_seq

    model = load(args.model_path)
    fcfg = FeatureConfig(
        add_physchem=not args.no_physchem,
        add_kmer_k=args.kmer_k if args.kmer_k >= 1 else 0,
        add_embedding_aaindex=not args.no_embed_aaindex,
        add_mhc=args.add_mhc,
    )
    X = build_features(df, args.seq_col, fcfg)
    y = df[args.label_col].astype(int).values

    probs = model.predict_proba(X)[:, 1]
    preds_path = os.path.join(args.output_dir, f'predictions_{model_name}_{dataset_base}.csv')
    pd.DataFrame({'prob': probs, 'label': y}).to_csv(preds_path, index=False)

    metrics = compute_metrics(y, probs, threshold=args.threshold)
    best_thr_f1, best_m_f1 = find_best_threshold(y, probs, metric='f1')
    best_thr_acc, best_m_acc = find_best_threshold(y, probs, metric='accuracy')
    metrics.update({
        'best_threshold_f1': best_thr_f1,
        'f1_best': best_m_f1.get('f1'),
        'accuracy_at_best_f1': best_m_f1.get('accuracy'),
        'best_threshold_accuracy': best_thr_acc,
        'accuracy_best': best_m_acc.get('accuracy'),
        'f1_at_best_accuracy': best_m_acc.get('f1'),
    })

    metrics.update({
        'model_name': model_name,
        'dataset': dataset_base,
        'label_column': args.label_col,
        'sequence_column': args.seq_col,
    })
    metrics_path = os.path.join(args.output_dir, f'metrics_{model_name}_{dataset_base}.json')
    save_metrics_json(metrics, metrics_path)

    print('Saved:')
    print(' -', preds_path)
    print(' -', metrics_path)


if __name__ == '__main__':
    main()
