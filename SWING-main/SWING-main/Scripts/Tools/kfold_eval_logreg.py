import os
import argparse
import sys
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

CURRENT_DIR = os.path.dirname(__file__)
SCRIPTS_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if SCRIPTS_ROOT not in sys.path:
    sys.path.append(SCRIPTS_ROOT)

from common.io import load_dataset, detect_columns
from common.features import build_features
from common.metrics import compute_metrics, find_best_threshold


def main():
    ap = argparse.ArgumentParser(description='ارزیابی k-fold برای Logistic Regression')
    ap.add_argument('--data_set', required=True)
    ap.add_argument('--seq_col', default=None)
    ap.add_argument('--label_col', default=None)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--random_state', type=int, default=42)
    ap.add_argument('--output_dir', default=os.path.join('Results', 'LogReg'))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_dataset(args.data_set)
    if args.seq_col is None or args.label_col is None or args.seq_col not in df.columns or args.label_col not in df.columns:
        auto_label, auto_seq = detect_columns(df)
        args.label_col = args.label_col or auto_label
        args.seq_col = args.seq_col or auto_seq

    X = build_features(df, args.seq_col)
    y = df[args.label_col].astype(int).values

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.random_state)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=1)),
        ])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]

        m = compute_metrics(y_test, probs, threshold=0.5)
        best_thr_f1, best_m_f1 = find_best_threshold(y_test, probs, metric='f1')
        best_thr_acc, best_m_acc = find_best_threshold(y_test, probs, metric='accuracy')
        m.update({
            'best_threshold_f1': best_thr_f1,
            'f1_best': best_m_f1.get('f1'),
            'accuracy_at_best_f1': best_m_f1.get('accuracy'),
            'best_threshold_accuracy': best_thr_acc,
            'accuracy_best': best_m_acc.get('accuracy'),
            'f1_at_best_accuracy': best_m_acc.get('f1'),
        })
        fold_metrics.append(m)
        print(f"Fold {fold_idx}/{args.k}: auc={m.get('auc'):.4f}, f1={m.get('f1'):.4f}, acc={m.get('accuracy'):.4f}")

    # خلاصهٔ تجمیعی
    def agg(key):
        vals = [m.get(key, np.nan) for m in fold_metrics]
        vals = np.array(vals, dtype=float)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    summary = {
        'auc_mean': agg('auc')[0], 'auc_std': agg('auc')[1],
        'f1_mean': agg('f1')[0], 'f1_std': agg('f1')[1],
        'accuracy_mean': agg('accuracy')[0], 'accuracy_std': agg('accuracy')[1],
        'precision_mean': agg('precision')[0], 'precision_std': agg('precision')[1],
        'recall_mean': agg('recall')[0], 'recall_std': agg('recall')[1],
        'avg_precision_mean': agg('avg_precision')[0], 'avg_precision_std': agg('avg_precision')[1],
    }

    with open(os.path.join(args.output_dir, 'kfold_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({'folds': fold_metrics, 'summary': summary}, f, indent=2)

    print('Saved summary to', os.path.join(args.output_dir, 'kfold_metrics.json'))


if __name__ == '__main__':
    main()