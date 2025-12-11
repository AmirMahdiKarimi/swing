import os
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump

# اضافه کردن Scripts به sys.path برای ایمپورت‌های پایدار هنگام اجرای مستقیم اسکریپت
CURRENT_DIR = os.path.dirname(__file__)
SCRIPTS_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if SCRIPTS_ROOT not in sys.path:
    sys.path.append(SCRIPTS_ROOT)

from common.io import load_dataset, train_test_split_df, detect_columns
from common.metrics import compute_metrics, save_metrics_json, find_best_threshold
from common.features import build_features, FeatureConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_set', required=True, help='مسیر CSV دیتاست')
    ap.add_argument('--seq_col', default=None, help='نام ستون دنباله؛ اگر خالی باشد خودکار کشف می‌شود')
    ap.add_argument('--label_col', default=None, help='نام ستون برچسب دودویی؛ اگر خالی باشد خودکار کشف می‌شود')
    ap.add_argument('--output_dir', default=os.path.join('Results', 'LogReg'), help='پوشهٔ خروجی')
    ap.add_argument('--test_size', type=float, default=0.5)
    ap.add_argument('--random_state', type=int, default=42)
    ap.add_argument('--kmer_k', type=int, default=2, help='k برای k-mer (0 یعنی غیرفعال)')
    ap.add_argument('--no_physchem', action='store_true', help='غیرفعال کردن ویژگی‌های فیزیکوشیمیایی')
    ap.add_argument('--no_embed_aaindex', action='store_true', help='غیرفعال کردن embedding ساده AAindex')
    ap.add_argument('--add_mhc', action='store_true', help='افزودن ویژگی وان‌هات برای ستون MHC در صورت وجود')
    ap.add_argument('--oversample', choices=['none', 'random'], default='none', help='مدل اورسمپلینگ برای کلاس مثبت')
    ap.add_argument('--add_anchor', action='store_true', help='افزودن ویژگی‌های anchor برای موقعیت‌های کلیدی دنباله')
    ap.add_argument('--anchor_strict', action='store_true', help='افزودن سیگنال‌های تخصصی anchor برای کلاس I')
    ap.add_argument('--add_blosum', action='store_true', help='Add BLOSUM62 embeddings')
    ap.add_argument('--add_protbert', action='store_true', help='Add ProtBert embeddings')
    ap.add_argument('--mhc_grouping', type=str, default='none', choices=['none', 'locus', 'two_digit'], help='MHC grouping strategy')
    ap.add_argument('--target_accuracy', type=float, default=None, help='اگر تنظیم شود، آستانه‌ای را پیدا می‌کند که به این دقت برسد')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = 'LogReg'
    dataset_base = os.path.splitext(os.path.basename(args.data_set))[0]

    df = load_dataset(args.data_set)
    if args.seq_col is None or args.label_col is None or args.seq_col not in df.columns or args.label_col not in df.columns:
        auto_label, auto_seq = detect_columns(df)
        args.label_col = args.label_col or auto_label
        args.seq_col = args.seq_col or auto_seq

    df_train, df_test = train_test_split_df(df, label_col=args.label_col, test_size=args.test_size, random_state=args.random_state)
    fcfg = FeatureConfig(
        add_physchem=not args.no_physchem,
        add_kmer_k=args.kmer_k if args.kmer_k >= 1 else 0,
        add_embedding_aaindex=not args.no_embed_aaindex,
        add_mhc=args.add_mhc,
        add_anchor=args.add_anchor,
        anchor_strict=args.anchor_strict,
        add_blosum=args.add_blosum,
        add_protbert=args.add_protbert,
        mhc_grouping=args.mhc_grouping,
    )
    X_train = build_features(df_train, args.seq_col, fcfg)
    y_train = df_train[args.label_col].astype(int).values
    X_test = build_features(df_test, args.seq_col, fcfg)
    y_test = df_test[args.label_col].astype(int).values

    # اورسمپلینگ اختیاری برای کلاس مثبت
    if args.oversample == 'random':
        try:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=args.random_state)
            X_train, y_train = ros.fit_resample(X_train, y_train)
        except Exception as e:
            print('Oversampling failed or imbalanced-learn not installed:', e)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=1)),
    ])

    pipe.fit(X_train, y_train)

    # ذخیرهٔ مدل با نام قابل تشخیص
    model_path = os.path.join(args.output_dir, f'model_{model_name}.joblib')
    dump(pipe, model_path)

    # پیش‌بینی و ذخیرهٔ خروجی تست
    probs = pipe.predict_proba(X_test)[:, 1]
    preds_path = os.path.join(args.output_dir, f'predictions_{model_name}_{dataset_base}.csv')
    pd.DataFrame({'prob': probs, 'label': y_test}).to_csv(preds_path, index=False)

    # محاسبهٔ متریک‌ها و ذخیره
    metrics = compute_metrics(y_test, probs, threshold=0.5)
    # جستجوی بهترین threshold بر اساس F1 و Accuracy
    best_thr_f1, best_m_f1 = find_best_threshold(y_test, probs, metric='f1')
    best_thr_acc, best_m_acc = find_best_threshold(y_test, probs, metric='accuracy')
    # اگر کاربر دقت هدف داده باشد، آستانه متناظر را پیدا کنیم
    if args.target_accuracy is not None:
        from common.metrics import find_threshold_for_accuracy
        thr_tgt, m_tgt = find_threshold_for_accuracy(y_test, probs, target_acc=args.target_accuracy)
        metrics.update({
            'target_accuracy': args.target_accuracy,
            'threshold_target_accuracy': thr_tgt,
            'metrics_at_target_accuracy': m_tgt,
        })
    metrics.update({
        'best_threshold_f1': best_thr_f1,
        'f1_best': best_m_f1.get('f1'),
        'accuracy_at_best_f1': best_m_f1.get('accuracy'),
        'best_threshold_accuracy': best_thr_acc,
        'accuracy_best': best_m_acc.get('accuracy'),
        'f1_at_best_accuracy': best_m_acc.get('f1'),
    })

    # افزودن اطلاعات مدل و دیتاست به متریک‌ها
    metrics.update({
        'model_name': model_name,
        'dataset': dataset_base,
        'label_column': args.label_col,
        'sequence_column': args.seq_col,
    })

    metrics_path = os.path.join(args.output_dir, f'metrics_{model_name}_{dataset_base}.json')
    save_metrics_json(metrics, metrics_path)

    # ذخیرهٔ متاداده برای تشخیص بهتر خروجی‌ها
    meta = {
        'model_name': model_name,
        'dataset': dataset_base,
        'label_column': args.label_col,
        'sequence_column': args.seq_col,
        'test_size': args.test_size,
        'random_state': args.random_state,
        'feature_set': (
            'aa20_counts+freq+length'
            + (f'+kmer{k}' if (k := (args.kmer_k if args.kmer_k >= 1 else 0)) else '')
            + ('' if args.no_physchem else '+physchem')
            + ('' if args.no_embed_aaindex else '+aaindex')
            + ('+blosum' if args.add_blosum else '')
            + ('+protbert' if args.add_protbert else '')
            + ('+mhc' if args.add_mhc else '')
            + (f'+mhc_group:{args.mhc_grouping}' if args.add_mhc and args.mhc_grouping != 'none' else '')
            + ('+anchor' if args.add_anchor else '')
            + ('+anchor_strict' if args.add_anchor and args.anchor_strict else '')
        ),
        'artifacts': {
            'model_path': model_path,
            'predictions_path': preds_path,
            'metrics_path': metrics_path,
        }
    }
    with open(os.path.join(args.output_dir, f'meta_{model_name}_{dataset_base}.json'), 'w', encoding='utf-8') as f:
        import json
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print('Saved:')
    print(' -', model_path)
    print(' -', preds_path)
    print(' -', metrics_path)
    print(' -', os.path.join(args.output_dir, f'meta_{model_name}_{dataset_base}.json'))


if __name__ == '__main__':
    main()



