import os
import argparse
import sys

import numpy as np
import pandas as pd
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
    ap.add_argument('--output_dir', default=os.path.join('Results', 'CatBoost'), help='پوشهٔ خروجی')
    ap.add_argument('--test_size', type=float, default=0.5)
    ap.add_argument('--random_state', type=int, default=42)
    ap.add_argument('--kmer_k', type=int, default=3, help='k برای k-mer (0 یعنی غیرفعال)')
    ap.add_argument('--no_physchem', action='store_true', help='غیرفعال کردن ویژگی‌های فیزیکوشیمیایی')
    ap.add_argument('--no_embed_aaindex', action='store_true', help='غیرفعال کردن embedding ساده AAindex')
    ap.add_argument('--add_anchor', action='store_true', help='افزودن ویژگی‌های anchor برای موقعیت‌های کلیدی دنباله')
    ap.add_argument('--anchor_strict', action='store_true', help='افزودن سیگنال‌های تخصصی anchor برای کلاس I')
    ap.add_argument('--add_blosum', action='store_true', help='افزودن امبدینگ BLOSUM62 (میانگین بردار سطرها)')
    ap.add_argument('--add_protbert', action='store_true', help='Use ProtBert embeddings')
    ap.add_argument('--add_mhc', action='store_true', help='وان‌هات MHC به‌صورت ویژگی عددی در صورت نیاز')
    ap.add_argument('--use_mhc_raw', action='store_true', help='افزودن ستون MHC به‌صورت دسته‌ای خام برای CatBoost در صورت وجود')
    ap.add_argument('--mhc_grouping', choices=['none', 'locus', 'two_digit'], default='none', help='کاهش کاردینالیتی MHC با گروه‌بندی برای وان‌هات یا خام')
    ap.add_argument('--target_accuracy', type=float, default=None, help='اگر تنظیم شود، آستانه‌ای را پیدا می‌کند که به این دقت برسد')
    ap.add_argument('--oversample', choices=['none', 'random'], default='none', help='اورسمپلینگ کلاس مثبت')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = 'CatBoost'
    dataset_base = os.path.splitext(os.path.basename(args.data_set))[0]

    df = load_dataset(args.data_set)
    if args.seq_col is None or args.label_col is None or args.seq_col not in df.columns or args.label_col not in df.columns:
        auto_label, auto_seq = detect_columns(df)
        args.label_col = args.label_col or auto_label
        args.seq_col = args.seq_col or auto_seq

    df_train, df_test = train_test_split_df(df, label_col=args.label_col, test_size=args.test_size, random_state=args.random_state)

    # ویژگی‌ها را بدون وان‌هات MHC می‌سازیم تا نسخهٔ خام را به‌صورت دسته‌ای اضافه کنیم
    fcfg = FeatureConfig(
        add_physchem=not args.no_physchem,
        add_kmer_k=args.kmer_k if args.kmer_k >= 1 else 0,
        add_embedding_aaindex=not args.no_embed_aaindex,
        add_mhc=(args.add_mhc and not args.use_mhc_raw),
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

    # به DataFrame تبدیل کنید تا بتوانیم ویژگی دسته‌ای خام را اضافه کنیم
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    cat_features_idx = []
    if args.use_mhc_raw and 'MHC' in df_train.columns:
        # پاکسازی کامل با تبدیل مستقیم هر مقدار به رشته و جایگزینی NaN
        # گروه‌بندی اختیاری برای کاهش کاردینالیتی
        from common.features import _mhc_group_value
        mhc_train = (
            df_train['MHC']
            .fillna('UNK')
            .apply(lambda v: _mhc_group_value(v, args.mhc_grouping))
            .fillna('UNK')
            .astype(str)
            .replace({'nan': 'UNK', 'NaN': 'UNK', 'None': 'UNK'})
        )
        mhc_test = (
            df_test['MHC']
            .fillna('UNK')
            .apply(lambda v: _mhc_group_value(v, args.mhc_grouping))
            .fillna('UNK')
            .astype(str)
            .replace({'nan': 'UNK', 'NaN': 'UNK', 'None': 'UNK'})
        )
        X_train_df['MHC'] = mhc_train
        X_test_df['MHC'] = mhc_test
        # اطمینان از نبود رشته‌های نامعتبر
        X_train_df['MHC'] = X_train_df['MHC'].fillna('UNK').replace({'nan': 'UNK', 'NaN': 'UNK', 'None': 'UNK'})
        X_test_df['MHC'] = X_test_df['MHC'].fillna('UNK').replace({'nan': 'UNK', 'NaN': 'UNK', 'None': 'UNK'})
        cat_features_idx = [X_train_df.columns.get_loc('MHC')]

    # اورسمپلینگ اختیاری
    if args.oversample == 'random':
        try:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=args.random_state)
            X_train_df, y_train = ros.fit_resample(X_train_df, y_train)
        except Exception as e:
            print('Oversampling failed or imbalanced-learn not installed:', e)

    # ساخت و آموزش CatBoost
    try:
        from catboost import CatBoostClassifier
    except Exception as e:
        print('CatBoost is not installed. Please install with: pip install catboost')
        raise

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=args.random_state,
        auto_class_weights='Balanced',
        od_type='Iter',
        od_wait=50,
        verbose=False,
    )

    model.fit(X_train_df, y_train, eval_set=(X_test_df, y_test), cat_features=cat_features_idx, use_best_model=True, verbose=False)

    # ذخیرهٔ مدل
    model_path = os.path.join(args.output_dir, f'model_{model_name}.joblib')
    dump(model, model_path)

    # پیش‌بینی و ذخیره
    probs = model.predict_proba(X_test_df)[:, 1]
    preds_path = os.path.join(args.output_dir, f'predictions_{model_name}_{dataset_base}.csv')
    pd.DataFrame({'prob': probs, 'label': y_test}).to_csv(preds_path, index=False)

    # محاسبهٔ متریک‌ها
    metrics = compute_metrics(y_test, probs, threshold=0.5)
    best_thr_f1, best_m_f1 = find_best_threshold(y_test, probs, metric='f1')
    best_thr_acc, best_m_acc = find_best_threshold(y_test, probs, metric='accuracy')
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

    metrics.update({
        'model_name': model_name,
        'dataset': dataset_base,
        'label_column': args.label_col,
        'sequence_column': args.seq_col,
    })
    metrics_path = os.path.join(args.output_dir, f'metrics_{model_name}_{dataset_base}.json')
    save_metrics_json(metrics, metrics_path)

    # متاداده برای تشخیص بهتر
    feature_set = (
        'aa20_counts+freq+length'
        + (f'+kmer{k}' if (k := (args.kmer_k if args.kmer_k >= 1 else 0)) else '')
        + ('' if args.no_physchem else '+physchem')
        + ('' if args.no_embed_aaindex else '+aaindex')
        + ('+blosum' if args.add_blosum else '')
        + ('+protbert' if args.add_protbert else '')
        + ('+anchor' if args.add_anchor else '')
        + ('+anchor_strict' if args.add_anchor and args.anchor_strict else '')
        + ('+mhc_raw' if (args.use_mhc_raw and 'MHC' in df.columns) else '')
        + (f'+mhc_group:{args.mhc_grouping}' if (args.use_mhc_raw or args.add_mhc) and args.mhc_grouping != 'none' else '')
    )
    meta = {
        'model_name': model_name,
        'dataset': dataset_base,
        'label_column': args.label_col,
        'sequence_column': args.seq_col,
        'test_size': args.test_size,
        'random_state': args.random_state,
        'feature_set': feature_set,
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

