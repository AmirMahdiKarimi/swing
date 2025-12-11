from typing import Tuple, Optional

import pandas as pd
import numpy as np


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    # تشخیص ستون برچسب دودویی
    label_col: Optional[str] = None
    if 'Label' in df.columns:
        vals = pd.to_numeric(df['Label'], errors='coerce')
        uniq = set(pd.Series(vals).dropna().astype(int).unique().tolist())
        if uniq.issubset({0, 1}) and len(uniq) > 0:
            label_col = 'Label'

    if label_col is None:
        for c in df.columns:
            vals = pd.to_numeric(df[c], errors='coerce')
            uniq = set(pd.Series(vals).dropna().astype(int).unique().tolist())
            if uniq.issubset({0, 1}) and len(uniq) > 0:
                label_col = c
                break

    if label_col is None:
        raise ValueError('ستون برچسب دودویی (0/1) یافت نشد.')

    # تشخیص ستون دنباله (ترجیح: Sequence سپس Epitope سپس Long_Sequence)
    seq_candidates = ['Sequence', 'Epitope', 'Long_Sequence']
    seq_col: Optional[str] = None
    for c in seq_candidates:
        if c in df.columns:
            if df[c].dtype == object:
                seq_col = c
                break

    if seq_col is None:
        # انتخاب اولین ستون متنی
        for c in df.columns:
            if df[c].dtype == object:
                seq_col = c
                break

    if seq_col is None:
        raise ValueError('ستون دنبالهٔ متنی یافت نشد.')

    return label_col, seq_col


def train_test_split_df(df: pd.DataFrame, label_col: str, test_size: float = 0.5, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Stratified split by label
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    y = df[label_col].values
    idx_train, idx_test = next(sss.split(df, y))
    return df.iloc[idx_train].copy(), df.iloc[idx_test].copy()