import os
import argparse
import math
import json
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedKFold

# PyTorch components
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# Optional plotting support
try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    _MATPLOTLIB_AVAILABLE = False

# Reuse SWING encoders
# NOTE: Do not import scv.py because it parses CLI args on import.
# We re-implement get_window_encodings here to avoid side-effects.


# ---------------------------
# Data and Tokenization
# ---------------------------
DIGITS_VOCAB = {str(i): i for i in range(10)}  # 0..9
CLS_TOKEN_ID = 10
VOCAB_SIZE = 11  # 0..9 + CLS


def tokenize_digit_string(s: str):
    return [DIGITS_VOCAB.get(ch, 9) for ch in s]  # default to padding 9

# زمان‌بندی خوانا: h:mm:ss یا m:ss
def _fmt_time(seconds: float) -> str:
    total = int(seconds)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


class DigitSequenceDataset(Dataset):
    def __init__(self, sequences, labels, max_len):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.max_len = max_len

        tokenized = []
        for seq in sequences:
            toks = tokenize_digit_string(seq)
            # prepend CLS
            toks = [CLS_TOKEN_ID] + toks
            # pad/truncate
            if len(toks) < max_len:
                toks = toks + [9] * (max_len - len(toks))
            else:
                toks = toks[:max_len]
            tokenized.append(toks)
        self.inputs = torch.tensor(tokenized, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


# ---------------------------
# Model
# ---------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: [B, L]
        bsz, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        return self.pos_embed(positions)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, need_weights=False):
        attn_out, attn_weights = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=need_weights)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights


class InteractionTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEmbedding(max_len=max_len, d_model=d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.cls_head = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, x, key_padding_mask=None, collect_attention=False):
        # x: [B, L] token ids, with CLS at index 0
        tok = self.token_embed(x)
        pos = self.pos_embed(x)
        h = tok + pos
        attn_list = []
        for layer in self.layers:
            h, attn = layer(h, key_padding_mask=key_padding_mask, need_weights=collect_attention)
            if collect_attention:
                attn_list.append(attn)  # [B, heads, L, L]
        # Use CLS representation (position 0)
        cls_h = h[:, 0, :]
        logits = self.cls_head(self.dropout(cls_h)).squeeze(-1)
        if collect_attention:
            return logits, attn_list
        return logits


# ---------------------------
# Training / Evaluation
# ---------------------------
@dataclass
class TrainConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.2
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 8
    folds: int = 5
    padding_score: int = 9


def make_key_padding_mask(batch_inputs: torch.Tensor, pad_id: int = 9):
    # batch_inputs: [B, L] longs
    return batch_inputs.eq(pad_id)


def train_one_fold(model, train_loader, val_loader, epochs, lr, device, grad_accum_steps=1, log_interval=0, limit_train_batches: int = 0, pos_weight: torch.Tensor = None):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else None
    best_val = -1.0
    best_state = None
    for ep in range(epochs):
        ep_t0 = time.perf_counter()
        model.train()
        opt.zero_grad()
        accum = 0
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            mask = make_key_padding_mask(x)
            logits = model(x, key_padding_mask=mask)
            if criterion is not None:
                loss = criterion(logits, y)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            accum += 1
            running_loss += float(loss.item())
            if log_interval and ((batch_idx + 1) % log_interval == 0):
                step_elapsed = time.perf_counter() - ep_t0
                print(f"Epoch {ep+1}/{epochs} - step {batch_idx+1} - loss {loss.item():.4f} - زمان={_fmt_time(step_elapsed)}", flush=True)
            if accum % max(1, grad_accum_steps) == 0:
                opt.step()
                opt.zero_grad()
            if limit_train_batches and (batch_idx + 1) >= limit_train_batches:
                break
        # quick val
        model.eval()
        all_logits = []
        all_y = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                mask = make_key_padding_mask(x)
                logits = model(x, key_padding_mask=mask)
                all_logits.append(logits.detach().cpu())
                all_y.append(y.detach().cpu())
        probs = torch.sigmoid(torch.cat(all_logits)).numpy()
        labels = torch.cat(all_y).numpy()
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = 0.0
        avg_train_loss = running_loss / max(1, (batch_idx + 1))
        ep_elapsed = time.perf_counter() - ep_t0
        print(f"Epoch {ep+1}/{epochs} done - train_loss={avg_train_loss:.4f} - val_auc={auc:.4f} - زمان={_fmt_time(ep_elapsed)}", flush=True)
        if auc > best_val:
            best_val = auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def evaluate_fold(model, loader, device, threshold_override: float = None, use_best_for_primary: bool = False):
    eval_t0 = time.perf_counter()
    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            mask = make_key_padding_mask(x)
            logits = model(x, key_padding_mask=mask)
            all_logits.append(logits.detach().cpu())
            all_y.append(y.detach().cpu())
    probs = torch.sigmoid(torch.cat(all_logits)).numpy()
    labels = torch.cat(all_y).numpy()
    auc = roc_auc_score(labels, probs)
    eval_elapsed = time.perf_counter() - eval_t0
    print(f"[Eval] مدت‌زمان ارزیابی: {_fmt_time(eval_elapsed)}", flush=True)
    # Metrics at default threshold = 0.5 (safe for short runs)
    # Primary threshold (can be overridden or set to best)
    threshold_primary = 0.5
    if threshold_override is not None:
        threshold_primary = float(threshold_override)
    elif use_best_for_primary:
        # temporarily set; will be finalized after best_threshold computed
        threshold_primary = None
    preds_primary = (probs >= 0.5).astype(np.int32)
    f1 = f1_score(labels, preds_primary, zero_division=0)
    precision = precision_score(labels, preds_primary, zero_division=0)
    recall = recall_score(labels, preds_primary, zero_division=0)
    avg_prec = average_precision_score(labels, probs)
    # Optimize decision threshold via PR curve to maximize F1
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(labels, probs)
    if len(pr_thresholds) > 0:
        f1_vals = 2 * pr_precision[:-1] * pr_recall[:-1] / (pr_precision[:-1] + pr_recall[:-1] + 1e-8)
        idx_best = int(np.nanargmax(f1_vals))
        best_threshold = float(pr_thresholds[idx_best])
    else:
        best_threshold = 0.5
    # finalize primary threshold if set to use best
    if threshold_primary is None:
        threshold_primary = best_threshold
    # recompute primary metrics at chosen threshold
    preds_primary = (probs >= threshold_primary).astype(np.int32)
    f1 = f1_score(labels, preds_primary, zero_division=0)
    precision = precision_score(labels, preds_primary, zero_division=0)
    recall = recall_score(labels, preds_primary, zero_division=0)
    accuracy = float((preds_primary == labels).mean())
    preds_best = (probs >= best_threshold).astype(np.int32)
    f1_best = f1_score(labels, preds_best, zero_division=0)
    precision_best = precision_score(labels, preds_best, zero_division=0)
    recall_best = recall_score(labels, preds_best, zero_division=0)
    accuracy_best = float((preds_best == labels).mean())
    return {
        "auc": float(auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "avg_precision": float(avg_prec),
        "best_threshold": float(best_threshold),
        "f1_best": float(f1_best),
        "precision_best": float(precision_best),
        "recall_best": float(recall_best),
        "accuracy": float(accuracy),
        "accuracy_best": float(accuracy_best),
        "threshold_used": float(threshold_primary),
    }


def _save_curves_for_fold(model, loader, device, out_dir: str, fold_idx: int):
    """Save ROC and PR curves as PNGs for the given fold."""
    if not _MATPLOTLIB_AVAILABLE:
        return
    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            mask = make_key_padding_mask(x)
            logits = model(x, key_padding_mask=mask)
            all_logits.append(logits.detach().cpu())
            all_y.append(y.detach().cpu())
    probs = torch.sigmoid(torch.cat(all_logits)).numpy()
    labels = torch.cat(all_y).numpy()
    # ROC
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Curve (Fold {fold_idx+1})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'ROC_curve_fold_{fold_idx+1}.png'))
    plt.close()
    # PR
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label='PR')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve (Fold {fold_idx+1})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'PR_curve_fold_{fold_idx+1}.png'))
    plt.close()


def collect_attention(model, loader, device, max_batches=10):
    model.eval()
    attn_accum = None
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mask = make_key_padding_mask(x)
            logits, attn_list = model(x, key_padding_mask=mask, collect_attention=True)
            # take last layer attention; support both [B,H,L,L] and [B,L,L]
            last = attn_list[-1]
            if last.dim() == 4:
                # per-head weights: [B, H, L, L]
                cls_attn = last[:, :, 0, :]  # [B, H, L]
                mean_heads = cls_attn.mean(dim=1)  # [B, L]
            elif last.dim() == 3:
                # averaged over heads: [B, L, L]
                mean_heads = last[:, 0, :]  # [B, L]
            else:
                # unexpected shape, skip
                continue
            mean_heads = mean_heads.detach().cpu().numpy()
            attn_accum = mean_heads if attn_accum is None else np.vstack([attn_accum, mean_heads])
            count += 1
            if count >= max_batches:
                break
    if attn_accum is None:
        return None
    return attn_accum  # [N, L]


def main():
    run_t0 = time.perf_counter()
    parser = argparse.ArgumentParser("Transformer-based Class I pMHC with attention interpretability")
    parser.add_argument('--data_set', required=False, default=os.path.join('..', 'Data', 'ClassI_Model', 'ClassI_training_210.csv'))
    parser.add_argument('--output_dir', required=False, default=os.path.join('..', 'Results', 'Transformer_ClassI'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--padding_score', type=int, default=9)
    parser.add_argument('--metric', type=str, default='polarity', choices=['polarity', 'hydrophobicity'])
    parser.add_argument('--max_len', type=int, default=2048, help='Sequence max length including CLS; will truncate longer encodings')
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    parser.add_argument('--amp', action='store_true', help='Use mixed precision (CUDA only)')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Accumulate gradients to reduce memory per step')
    parser.add_argument('--light', action='store_true', help='Lighter model: d_model=64, n_heads=2, n_layers=1, d_ff=128')
    parser.add_argument('--train_frac', type=float, default=1.0, help='Fraction of data to use (e.g., 0.5)')
    parser.add_argument('--log_interval', type=int, default=0, help='Print training status every N steps (0=off)')
    # New flags: evaluation-only mode and checkpoint saving
    # Throughput tuning
    parser.add_argument('--num_workers', type=int, default=max(1, ((os.cpu_count() or 2) // 2)), help='DataLoader workers for parallel batch preparation')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='DataLoader prefetch factor (only effective when num_workers>0)')
    parser.add_argument('--limit_train_batches', type=int, default=0, help='Limit training batches per epoch for faster runs (0=disable)')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation only: load saved fold models from output_dir and compute metrics')
    parser.add_argument('--save_checkpoints', action='store_true', help='Save best model state_dict per fold into output_dir', default=True)
    parser.add_argument('--threshold', type=float, default=None, help='Global decision threshold for classification (overrides 0.5)')
    parser.add_argument('--use_best_threshold', action='store_true', help='Use per-fold F1-optimized threshold for primary metrics')
    args = parser.parse_args()

    # Resolve paths robustly: prefer CWD for user-relative paths; fallback to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()

    # Resolve dataset path
    if os.path.isabs(args.data_set):
        data_path_resolved = args.data_set
    else:
        candidates = [
            os.path.normpath(os.path.join(cwd, args.data_set)),
            os.path.normpath(os.path.join(script_dir, args.data_set)),
        ]
        data_path_resolved = None
        for cand in candidates:
            if os.path.exists(cand):
                data_path_resolved = cand
                break
        if data_path_resolved is None:
            # Last chance: if the provided is a known default relative to project root
            default_rel = os.path.join('SWING-main', 'SWING-main', 'Data', 'ClassI_Model', 'ClassI_training_210.csv')
            fallback = os.path.normpath(os.path.join(cwd, default_rel))
            if os.path.exists(fallback):
                data_path_resolved = fallback
            else:
                raise FileNotFoundError(
                    f"Could not resolve data_set: tried {candidates + [fallback]}"
                )
    args.data_set = data_path_resolved

    # Resolve output directory
    if os.path.isabs(args.output_dir):
        out_dir_resolved = args.output_dir
    else:
        # Prefer keeping outputs under CWD for clarity
        out_dir_resolved = os.path.normpath(os.path.join(cwd, args.output_dir))
    args.output_dir = out_dir_resolved
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Paths] data_set={args.data_set}")
    print(f"[Paths] output_dir={args.output_dir}")

    # Persist run config will be done AFTER applying any light overrides

    # Load data
    df = pd.read_csv(args.data_set)
    if args.train_frac < 1.0:
        df = df.sample(frac=max(0.05, args.train_frac), random_state=42).reset_index(drop=True)
    # Expect columns: Epitope, Sequence, Hit
    if not {'Epitope', 'Sequence', 'Hit'}.issubset(df.columns):
        raise ValueError(f"Expected columns 'Epitope', 'Sequence', 'Hit' in {args.data_set}, found: {list(df.columns)}")

    # Build AA score dictionary (polarity or hydrophobicity) and generate encodings
    if args.metric == 'polarity':
        AA_scores = {'A':8.1,'R':10.5,'N':11.6,'D':13.0,'C':5.5,'E':12.3,'Q':10.5,'G':9.0,'H':10.4,'I':5.2,
                     'L':4.9,'K':11.3,'M':5.7,'F':5.2,'P':8.0,'S':9.2,'T':8.6,'W':5.4,'Y':6.2,'V':5.9}
    else:
        AA_scores = {'A': 5.33, 'R': 4.18, 'D': 3.59, 'N': 3.59, 'C': 7.93, 'Q': 3.87, 'E': 3.65, 'G': 4.48, 'H': 5.1, 'I': 8.83,
                     'L': 8.47, 'K': 2.95, 'M': 8.95, 'F': 9.03, 'P': 3.87, 'S': 4.09, 'T': 4.49, 'W': 7.66, 'Y': 5.89, 'V': 7.63}

    AAs = list(AA_scores.keys())
    aa_score_dict = {}
    for i in range(len(AAs)):
        for j in range(len(AAs)-i):
            AA_pair = AAs[i]+AAs[j+i]
            AA_pair_score = round(abs(AA_scores[AAs[i]]-AA_scores[AAs[j+i]]))
            aa_score_dict[AA_pair] = AA_pair_score
            aa_score_dict[AA_pair[::-1]] = AA_pair_score

    def get_window_encodings(df_local: pd.DataFrame, padding_score: int = 9):
        total_encodings = []
        n_rows = len(df_local.index)
        enc_t0 = time.perf_counter()
        print(f"[Enc] شروع تولید انکودینگ‌ها برای {n_rows} ردیف...", flush=True)
        for i in (df_local.index):
            mut_window = df_local['Epitope'].iloc[i]
            interactor = df_local['Sequence'].iloc[i]
            PPI_encoding = ''
            its = 0
            for _ in range(len(interactor)):
                window_scores = ''
                for k in range(len(mut_window)):
                    try:
                        pair = mut_window[k]+interactor[k+its]
                        score = aa_score_dict[pair]
                    except:
                        score = padding_score
                    window_scores = window_scores + str(score)
                its += 1
                PPI_encoding = PPI_encoding + str(window_scores)
            total_encodings.append(PPI_encoding)
            if (len(total_encodings) % 10000) == 0:
                print(f"[Enc] پیشرفت: {len(total_encodings)}/{n_rows}", flush=True)
        enc_elapsed = time.perf_counter() - enc_t0
        print(f"[Enc] تولید انکودینگ‌ها تمام شد (زمان: {_fmt_time(enc_elapsed)}).", flush=True)
        return total_encodings

    encodings = get_window_encodings(df, padding_score=args.padding_score)
    print(f"[Prep] تعداد انکودینگ‌ها: {len(encodings)}", flush=True)
    labels = df['Hit'].astype(int).tolist()

    # Determine max length budget (include CLS)
    max_len_observed = 1 + max(len(s) for s in encodings)
    max_len = min(args.max_len, max_len_observed)
    print(f"[Prep] بیشینه طول با CLS: {max_len_observed} → بودجه طول: {max_len}", flush=True)

    X_all = np.array(encodings)
    y_all = np.array(labels)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Device log and performance tweaks
    print(f"[Device] Using {device}", flush=True)
    if device.type == 'cuda':
        try:
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    # Apply light config if requested
    if args.light:
        args.d_model = 64
        args.n_heads = 2
        args.n_layers = 1
        args.d_ff = 128

    # Now persist the EFFECTIVE run config (after any light overrides)
    run_cfg_path = os.path.join(args.output_dir, 'run_config.json')
    run_cfg = {
        'folds': args.folds,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'padding_score': args.padding_score,
        'metric': args.metric,
        'max_len': args.max_len,
        'light': args.light,
    }
    try:
        with open(run_cfg_path, 'w') as f:
            json.dump(run_cfg, f, indent=2)
    except Exception:
        pass

    # In eval-only mode, align architecture with the saved training config
    # to avoid checkpoint shape mismatches.
    if args.eval_only:
        cfg_path = os.path.join(args.output_dir, 'run_config.json')
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r') as f:
                    cfg = json.load(f)
                # Override model hyperparameters from saved config
                args.d_model = int(cfg.get('d_model', args.d_model))
                args.n_heads = int(cfg.get('n_heads', args.n_heads))
                args.n_layers = int(cfg.get('n_layers', args.n_layers))
                args.d_ff = int(cfg.get('d_ff', args.d_ff))
                args.dropout = float(cfg.get('dropout', args.dropout))
                # Keep max_len consistent with training if present
                args.max_len = int(cfg.get('max_len', args.max_len))
            except Exception as e:
                print(f"Warning: failed to load run_config.json for eval_only: {e}", flush=True)

    fold_metrics = []
    attn_summaries = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        fold_t0 = time.perf_counter()
        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]

        train_ds = DigitSequenceDataset(X_train, y_train, max_len=max_len)
        test_ds = DigitSequenceDataset(X_test, y_test, max_len=max_len)

        # High-throughput DataLoader settings
        pin_mem = (device.type == 'cuda')
        nw = max(0, int(args.num_workers))
        pf = int(args.prefetch_factor)
        dl_common = dict(num_workers=nw, pin_memory=pin_mem, persistent_workers=(nw > 0))
        if nw > 0:
            dl_common['prefetch_factor'] = max(2, pf)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_common)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **dl_common)

        print(f"شروع فولد {fold_idx+1}/{args.folds} - {'ارزیابی' if args.eval_only else 'آموزش'}: {len(train_ds)}، تست: {len(test_ds)}", flush=True)

        if args.eval_only:
            print(f"[Fold {fold_idx+1}] بارگذاری چک‌پوینت و آماده‌سازی مدل برای ارزیابی...", flush=True)
            # Load saved checkpoint for this fold and infer architecture to avoid mismatches
            ckpt_path = os.path.join(args.output_dir, f'model_fold_{fold_idx+1}.pt')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found for fold {fold_idx+1}: {ckpt_path}. Train once with --save_checkpoints to create it.")
            state = torch.load(ckpt_path, map_location=device)
            # Infer d_model from classifier head or first layer norm
            inferred_d_model = args.d_model
            try:
                inferred_d_model = int(state['cls_head.weight'].shape[1])
            except Exception:
                if 'layers.0.norm1.weight' in state:
                    inferred_d_model = int(state['layers.0.norm1.weight'].shape[0])
            # Infer d_ff from first FF layer if available
            inferred_d_ff = args.d_ff
            if 'layers.0.ff.0.weight' in state:
                inferred_d_ff = int(state['layers.0.ff.0.weight'].shape[0])
            # Infer n_layers by counting encoder layers in the state dict
            inferred_n_layers = args.n_layers
            try:
                inferred_n_layers = 1 + max(int(k.split('.')[1]) for k in state.keys() if k.startswith('layers.'))
            except Exception:
                pass
            # Override args with inferred values
            args.d_model = inferred_d_model
            args.d_ff = inferred_d_ff
            args.n_layers = inferred_n_layers

            # Recreate model with inferred architecture and load weights
            model = InteractionTransformer(
                vocab_size=VOCAB_SIZE,
                max_len=max_len,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                d_ff=args.d_ff,
                dropout=args.dropout,
            )
            model.load_state_dict(state)
            model.to(device)
            best_val = float('nan')
        else:
            model = InteractionTransformer(
                vocab_size=VOCAB_SIZE,
                max_len=max_len,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                d_ff=args.d_ff,
                dropout=args.dropout,
            )
            # Compute class imbalance for pos_weight
            pos_weight_tensor = None
            try:
                n_pos = int(y_train.sum())
                n_total = int(len(y_train))
                n_neg = max(0, n_total - n_pos)
                if n_pos > 0:
                    pw = float(n_neg) / float(n_pos)
                    # clamp to reasonable range
                    pw = float(np.clip(pw, 1.0, 50.0))
                    pos_weight_tensor = torch.tensor(pw, device=device)
            except Exception:
                pos_weight_tensor = None

            # Training path (with AMP support optional)
            if args.amp and device.type == 'cuda':
                scaler = GradScaler('cuda')
                model.to(device)
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) if pos_weight_tensor is not None else None
                best_val = -1.0
                best_state = None
                for ep in range(args.epochs):
                    ep_t0 = time.perf_counter()
                    model.train()
                    running_loss = 0.0
                    for batch_idx, (x, y) in enumerate(train_loader):
                        x = x.to(device)
                        y = y.to(device)
                        mask = make_key_padding_mask(x)
                        opt.zero_grad()
                        with autocast('cuda'):
                            logits = model(x, key_padding_mask=mask)
                            if criterion is not None:
                                loss = criterion(logits, y)
                            else:
                                loss = F.binary_cross_entropy_with_logits(logits, y)
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                        running_loss += float(loss.item())
                        if args.log_interval and ((batch_idx + 1) % args.log_interval == 0):
                            step_elapsed = time.perf_counter() - ep_t0
                            print(f"[AMP] Epoch {ep+1}/{args.epochs} - step {batch_idx+1} - loss {loss.item():.4f} - زمان={_fmt_time(step_elapsed)}", flush=True)
                        if args.limit_train_batches and (batch_idx + 1) >= args.limit_train_batches:
                            break
                    # validate after epoch (outside batch loop)
                    model.eval()
                    all_logits = []
                    all_y = []
                    with torch.no_grad():
                        for x, y in test_loader:
                            x = x.to(device)
                            y = y.to(device)
                            mask = make_key_padding_mask(x)
                            logits = model(x, key_padding_mask=mask)
                            all_logits.append(logits.detach().cpu())
                            all_y.append(y.detach().cpu())
                    probs = torch.sigmoid(torch.cat(all_logits)).numpy()
                    labels = torch.cat(all_y).numpy()
                    try:
                        auc = roc_auc_score(labels, probs)
                    except Exception:
                        auc = 0.0
                    avg_train_loss = running_loss / max(1, (batch_idx + 1))
                    ep_elapsed = time.perf_counter() - ep_t0
                    print(f"[AMP] Epoch {ep+1}/{args.epochs} done - train_loss={avg_train_loss:.4f} - val_auc={auc:.4f} - زمان={_fmt_time(ep_elapsed)}", flush=True)
                    if auc > best_val:
                        best_val = auc
                        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            # restore best after AMP training
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        model, best_val = train_one_fold(
            model, train_loader, test_loader,
            epochs=args.epochs, lr=args.lr, device=device,
            grad_accum_steps=args.grad_accum_steps,
            log_interval=args.log_interval,
            limit_train_batches=args.limit_train_batches,
            pos_weight=pos_weight_tensor
        )
        # Save checkpoint per fold (best state) if requested
        if args.save_checkpoints:
            ckpt_path = os.path.join(args.output_dir, f'model_fold_{fold_idx+1}.pt')
            torch.save(model.state_dict(), ckpt_path)
        print(f"[Fold {fold_idx+1}] در حال محاسبه متریک‌های تست...", flush=True)
        metrics_fold = evaluate_fold(
            model, test_loader, device=device,
            threshold_override=args.threshold,
            use_best_for_primary=args.use_best_threshold
        )
        # In eval-only, we set val_auc to the test auc for reporting consistency
        metrics_fold['val_auc'] = float(metrics_fold['auc']) if args.eval_only else float(best_val)
        fold_metrics.append(metrics_fold)
        try:
            print(f"[Fold {fold_idx+1}] نتایج: auc={metrics_fold['auc']:.4f}, f1={metrics_fold['f1']:.4f}, precision={metrics_fold['precision']:.4f}, recall={metrics_fold['recall']:.4f}, accuracy={metrics_fold['accuracy']:.4f} (threshold_used={metrics_fold['threshold_used']:.3f})", flush=True)
            print(f"[Fold {fold_idx+1}] آستانه‌بهینه={metrics_fold['best_threshold']:.3f}، f1_best={metrics_fold['f1_best']:.4f}, accuracy_best={metrics_fold['accuracy_best']:.4f}", flush=True)
        except Exception:
            pass

        # Save ROC/PR curves for this fold
        try:
            _save_curves_for_fold(model, test_loader, device=device, out_dir=args.output_dir, fold_idx=fold_idx)
            print(f"[Fold {fold_idx+1}] نمودارهای ROC/PR ذخیره شد.", flush=True)
        except Exception as e:
            print(f"Failed to save ROC/PR curves for fold {fold_idx+1}: {e}", flush=True)

        # Collect attention on a subset of test batches
        print(f"[Fold {fold_idx+1}] جمع‌آوری خلاصه توجه (تا 10 بچ) ...", flush=True)
        attn = collect_attention(model, test_loader, device=device, max_batches=10)
        if attn is not None:
            # Average across samples to get a single vector
            mean_attn = attn.mean(axis=0).tolist()
        else:
            mean_attn = None
        attn_summaries.append({
            'fold': fold_idx,
            'mean_cls_attention': mean_attn,
        })
        fold_elapsed = time.perf_counter() - fold_t0
        print(f"[Fold {fold_idx+1}] اتمام فولد (زمان: {_fmt_time(fold_elapsed)}).", flush=True)

    # Aggregate metrics
    agg = {}
    for key in ['auc', 'f1', 'precision', 'recall', 'accuracy', 'avg_precision', 'val_auc', 'f1_best', 'precision_best', 'recall_best', 'accuracy_best', 'best_threshold']:
        vals = [m[key] for m in fold_metrics]
        agg[key] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'n': len(vals)
        }

    # Print aggregate stats to terminal for quick visibility
    try:
        print('[Summary] آمار تجمیعی فولدها (mean ± std):', flush=True)
        for k in ['auc', 'f1', 'precision', 'recall', 'accuracy', 'avg_precision']:
            if k in agg:
                print(f" - {k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f} (n={agg[k]['n']})", flush=True)
        print('[Summary] آستانه‌بهینه و امتیازهای متناظر:', flush=True)
        for k in ['best_threshold', 'f1_best', 'accuracy_best']:
            if k in agg:
                print(f" - {k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f} (n={agg[k]['n']})", flush=True)
    except Exception:
        pass

    # Save raw and summary
    metrics_path = os.path.join(args.output_dir, 'fold_metrics.json')
    attn_path = os.path.join(args.output_dir, 'attention_summaries.json')
    with open(metrics_path, 'w') as f:
        json.dump({'fold_metrics': fold_metrics, 'aggregate': agg}, f, indent=2)
    with open(attn_path, 'w') as f:
        json.dump({'attention': attn_summaries, 'max_len': max_len}, f, indent=2)
    print("Saved:", flush=True)
    print(f" - {metrics_path}", flush=True)
    print(f" - {attn_path} ", flush=True)

    # Markdown report
    md_lines = []
    md_lines.append('# Transformer Class I SCV Summary')
    md_lines.append('')
    md_lines.append(f"Data: `{args.data_set}` | Folds: `{args.folds}` | Epochs: `{args.epochs}`")
    md_lines.append(f"Max len (incl CLS): `{max_len}` | Device: `{device}`")
    md_lines.append('')
    md_lines.append('## Metrics (mean ± std)')
    for k, v in agg.items():
        md_lines.append(f"- {k}: {v['mean']:.4f} ± {v['std']:.4f} (n={v['n']})")
    md_lines.append('')
    md_lines.append('## Threshold Optimization')
    md_lines.append(f"- best_threshold (mean±std): {agg['best_threshold']['mean']:.4f} ± {agg['best_threshold']['std']:.4f}")
    md_lines.append(f"- f1_best (mean±std): {agg['f1_best']['mean']:.4f} ± {agg['f1_best']['std']:.4f}")
    md_lines.append(f"- accuracy_best (mean±std): {agg['accuracy_best']['mean']:.4f} ± {agg['accuracy_best']['std']:.4f}")
    md_lines.append('')
    md_lines.append('## Attention (CLS → tokens)')
    md_lines.append('- Saved mean per-fold CLS attention over sequence tokens to `attention_summaries.json`.')
    md_lines.append('- Higher values indicate tokens the model focuses on for classification.')
    md_lines.append('')
    md_lines.append('## Curves')
    md_lines.append('- Per-fold ROC and PR curves are saved as PNGs in the output directory.')
    md_lines.append('')
    md_lines.append('## Notes')
    md_lines.append('- Tokens are digits 0–9 (score differences) with 9 as padding; a CLS token is prepended.')
    md_lines.append('- Interpretability derives from last-layer multi-head self-attention averaged over heads.')
    md_lines.append('- Classification uses the CLS representation via a linear head.')

    with open(os.path.join(args.output_dir, 'Transformer_ClassI_summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    print('Saved:')
    print(' -', os.path.join(args.output_dir, 'fold_metrics.json'))
    print(' -', os.path.join(args.output_dir, 'attention_summaries.json'))
    print(' -', os.path.join(args.output_dir, 'Transformer_ClassI_summary.md'))
    run_elapsed = time.perf_counter() - run_t0
    print(f"[Run] مدت‌زمان کل اجرا: {_fmt_time(run_elapsed)}", flush=True)


if __name__ == '__main__':
    main()