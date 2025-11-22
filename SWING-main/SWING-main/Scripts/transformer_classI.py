import os
import argparse
import math
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

# PyTorch components
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

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


def train_one_fold(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = make_key_padding_mask(x)
            logits = model(x, key_padding_mask=mask)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
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
        if auc > best_val:
            best_val = auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def evaluate_fold(model, loader, device):
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
    preds = (probs >= 0.5).astype(np.int32)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    avg_prec = average_precision_score(labels, probs)
    fpr, tpr, thresh = precision_recall_curve(labels, probs)
    return {
        "auc": float(auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "avg_precision": float(avg_prec),
    }


def collect_attention(model, loader, device, max_batches=10):
    model.eval()
    attn_accum = None
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mask = make_key_padding_mask(x)
            logits, attn_list = model(x, key_padding_mask=mask, collect_attention=True)
            # take last layer attention, average heads; focus on CLS row
            last = attn_list[-1]  # [B, H, L, L]
            cls_attn = last[:, :, 0, :]  # [B, H, L]
            mean_heads = cls_attn.mean(dim=1)  # [B, L]
            mean_heads = mean_heads.detach().cpu().numpy()
            attn_accum = mean_heads if attn_accum is None else np.vstack([attn_accum, mean_heads])
            count += 1
            if count >= max_batches:
                break
    if attn_accum is None:
        return None
    return attn_accum  # [N, L]


def main():
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data_set)
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
        return total_encodings

    encodings = get_window_encodings(df, padding_score=args.padding_score)
    labels = df['Hit'].astype(int).tolist()

    # Determine max length budget (include CLS)
    max_len_observed = 1 + max(len(s) for s in encodings)
    max_len = min(args.max_len, max_len_observed)

    X_all = np.array(encodings)
    y_all = np.array(labels)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fold_metrics = []
    attn_summaries = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]

        train_ds = DigitSequenceDataset(X_train, y_train, max_len=max_len)
        test_ds = DigitSequenceDataset(X_test, y_test, max_len=max_len)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = InteractionTransformer(
            vocab_size=VOCAB_SIZE,
            max_len=max_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
        )

        # Optional AMP support within training
        if args.amp and device.type == 'cuda':
            scaler = GradScaler()
            # Lightweight wrapper: manual training here using AMP
            model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            best_val = -1.0
            best_state = None
            for ep in range(args.epochs):
                model.train()
                for x, y in train_loader:
                    x = x.to(device)
                    y = y.to(device)
                    mask = make_key_padding_mask(x)
                    opt.zero_grad()
                    with autocast():
                        logits = model(x, key_padding_mask=mask)
                        loss = F.binary_cross_entropy_with_logits(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                # validate
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
                if auc > best_val:
                    best_val = auc
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if best_state is not None:
                model.load_state_dict(best_state)
        else:
            model, best_val = train_one_fold(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)
        metrics_fold = evaluate_fold(model, test_loader, device=device)
        metrics_fold['val_auc'] = float(best_val)
        fold_metrics.append(metrics_fold)

        # Collect attention on a subset of test batches
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

    # Aggregate metrics
    agg = {}
    for key in ['auc', 'f1', 'precision', 'recall', 'avg_precision', 'val_auc']:
        vals = [m[key] for m in fold_metrics]
        agg[key] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'n': len(vals)
        }

    # Save raw and summary
    with open(os.path.join(args.output_dir, 'fold_metrics.json'), 'w') as f:
        json.dump({'fold_metrics': fold_metrics, 'aggregate': agg}, f, indent=2)
    with open(os.path.join(args.output_dir, 'attention_summaries.json'), 'w') as f:
        json.dump({'attention': attn_summaries, 'max_len': max_len}, f, indent=2)

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
    md_lines.append('## Attention (CLS → tokens)')
    md_lines.append('- Saved mean per-fold CLS attention over sequence tokens to `attention_summaries.json`.')
    md_lines.append('- Higher values indicate tokens the model focuses on for classification.')
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


if __name__ == '__main__':
    main()