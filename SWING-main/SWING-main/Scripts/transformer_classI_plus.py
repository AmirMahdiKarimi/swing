import os
import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# ---------------------------
# Data and Tokenization
# ---------------------------
DIGITS_VOCAB = {str(i): i for i in range(10)}
CLS_TOKEN_ID = 10
VOCAB_SIZE = 11


def tokenize_digit_string(s: str):
    return [DIGITS_VOCAB.get(ch, 9) for ch in s]


class DigitSequenceDataset(Dataset):
    def __init__(self, sequences, labels, max_len):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.max_len = max_len
        tokenized = []
        for seq in sequences:
            toks = tokenize_digit_string(seq)
            toks = [CLS_TOKEN_ID] + toks
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
        bsz, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        return self.pos_embed(positions)


class RelativePositionBias(nn.Module):
    def __init__(self, max_distance: int = 128):
        super().__init__()
        self.max_distance = max_distance
        self.num_buckets = 2 * max_distance + 1
        self.emb = nn.Embedding(self.num_buckets, 1)

    def forward(self, seq_len: int, device: torch.device):
        # distance matrix: j - i in [-max_distance, max_distance]
        idx = torch.arange(seq_len, device=device)
        rel = idx.unsqueeze(0) - idx.unsqueeze(1)  # [L, L]
        rel = rel.clamp(-self.max_distance, self.max_distance) + self.max_distance
        bias = self.emb(rel)  # [L, L, 1]
        return bias.squeeze(-1)  # [L, L], shared across heads


class TransformerBlockPlus(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, rel_pos_bias: RelativePositionBias = None):
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
        self.rel_pos_bias = rel_pos_bias

    def forward(self, x, key_padding_mask=None, need_weights=False):
        # PreNorm transformer with optional relative position bias
        x_norm = self.norm1(x)
        bias_mask = None
        if self.rel_pos_bias is not None:
            seq_len = x_norm.size(1)
            bias_mask = self.rel_pos_bias(seq_len, device=x_norm.device)
        attn_out, attn_weights = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=bias_mask,
            average_attn_weights=False,
        )
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x, attn_weights


class InteractionTransformerPlus(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float, relative_pos: bool = False, rp_max_dist: int = 128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEmbedding(max_len=max_len, d_model=d_model)
        rel_bias = RelativePositionBias(max_distance=rp_max_dist) if relative_pos else None
        self.layers = nn.ModuleList([
            TransformerBlockPlus(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, rel_pos_bias=rel_bias)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, x, key_padding_mask=None, collect_attention=False):
        tok = self.token_embed(x)
        pos = self.pos_embed(x)
        h = tok + pos
        attn_list = []
        for layer in self.layers:
            h, attn = layer(h, key_padding_mask=key_padding_mask, need_weights=collect_attention)
            if collect_attention:
                attn_list.append(attn)  # [B, heads, L, L]
        cls_h = h[:, 0, :]
        logits = self.head(self.dropout(cls_h)).squeeze(-1)
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
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.2
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 8
    folds: int = 5
    padding_score: int = 9


def make_key_padding_mask(batch_inputs: torch.Tensor, pad_id: int = 9):
    return batch_inputs.eq(pad_id)


def train_one_fold(model, train_loader, val_loader, epochs, lr, device, grad_accum_steps=1, log_interval=0):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        accum = 0
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            mask = make_key_padding_mask(x)
            logits = model(x, key_padding_mask=mask)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            accum += 1
            running_loss += float(loss.item())
            if log_interval and ((batch_idx + 1) % log_interval == 0):
                print(f"Epoch {ep+1}/{epochs} - step {batch_idx+1} - loss {loss.item():.4f}", flush=True)
            if accum % max(1, grad_accum_steps) == 0:
                opt.step()
                opt.zero_grad()
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
        print(f"Epoch {ep+1}/{epochs} done - train_loss={avg_train_loss:.4f} - val_auc={auc:.4f}", flush=True)
        if auc > best_val:
            best_val = auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
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
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    avg_prec = average_precision_score(labels, probs)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(labels, probs)
    if len(pr_thresholds) > 0:
        f1_vals = 2 * pr_precision[:-1] * pr_recall[:-1] / (pr_precision[:-1] + pr_recall[:-1] + 1e-8)
        idx_best = int(np.nanargmax(f1_vals))
        best_threshold = float(pr_thresholds[idx_best])
    else:
        best_threshold = 0.5
    preds_best = (probs >= best_threshold).astype(np.int32)
    f1_best = f1_score(labels, preds_best, zero_division=0)
    precision_best = precision_score(labels, preds_best, zero_division=0)
    recall_best = recall_score(labels, preds_best, zero_division=0)
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
            last = attn_list[-1]
            if last.dim() == 4:
                cls_attn = last[:, :, 0, :]
                mean_heads = cls_attn.mean(dim=1)
            elif last.dim() == 3:
                mean_heads = last[:, 0, :]
            else:
                continue
            mean_heads = mean_heads.detach().cpu().numpy()
            attn_accum = mean_heads if attn_accum is None else np.vstack([attn_accum, mean_heads])
            count += 1
            if count >= max_batches:
                break
    if attn_accum is None:
        return None
    return attn_accum


def main():
    parser = argparse.ArgumentParser("Transformer+ Class I with Relative Positional Bias and PreNorm")
    parser.add_argument('--data_set', required=False, default=os.path.join('..', 'Data', 'ClassI_Model', 'ClassI_training_210.csv'))
    parser.add_argument('--output_dir', required=False, default=os.path.join('..', 'Results', 'Transformer_ClassI_Plus'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--padding_score', type=int, default=9)
    parser.add_argument('--metric', type=str, default='polarity', choices=['polarity', 'hydrophobicity'])
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--light', action='store_true')
    parser.add_argument('--train_frac', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=0)
    parser.add_argument('--relative_pos', action='store_true')
    parser.add_argument('--rp_max_dist', type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_set)
    if args.train_frac < 1.0:
        df = df.sample(frac=max(0.05, args.train_frac), random_state=42).reset_index(drop=True)
    if not {'Epitope', 'Sequence', 'Hit'}.issubset(df.columns):
        raise ValueError(f"Expected columns 'Epitope', 'Sequence', 'Hit' in {args.data_set}, found: {list(df.columns)}")

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

    if args.light:
        args.d_model = 64
        args.n_heads = 2
        args.n_layers = 2
        args.d_ff = 128

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

        print(f"شروع فولد {fold_idx+1}/{args.folds} - آموزش: {len(train_ds)}، اعتبارسنجی: {len(test_ds)}", flush=True)

        model = InteractionTransformerPlus(
            vocab_size=VOCAB_SIZE,
            max_len=max_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            relative_pos=args.relative_pos,
            rp_max_dist=args.rp_max_dist,
        )

        if args.amp and device.type == 'cuda':
            scaler = GradScaler()
            model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            best_val = -1.0
            best_state = None
            for ep in range(args.epochs):
                model.train()
                running_loss = 0.0
                for batch_idx, (x, y) in enumerate(train_loader):
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
                    running_loss += float(loss.item())
                    if args.log_interval and ((batch_idx + 1) % args.log_interval == 0):
                        print(f"[AMP+] Epoch {ep+1}/{args.epochs} - step {batch_idx+1} - loss {loss.item():.4f}", flush=True)
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
                print(f"[AMP+] Epoch {ep+1}/{args.epochs} done - train_loss={avg_train_loss:.4f} - val_auc={auc:.4f}", flush=True)
                if auc > best_val:
                    best_val = auc
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if best_state is not None:
                model.load_state_dict(best_state)
        else:
            model, best_val = train_one_fold(
                model, train_loader, test_loader,
                epochs=args.epochs, lr=args.lr, device=device,
                grad_accum_steps=args.grad_accum_steps,
                log_interval=args.log_interval
            )

        metrics_fold = evaluate_fold(model, test_loader, device=device)
        metrics_fold['val_auc'] = float(best_val)
        fold_metrics.append(metrics_fold)

        attn = collect_attention(model, test_loader, device=device, max_batches=10)
        if attn is not None:
            mean_attn = attn.mean(axis=0).tolist()
        else:
            mean_attn = None
        attn_summaries.append({
            'fold': fold_idx,
            'mean_cls_attention': mean_attn,
        })

    agg = {}
    for key in ['auc', 'f1', 'precision', 'recall', 'avg_precision', 'val_auc', 'f1_best', 'precision_best', 'recall_best', 'best_threshold']:
        vals = [m[key] for m in fold_metrics]
        agg[key] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'n': len(vals)
        }

    with open(os.path.join(args.output_dir, 'fold_metrics.json'), 'w') as f:
        json.dump({'fold_metrics': fold_metrics, 'aggregate': agg}, f, indent=2)
    with open(os.path.join(args.output_dir, 'attention_summaries.json'), 'w') as f:
        json.dump({'attention': attn_summaries, 'max_len': max_len}, f, indent=2)

    md_lines = []
    md_lines.append('# Transformer+ Class I SCV Summary')
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
    md_lines.append('')
    md_lines.append('## Attention (CLS → tokens)')
    md_lines.append('- Saved mean per-fold CLS attention over sequence tokens to `attention_summaries.json`.')
    md_lines.append('- Relative positional bias encourages locality where relevant.')
    md_lines.append('')
    md_lines.append('## Notes')
    md_lines.append('- Uses PreNorm residuals and a 2-layer MLP head with GELU.')
    md_lines.append('- Optional relative positional bias injected via additive attention mask.')

    with open(os.path.join(args.output_dir, 'Transformer_ClassI_Plus_summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    print('Saved:')
    print(' -', os.path.join(args.output_dir, 'fold_metrics.json'))
    print(' -', os.path.join(args.output_dir, 'attention_summaries.json'))
    print(' -', os.path.join(args.output_dir, 'Transformer_ClassI_Plus_summary.md'))


if __name__ == '__main__':
    main()