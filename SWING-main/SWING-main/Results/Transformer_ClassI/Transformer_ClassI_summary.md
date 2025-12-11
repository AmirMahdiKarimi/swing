# Transformer Class I SCV Summary

Data: `SWING-main/SWING-main/Data/ClassI_Model/ClassI_training_210.csv` | Folds: `5` | Epochs: `8`
Max len (incl CLS): `2048` | Device: `cuda`

## Metrics (mean ± std)
- auc: 0.8359 ± 0.0145 (n=5)
- f1: 0.1818 ± 0.1658 (n=5)
- precision: 0.3401 ± 0.2833 (n=5)
- recall: 0.1359 ± 0.1340 (n=5)
- avg_precision: 0.3969 ± 0.0319 (n=5)
- val_auc: 0.8359 ± 0.0145 (n=5)
- f1_best: 0.4659 ± 0.0219 (n=5)
- precision_best: 0.4111 ± 0.0342 (n=5)
- recall_best: 0.5435 ± 0.0421 (n=5)
- best_threshold: 0.2466 ± 0.0830 (n=5)

## Threshold Optimization
- best_threshold (mean±std): 0.2466 ± 0.0830
- f1_best (mean±std): 0.4659 ± 0.0219

## Attention (CLS → tokens)
- Saved mean per-fold CLS attention over sequence tokens to `attention_summaries.json`.
- Higher values indicate tokens the model focuses on for classification.

## Curves
- Per-fold ROC and PR curves are saved as PNGs in the output directory.

## Notes
- Tokens are digits 0–9 (score differences) with 9 as padding; a CLS token is prepended.
- Interpretability derives from last-layer multi-head self-attention averaged over heads.
- Classification uses the CLS representation via a linear head.