# Transformer Class I SCV Summary

Data: `../Data/ClassI_Model/ClassI_training_210.csv` | Folds: `2` | Epochs: `1`
Max len (incl CLS): `512` | Device: `cpu`

## Metrics (mean ± std)
- auc: 0.6431 ± 0.0527 (n=2)
- f1: 0.0000 ± 0.0000 (n=2)
- precision: 0.0000 ± 0.0000 (n=2)
- recall: 0.0000 ± 0.0000 (n=2)
- avg_precision: 0.1508 ± 0.0309 (n=2)
- val_auc: 0.6431 ± 0.0527 (n=2)

## Attention (CLS → tokens)
- Saved mean per-fold CLS attention over sequence tokens to `attention_summaries.json`.
- Higher values indicate tokens the model focuses on for classification.

## Notes
- Tokens are digits 0–9 (score differences) with 9 as padding; a CLS token is prepended.
- Interpretability derives from last-layer multi-head self-attention averaged over heads.
- Classification uses the CLS representation via a linear head.