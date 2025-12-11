import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# تنظیمات اولیه
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# خواندن نتایج تمام مدل‌ها
models = ['LogReg', 'XGBoost', 'LightGBM', 'CatBoost']
results = {}

for model in models:
    metrics_path = f'Results/{model}/metrics_{model}_ClassI_training_210.json'
    preds_path = f'Results/{model}/predictions_{model}_ClassI_training_210.csv'
    
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    preds_df = pd.read_csv(preds_path)
    
    results[model] = {
        'metrics': metrics,
        'predictions': preds_df
    }

# ایجاد نمودارهای اصلی (ROC, PR, Metrics)
fig_main, axes_main = plt.subplots(1, 3, figsize=(18, 6))
fig_main.suptitle('مقایسه عملکرد مدل‌ها: ROC، PR و متریک‌ها', fontsize=16, fontweight='bold')

# 1. نمودار ROC Curve
for model in models:
    preds = results[model]['predictions']
    fpr, tpr, _ = roc_curve(preds['label'], preds['prob'])
    roc_auc = auc(fpr, tpr)
    
    axes_main[0].plot(fpr, tpr, lw=2, label=f'{model} (AUC = {roc_auc:.3f})')

axes_main[0].plot([0, 1], [0, 1], 'k--', lw=2)
axes_main[0].set_xlim([0.0, 1.0])
axes_main[0].set_ylim([0.0, 1.05])
axes_main[0].set_xlabel('نرخ مثبت کاذب (False Positive Rate)')
axes_main[0].set_ylabel('نرخ مثبت واقعی (True Positive Rate)')
axes_main[0].set_title('منحنی ROC')
axes_main[0].legend(loc="lower right")
axes_main[0].grid(True, alpha=0.3)

# 2. نمودار Precision-Recall Curve
for model in models:
    preds = results[model]['predictions']
    precision, recall, _ = precision_recall_curve(preds['label'], preds['prob'])
    avg_precision = results[model]['metrics']['avg_precision']
    
    axes_main[1].plot(recall, precision, lw=2, label=f'{model} (AP = {avg_precision:.3f})')

axes_main[1].set_xlim([0.0, 1.0])
axes_main[1].set_ylim([0.0, 1.05])
axes_main[1].set_xlabel('Recall')
axes_main[1].set_ylabel('Precision')
axes_main[1].set_title('منحنی Precision-Recall')
axes_main[1].legend(loc="lower left")
axes_main[1].grid(True, alpha=0.3)

# 3. مقایسه متریک‌های اصلی
metrics_to_compare = ['auc', 'f1', 'precision', 'recall', 'accuracy']
metric_names = ['AUC', 'F1 Score', 'Precision', 'Recall', 'Accuracy']

for i, metric in enumerate(metrics_to_compare):
    values = [results[model]['metrics'][metric] for model in models]
    axes_main[2].bar(np.arange(len(models)) + i*0.15, values, width=0.15, label=metric_names[i])

axes_main[2].set_xticks(np.arange(len(models)) + 0.3)
axes_main[2].set_xticklabels(models)
axes_main[2].set_ylabel('مقدار متریک')
axes_main[2].set_title('مقایسه متریک‌های عملکرد')
axes_main[2].legend()
axes_main[2].grid(True, alpha=0.3, axis='y')

# 4. ماتریس‌های confusion
confusion_data = []
for model in models:
    cm = results[model]['metrics']['confusion_matrix']
    confusion_data.append({
        'Model': model,
        'TN': cm['tn'],
        'FP': cm['fp'],
        'FN': cm['fn'],
        'TP': cm['tp']
    })

confusion_df = pd.DataFrame(confusion_data)
confusion_df.set_index('Model', inplace=True)

# رسم ماتریس‌های confusion در شکل جداگانه
fig_cm, axes_cm = plt.subplots(1, 4, figsize=(20, 5))
for i, model in enumerate(models):
    cm_data = confusion_df.loc[model]
    cm_matrix = np.array([[cm_data['TN'], cm_data['FP']],
                          [cm_data['FN'], cm_data['TP']]])
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=axes_cm[i], cbar=False)
    axes_cm[i].set_title(f'Confusion Matrix - {model}')
    axes_cm[i].set_xlabel('پیش‌بینی')
    axes_cm[i].set_ylabel('واقعیت')

# ذخیره خروجی‌ها
out_main = os.path.join('Results', 'model_comparison_analysis.png')
out_cm = os.path.join('Results', 'confusion_matrices.png')
fig_main.tight_layout()
fig_main.savefig(out_main, dpi=300, bbox_inches='tight')
fig_cm.tight_layout()
fig_cm.savefig(out_cm, dpi=300, bbox_inches='tight')
print(f"Saved analysis figures to: {out_main} and {out_cm}")
