import os
import numpy as np

def load_array(path):
    try:
        return np.load(path)
    except Exception as e:
        print(f"[WARN] Could not load '{path}': {e}")
        return None

def summarize_metric(name, arr):
    if arr is None:
        return f"- {name}: not available"
    try:
        arr = np.array(arr).astype(float)
        return f"- {name}: mean={arr.mean():.4f}, std={arr.std(ddof=1):.4f}, n={arr.size}"
    except Exception as e:
        return f"- {name}: error computing summary: {e}"

def main():
    scv_dir = r"c:\Users\Amir\Desktop\نامه ی بی پایان\SWING\output\output\scv"
    results_dir = r"c:\Users\Amir\Desktop\نامه ی بی پایان\SWING\SWING-main\SWING-main\Results"

    # Filenames for Class I SCV
    prefix = "ClassI_SCV_210"
    files = {
        "AUC (train-test splits)": os.path.join(scv_dir, f"all_tts_aucs_{prefix}.npy"),
        "AUC (permutation)": os.path.join(scv_dir, f"all_permy_aucs_{prefix}.npy"),
        "Average precision": os.path.join(scv_dir, f"all_avg_precisions_{prefix}.npy"),
        "Best thresholds": os.path.join(scv_dir, f"all_best_thresh_{prefix}.npy"),
        "F1 scores": os.path.join(scv_dir, f"all_f1_scores_{prefix}.npy"),
        "Precisions": os.path.join(scv_dir, f"all_precisions_{prefix}.npy"),
        "Recalls": os.path.join(scv_dir, f"all_recalls_{prefix}.npy"),
    }

    loaded = {name: load_array(path) for name, path in files.items()}

    lines = []
    lines.append("SWING Class I SCV Summary (ClassI_SCV_210)")
    lines.append("")
    for name, arr in loaded.items():
        lines.append(summarize_metric(name, arr))

    # Attempt quick comparison to paper's reported AUC ~0.72 for Class I SCV
    tts = loaded.get("AUC (train-test splits)")
    if tts is not None:
        try:
            mean_auc = float(np.mean(tts))
            lines.append("")
            lines.append(f"- Comparison: reported Class I SCV AUC ≈ 0.72; observed mean AUC = {mean_auc:.4f}")
        except Exception as e:
            lines.append(f"- Comparison: error computing observed mean AUC: {e}")

    out_md = os.path.join(results_dir, "ClassI_SCV_210_summary.md")
    try:
        os.makedirs(results_dir, exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[INFO] Summary written to: {out_md}")
    except Exception as e:
        print(f"[ERROR] Could not write summary: {e}")

    # Also print to stdout for convenience
    print("\n".join(lines))

if __name__ == "__main__":
    main()