# =============================================================================
# main.py — Main runner: model comparison, evaluation report, visualization
# Usage:  python main.py [--data-dir <data directory>]
# =============================================================================

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, classification_report,
)

from preprocess import run_preprocessing_cached
from models import train_all_models

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Plot style
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = [
    "#4C6EF5",   # KNN            — indigo
    "#F76707",   # Decision Tree  — orange
    "#2F9E44",   # Random Forest  — green
    "#AE3EC9",   # Logistic Reg.  — purple
    "#E03131",   # GBT (LightGBM) — red
]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

OUTPUT_DIR = "output_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# English model name mapping (for report_df index / labels)
MODEL_NAME_EN = {
    "KNN":              "KNN",
    "决策树":           "Decision Tree",
    "随机森林":         "Random Forest",
    "逻辑回归":         "Logistic Regression",
    "GBT (LightGBM)":  "GBT (LightGBM)",
}

# ──────────────────────────────────────────────────────────────────────────────
# Batched inference helper
# ──────────────────────────────────────────────────────────────────────────────

_BATCH_SIZE = 50_000   # ~50k rows per batch — prevents OOM for KNN on large test sets


def _predict_in_batches(model, X: np.ndarray, proba: bool = False) -> np.ndarray:
    """Run inference in fixed-size batches to limit peak memory usage."""
    results = []
    for start in range(0, X.shape[0], _BATCH_SIZE):
        batch = X[start : start + _BATCH_SIZE]
        results.append(model.predict_proba(batch)[:, 1] if proba else model.predict(batch))
    return np.concatenate(results)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Compute Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC on test set."""
    print(f"    -> Inferring (batch={_BATCH_SIZE:,})...", end=" ", flush=True)
    y_pred  = _predict_in_batches(model, X_test, proba=False)
    y_score = (
        _predict_in_batches(model, X_test, proba=True)
        if hasattr(model, "predict_proba")
        else (model.decision_function(X_test) if hasattr(model, "decision_function")
              else y_pred.astype(float))
    )
    print("done")

    en_name = MODEL_NAME_EN.get(name, name)
    return {
        "Model":     en_name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_test, y_score),
        "PR-AUC":    average_precision_score(y_test, y_score),
        "_y_score":  y_score,
        "_y_pred":   y_pred,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ROC curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(results: list, y_test: np.ndarray) -> None:
    """Plot ROC curves for all five models and save to output_figures/roc_curves.png."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, res in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, res["_y_score"])
        ax.plot(fpr, tpr, color=PALETTE[i], lw=2,
                label=f"{res['Model']}  (AUC={res['ROC-AUC']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier (AUC=0.5)")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title("ROC Curve Comparison (5 Models)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  ROC curve saved -> {path}")


# ──────────────────────────────────────────────────────────────────────────────
# PR curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_pr_curves(results: list, y_test: np.ndarray) -> None:
    """Plot Precision-Recall curves for all five models."""
    baseline = y_test.mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, res in enumerate(results):
        prec, rec, _ = precision_recall_curve(y_test, res["_y_score"])
        ax.plot(rec, prec, color=PALETTE[i], lw=2,
                label=f"{res['Model']}  (AP={res['PR-AUC']:.4f})")
    ax.axhline(y=baseline, color="k", linestyle="--", lw=1, alpha=0.5,
               label=f"Random Classifier (AP={baseline:.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve Comparison (5 Models)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    path = os.path.join(OUTPUT_DIR, "pr_curves.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  PR curve saved -> {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Metrics heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_metrics_heatmap(report_df: pd.DataFrame) -> None:
    """Plot a 5-model x 6-metric color heatmap."""
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    heat_data = report_df.set_index("Model")[metric_cols]
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heat_data, annot=True, fmt=".4f", cmap="YlGn",
                vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Score"})
    ax.set_title("Model Performance Heatmap (5 Models)", fontsize=14, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
    path = os.path.join(OUTPUT_DIR, "metrics_heatmap.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  Heatmap saved -> {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Feature importance
# ──────────────────────────────────────────────────────────────────────────────

# English labels for all engineered features
FEAT_NAME_EN = {
    "sess_view_cnt":           "Session View Count",
    "sess_cart_cnt":           "Session Cart Count",
    "sess_duration_sec":       "Session Duration (sec)",
    "sess_unique_items":       "Session Unique Items",
    "sess_cart_view_ratio":    "Cart/View Ratio",
    "user_recency_sec":        "User Recency (sec)",
    "user_freq_total":         "User Total Frequency",
    "user_cart_freq":          "User Cart Frequency",
    "user_decayed_view":       "Decayed View Score",
    "user_decayed_cart":       "Decayed Cart Score",
    "user_cat_breadth":        "Category Breadth",
    "user_cat_concentration":  "Category Concentration (HHI)",
}


def plot_feature_importance(trained_models: dict, feature_names: list) -> None:
    """Bar chart of Top-15 feature importances for tree-based models."""
    imp_models = {
        MODEL_NAME_EN.get(k, k): v
        for k, v in trained_models.items()
        if hasattr(v, "feature_importances_")
    }
    n = len(imp_models)
    if n == 0:
        print("  No models support feature_importances_, skipping.")
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]

    color_map = {
        "Decision Tree":   PALETTE[1],
        "Random Forest":   PALETTE[2],
        "GBT (LightGBM)":  PALETTE[4],
    }

    for ax, (name, model) in zip(axes, imp_models.items()):
        imp = model.feature_importances_
        imp_norm = imp / (imp.sum() + 1e-10)
        idx = np.argsort(imp_norm)[::-1][:15]
        labels = [FEAT_NAME_EN.get(feature_names[i], feature_names[i]) for i in idx]
        vals   = imp_norm[idx]

        ax.barh(range(len(labels)), vals[::-1], color=color_map.get(name, "#4C6EF5"))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels[::-1], fontsize=9)
        ax.set_xlabel("Normalized Importance", fontsize=10)
        ax.set_title(f"{name}\nTop-15 Feature Importance", fontsize=11, fontweight="bold")
        ax.set_xlim(0, vals.max() * 1.25)

    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  Feature importance chart saved -> {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Console report
# ──────────────────────────────────────────────────────────────────────────────

def print_report(report_df: pd.DataFrame) -> None:
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    display = report_df[["Model"] + metric_cols].copy()
    sep = "=" * 90
    print(f"\n{sep}")
    print("  Purchase Intent Prediction — Model Comparison Report (Retailrocket Dataset)")
    print(sep)
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(sep)
    print("\n  Best model per metric:")
    for col in metric_cols:
        best_idx   = display[col].idxmax()
        best_model = display.loc[best_idx, "Model"]
        best_val   = display.loc[best_idx, col]
        print(f"    {col:<14}: {best_model}  ({best_val:.4f})")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(data_dir: str = "data") -> None:
    print("=" * 60)
    print("  Phase 1 — User Purchase Intent: 5-Model Comparison")
    print("=" * 60)

    # Step 1: Preprocessing
    X_train, X_test, y_train, y_test, feature_names, scaler = run_preprocessing_cached(data_dir)

    # Step 2: Train / load models
    print("\n" + "=" * 60)
    print("  Training models (loads from cache if available)")
    print("=" * 60)
    trained_models = train_all_models(X_train, y_train)

    # Step 3: Evaluate
    print("\n" + "=" * 60)
    print("  Evaluating models")
    print("=" * 60)
    results = []
    for name, model in trained_models.items():
        res = evaluate_model(name, model, X_test, y_test)
        results.append(res)
        en_name = MODEL_NAME_EN.get(name, name)
        print(f"\n  [{en_name}] Classification Report:")
        print(classification_report(
            y_test, res["_y_pred"],
            target_names=["No Purchase (0)", "Purchase (1)"],
            digits=4,
        ))

    # Step 4: Build summary DataFrame
    report_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ])

    # Step 5: Console report + CSV
    print_report(report_df)
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison_report.csv")
    report_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  Report saved -> {csv_path}")

    # Step 6: Visualizations
    print("\n" + "=" * 60)
    print("  Generating charts")
    print("=" * 60)
    plot_roc_curves(results, y_test)
    plot_pr_curves(results, y_test)
    plot_metrics_heatmap(report_df)
    plot_feature_importance(trained_models, feature_names)

    print("\nAll done!")
    print(f"  Charts : ./{OUTPUT_DIR}/")
    print(f"  Models : ./saved_models/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Purchase Intent Prediction — Model Comparison")
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing Retailrocket CSV files (default: ./data/)")
    args = parser.parse_args()
    main(data_dir=args.data_dir)
