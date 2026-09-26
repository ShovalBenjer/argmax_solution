"""
Evaluation Analysis Module for Diet Classification System

This module provides comprehensive analysis and visualization of diet classification
evaluation results. It generates publication-quality visualizations and detailed
performance metrics for both keto and vegan classifiers.

The module performs the following analyses:
- Confusion matrix visualization for both classifiers
- ROC curve analysis with AUC calculation
- Detailed classification reports with precision, recall, and F1 scores
- Misclassification analysis for root cause investigation
- Performance comparison between different model configurations

Key Features:
- Publication-quality visualizations using matplotlib and seaborn
- Comprehensive performance metrics calculation
- Interactive dashboard-style output for Jupyter notebooks
- Detailed misclassification analysis for model improvement
- Support for both binary classification tasks (keto/vegan)

Visualizations Generated:
- Confusion matrices for keto and vegan classifiers
- ROC curves with AUC scores (when probability scores available)
- Classification reports with detailed metrics
- Misclassification examples for analysis

Dependencies:
    - pandas: Data manipulation and analysis
    - matplotlib: Core plotting functionality
    - seaborn: Enhanced statistical visualizations
    - sklearn: Performance metrics calculation
    - IPython: Interactive display in notebooks

Usage:
    This module is designed to be run after executing the evaluation pipeline
    to analyze and visualize the results. It expects a predictions.csv file
    in the evaluation_results directory.

Example:
    >>> # After running evaluation pipeline
    >>> python nb/src/run_final_evaluation.py
    >>> # Then analyze results
    >>> python nb/src/evaluation_analysis.py
    >>> # Or import in Jupyter notebook
    >>> exec(open('nb/src/evaluation_analysis.py').read())
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pd
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

# Use a modern, publication-quality plotting style
sns.set_theme(style="whitegrid", palette="muted")

# --- Configuration ---
RESULTS_PATH = Path("nb/src/data/evaluation_results/")  # Directory now
PREDICTIONS_FILE = (
    RESULTS_PATH / "predictions.csv"
)  # Assuming run_final_evaluation saves this

if not PREDICTIONS_FILE.exists():
    print("Evaluation predictions file not found at " + str(PREDICTIONS_FILE) + ".")
    print("Please run `run_final_evaluation.py` to generate the results.")
else:
    df_results = pd.read_csv(PREDICTIONS_FILE)

    # --- 1. Generate and Display SOTA Evaluation Dashboard ---
    display(Markdown("## Model Evaluation Dashboard"))

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle(
        "Comprehensive Evaluation of Keto & Vegan Classifiers",
        fontsize=24,
        weight="bold",
    )

    # --- KETO ANALYSIS ---

    # Keto Confusion Matrix
    cm_keto = confusion_matrix(df_results["is_keto_true"], df_results["is_keto_pred"])
    sns.heatmap(
        cm_keto,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0, 0],
        cbar=False,
        annot_kws={"size": 16},
    )
    axes[0, 0].set_title("Keto Classifier Confusion Matrix", fontsize=16)
    axes[0, 0].set_xlabel("Predicted Label", fontsize=12)
    axes[0, 0].set_ylabel("True Label", fontsize=12)
    axes[0, 0].set_xticklabels(["Not Keto", "Keto"])
    axes[0, 0].set_yticklabels(["Not Keto", "Keto"])

    # Keto ROC Curve
    # Note: Requires probability scores. If not available, this part will be skipped.
    if "keto_pred_proba" in df_results.columns:
        fpr_keto, tpr_keto, _ = roc_curve(
            df_results["is_keto_true"], df_results["keto_pred_proba"]
        )
        roc_auc_keto = auc(fpr_keto, tpr_keto)
        axes[0, 1].plot(
            fpr_keto,
            tpr_keto,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {roc_auc_keto:0.2f})",
        )
        axes[0, 1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axes[0, 1].set_title("Keto Classifier ROC Curve", fontsize=16)
        axes[0, 1].set_xlabel("False Positive Rate")
        axes[0, 1].set_ylabel("True Positive Rate")
        axes[0, 1].legend(loc="lower right")
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "ROC Curve requires prediction probabilities.",
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )

    # --- VEGAN ANALYSIS ---

    # Vegan Confusion Matrix
    cm_vegan = confusion_matrix(
        df_results["is_vegan_true"], df_results["is_vegan_pred"]
    )
    sns.heatmap(
        cm_vegan,
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=axes[1, 0],
        cbar=False,
        annot_kws={"size": 16},
    )
    axes[1, 0].set_title("Vegan Classifier Confusion Matrix", fontsize=16)
    axes[1, 0].set_xlabel("Predicted Label", fontsize=12)
    axes[1, 0].set_ylabel("True Label", fontsize=12)
    axes[1, 0].set_xticklabels(["Not Vegan", "Vegan"])
    axes[1, 0].set_yticklabels(["Not Vegan", "Vegan"])

    # Vegan ROC Curve
    if "vegan_pred_proba" in df_results.columns:
        fpr_vegan, tpr_vegan, _ = roc_curve(
            df_results["is_vegan_true"], df_results["vegan_pred_proba"]
        )
        roc_auc_vegan = auc(fpr_vegan, tpr_vegan)
        axes[1, 1].plot(
            fpr_vegan,
            tpr_vegan,
            color="darkgreen",
            lw=2,
            label=f"ROC curve (AUC = {roc_auc_vegan:0.2f})",
        )
        axes[1, 1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axes[1, 1].set_title("Vegan Classifier ROC Curve", fontsize=16)
        axes[1, 1].set_xlabel("False Positive Rate")
        axes[1, 1].set_ylabel("True Positive Rate")
        axes[1, 1].legend(loc="lower right")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "ROC Curve requires prediction probabilities.",
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- 2. Detailed Classification Reports ---
    display(Markdown("### Keto Classification Report"))
    report_keto = classification_report(
        df_results["is_keto_true"],
        df_results["is_keto_pred"],
        target_names=["Not Keto", "Keto"],
        output_dict=True,
    )
    display(pd.DataFrame(report_keto).transpose().round(3))

    display(Markdown("### Vegan Classification Report"))
    report_vegan = classification_report(
        df_results["is_vegan_true"],
        df_results["is_vegan_pred"],
        target_names=["Not Vegan", "Vegan"],
        output_dict=True,
    )
    display(pd.DataFrame(report_vegan).transpose().round(3))

    # --- 3. SOTA Misclassification Analysis ---
    display(Markdown("### Root Cause Analysis of Misclassifications"))
    misclass_keto = df_results[df_results["is_keto_true"] != df_results["is_keto_pred"]]
    misclass_vegan = df_results[
        df_results["is_vegan_true"] != df_results["is_vegan_pred"]
    ]

    if not misclass_keto.empty:
        display(Markdown("**Keto Misclassifications:**"))
        display(
            misclass_keto[
                ["title", "ingredients", "is_keto_true", "is_keto_pred"]
            ].head()
        )
    else:
        display(Markdown("No Keto misclassifications found!"))

    if not misclass_vegan.empty:
        display(Markdown("**Vegan Misclassifications:**"))
        display(
            misclass_vegan[
                ["title", "ingredients", "is_vegan_true", "is_vegan_pred"]
            ].head()
        )
    else:
        display(Markdown("No Vegan misclassifications found!"))
