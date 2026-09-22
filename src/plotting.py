"""Shared plotting utilities for the Automobile Price Prediction project.

Kept separate from the notebook so every chart uses the same visual style
(colors, fonts, layout) and so the plotting code isn't duplicated across cells.
None of these functions compute metrics — they only visualize values that were
already computed from the model (e.g. in `evaluation.py` or the notebook).
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Semantic, restrained color palette — consistent with the other project in
# this portfolio (blue = baseline/default, orange = the alternate configuration).
MODEL_COLORS = {
    "default": "#3B82F6",  # blue  — default-parameter Random Forest
    "tuned": "#F97316",    # orange — RandomizedSearchCV-tuned Random Forest
}
ACCENT_COLOR = "#0EA5A0"  # teal, used for single-series charts (feature importance, price distribution)


def apply_style():
    """Consistent, professional matplotlib/seaborn config for every chart."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#444444",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": False,
    })


def plot_feature_importance(importances, save_path, top_n=10,
                             title="Top 10 Feature Importances — Tuned Random Forest"):
    """Horizontal bar chart of feature importances, largest on top, with exact
    values annotated (nothing rounded away)."""
    apply_style()
    top = importances.head(top_n).iloc[::-1]  # smallest first so largest ends up on top when plotted

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(top.index, top.values, color=ACCENT_COLOR)
    for bar, v in zip(bars, top.values):
        ax.annotate(
            f"{v:.3f}", xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0), textcoords="offset points", va="center", fontsize=9.5,
        )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.set_xlim(0, top.values.max() * 1.15)

    fig.tight_layout()
    fig.savefig(save_path)
    return fig


def plot_actual_vs_predicted(y_test, preds, save_path, r2=None,
                              title="Actual vs Predicted Price — Tuned Random Forest"):
    """Scatter of actual vs. predicted price with a perfect-prediction
    reference line and (optionally) the test R² annotated directly on the chart."""
    apply_style()
    lo = min(y_test.min(), preds.min())
    hi = max(y_test.max(), preds.max())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, preds, alpha=0.25, s=10, color=MODEL_COLORS["tuned"], linewidths=0)
    ax.plot([lo, hi], [lo, hi], color="#444444", linestyle="--", linewidth=1.8, label="Perfect Prediction")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual Price (£)")
    ax.set_ylabel("Predicted Price (£)")
    ax.set_title(title)
    ax.legend(loc="upper left")

    if r2 is not None:
        ax.text(
            0.97, 0.05, f"Test R² = {r2:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc"),
        )

    fig.tight_layout()
    fig.savefig(save_path)
    return fig


def plot_price_distribution(prices, save_path, title="Distribution of Car Prices (Cleaned Dataset)"):
    """Histogram + KDE of the price column, with the median marked."""
    apply_style()
    median = prices.median()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(prices, bins=50, kde=True, color=ACCENT_COLOR, ax=ax)
    ax.axvline(median, color="#444444", linestyle=":", linewidth=1.6, label=f"Median = £{median:,.0f}")
    ax.set_xlabel("Price (£)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    return fig


def plot_model_comparison(results_df, save_path,
                           title="Baseline vs Tuned — Test Set Comparison"):
    """Small-multiple grid comparing Default vs Tuned on R², MAE, RMSE and
    RMSLE. Each metric gets its own panel since they're on different scales
    (a single shared axis would make R² and RMSLE unreadable next to £ values)."""
    apply_style()
    by_model = results_df.set_index("Model")
    default_name = [n for n in by_model.index if "Default" in n][0]
    tuned_name = [n for n in by_model.index if "Tuned" in n][0]

    panels = [
        ("Test R2", "Test R²", "{:.3f}"),
        ("Test MAE", "Test MAE (£)", "{:,.0f}"),
        ("Test RMSE", "Test RMSE (£)", "{:,.0f}"),
        ("Test RMSLE", "Test RMSLE", "{:.3f}"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (col, label, fmt) in zip(axes, panels):
        values = [by_model.loc[default_name, col], by_model.loc[tuned_name, col]]
        bars = ax.bar(["Default", "Tuned"], values,
                       color=[MODEL_COLORS["default"], MODEL_COLORS["tuned"]], width=0.55)
        for bar, v in zip(bars, values):
            ax.annotate(
                fmt.format(v), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9.5,
            )
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, max(values) * 1.2)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    return fig
