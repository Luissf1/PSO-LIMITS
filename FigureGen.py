"""
Generate publication-ready figures for the velocity limit article.

Reads the most recent JSON from results/ and produces six figures
in both PDF (vector) and PNG (raster, 300 DPI) formats.
"""

import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.experiments.limit_study import compute_composite_score
from src.utils.statistics import all_pairs_wilcoxon


# ============================================================================
# Configuration
# ============================================================================
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

FAMILY_COLORS = {
    "baseline": "#6C757D",   # slate gray (unclamped control)
    "fixed":    "#E63946",   # red
    "fraction": "#2A9D8F",   # teal
    "dynamic":  "#F4A261",   # orange
    "adaptive": "#264653",   # dark blue
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


# ============================================================================
# Load results
# ============================================================================
def load_latest_results():
    """Load the most recent detailed JSON from results/."""
    files = glob.glob("results/**/detailed_*.json", recursive=True)
    if not files:
        raise FileNotFoundError(
            "No detailed JSON found in results/. "
            "Run `python main.py --mode full --output results/full_v5` first."
        )
    latest = max(files, key=lambda p: Path(p).stat().st_ctime)
    print(f"Loading results from: {latest}")

    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    df = pd.DataFrame([
        {
            "name": r["name"],
            "family": r["category"],
            "factor": r["factor"],
            "mean_fitness": r["mean_fitness"],
            "std_fitness": r["std_fitness"],
            "median_fitness": r["median_fitness"],
            "q1_fitness": r["q1_fitness"],
            "q3_fitness": r["q3_fitness"],
            "min_fitness": r["min_fitness"],
            "max_fitness": r["max_fitness"],
            "mean_iterations": r["mean_iterations"],
            "std_iterations": r["std_iterations"],
            "all_fitness": r["all_fitness"],
        }
        for r in results
    ])

    # Use the exact same canonical composite score function as limit_study.py
    df = compute_composite_score(df)
    df["composite"] = df["composite_score"]
    return df, results


def _family_legend_handles(df):
    present = [f for f in ["adaptive", "fraction", "dynamic", "fixed", "baseline"]
               if f in set(df["family"])]
    return [mpatches.Patch(color=FAMILY_COLORS[f], label=f.capitalize())
            for f in present]


# ============================================================================
# Figure 1: Composite score ranking (horizontal bar chart)
# ============================================================================
def fig1_composite_ranking(df):
    fig, ax = plt.subplots(figsize=(10, 6.5))

    colors = [FAMILY_COLORS[f] for f in df["family"]]
    bars = ax.barh(df["name"], df["composite"], color=colors,
                   edgecolor="black", linewidth=0.5)

    for bar, score in zip(bars, df["composite"]):
        ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Composite Score")
    ax.set_ylabel("Strategy")
    ax.set_title("Composite Score Ranking of Velocity Limit Strategies")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(handles=_family_legend_handles(df), loc="lower right", title="Family")

    plt.tight_layout()
    _save(fig, "fig1_composite_ranking")


# ============================================================================
# Figure 2: Mean fitness by family (grouped bars)
# ============================================================================
def fig2_family_comparison(df):
    fig, ax = plt.subplots(figsize=(9, 5))

    family_order = [f for f in ["fixed", "fraction", "dynamic", "adaptive", "baseline"]
                    if f in set(df["family"])]
    grouped = df.groupby("family").agg({
        "mean_fitness": ["mean", "std"],
    }).reindex(family_order)

    means = grouped[("mean_fitness", "mean")].values
    stds = np.nan_to_num(grouped[("mean_fitness", "std")].values, nan=0.0)
    colors = [FAMILY_COLORS[f] for f in family_order]

    bars = ax.bar(family_order, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

    y_offset = max(means) * 0.03 if max(means) > 0 else 0.001
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + y_offset,
                f"{mean:.4f}", ha="center", fontsize=9)

    ax.set_ylim(0, max(means + stds) * 1.18)
    ax.set_ylabel("Mean Fitness (lower is better)")
    ax.set_xlabel("Family")
    ax.set_title("Mean Fitness by Velocity Limit Family")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    _save(fig, "fig2_family_comparison")


# ============================================================================
# Figure 3: Convergence speed vs fitness (scatter with non-overlapping labels)
# ============================================================================
def fig3_convergence_vs_fitness(df):
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    colors = [FAMILY_COLORS[f] for f in df["family"]]
    ax.scatter(
        df["mean_iterations"], df["mean_fitness"],
        s=df["composite"] * 450 + 60,
        c=colors, edgecolors="black", linewidth=0.5,
        alpha=0.85, zorder=3,
    )

    # Sort by x then y so nearby labels alternate vertical offsets cleanly
    df_sorted = df.sort_values(["mean_iterations", "mean_fitness"]).reset_index(drop=True)
    offsets = [(6, 6), (6, -11), (-45, 8), (6, 12), (-45, -11)]
    for idx, row in df_sorted.iterrows():
        dx, dy = offsets[idx % len(offsets)]
        ax.annotate(row["name"],
                    (row["mean_iterations"], row["mean_fitness"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, alpha=0.9)

    ax.margins(x=0.14, y=0.14)
    ax.set_xlabel("Mean Iterations to Convergence")
    ax.set_ylabel("Mean Fitness (lower is better)")
    ax.set_title("Convergence Speed vs Solution Quality")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(handles=_family_legend_handles(df), loc="upper right", title="Family")

    plt.tight_layout()
    _save(fig, "fig3_convergence_vs_fitness")


# ============================================================================
# Figure 4: Wilcoxon significance heatmap (Holm-Bonferroni corrected, ASCII '*')
# ============================================================================
def fig4_wilcoxon_heatmap(df, results):
    names = df["name"].tolist()
    n = len(names)
    name_to_fit = {r["name"]: r["all_fitness"] for r in results}
    ordered_matrix = np.array([name_to_fit[name] for name in names])

    _, adj_p_values = all_pairs_wilcoxon(names, ordered_matrix)

    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    im = ax.imshow(adj_p_values, cmap="RdYlGn_r", vmin=0, vmax=0.1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Pairwise Wilcoxon Signed-Rank Test (Holm-Adjusted p-values)")

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Adjusted p-value (green = significant difference)")

    # Use standard ASCII '*' so PDF serif fonts render cleanly without missing glyphs
    for i in range(n):
        for j in range(n):
            if i != j and adj_p_values[i, j] < 0.05:
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")

    plt.tight_layout()
    _save(fig, "fig4_wilcoxon_heatmap")


# ============================================================================
# Figure 5: Fraction sensitivity (line plot with error bars)
# ============================================================================
def fig5_fraction_sensitivity(df):
    frac_df = df[df["family"] == "fraction"].copy().sort_values("factor")

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.errorbar(frac_df["factor"], frac_df["mean_fitness"],
                yerr=frac_df["std_fitness"],
                fmt="o-", capsize=5, color=FAMILY_COLORS["fraction"],
                linewidth=2, markersize=8, markeredgecolor="black")

    for _, row in frac_df.iterrows():
        ax.annotate(f"{row['name']}",
                    (row["factor"], row["mean_fitness"]),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8, alpha=0.85)

    ax.margins(x=0.1, y=0.15)
    ax.set_xlabel("Fraction Factor")
    ax.set_ylabel("Mean Fitness (lower is better)")
    ax.set_title("Sensitivity of Performance to the Velocity Factor")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    _save(fig, "fig5_fraction_sensitivity")


# ============================================================================
# Figure 6: Fitness distribution boxplot
# ============================================================================
def fig6_fitness_boxplot(df):
    fig, ax = plt.subplots(figsize=(12.5, 6))

    data = [np.array(f) for f in df["all_fitness"]]
    names = df["name"].tolist()
    colors = [FAMILY_COLORS[f] for f in df["family"]]

    bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="yellow",
                                   markeredgecolor="black", markersize=5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Fitness Across 30 Runs")
    ax.set_title("Distribution of Fitness Values Across 30 Independent Runs")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(handles=_family_legend_handles(df), loc="upper left", title="Family")

    plt.tight_layout()
    _save(fig, "fig6_fitness_boxplot")


# ============================================================================
# Helper: save figure in PDF and PNG
# ============================================================================
def _save(fig, name):
    pdf_path = FIGURE_DIR / f"{name}.pdf"
    png_path = FIGURE_DIR / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  -> {pdf_path}")
    print(f"  -> {png_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("Loading results...")
    df, results = load_latest_results()

    print(f"\nLoaded {len(df)} strategies:")
    print(df[["rank", "name", "mean_fitness", "std_fitness", "mean_iterations", "composite"]]
          .to_string(index=False))

    print("\nGenerating figures:")
    fig1_composite_ranking(df)
    fig2_family_comparison(df)
    fig3_convergence_vs_fitness(df)
    fig4_wilcoxon_heatmap(df, results)
    fig5_fraction_sensitivity(df)
    fig6_fitness_boxplot(df)

    print(f"\nAll figures saved to: {FIGURE_DIR}/")


if __name__ == "__main__":
    main()