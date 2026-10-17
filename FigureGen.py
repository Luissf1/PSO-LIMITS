"""
Generate publication-ready figures for the velocity limit article.

Reads the most recent JSON from results/full_v4/ and produces six figures
in both PDF (vector) and PNG (raster, 300 DPI) formats.
"""

import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats


# ============================================================================
# Configuration
# ============================================================================
FIGURE_DIR = Path('figures')
FIGURE_DIR.mkdir(exist_ok=True)

# Color palette per family
FAMILY_COLORS = {
    'fixed':    '#E63946',   # red
    'fraction': '#2A9D8F',   # teal
    'dynamic':  '#F4A261',   # orange
    'adaptive': '#264653',   # dark blue
}

# Common style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})


# ============================================================================
# Load results
# ============================================================================
def load_latest_results():
    """Load the most recent detailed JSON from results/full_v4/."""
    files = glob.glob('results/full_v4/detailed_*.json')
    if not files:
        raise FileNotFoundError(
            "No detailed JSON found in results/full_v4/. "
            "Run `python main.py --mode full --output results/full_v4` first."
        )
    latest = max(files, key=lambda p: Path(p).stat().st_ctime)
    print(f"Loading results from: {latest}")

    with open(latest) as f:
        data = json.load(f)

    results = data['results']
    df = pd.DataFrame([
        {
            'name': r['name'],
            'family': r['category'],
            'factor': r['factor'],
            'mean_fitness': r['mean_fitness'],
            'std_fitness': r['std_fitness'],
            'median_fitness': r['median_fitness'],
            'q1_fitness': r['q1_fitness'],
            'q3_fitness': r['q3_fitness'],
            'min_fitness': r['min_fitness'],
            'max_fitness': r['max_fitness'],
            'mean_iterations': r['mean_iterations'],
            'std_iterations': r['std_iterations'],
            'all_fitness': r['all_fitness'],
        }
        for r in results
    ])

    # Add composite score (same formula as in limit_study.py)
    df['rank_fitness'] = df['mean_fitness'].rank(method='min')
    df['rank_std'] = df['std_fitness'].rank(method='min')
    df['rank_iter'] = df['mean_iterations'].rank(method='min')
    N = len(df)
    df['composite'] = 1 - (
        (df['rank_fitness'] + df['rank_std'] + df['rank_iter']) / (3 * N)
    )
    df = df.sort_values('composite', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    return df, results


# ============================================================================
# Figure 1: Composite score ranking (horizontal bar chart)
# ============================================================================
def fig1_composite_ranking(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [FAMILY_COLORS[f] for f in df['family']]
    bars = ax.barh(df['name'], df['composite'], color=colors,
                   edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, score in zip(bars, df['composite']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{score:.3f}', va='center', fontsize=9)

    ax.set_xlabel('Composite Score')
    ax.set_ylabel('Strategy')
    ax.set_title('Composite Score Ranking of Velocity Limit Strategies')
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Legend
    patches = [mpatches.Patch(color=c, label=f.capitalize())
               for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=patches, loc='lower right', title='Family')

    plt.tight_layout()
    _save(fig, 'fig1_composite_ranking')


# ============================================================================
# Figure 2: Mean fitness by family (grouped bars)
# ============================================================================
def fig2_family_comparison(df):
    fig, ax = plt.subplots(figsize=(9, 5))

    # Group by family
    family_order = ['fixed', 'fraction', 'dynamic', 'adaptive']
    grouped = df.groupby('family').agg({
        'mean_fitness': ['mean', 'std'],
    }).reindex(family_order)

    means = grouped[('mean_fitness', 'mean')].values
    stds = grouped[('mean_fitness', 'std')].values
    colors = [FAMILY_COLORS[f] for f in family_order]

    bars = ax.bar(family_order, means, yerr=stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f'{mean:.4f}', ha='center', fontsize=9)

    ax.set_ylabel('Mean Fitness (lower is better)')
    ax.set_xlabel('Family')
    ax.set_title('Mean Fitness by Velocity Limit Family')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    _save(fig, 'fig2_family_comparison')


# ============================================================================
# Figure 3: Convergence speed vs fitness (scatter)
# ============================================================================
def fig3_convergence_vs_fitness(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [FAMILY_COLORS[f] for f in df['family']]

    scatter = ax.scatter(
        df['mean_iterations'], df['mean_fitness'],
        s=df['composite'] * 500 + 50,
        c=colors, edgecolors='black', linewidth=0.5,
        alpha=0.85, zorder=3,
    )

    for _, row in df.iterrows():
        ax.annotate(row['name'],
                    (row['mean_iterations'], row['mean_fitness']),
                    xytext=(6, 6), textcoords='offset points',
                    fontsize=8, alpha=0.8)

    ax.set_xlabel('Mean Iterations to Convergence')
    ax.set_ylabel('Mean Fitness (lower is better)')
    ax.set_title('Convergence Speed vs Solution Quality')
    ax.grid(True, alpha=0.3, linestyle='--')

    patches = [mpatches.Patch(color=c, label=f.capitalize())
               for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=patches, loc='upper right', title='Family')

    plt.tight_layout()
    _save(fig, 'fig3_convergence_vs_fitness')


# ============================================================================
# Figure 4: Wilcoxon significance heatmap
# ============================================================================
def fig4_wilcoxon_heatmap(df, results):
    names = df['name'].tolist()
    n = len(names)
    p_values = np.ones((n, n))

    # Fit matrix
    fit_matrix = np.array([r['all_fitness'] for r in results])
    name_to_idx = {r['name']: i for i, r in enumerate(results)}

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                p_values[i, j] = 1.0
                continue
            ia = name_to_idx[a]
            ib = name_to_idx[b]
            try:
                _, p = stats.wilcoxon(fit_matrix[ia], fit_matrix[ib])
            except ValueError:
                p = 1.0
            p_values[i, j] = p

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(p_values, cmap='RdYlGn_r', vmin=0, vmax=0.1, aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title('Pairwise Wilcoxon Signed-Rank Test (p-values)')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.04)
    cbar.set_label('p-value (green = significant difference)')

    # Annotate significant pairs
    for i in range(n):
        for j in range(n):
            if i != j and p_values[i, j] < 0.05:
                ax.text(j, i, '★', ha='center', va='center',
                        fontsize=6, color='white')

    plt.tight_layout()
    _save(fig, 'fig4_wilcoxon_heatmap')


# ============================================================================
# Figure 5: Fraction sensitivity (line plot with error bars)
# ============================================================================
def fig5_fraction_sensitivity(df):
    frac_df = df[df['family'] == 'fraction'].copy()
    frac_df = frac_df.sort_values('factor')

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(frac_df['factor'], frac_df['mean_fitness'],
                yerr=frac_df['std_fitness'],
                fmt='o-', capsize=5, color=FAMILY_COLORS['fraction'],
                linewidth=2, markersize=8, markeredgecolor='black')

    for _, row in frac_df.iterrows():
        ax.annotate(f"{row['name']}",
                    (row['factor'], row['mean_fitness']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=8, alpha=0.8)

    ax.set_xlabel('Fraction Factor')
    ax.set_ylabel('Mean Fitness (lower is better)')
    ax.set_title('Sensitivity of Performance to the Velocity Factor')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    _save(fig, 'fig5_fraction_sensitivity')


# ============================================================================
# Figure 6: Fitness distribution boxplot
# ============================================================================
def fig6_fitness_boxplot(df):
    fig, ax = plt.subplots(figsize=(12, 6))

    data = [np.array(f) for f in df['all_fitness']]
    names = df['name'].tolist()
    colors = [FAMILY_COLORS[f] for f in df['family']]

    bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='yellow',
                                   markeredgecolor='black', markersize=5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Fitness Across 30 Runs')
    ax.set_title('Distribution of Fitness Values Across 30 Independent Runs')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    patches = [mpatches.Patch(color=c, label=f.capitalize())
               for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=patches, loc='upper right', title='Family')

    plt.tight_layout()
    _save(fig, 'fig6_fitness_boxplot')


# ============================================================================
# Helper: save figure in PDF and PNG
# ============================================================================
def _save(fig, name):
    pdf_path = FIGURE_DIR / f'{name}.pdf'
    png_path = FIGURE_DIR / f'{name}.png'
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  → {pdf_path}")
    print(f"  → {png_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("Loading results...")
    df, results = load_latest_results()

    print(f"\nLoaded {len(df)} strategies:")
    print(df[['rank', 'name', 'mean_fitness', 'mean_iterations', 'composite']]
          .to_string(index=False))

    print("\nGenerating figures:")
    fig1_composite_ranking(df)
    fig2_family_comparison(df)
    fig3_convergence_vs_fitness(df)
    fig4_wilcoxon_heatmap(df, results)
    fig5_fraction_sensitivity(df)
    fig6_fitness_boxplot(df)

    print(f"\nAll figures saved to: {FIGURE_DIR}/")
    print("\nFiles generated:")
    for f in sorted(FIGURE_DIR.glob('fig*.png')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()