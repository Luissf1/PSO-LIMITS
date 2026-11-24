"""
Run omnibus Friedman test, Holm-Bonferroni adjusted pairwise Wilcoxon tests,
and Cliff's delta effect sizes on the latest experimental JSON.
"""

import json
import glob
from pathlib import Path
from src.utils.statistics import friedman_test, all_pairs_wilcoxon
import numpy as np


def main():
    files = glob.glob("results/**/detailed_*.json", recursive=True)
    if not files:
        raise FileNotFoundError("No detailed_*.json found under results/.")
    latest = max(files, key=lambda p: Path(p).stat().st_ctime)
    print(f"Analyzing: {latest}")

    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    names = [r["name"] for r in results]
    fitness_matrix = np.array([r["all_fitness"] for r in results])

    # 1. Omnibus Friedman test
    friedman = friedman_test(fitness_matrix)
    print(f"\n=== Friedman Omnibus Test ===")
    print(f"chi2({friedman['df']}) = {friedman['statistic']:.4f}, "
          f"p = {friedman['p_value']:.4e}, "
          f"Kendall's W = {friedman['kendalls_w']:.4f}")

    # 2. Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction
    df_pairs, _ = all_pairs_wilcoxon(names, fitness_matrix)
    out_csv = Path(latest).parent / "pairwise_wilcoxon_holm.csv"
    df_pairs.to_csv(out_csv, index=False)
    print(f"Saved {len(df_pairs)} Holm-adjusted pairwise tests to: {out_csv}")

    # 3. Print comparisons involving 'clerc'
    clerc_rows = df_pairs[
        (df_pairs["strategy_a"] == "clerc") | (df_pairs["strategy_b"] == "clerc")
    ]
    print("\n=== Pairwise Comparisons vs. 'clerc' (Holm-Bonferroni corrected) ===")
    print(clerc_rows.to_string(index=False))


if __name__ == "__main__":
    main()