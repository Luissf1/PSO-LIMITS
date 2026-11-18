"""
Statistical tests for comparing velocity limit strategies.
Includes Friedman omnibus test, pairwise Wilcoxon signed-rank tests with
Holm-Bonferroni multiple-comparison correction, and Cliff's delta effect size.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List


def friedman_test(fitness_matrix: np.ndarray) -> Dict:
    """Friedman test for multiple related samples (shape: n_strategies x n_runs)."""
    stat, p = stats.friedmanchisquare(*fitness_matrix)
    n_strategies, n_runs = fitness_matrix.shape
    kendall_w = float(stat / (n_runs * (n_strategies - 1)))
    return {
        "statistic": float(stat),
        "df": int(n_strategies - 1),
        "p_value": float(p),
        "kendalls_w": kendall_w,
    }


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta non-parametric effect size in [-1, 1]."""
    x = np.asarray(x)
    y = np.asarray(y)
    return float(np.mean(np.sign(x[:, None] - y[None, :])))


def wilcoxon_pairwise(fitness_a: np.ndarray,
                      fitness_b: np.ndarray) -> Dict:
    """Wilcoxon signed-rank test between two paired strategies."""
    diff = np.asarray(fitness_a) - np.asarray(fitness_b)
    if np.allclose(diff, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "cliffs_delta": 0.0}
    stat, p = stats.wilcoxon(fitness_a, fitness_b, zero_method="wilcox")
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "cliffs_delta": cliffs_delta(fitness_a, fitness_b),
    }


def holm_bonferroni(p_values: List[float]) -> np.ndarray:
    """Apply Holm-Bonferroni step-down correction to a list of p-values."""
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    order = np.argsort(p_arr)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        corr = p_arr[idx] * (m - rank)
        running_max = max(running_max, corr)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def all_pairs_wilcoxon(names: List[str],
                       fitness_matrix: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute all pairwise Wilcoxon signed-rank tests with Holm-Bonferroni
    correction and Cliff's delta. Returns (summary_df, adjusted_p_matrix).
    """
    n = len(names)
    raw_p_mat = np.ones((n, n), dtype=float)
    adj_p_mat = np.ones((n, n), dtype=float)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            res = wilcoxon_pairwise(fitness_matrix[i], fitness_matrix[j])
            raw_p_mat[i, j] = raw_p_mat[j, i] = res["p_value"]
            pairs.append({
                "strategy_a": names[i],
                "strategy_b": names[j],
                "i": i,
                "j": j,
                "wilcoxon_stat": res["statistic"],
                "p_raw": res["p_value"],
                "cliffs_delta": res["cliffs_delta"],
            })

    if pairs:
        adj_ps = holm_bonferroni([p["p_raw"] for p in pairs])
        for p_dict, p_adj in zip(pairs, adj_ps):
            p_dict["p_holm"] = float(p_adj)
            p_dict["significant_0.05"] = bool(p_adj < 0.05)
            i, j = p_dict["i"], p_dict["j"]
            adj_p_mat[i, j] = adj_p_mat[j, i] = float(p_adj)

    df_pairs = pd.DataFrame(pairs).drop(columns=["i", "j"], errors="ignore")
    return df_pairs, adj_p_mat


def confidence_interval(data: np.ndarray, alpha: float = 0.05) -> tuple:
    """Return (mean, lower, upper) of the 95% CI."""
    n = len(data)
    mean = float(np.mean(data))
    sem = float(stats.sem(data))
    h = sem * stats.t.ppf(1 - alpha / 2, n - 1)
    return mean, mean - h, mean + h