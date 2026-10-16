"""
Statistical tests for comparing velocity limit strategies.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List


def friedman_test(fitness_matrix: np.ndarray) -> Dict:
    """
    Friedman test for multiple related samples.

    Parameters
    ----------
    fitness_matrix : np.ndarray
        Shape (n_strategies, n_runs).

    Returns
    -------
    dict
        {'statistic': float, 'p_value': float}
    """
    stat, p = stats.friedmanchisquare(*fitness_matrix)
    return {"statistic": float(stat), "p_value": float(p)}


def wilcoxon_pairwise(fitness_a: np.ndarray,
                      fitness_b: np.ndarray) -> Dict:
    """Wilcoxon signed-rank test between two strategies."""
    stat, p = stats.wilcoxon(fitness_a, fitness_b)
    return {"statistic": float(stat), "p_value": float(p)}


def confidence_interval(data: np.ndarray, alpha: float = 0.05) -> tuple:
    """Return (mean, lower, upper) of the 95% CI."""
    n = len(data)
    mean = float(np.mean(data))
    sem = float(stats.sem(data))
    h = sem * stats.t.ppf(1 - alpha / 2, n - 1)
    return mean, mean - h, mean + h