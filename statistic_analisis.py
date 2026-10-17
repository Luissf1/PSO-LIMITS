import json, glob
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

# Cargar el JSON más reciente
files = glob.glob('results/full_v4/detailed_*.json')
latest = max(files, key=lambda p: __import__('pathlib').Path(p).stat().st_ctime)
with open(latest) as f:
    data = json.load(f)

# Extraer matriz de fitness (16 estrategias × 30 runs)
fitness_matrix = np.array([r['all_fitness'] for r in data['results']])

# Friedman test
stat, p = friedmanchisquare(*fitness_matrix)
print(f"Friedman statistic: {stat:.2f}")
print(f"Friedman p-value: {p:.6f}")

# Wilcoxon: clerc vs fixed_0.05
clerc_idx = [r['name'] for r in data['results']].index('clerc')
fixed_idx = [r['name'] for r in data['results']].index('fixed_0.05')
stat_w, p_w = wilcoxon(fitness_matrix[clerc_idx], fitness_matrix[fixed_idx])
print(f"Wilcoxon clerc vs fixed_0.05: stat={stat_w:.2f}, p={p_w:.6f}")

# Cliff's delta
clerc_fit = fitness_matrix[clerc_idx]
fixed_fit = fitness_matrix[fixed_idx]
n1, n2 = len(clerc_fit), len(fixed_fit)
delta = np.mean([np.sign(a - b) for a in clerc_fit for b in fixed_fit])
print(f"Cliff's delta: {delta:.4f}")