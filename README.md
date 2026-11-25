# Velocity Limit Strategies in PSO for Quadrotor PID Tuning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Research-Article-red.svg)]()

## Overview

This repository contains the complete source code, configuration files, statistical analysis scripts, and figure generators for a systematic study of **velocity-handling strategies** in Particle Swarm Optimization (PSO) applied to tuning a 12-gain PID controller for a 12-state Newton–Euler quadrotor UAV model.

The study benchmarks **19 velocity-handling configurations** grouped into four families plus an unclamped baseline:

| Family | Configurations | Formula / Rule |
| :--- | :--- | :--- |
| **Fraction** | `fraction_0.05` ... `fraction_1.00` (9 levels) | $\|v_{ij}\| \le \alpha \cdot (x_{j,\max} - x_{j,\min})$ |
| **Dynamic** | `dynamic_0.10`, `dynamic_0.20`, `dynamic_0.30` | $\|v_{ij}(t)\| \le \alpha (1 - t/T)(x_{j,\max} - x_{j,\min})$ |
| **Adaptive** | `clerc`, `clerc_adaptive` | Canonical Clerc–Kennedy constriction ($c_1=c_2=2.05, \chi=0.7298, w=1.0$) |
| **Fixed** | `fixed_0.05`, `fixed_0.10`, `fixed_0.20`, `fixed_0.50` | $\|v_{ij}\| \le \alpha$ (absolute bound) |
| **Baseline** | `unclamped` | Standard inertia-weight PSO ($w: 0.9 \to 0.4$, no velocity cap) |

Each configuration is evaluated across **30 independent runs** (seeds `42` to `71`) on a coupled 4-DOF step maneuver (`E6`: $\mathbf{r} = [1.5\text{ m}, 0.15\text{ rad}, -0.15\text{ rad}, 0.524\text{ rad}]^T$) using a strictly discriminative fitness function aggregating ITSE, IAE, worst-case overshoot, and worst-case 5%-band settling time across all four controlled channels ($z, \phi, \theta, \psi$).

## Repository Structure

```text
PSO-LIMITS/
├── config/
│   ├── limit_strategies.yaml      # Definitions for all 19 PSO configurations & experiment settings
│   └── scenarios.yaml             # Quadrotor step maneuvers (E1–E6, default: E6)
├── src/
│   ├── controllers/pid.py         # 12-gain decoupled PID controller with anti-windup & saturation
│   ├── models/quadrotor.py        # 12-state rigid-body Newton–Euler dynamics (RK4, dt = 0.01 s)
│   ├── optimizers/pso.py          # Canonical & constricted PSO with 19 velocity-handling rules
│   ├── experiments/limit_study.py # 30-run batch runner & composite rank scorer
│   └── utils/
│       ├── fitness.py             # Multi-channel time-domain fitness evaluator
│       ├── metrics.py             # ITSE, IAE, overshoot, and settling time metrics
│       └── statistics.py          # Friedman test, Kendall's W, Holm–Bonferroni Wilcoxon, Cliff's delta
├── main.py                        # CLI entry point to execute the benchmark study
├── statistic_analisis.py          # Runs omnibus Friedman & Holm-corrected pairwise Wilcoxon tests
├── validate_robustness.py         # Out-of-sample perturbation tests (+20% mass, noise, wind gust)
├── FigureGen.py                   # Generates publication-ready Figures 1–6 (PNG & PDF)
└── requirements.txt               # Python dependencies
```

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Luissf1/PSO-LIMITS.git
cd PSO-LIMITS

# 2. Create and activate a virtual environment
python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (cmd):
venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Reproducing the Results

```bash
# 1. Run the full 19-configuration × 30-run benchmark
python main.py --mode full --output results/full_v5

# 2. Run statistical significance tests (Friedman + Holm–Bonferroni Wilcoxon + Cliff's delta)
python statistic_analisis.py

# 3. Run out-of-sample robustness validation (+20% mass, sensor noise, wind gust)
python validate_robustness.py

# 4. Regenerate Figures 1–6 in figures/
python FigureGen.py
```

## Key Findings

* **Omnibus Significance:** Friedman $\chi^2(18) = 287.35$, $p = 1.91 \times 10^{-50}$, Kendall's $W = 0.532$.
* **Top Composite Rank (`0.684`):** Moderate fraction-based limits (`fraction_0.15` and `fraction_0.20`) achieve the highest composite score by combining strong tracking accuracy ($0.2640\text{--}0.2655$) with $45\%\text{--}50\%$ lower run-to-run standard deviation ($\sigma = 0.0083\text{--}0.0091$) than unclamped PSO ($\sigma = 0.0166$).
* **Statistical Equivalence Tier:** After Holm–Bonferroni correction across all 171 pairs, `unclamped` ($0.2587$), `fraction_0.50` ($0.2613$), canonical `clerc` constriction ($0.2615$), and fraction bounds in $[0.10, 1.00]$ are statistically indistinguishable ($p_{\text{holm}} = 1.000$).
* **Fixed Bounds:** Unscaled fixed absolute bounds (`fixed_0.05`–`fixed_0.50`) degrade fitness significantly ($p_{\text{holm}} < 10^{-5}$, Cliff's $\delta = -1.000$ for `fixed_0.05`–`0.20`) due to premature stagnation across heterogeneous PID gain ranges.