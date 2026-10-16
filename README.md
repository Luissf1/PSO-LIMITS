# Velocity Limit Strategies in PSO for Quadrotor PID Tuning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Research-Article-red.svg)]()

## Overview

This repository contains the source code for a systematic study of **velocity limit strategies** in Particle Swarm Optimization (PSO) applied to the tuning of PID controllers for a quadrotor UAV.

The study compares 16 velocity limit strategies grouped into four families:

| Family | Strategies | Description |
|---|---|---|
| **Fraction** | `fraction_0.05` ... `fraction_0.50` | `v_max = factor × (var_max − var_min)` |
| **Dynamic** | `dynamic_0.10`, `dynamic_0.20`, `dynamic_0.30` | `v_max(t) = factor × (1 − t/T) × (var_max − var_min)` |
| **Adaptive** | `clerc`, `clerc_adaptive` | `v ← χ · v` (Clerc's constriction) |
| **Fixed** | `fixed_0.05` ... `fixed_0.50` | `\|v\| ≤ factor` (absolute) |

Each strategy is evaluated across 30 independent runs on a quadrotor PID tuning task, using a multi-objective fitness function that combines ITSE, IAE, overshoot, and settling time.

## Objective

To determine, through systematic experimentation, which velocity limit strategy offers the best trade-off between:

- **Solution quality** (mean fitness across runs)
- **Robustness** (standard deviation across runs)
- **Convergence speed** (mean iterations to convergence)
- **Success rate** (percentage of runs that hit the target fitness)

## Installation

### Requirements

- Python 3.9 or higher
- NumPy, SciPy, pandas, matplotlib, seaborn, PyYAML, openpyxl, tqdm

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Luissf1/PSO-LIMITS.git
cd PSO-LIMITS

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
#    Windows (PowerShell):
venv\Scripts\Activate.ps1
#    Windows (cmd):
venv\Scripts\activate.bat
#    Linux / macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt