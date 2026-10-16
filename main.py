"""
Main entry point for the velocity limit study.
"""

import argparse
import numpy as np
from pathlib import Path
import time

from src.experiments.limit_study import SystematicLimitStudy


def main():
    parser = argparse.ArgumentParser(
        description="Systematic study of velocity limits in PSO"
    )
    parser.add_argument("--config", default="config/limit_strategies.yaml")
    parser.add_argument("--scenario", default="config/scenarios.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--runs", type=int, default=None,
                        help="Override number of runs per strategy")
    parser.add_argument("--mode", choices=["full", "quick", "analyze"],
                        default="full")
    args = parser.parse_args()

    if args.mode == "analyze":
        from scripts.analyze import run_analysis
        run_analysis(args.output)
        return

    study = SystematicLimitStudy(
        config_path=args.config,
        scenario_path=args.scenario,
        output_dir=args.output,
    )

    if args.mode == "quick":
        # Only run 3 strategies for a quick smoke test
        study.strategies = study.strategies[:3]

    t0 = time.time()
    df = study.run(n_runs=args.runs)
    elapsed = time.time() - t0

    print(f"\nTotal time: {elapsed/60:.2f} minutes")
    print(df[["rank", "name", "mean_fitness", "std_fitness",
              "success_rate", "mean_iterations", "composite_score"]])


if __name__ == "__main__":
    main()