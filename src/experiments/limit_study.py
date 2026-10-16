"""
Systematic study of velocity limit strategies in PSO for quadrotor PID tuning.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from ..optimizers.pso import PSOConfig, ParticleSwarmOptimizer
from ..utils.fitness import QuadrotorFitness

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SystematicLimitStudy:
    """Run the systematic comparison of velocity limit strategies."""

    def __init__(self,
                 config_path: str = "config/limit_strategies.yaml",
                 scenario_path: str = "config/scenarios.yaml",
                 output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        with open(scenario_path, "r", encoding="utf-8") as f:
            self.scenarios = yaml.safe_load(f)

        self.strategies = self.config["strategies"]
        self.exp = self.config["experiment"]
        self.default_scenario = self.scenarios["default_scenario"]
        self.scenario = self.scenarios["scenarios"][self.default_scenario]

        # Search bounds for the 12 gains
        self.var_min = np.array([
            5.0,  0.1,  1.0,    # z
            5.0,  0.1,  1.0,    # phi
            5.0,  0.1,  1.0,    # theta
            3.0,  0.05, 0.5,    # psi
        ])
        self.var_max = np.array([
            40.0, 10.0, 20.0,   # z
            60.0, 5.0,  15.0,   # phi
            60.0, 5.0,  15.0,   # theta
            40.0, 3.0,  10.0,   # psi
        ])

        # Fitness evaluator
        self.fitness = QuadrotorFitness(
            reference=np.array(self.scenario["reference"]),
            initial_state=np.array(self.scenario["initial_state"]),
        )

    # ------------------------------------------------------------------
    def evaluate_strategy(self, strategy: Dict,
                          n_runs: Optional[int] = None) -> Dict:
        """Run a single strategy n_runs times and return aggregate stats."""
        n_runs = n_runs or self.exp["runs_per_strategy"]
        name = strategy["name"]

        logger.info(f"Strategy: {name} ({strategy['type']}, "
                    f"factor={strategy['factor']})")

        pso_config = PSOConfig(
            num_particles=self.exp["population_size"],
            max_iterations=self.exp["max_iterations"],
            velocity_limit_strategy=strategy["type"],
            velocity_limit_factor=strategy["factor"],
            tolerance=self.exp["early_stop_tolerance"],
            patience=self.exp["early_stop_patience"],
        )

        fitnesses = []
        iterations = []
        best_positions = []

        for run in range(n_runs):
            # Different seed per run for statistical independence
            pso_config.random_seed = self.exp["random_seed"] + run
            optimizer = ParticleSwarmOptimizer(pso_config,
                                               self.var_min, self.var_max)
            result = optimizer.optimize(self.fitness.evaluate)

            fitnesses.append(result["best_fitness"])
            iterations.append(result["num_iterations"])
            best_positions.append(result["best_position"].tolist())

            logger.info(f"  run {run+1}/{n_runs}: "
                        f"fitness={result['best_fitness']:.6f}, "
                        f"iters={result['num_iterations']}")

        fitnesses = np.array(fitnesses)

        return {
            "name": name,
            "type": strategy["type"],
            "factor": strategy["factor"],
            "category": strategy["category"],
            "description": strategy["description"],
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "min_fitness": float(np.min(fitnesses)),
            "max_fitness": float(np.max(fitnesses)),
            "median_fitness": float(np.median(fitnesses)),
            "q1_fitness": float(np.percentile(fitnesses, 25)),
            "q3_fitness": float(np.percentile(fitnesses, 75)),
            "success_rate": float(np.mean(fitnesses < self.exp["target_fitness"]) * 100),
            "mean_iterations": float(np.mean(iterations)),
            "std_iterations": float(np.std(iterations)),
            "all_fitness": fitnesses.tolist(),
            "all_iterations": iterations,
            "all_positions": best_positions,
        }

    # ------------------------------------------------------------------
    def run(self, n_runs: Optional[int] = None) -> pd.DataFrame:
        """Run all strategies."""
        logger.info("=" * 70)
        logger.info(f"Running {len(self.strategies)} strategies")
        logger.info(f"Scenario: {self.default_scenario}")
        logger.info("=" * 70)

        results = []
        for i, strategy in enumerate(self.strategies):
            logger.info(f"[{i+1}/{len(self.strategies)}]")
            results.append(self.evaluate_strategy(strategy, n_runs))
            self._save_intermediate(results)

        df = self._to_dataframe(results)
        df = self._rank(df)
        self._save_final(df, results)

        return df

    # ------------------------------------------------------------------
    def _to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        rows = []
        for r in results:
            row = {k: v for k, v in r.items() if not isinstance(v, list)}
            rows.append(row)
        return pd.DataFrame(rows)

    def _rank(self, df: pd.DataFrame) -> pd.DataFrame:
        w = self.config["ranking"]["weights"]

        # Lower is better for fitness
        f = df["mean_fitness"]
        df["rank_fitness"] = f.rank(method="min")
        df["rank_std"] = df["std_fitness"].rank(method="min")
        df["rank_success"] = df["success_rate"].rank(method="min",
                                                     ascending=False)
        df["rank_iter"] = df["mean_iterations"].rank(method="min")

        max_rank = len(df)
        df["composite_score"] = 1.0 - (
            w["mean_fitness"] * (df["rank_fitness"] / max_rank)
            + w["std_fitness"] * (df["rank_std"] / max_rank)
            + w["success_rate"] * (df["rank_success"] / max_rank)
            + w["convergence_speed"] * (df["rank_iter"] / max_rank)
        )

        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    # ------------------------------------------------------------------
    def _save_intermediate(self, results: List[Dict]) -> None:
        with open(self.output_dir / "intermediate.pkl", "wb") as f:
            pickle.dump(results, f)

    def _save_final(self, df: pd.DataFrame, results: List[Dict]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = df.drop(columns=["all_fitness", "all_iterations",
                                   "all_positions"], errors="ignore")
        summary.to_csv(self.output_dir / f"summary_{timestamp}.csv", index=False)

        with open(self.output_dir / f"detailed_{timestamp}.json", "w") as f:
            json.dump({
                "timestamp": timestamp,
                "scenario": self.default_scenario,
                "config": self.config,
                "results": results,
            }, f, indent=2)

        logger.info(f"Saved results to {self.output_dir}")