"""
Particle Swarm Optimization with configurable velocity limit strategies.

Supports 16 strategies grouped into 4 families:
  - fraction: v_max = factor * (var_max - var_min)
  - dynamic:  v_max(t) = factor * (1 - t/T) * (var_max - var_min)
  - clerc:    v <- chi * v
  - fixed:    |v| <= factor
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class PSOConfig:
    """Configuration for the PSO optimizer."""
    num_particles: int = 30
    max_iterations: int = 1000
    w_min: float = 0.4
    w_max: float = 0.9
    c1: float = 1.496
    c2: float = 1.496
    velocity_limit_strategy: str = "fraction"
    velocity_limit_factor: float = 0.20
    tolerance: float = 1e-6
    patience: int = 50
    random_seed: int = 42


class VelocityLimit:
    """Velocity limit strategies."""

    @staticmethod
    def apply(velocity: np.ndarray,
              position: np.ndarray,
              var_min: np.ndarray,
              var_max: np.ndarray,
              config: PSOConfig,
              iteration: int,
              **kwargs) -> np.ndarray:
        """Dispatch to the correct strategy."""
        strategy = config.velocity_limit_strategy
        if strategy == "fraction":
            return VelocityLimit._fraction(velocity, var_min, var_max, config)
        if strategy == "dynamic":
            return VelocityLimit._dynamic(velocity, var_min, var_max,
                                          config, iteration)
        if strategy == "clerc":
            return VelocityLimit._clerc(velocity, config)
        if strategy == "clerc_adaptive":
            return VelocityLimit._clerc_adaptive(velocity, config, **kwargs)
        if strategy == "fixed":
            return VelocityLimit._fixed(velocity, config)
        raise ValueError(f"Unknown strategy: {strategy}")

    # ------------------------------------------------------------------
    @staticmethod
    def _fraction(v, vmin, vmax, config):
        bound = config.velocity_limit_factor * (vmax - vmin)
        return np.clip(v, -bound, bound)

    @staticmethod
    def _dynamic(v, vmin, vmax, config, iteration):
        progress = iteration / max(config.max_iterations, 1)
        factor = config.velocity_limit_factor * (1.0 - progress)
        bound = factor * (vmax - vmin)
        return np.clip(v, -bound, bound)

    @staticmethod
    def _clerc(v, config):
        phi = config.c1 + config.c2
        kappa = 1.0
        denom = abs(2.0 - phi - np.sqrt(max(phi * phi - 4.0 * phi, 1e-12)))
        chi = 2.0 * kappa / denom
        return v * chi

    @staticmethod
    def _clerc_adaptive(v, config, **kwargs):
        # Adaptive chi: stronger damping when diversity is low
        diversity = kwargs.get("diversity", 1.0)
        base_chi = 0.7298
        scale = 0.9 + 0.2 * diversity  # range roughly [0.9, 1.1]
        return v * base_chi * scale

    @staticmethod
    def _fixed(v, config):
        bound = config.velocity_limit_factor
        return np.clip(v, -bound, bound)


class Particle:
    """A single particle in the swarm."""

    def __init__(self, position, velocity, var_min, var_max):
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.var_min = var_min
        self.var_max = var_max
        self.best_position = position.copy()
        self.best_fitness = float("inf")
        self.fitness = float("inf")


class ParticleSwarmOptimizer:
    """PSO optimizer with configurable velocity limit."""

    def __init__(self,
                 config: PSOConfig,
                 var_min: np.ndarray,
                 var_max: np.ndarray):
        self.config = config
        self.var_min = np.asarray(var_min, dtype=float)
        self.var_max = np.asarray(var_max, dtype=float)
        self._rng = np.random.RandomState(config.random_seed)

        self.swarm = []
        self.global_best_position = None
        self.global_best_fitness = float("inf")
        self.convergence_history = []
        self.velocity_magnitudes = []
        self.diversity_history = []

    # ------------------------------------------------------------------
    def _initialize_swarm(self):
        self.swarm = []
        range_size = self.var_max - self.var_min
        for _ in range(self.config.num_particles):
            pos = self._rng.uniform(self.var_min, self.var_max)
            vel = self._rng.uniform(-0.1 * range_size, 0.1 * range_size)
            self.swarm.append(Particle(pos, vel, self.var_min, self.var_max))

    def _compute_diversity(self) -> float:
        positions = np.array([p.position for p in self.swarm])
        mean_pos = positions.mean(axis=0)
        spread = np.linalg.norm(positions - mean_pos, axis=1).mean()
        norm = np.linalg.norm(self.var_max - self.var_min)
        return float(spread / max(norm, 1e-12))

    # ------------------------------------------------------------------
    def optimize(self,
                 fitness_function: Callable,
                 verbose: bool = False) -> Dict:
        """Run the PSO."""
        self._initialize_swarm()
        self.convergence_history = []
        self.velocity_magnitudes = []
        self.diversity_history = []
        self.global_best_fitness = float("inf")

        # Initial evaluation
        for p in self.swarm:
            p.fitness = fitness_function(p.position)
            if p.fitness < p.best_fitness:
                p.best_fitness = p.fitness
                p.best_position = p.position.copy()
            if p.fitness < self.global_best_fitness:
                self.global_best_fitness = p.fitness
                self.global_best_position = p.position.copy()

        self.convergence_history.append(self.global_best_fitness)

        # Main loop
        stagnation = 0
        for iteration in range(1, self.config.max_iterations + 1):
            diversity = self._compute_diversity()
            self.diversity_history.append(diversity)

            w = self.config.w_max - (self.config.w_max - self.config.w_min) * (
                iteration / self.config.max_iterations
            )

            for p in self.swarm:
                r1 = self._rng.rand()
                r2 = self._rng.rand()
                cognitive = self.config.c1 * r1 * (p.best_position - p.position)
                social = self.config.c2 * r2 * (self.global_best_position - p.position)
                p.velocity = w * p.velocity + cognitive + social

                self.velocity_magnitudes.append(float(np.mean(np.abs(p.velocity))))

                p.velocity = VelocityLimit.apply(
                    p.velocity, p.position,
                    self.var_min, self.var_max,
                    self.config, iteration,
                    diversity=diversity,
                )

                p.position = np.clip(p.position + p.velocity,
                                     self.var_min, self.var_max)

                p.fitness = fitness_function(p.position)
                if p.fitness < p.best_fitness:
                    p.best_fitness = p.fitness
                    p.best_position = p.position.copy()
                if p.fitness < self.global_best_fitness:
                    self.global_best_fitness = p.fitness
                    self.global_best_position = p.position.copy()
                    stagnation = 0
                else:
                    stagnation += 1

            self.convergence_history.append(self.global_best_fitness)

            if stagnation >= self.config.patience:
                if verbose:
                    print(f"Early stop at iteration {iteration}")
                break

            if verbose and iteration % 50 == 0:
                print(f"Iter {iteration}: best = {self.global_best_fitness:.6f}")

        return {
            "best_position": self.global_best_position,
            "best_fitness": self.global_best_fitness,
            "convergence": self.convergence_history,
            "velocity_magnitudes": self.velocity_magnitudes,
            "diversity": self.diversity_history,
            "num_iterations": len(self.convergence_history),
        }