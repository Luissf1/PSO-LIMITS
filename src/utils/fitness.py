"""
Fitness function for quadrotor PID tuning.

Design goals:
1. Discriminate strictly between controllers across all 4 channels (z, phi, theta, psi).
2. Avoid flat zero-fitness plateaus (m_ref = 0.0) so gradient/selection pressure persists.
3. Return a bounded value in [0, 1] where lower is better.
"""

import numpy as np
from typing import Dict
from ..controllers.pid import PIDGains, PIDController
from ..models.quadrotor import QuadrotorDynamics, QuadrotorParameters
from .metrics import PerformanceMetrics, _trapz


FAILURE_FITNESS = 1.0

# Weight coefficients (sum = 1.0)
WEIGHTS = {
    "itse": 0.45,
    "iae": 0.30,
    "overshoot": 0.15,
    "settling": 0.10,
}

# Zero reference offsets eliminate the flat f = 0.0 dead-zone plateau
REF = {
    "itse": 0.0,
    "iae": 0.0,
    "overshoot": 0.0,
    "settling": 0.0,
}

SCALE = {
    "itse": 0.5,
    "iae": 1.0,
    "overshoot": 10.0,
    "settling": 2.0,
}


class QuadrotorFitness:
    """Fitness evaluator for a 12-gain PID controller."""

    def __init__(self,
                 reference: np.ndarray = None,
                 initial_state: np.ndarray = None,
                 simulation_time: float = 10.0,
                 dt: float = 0.01):
        self.reference = np.asarray(
            reference if reference is not None else [1.5, 0.15, -0.15, 0.524],
            dtype=float)
        self.initial_state = np.asarray(
            initial_state if initial_state is not None else np.zeros(12),
            dtype=float)
        self.simulation_time = simulation_time
        self.dt = dt
        self.model = QuadrotorDynamics(QuadrotorParameters())

    def evaluate(self, gains_array: np.ndarray) -> float:
        try:
            gains = PIDGains.from_array(gains_array)
        except (ValueError, TypeError):
            return FAILURE_FITNESS

        if np.any(np.isnan(gains_array)) or np.any(np.abs(gains_array) > 1e6):
            return FAILURE_FITNESS

        try:
            metrics = self._simulate_and_measure(gains)
        except Exception:
            return FAILURE_FITNESS

        if metrics is None:
            return FAILURE_FITNESS

        for k, v in metrics.items():
            if not np.isfinite(v) or v < 0:
                return FAILURE_FITNESS

        return self._combine(metrics)

    def _simulate_and_measure(self, gains: PIDGains) -> Dict[str, float]:
        controller = PIDController(gains, dt=self.dt, mass=1.0, gravity=9.81)

        def control_law(t, state):
            z, phi, theta, psi = state[2], state[3], state[4], state[5]
            measured = np.array([z, phi, theta, psi])
            return controller.compute(self.reference, measured)

        result = self.model.simulate(
            control_law,
            initial_state=self.initial_state,
            t_span=(0.0, self.simulation_time),
            dt=self.dt,
        )

        if not result["success"]:
            return None

        t = result["t"]
        states = result["state"]

        if not np.isfinite(states).all():
            return None
        if np.any(np.abs(states[:6, :]) > 100):
            return None

        z, phi, theta, psi = states[2, :], states[3, :], states[4, :], states[5, :]
        ref = self.reference

        # Combined L1 tracking error across all 4 controlled channels
        err = (np.abs(z - ref[0])
               + np.abs(phi - ref[1])
               + np.abs(theta - ref[2])
               + np.abs(psi - ref[3]))

        itse = float(_trapz(t * err ** 2, t))
        iae = float(_trapz(err, t))

        # Worst-case percentage overshoot and 5%-band settling time across all 4 channels
        channels = [(z, ref[0]), (phi, ref[1]), (theta, ref[2]), (psi, ref[3])]
        overshoot = max(
            PerformanceMetrics.overshoot(t, y, r) for y, r in channels
        )
        settling = max(
            PerformanceMetrics.settling_time(t, y, r, tolerance=0.05) for y, r in channels
        )

        return {
            "itse": itse,
            "iae": iae,
            "overshoot": overshoot,
            "settling": settling,
        }

    def _combine(self, metrics: Dict[str, float]) -> float:
        """
        Combine metrics into a strictly discriminative fitness value in [0, 1).
        """
        def penalty(value, ref, scale):
            excess = max(0.0, value - ref)
            return np.log1p(excess / scale)

        p_itse = penalty(metrics["itse"], REF["itse"], SCALE["itse"])
        p_iae = penalty(metrics["iae"], REF["iae"], SCALE["iae"])
        p_os = penalty(metrics["overshoot"], REF["overshoot"], SCALE["overshoot"])
        p_st = penalty(metrics["settling"], REF["settling"], SCALE["settling"])

        raw = (WEIGHTS["itse"] * p_itse
               + WEIGHTS["iae"] * p_iae
               + WEIGHTS["overshoot"] * p_os
               + WEIGHTS["settling"] * p_st)

        return float(np.tanh(raw))