"""
Fitness function for quadrotor PID tuning.

Design goals:
  1. Discriminate between good, fair, and bad controllers.
  2. Not be dominated by any single metric.
  3. Return a value in [0, 1] where lower is better.

The formula uses:
  - ITSE and IAE as primary accuracy metrics.
  - Overshoot and settling time as soft constraints (only penalized
    when they exceed a threshold).
"""

import numpy as np
from typing import Dict
from ..controllers.pid import PIDGains, PIDController
from ..models.quadrotor import QuadrotorDynamics, QuadrotorParameters


FAILURE_FITNESS = 1.0

# Weight coefficients (sum = 1.0)
WEIGHTS = {
    "itse": 0.45,
    "iae": 0.30,
    "overshoot": 0.15,
    "settling": 0.10,
}

# Reference values for a "good" controller
# Metrics below these references contribute nothing to the penalty
REF = {
    "itse": 0.2,         # ITSE of a well-tuned controller
    "iae": 1.0,          # IAE of a well-tuned controller
    "overshoot": 5.0,    # 5% overshoot is fine
    "settling": 3.0,     # 3 seconds is fine
}

# Scale constants for log-normalization of excess
SCALE = {
    "itse": 1.0,
    "iae": 2.0,
    "overshoot": 20.0,
    "settling": 5.0,
}


class QuadrotorFitness:
    """Fitness evaluator for a 12-gain PID controller."""

    def __init__(self,
                 reference: np.ndarray = None,
                 initial_state: np.ndarray = None,
                 simulation_time: float = 10.0,
                 dt: float = 0.01):
        self.reference = np.asarray(
            reference if reference is not None else [1.0, 0.0, 0.0, 0.0],
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

        z = states[2, :]
        phi = states[3, :]
        theta = states[4, :]
        psi = states[5, :]

        ref = self.reference

        err = (np.abs(z - ref[0])
               + np.abs(phi - ref[1])
               + np.abs(theta - ref[2])
               + np.abs(psi - ref[3]))

        itse = float(np.trapz(t * err ** 2, t))
        iae = float(np.trapz(err, t))

        overshoot = 0.0
        for y, r in [(z, ref[0]), (phi, ref[1]),
                     (theta, ref[2]), (psi, ref[3])]:
            if abs(r) > 1e-6:
                peak = float(np.max(y / r))
                if peak > 1.0:
                    overshoot = max(overshoot, (peak - 1.0) * 100.0)

        # Settling time: use a 5% band (more realistic than 2%)
        settling = 0.0
        for y, r in [(z, ref[0]), (phi, ref[1]),
                     (theta, ref[2]), (psi, ref[3])]:
            if abs(r) > 1e-6:
                error = np.abs(y - r)
                band = 0.05 * abs(r)
                outside = error > band
                if np.any(outside):
                    last_outside = np.max(np.where(outside)[0])
                    settling = max(settling, float(t[last_outside]))

        return {
            "itse": itse,
            "iae": iae,
            "overshoot": overshoot,
            "settling": settling,
        }

    def _combine(self, metrics: Dict[str, float]) -> float:
        """
        Combine metrics into a single fitness value.

        Only the EXCESS above the reference contributes to the penalty.
        A well-tuned controller gets a fitness close to 0.
        A poorly-tuned controller gets a fitness close to 1.
        """
        def penalty(value, ref, scale):
            """Excess above reference, log-normalized."""
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

        # Normalize to [0, 1]
        return float(np.tanh(raw))