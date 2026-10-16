"""
Control performance metrics for quadrotor PID tuning.
"""

import numpy as np
from typing import Dict


class PerformanceMetrics:
    """Collection of standard control performance metrics."""

    @staticmethod
    def settling_time(t: np.ndarray,
                      y: np.ndarray,
                      reference: float,
                      tolerance: float = 0.02) -> float:
        """Time to enter and stay within a tolerance band."""
        t = np.asarray(t).flatten()
        y = np.asarray(y).flatten()
        error = np.abs(y - reference)
        band = tolerance * abs(reference) if reference != 0 else tolerance

        inside = error <= band
        if not np.any(inside):
            return float(t[-1])

        for i in range(len(t) - 1, -1, -1):
            if inside[i] and np.all(inside[i:]):
                return float(t[i])
        return float(t[-1])

    @staticmethod
    def overshoot(t: np.ndarray, y: np.ndarray, reference: float) -> float:
        """Percentage overshoot relative to reference."""
        if reference == 0:
            return 0.0
        y = np.asarray(y).flatten()
        peak = np.max(y / reference)
        return float(max(0.0, (peak - 1.0) * 100.0))

    @staticmethod
    def rise_time(t: np.ndarray, y: np.ndarray, reference: float,
                  low: float = 0.1, high: float = 0.9) -> float:
        """Time from 10% to 90% of the reference."""
        if reference == 0:
            return float(t[-1])
        t = np.asarray(t).flatten()
        y = np.asarray(y).flatten()
        y_norm = y / reference

        t_low = None
        t_high = None
        for i, val in enumerate(y_norm):
            if t_low is None and val >= low:
                t_low = t[i]
            if t_low is not None and t_high is None and val >= high:
                t_high = t[i]
                break
        if t_low is None or t_high is None:
            return float(t[-1])
        return float(t_high - t_low)

    @staticmethod
    def iae(t: np.ndarray, y: np.ndarray, reference: float) -> float:
        """Integral of Absolute Error."""
        return float(np.trapz(np.abs(y - reference), t))

    @staticmethod
    def ise(t: np.ndarray, y: np.ndarray, reference: float) -> float:
        """Integral of Squared Error."""
        return float(np.trapz((y - reference) ** 2, t))

    @staticmethod
    def itae(t: np.ndarray, y: np.ndarray, reference: float) -> float:
        """Integral of Time-weighted Absolute Error."""
        return float(np.trapz(t * np.abs(y - reference), t))

    @staticmethod
    def itse(t: np.ndarray, y: np.ndarray, reference: float) -> float:
        """Integral of Time-weighted Squared Error."""
        return float(np.trapz(t * (y - reference) ** 2, t))

    @staticmethod
    def rmse(t: np.ndarray, y: np.ndarray, reference: float) -> float:
        """Root Mean Square Error."""
        return float(np.sqrt(np.mean((y - reference) ** 2)))

    @staticmethod
    def control_effort(u: np.ndarray) -> float:
        """Sum of squared control inputs."""
        return float(np.sum(u ** 2))