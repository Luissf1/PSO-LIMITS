"""
Cascaded PID controller for a quadrotor.

Structure:
  - Outer loop (position): altitude -> U1 (thrust)
  - Inner loop (attitude): roll, pitch, yaw -> U2, U3, U4 (torques)

This is the standard control architecture for a quadrotor.
"""

import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class PIDGains:
    """12 gains: 3 for altitude, 3 for each of roll/pitch/yaw."""
    kp_z: float = 30.0
    ki_z: float = 5.0
    kd_z: float = 15.0
    kp_phi: float = 40.0
    ki_phi: float = 2.0
    kd_phi: float = 8.0
    kp_theta: float = 40.0
    ki_theta: float = 2.0
    kd_theta: float = 8.0
    kp_psi: float = 20.0
    ki_psi: float = 1.0
    kd_psi: float = 5.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.kp_z, self.ki_z, self.kd_z,
            self.kp_phi, self.ki_phi, self.kd_phi,
            self.kp_theta, self.ki_theta, self.kd_theta,
            self.kp_psi, self.ki_psi, self.kd_psi,
        ], dtype=float)

    @classmethod
    def from_array(cls, gains: np.ndarray) -> "PIDGains":
        gains = np.asarray(gains, dtype=float).flatten()
        if gains.size != 12:
            raise ValueError(f"Expected 12 gains, got {gains.size}")
        return cls(*gains)

    def to_dict(self) -> dict:
        return asdict(self)


class PIDController:
    """Cascaded PID controller with gravity feed-forward."""

    def __init__(self,
                 gains: PIDGains,
                 dt: float = 0.01,
                 mass: float = 1.0,
                 gravity: float = 9.81,
                 integrator_limit: float = 5.0):
        self.gains = gains
        self.dt = dt
        self.mass = mass
        self.gravity = gravity
        self.integrator_limit = integrator_limit

        self._integrators = np.zeros(4)
        self._prev_errors = np.zeros(4)
        self._prev_derivative = np.zeros(4)

    def reset(self):
        self._integrators = np.zeros(4)
        self._prev_errors = np.zeros(4)
        self._prev_derivative = np.zeros(4)

    def compute(self,
                references: np.ndarray,
                states: np.ndarray) -> np.ndarray:
        """
        Compute [U1, U2, U3, U4].

        U1 is thrust (>= 0), U2..U4 are torques.
        """
        references = np.asarray(references, dtype=float).flatten()
        states = np.asarray(states, dtype=float).flatten()

        errors = references - states

        # Derivative
        raw_deriv = (errors - self._prev_errors) / self.dt
        # Low-pass filter to avoid noise
        derivative = 0.3 * raw_deriv + 0.7 * self._prev_derivative

        # Integral with anti-windup
        self._integrators += errors * self.dt
        self._integrators = np.clip(self._integrators,
                                    -self.integrator_limit,
                                    self.integrator_limit)

        kp = np.array([self.gains.kp_z, self.gains.kp_phi,
                       self.gains.kp_theta, self.gains.kp_psi])
        ki = np.array([self.gains.ki_z, self.gains.ki_phi,
                       self.gains.ki_theta, self.gains.ki_psi])
        kd = np.array([self.gains.kd_z, self.gains.kd_phi,
                       self.gains.kd_theta, self.gains.kd_psi])

        u = kp * errors + ki * self._integrators + kd * derivative

        # Gravity feed-forward on thrust channel
        u[0] += self.mass * self.gravity

        self._prev_errors = errors.copy()
        self._prev_derivative = derivative.copy()

        # Saturation
        u[0] = np.clip(u[0], 0.0, 30.0)      # U1: 0..30 N
        u[1:4] = np.clip(u[1:4], -5.0, 5.0)  # U2..U4: torques

        return u