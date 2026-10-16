"""
Quadrotor dynamic model based on Newton-Euler equations.

The simulation uses fixed-step integration to keep the PID controller
and the dynamics in sync. This avoids the discontinuity problems of
variable-step integrators like solve_ivp.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class QuadrotorParameters:
    """Physical parameters of a small quadrotor."""
    mass: float = 1.0
    gravity: float = 9.81
    inertia_xx: float = 0.02
    inertia_yy: float = 0.02
    inertia_zz: float = 0.04
    arm_length: float = 0.25
    aerodynamic_drag: float = 0.05


class QuadrotorDynamics:
    """Nonlinear quadrotor model with fixed-step simulation."""

    def __init__(self, params: Optional[QuadrotorParameters] = None):
        self.params = params or QuadrotorParameters()

    def dynamics(self,
                 state: np.ndarray,
                 control_inputs: np.ndarray) -> np.ndarray:
        """
        Compute the state derivative given the current state and controls.

        State: [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
        Controls: [U1 (thrust in N), U2, U3, U4 (torques in N*m)]
        """
        state = np.asarray(state, dtype=float).flatten()
        control_inputs = np.asarray(control_inputs, dtype=float).flatten()

        x, y, z, phi, theta, psi = state[0:6]
        vx, vy, vz, p, q, r = state[6:12]
        U1, U2, U3, U4 = control_inputs

        # Safety: prevent NaN and explosion
        if not np.isfinite(state).all():
            return np.zeros(12)
        if not np.isfinite(control_inputs).all():
            return np.zeros(12)

        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        # Thrust in body frame: [0, 0, U1]
        # Rotated to inertial frame
        Fx = (c_psi * s_theta * c_phi + s_psi * s_phi) * U1
        Fy = (s_psi * s_theta * c_phi - c_psi * s_phi) * U1
        Fz = (c_theta * c_phi) * U1

        m = self.params.mass
        g = self.params.gravity
        b = self.params.aerodynamic_drag

        # Linear accelerations
        ax = Fx / m - b * vx
        ay = Fy / m - b * vy
        az = Fz / m - g - b * vz

        # Angular accelerations
        Ixx = self.params.inertia_xx
        Iyy = self.params.inertia_yy
        Izz = self.params.inertia_zz
        l = self.params.arm_length

        p_dot = (l * U2 + q * r * (Iyy - Izz)) / Ixx
        q_dot = (l * U3 + p * r * (Izz - Ixx)) / Iyy
        r_dot = (U4 + p * q * (Ixx - Iyy)) / Izz

        return np.array([vx, vy, vz, p, q, r,
                         ax, ay, az, p_dot, q_dot, r_dot])

    def simulate(self,
                 control_law: Callable,
                 initial_state: np.ndarray,
                 t_span: Tuple[float, float] = (0.0, 10.0),
                 dt: float = 0.01) -> dict:
        """
        Fixed-step simulation with RK4 integration.

        Parameters
        ----------
        control_law : callable
            Function control_law(t, state) -> [U1, U2, U3, U4].
        initial_state : np.ndarray
            Initial 12-dim state.
        t_span : tuple
            (t0, tf).
        dt : float
            Integration step (should match the controller dt).

        Returns
        -------
        dict
            {'t': ..., 'state': ..., 'success': bool}
        """
        initial_state = np.asarray(initial_state, dtype=float).flatten()
        t0, tf = t_span
        n_steps = int((tf - t0) / dt) + 1
        t = np.linspace(t0, tf, n_steps)
        states = np.zeros((12, n_steps))
        states[:, 0] = initial_state

        y = initial_state.copy()
        for i in range(1, n_steps):
            ti = t[i - 1]
            u = control_law(ti, y)

            # RK4 integration
            k1 = self.dynamics(y, u)
            k2 = self.dynamics(y + 0.5 * dt * k1, u)
            k3 = self.dynamics(y + 0.5 * dt * k2, u)
            k4 = self.dynamics(y + dt * k3, u)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Safety check
            if not np.isfinite(y).all() or np.any(np.abs(y) > 1e4):
                states[:, i] = y
                return {
                    "success": False,
                    "t": t[:i + 1],
                    "state": states[:, :i + 1],
                    "message": "Diverged",
                }

            states[:, i] = y

        return {
            "success": True,
            "t": t,
            "state": states,
            "message": "OK",
        }