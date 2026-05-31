"""
Controlador PID para cuadrotor
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PIDGains:
    """Ganancias PID para los 4 canales"""
    # Altura (z)
    kp_z: float = 1.0
    ki_z: float = 0.1
    kd_z: float = 0.5
    
    # Roll (phi)
    kp_phi: float = 1.0
    ki_phi: float = 0.1
    kd_phi: float = 0.5
    
    # Pitch (theta)
    kp_theta: float = 1.0
    ki_theta: float = 0.1
    kd_theta: float = 0.5
    
    # Yaw (psi)
    kp_psi: float = 1.0
    ki_psi: float = 0.1
    kd_psi: float = 0.5
    
    def to_array(self) -> np.ndarray:
        """Convertir a array numpy"""
        return np.array([
            self.kp_z, self.ki_z, self.kd_z,
            self.kp_phi, self.ki_phi, self.kd_phi,
            self.kp_theta, self.ki_theta, self.kd_theta,
            self.kp_psi, self.ki_psi, self.kd_psi
        ])
    
    @classmethod
    def from_array(cls, gains: np.ndarray) -> 'PIDGains':
        """Crear desde array numpy"""
        return cls(
            kp_z=gains[0], ki_z=gains[1], kd_z=gains[2],
            kp_phi=gains[3], ki_phi=gains[4], kd_phi=gains[5],
            kp_theta=gains[6], ki_theta=gains[7], kd_theta=gains[8],
            kp_psi=gains[9], ki_psi=gains[10], kd_psi=gains[11]
        )


class PIDController:
    """Controlador PID para cuadrotor"""
    
    def __init__(self, gains: PIDGains, dt: float = 0.01):
        self.gains = gains
        self.dt = dt
        self._integrators = np.zeros(4)  # [z, phi, theta, psi]
        self._prev_errors = np.zeros(4)
    
    def compute(self, references: np.ndarray, states: np.ndarray) -> np.ndarray:
        """
        Calcular entradas de control
        
        Args:
            references: Referencias [z_des, phi_des, theta_des, psi_des]
            states: Estado actual [z, phi, theta, psi] desde el modelo
            
        Returns:
            Entradas de control [U1, U2, U3, U4]
        """
        z, phi, theta, psi = states
        z_des, phi_des, theta_des, psi_des = references
        
        # Calcular errores
        errors = np.array([
            z_des - z,
            phi_des - phi,
            theta_des - theta,
            psi_des - psi
        ])
        
        # Derivada del error (diferencias finitas)
        error_deriv = (errors - self._prev_errors) / self.dt
        
        # Actualizar integradores (con anti-windup)
        self._integrators += errors * self.dt
        self._integrators = np.clip(self._integrators, -10, 10)
        
        # Acciones PID
        gains_array = np.array([
            self.gains.kp_z, self.gains.kp_phi, 
            self.gains.kp_theta, self.gains.kp_psi
        ])
        gains_i = np.array([
            self.gains.ki_z, self.gains.ki_phi,
            self.gains.ki_theta, self.gains.ki_psi
        ])
        gains_d = np.array([
            self.gains.kd_z, self.gains.kd_phi,
            self.gains.kd_theta, self.gains.kd_psi
        ])
        
        u = gains_array * errors + gains_i * self._integrators + gains_d * error_deriv
        
        # Guardar para próxima iteración
        self._prev_errors = errors
        
        # Limitar controles
        u[0] = np.clip(u[0], 0, 30)  # U1: fuerza de empuje (positiva)
        u[1:4] = np.clip(u[1:4], -10, 10)  # Momentos
        
        return u
    
    def reset(self):
        """Resetear estado interno del controlador"""
        self._integrators = np.zeros(4)
        self._prev_errors = np.zeros(4)