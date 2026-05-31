"""
Modelo dinámico del cuadrotor
Basado en ecuaciones de Newton-Euler
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.integrate import solve_ivp


@dataclass
class QuadrotorParameters:
    """Parámetros físicos del cuadrotor"""
    mass: float = 1.0  # kg
    gravity: float = 9.81  # m/s²
    inertia_xx: float = 0.1  # kg·m²
    inertia_yy: float = 0.1  # kg·m²
    inertia_zz: float = 0.2  # kg·m²
    arm_length: float = 0.25  # m
    thrust_factor: float = 3.13e-5  # N·s²/rad²
    drag_factor: float = 7.5e-7  # N·m·s²/rad²
    aerodynamic_drag: float = 0.1  # N·s/m


class QuadrotorDynamics:
    """Modelo dinámico completo del cuadrotor"""
    
    def __init__(self, params: Optional[QuadrotorParameters] = None):
        self.params = params or QuadrotorParameters()
        self.state_names = ['x', 'y', 'z', 'phi', 'theta', 'psi',
                           'vx', 'vy', 'vz', 'p', 'q', 'r']
    
    def dynamics(self, t: float, state: np.ndarray, 
                 control_inputs: np.ndarray,
                 disturbances: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Ecuaciones de movimiento del cuadrotor
        
        Args:
            t: Tiempo actual (s)
            state: Vector de estado [posición(3), ángulos(3), velocidad(3), velocidad angular(3)]
            control_inputs: Entradas de control [U1, U2, U3, U4]
            disturbances: Perturbaciones externas
            
        Returns:
            Derivada del estado
        """
        # Desempaquetar estado
        x, y, z, phi, theta, psi = state[0:6]
        vx, vy, vz, p, q, r = state[6:12]
        
        # Extraer controles
        U1, U2, U3, U4 = control_inputs
        
        # Matriz de rotación (cuerpo a inercial)
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        
        # Fuerzas en el sistema inercial
        Fx = (s_psi * s_phi + c_psi * s_theta * c_phi) * U1
        Fy = (-c_psi * s_phi + s_psi * s_theta * c_phi) * U1
        Fz = (c_theta * c_phi) * U1
        
        # Aceleraciones lineales
        ax = Fx / self.params.mass - self.params.aerodynamic_drag * vx
        ay = Fy / self.params.mass - self.params.aerodynamic_drag * vy
        az = Fz / self.params.mass - self.params.gravity - self.params.aerodynamic_drag * vz
        
        # Aceleraciones angulares
        Ixx, Iyy, Izz = self.params.inertia_xx, self.params.inertia_yy, self.params.inertia_zz
        l = self.params.arm_length
        
        p_dot = (l * U2 + q * r * (Iyy - Izz)) / Ixx
        q_dot = (l * U3 + p * r * (Izz - Ixx)) / Iyy
        r_dot = (U4 + p * q * (Ixx - Iyy)) / Izz
        
        # Aplicar perturbaciones si existen
        if disturbances is not None:
            ax += disturbances[0]
            ay += disturbances[1]
            az += disturbances[2]
            p_dot += disturbances[3]
            q_dot += disturbances[4]
            r_dot += disturbances[5]
        
        return np.array([vx, vy, vz, p, q, r, ax, ay, az, p_dot, q_dot, r_dot])
    
    def simulate(self, control_law, initial_state: np.ndarray,
                t_span: Tuple[float, float] = (0, 10),
                t_eval: Optional[np.ndarray] = None,
                **kwargs) -> dict:
        """
        Simular la respuesta del cuadrotor
        
        Args:
            control_law: Función que devuelve entradas de control
            initial_state: Estado inicial
            t_span: Intervalo de tiempo (inicio, fin)
            t_eval: Tiempos para evaluar la solución
            
        Returns:
            Diccionario con resultados de simulación
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 2000)
        
        def wrapped_dynamics(t, state):
            control_inputs = control_law(t, state)
            return self.dynamics(t, state, control_inputs)
        
        solution = solve_ivp(
            wrapped_dynamics, t_span, initial_state,
            t_eval=t_eval, method='RK45',
            rtol=kwargs.get('rtol', 1e-6),
            atol=kwargs.get('atol', 1e-9),
            max_step=kwargs.get('max_step', 0.01)
        )
        
        return {
            'success': solution.success,
            't': solution.t,
            'state': solution.y,
            'message': solution.message
        }