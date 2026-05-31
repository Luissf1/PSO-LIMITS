"""
Métricas de desempeño para sistemas de control
"""

import numpy as np
from typing import Dict


class PerformanceMetrics:
    """Calculadora de métricas de desempeño"""
    
    @staticmethod
    def settling_time(t: np.ndarray, y: np.ndarray, 
                     reference: float, tolerance: float = 0.02) -> float:
        """
        Calcular tiempo de establecimiento
        
        Args:
            t: Vector de tiempo
            y: Señal de salida
            reference: Valor de referencia
            tolerance: Tolerancia (default 2%)
        """
        error = np.abs(y - reference)
        band = tolerance * np.abs(reference) if reference != 0 else tolerance
        
        # Buscar último punto donde el error sale de la banda
        inside_band = error <= band
        if not np.any(inside_band):
            return t[-1]
        
        # Encontrar el punto desde donde se mantiene dentro
        for i in range(len(t)-1, -1, -1):
            if inside_band[i] and np.all(inside_band[i:]):
                return t[i]
        
        return t[-1]
    
    @staticmethod
    def overshoot(t: np.ndarray, y: np.ndarray, 
                 reference: float) -> float:
        """Calcular sobrepico porcentual"""
        if reference <= 0:
            return 0.0
        
        y_norm = y / reference
        peak = np.max(y_norm)
        
        if peak > 1.0:
            return (peak - 1.0) * 100
        return 0.0
    
    @staticmethod
    def rise_time(t: np.ndarray, y: np.ndarray,
                 reference: float, 
                 percent_low: float = 0.1,
                 percent_high: float = 0.9) -> float:
        """Calcular tiempo de subida"""
        if reference <= 0:
            return t[-1]
        
        y_norm = y / reference
        t_low = None
        t_high = None
        
        for i, val in enumerate(y_norm):
            if t_low is None and val >= percent_low:
                t_low = t[i]
            if t_high is None and val >= percent_high:
                t_high = t[i]
                break
        
        if t_low is not None and t_high is not None:
            return t_high - t_low
        return t[-1]
    
    @staticmethod
    def rmse(t: np.ndarray, y: np.ndarray, reference: float) -> float:
        """Root Mean Square Error"""
        error = y - reference
        return np.sqrt(np.mean(error**2))
    
    @staticmethod
    def iae(t: np.ndarray, y: np.ndarray, reference: float) -> float:
        """Integral of Absolute Error"""
        error = np.abs(y - reference)
        return np.trapz(error, t)
    
    @staticmethod
    def itae(t: np.ndarray, y: np.ndarray, reference: float) -> float:
        """Integral of Time-weighted Absolute Error"""
        error = np.abs(y - reference)
        return np.trapz(t * error, t)
    
    @staticmethod
    def control_effort(control_signals: np.ndarray) -> float:
        """Esfuerzo de control (energía total)"""
        return np.sum(control_signals**2)
    
    @staticmethod
    def compute_all(t: np.ndarray, y: np.ndarray, 
                   reference: float,
                   tol_settling: float = 0.02) -> Dict[str, float]:
        """Calcular todas las métricas"""
        
        ts = PerformanceMetrics.settling_time(t, y, reference, tol_settling)
        mp = PerformanceMetrics.overshoot(t, y, reference)
        tr = PerformanceMetrics.rise_time(t, y, reference)
        rmse_val = PerformanceMetrics.rmse(t, y, reference)
        iae_val = PerformanceMetrics.iae(t, y, reference)
        itae_val = PerformanceMetrics.itae(t, y, reference)
        
        # Normalizar (valores típicos)
        return {
            'settling_time': ts,
            'settling_time_norm': min(ts / 5.0, 1.0),
            'overshoot': mp,
            'overshoot_norm': min(mp / 30.0, 1.0),
            'rise_time': tr,
            'rmse': rmse_val,
            'rmse_norm': min(rmse_val / 2.0, 1.0),
            'iae': iae_val,
            'itae': itae_val,
            'control_effort_norm': 0.5,  # Placeholder
            'final_error': np.abs(y[-1] - reference)
        }