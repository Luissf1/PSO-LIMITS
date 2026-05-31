"""
Experimento: Estudio del efecto de límites de velocidad en PSO para PID
"""

import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

from ..models.quadrotor import QuadrotorDynamics, QuadrotorParameters
from ..controllers.pid import PIDController, PIDGains
from ..optimizers.pso import PSOConfig, ParticleSwarmOptimizer
from ..utils.metrics import PerformanceMetrics


# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PIDFitness:
    """Función de fitness para optimización PID"""
    
    def __init__(self, dynamics: QuadrotorDynamics, 
                 reference: np.ndarray,
                 t_span: Tuple[float, float] = (0, 10),
                 weights: Dict[str, float] = None):
        self.dynamics = dynamics
        self.reference = reference
        self.t_span = t_span
        self.t_eval = np.linspace(t_span[0], t_span[1], 2000)
        
        # Pesos para diferentes métricas
        self.weights = weights or {
            'settling_time': 0.30,
            'overshoot': 0.25,
            'rmse': 0.25,
            'control_effort': 0.20
        }
    
    def __call__(self, gains_array: np.ndarray, **kwargs) -> float:
        """
        Evaluar fitness de un conjunto de ganancias PID
        """
        gains = PIDGains.from_array(gains_array)
        controller = PIDController(gains, dt=0.01)
        
        # Función de control para simulación
        def control_law(t, state):
            # Extraer estados relevantes
            z = state[2]  # Altura
            phi, theta, psi = state[3], state[4], state[5]
            states_pid = np.array([z, phi, theta, psi])
            
            return controller.compute(self.reference, states_pid)
        
        # Simular
        initial_state = np.zeros(12)
        result = self.dynamics.simulate(
            control_law, initial_state, 
            self.t_span, self.t_eval
        )
        
        if not result['success']:
            return 1000.0  # Penalización alta
        
        # Calcular métricas
        metrics = PerformanceMetrics.compute_all(
            result['t'], result['state'][2],  # Altura
            self.reference[0],  # Referencia de altura
            self.tol_settling=0.02
        )
        
        # Calcular fitness ponderado
        fitness = (
            self.weights['settling_time'] * metrics['settling_time_norm'] +
            self.weights['overshoot'] * metrics['overshoot_norm'] +
            self.weights['rmse'] * metrics['rmse_norm'] +
            self.weights['control_effort'] * metrics['control_effort_norm']
        )
        
        # Penalizar si no alcanza referencia
        if metrics['final_error'] > 0.05:
            fitness += 0.5 * (metrics['final_error'] / 0.05)
        
        return fitness


class VelocityLimitExperiment:
    """Experimento para estudiar efecto de límites de velocidad"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración del problema
        self.dynamics = QuadrotorDynamics(QuadrotorParameters())
        
        # Límites de búsqueda para ganancias PID
        self.var_min = np.array([
            0.5, 0.0001, 0.01,   # Kp_z, Ki_z, Kd_z
            0.01, 0.0001, 0.01,  # Kp_phi, Ki_phi, Kd_phi
            0.01, 0.0001, 0.01,  # Kp_theta, Ki_theta, Kd_theta
            0.01, 0.0001, 0.01   # Kp_psi, Ki_psi, Kd_psi
        ])
        
        self.var_max = np.array([
            30.0, 5.0, 15.0,    # Altura
            20.0, 0.5, 5.0,     # Roll
            20.0, 0.5, 5.0,     # Pitch
            20.0, 0.5, 5.0      # Yaw
        ])
        
        # Configuración base de PSO
        self.base_config = PSOConfig(
            num_particles=30,
            max_iterations=100,
            w_min=0.4,
            w_max=0.9,
            c1=2.0,
            c2=2.0,
            tolerance=1e-6,
            patience=15
        )
        
        # Estrategias a probar
        self.strategies = [
            ('fraction_0.05', 'fraction', 0.05),
            ('fraction_0.1', 'fraction', 0.1),
            ('fraction_0.2', 'fraction', 0.2),
            ('fraction_0.5', 'fraction', 0.5),
            ('dynamic_0.2', 'dynamic', 0.2),
            ('clerc', 'clerc', None),
            ('fixed_0.1', 'fixed', 0.1),
            ('fixed_0.5', 'fixed', 0.5),
        ]
    
    def run_experiment(self, n_runs: int = 30, 
                      reference: np.ndarray = None,
                      save_results: bool = True) -> pd.DataFrame:
        """
        Ejecutar experimento completo
        
        Args:
            n_runs: Número de ejecuciones por configuración
            reference: Referencia [z_des, phi_des, theta_des, psi_des]
            save_results: Guardar resultados en archivo
            
        Returns:
            DataFrame con resultados
        """
        if reference is None:
            reference = np.array([1.0, 0.0, 0.0, 0.0])  # Despegue simple
        
        logger.info("=" * 80)
        logger.info("EXPERIMENTO: EFECTO DE LÍMITES DE VELOCIDAD EN PSO")
        logger.info("=" * 80)
        logger.info(f"Referencia: {reference}")
        logger.info(f"Ejecuciones por estrategia: {n_runs}")
        logger.info(f"Estrategias a probar: {len(self.strategies)}")
        logger.info("=" * 80)
        
        # Almacenar resultados
        all_results = []
        
        for strategy_name, strategy_type, factor in self.strategies:
            logger.info(f"\n📊 Probando estrategia: {strategy_name}")
            logger.info(f"   Tipo: {strategy_type}, Factor: {factor}")
            
            # Configurar PSO con esta estrategia
            config = PSOConfig(
                num_particles=self.base_config.num_particles,
                max_iterations=self.base_config.max_iterations,
                w_min=self.base_config.w_min,
                w_max=self.base_config.w_max,
                c1=self.base_config.c1,
                c2=self.base_config.c2,
                velocity_limit_strategy=strategy_type,
                velocity_limit_factor=factor if factor else 0.2,
                tolerance=self.base_config.tolerance,
                patience=self.base_config.patience
            )
            
            # Ejecutar múltiples veces
            strategy_results = []
            
            for run in range(n_runs):
                logger.info(f"   Ejecución {run + 1}/{n_runs}")
                
                # Crear función de fitness
                fitness_func = PIDFitness(
                    self.dynamics, reference,
                    weights={'settling_time': 0.30, 'overshoot': 0.25,
                            'rmse': 0.25, 'control_effort': 0.20}
                )
                
                # Optimizar
                optimizer = ParticleSwarmOptimizer(config, self.var_min, self.var_max)
                result = optimizer.optimize(
                    fitness_func, 
                    fitness_kwargs={},
                    verbose=False
                )
                
                # Guardar resultados
                strategy_results.append({
                    'run': run,
                    'best_fitness': result['best_fitness'],
                    'iterations': result['num_iterations'],
                    'convergence_speed': len(result['convergence']),
                    'final_velocity': result['velocity_magnitudes'][-1] if result['velocity_magnitudes'] else 0,
                    **{f'gain_{i}': result['best_position'][i] for i in range(12)}
                })
                
                logger.info(f"      Fitness: {result['best_fitness']:.4f}, "
                          f"Iter: {result['num_iterations']}")
            
            # Estadísticas de esta estrategia
            df_strategy = pd.DataFrame(strategy_results)
            stats = {
                'strategy': strategy_name,
                'type': strategy_type,
                'factor': factor if factor else 0,
                'mean_fitness': df_strategy['best_fitness'].mean(),
                'std_fitness': df_strategy['best_fitness'].std(),
                'min_fitness': df_strategy['best_fitness'].min(),
                'max_fitness': df_strategy['best_fitness'].max(),
                'mean_iterations': df_strategy['iterations'].mean(),
                'success_rate': (df_strategy['best_fitness'] < 0.5).mean() * 100
            }
            
            all_results.append(stats)
            
            # Guardar resultados detallados de esta estrategia
            if save_results:
                df_strategy.to_csv(
                    self.output_dir / f"results_{strategy_name}.csv",
                    index=False
                )
        
        # Crear DataFrame resumen
        summary_df = pd.DataFrame(all_results)
        summary_df = summary_df.sort_values('mean_fitness')
        
        # Guardar resumen
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_df.to_csv(
                self.output_dir / f"summary_{timestamp}.csv",
                index=False
            )
            
            # Guardar también en Excel con formato
            with pd.ExcelWriter(self.output_dir / f"experiment_{timestamp}.xlsx") as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Guardar detalles de cada estrategia
                for strategy_name, _, _ in self.strategies:
                    file = self.output_dir / f"results_{strategy_name}.csv"
                    if file.exists():
                        df = pd.read_csv(file)
                        df.to_excel(writer, sheet_name=strategy_name[:31], index=False)
        
        # Mostrar mejores resultados
        logger.info("\n" + "=" * 80)
        logger.info("RESULTADOS FINALES")
        logger.info("=" * 80)
        logger.info("\nTop 3 mejores estrategias:")
        for idx, row in summary_df.head(3).iterrows():
            logger.info(f"  {idx+1}. {row['strategy']}: "
                       f"Fitness = {row['mean_fitness']:.4f} ± {row['std_fitness']:.4f}, "
                       f"Éxito = {row['success_rate']:.1f}%")
        
        logger.info("\nPeores estrategias:")
        for idx, row in summary_df.tail(3).iterrows():
            logger.info(f"  {row['strategy']}: "
                       f"Fitness = {row['mean_fitness']:.4f} ± {row['std_fitness']:.4f}")
        
        return summary_df