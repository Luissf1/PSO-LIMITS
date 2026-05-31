"""
Algoritmo de Optimización por Enjambre de Partículas (PSO)
Con enfoque en estudio de límites de velocidad
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple
from collections import defaultdict


@dataclass
class PSOConfig:
    """Configuración del algoritmo PSO"""
    # Parámetros principales
    num_particles: int = 30
    max_iterations: int = 100
    
    # Parámetros de inercia
    w_min: float = 0.4
    w_max: float = 0.9
    
    # Coeficientes de aceleración
    c1: float = 2.0  # Cognitivo
    c2: float = 2.0  # Social
    
    # Límites de velocidad - ¡Esta es la clave del estudio!
    velocity_limit_factor: float = 0.2  # Fracción del rango de búsqueda
    # Alternativas: 
    # - 'dynamic' : Límite decreciente
    # - 'clerc'   : Constricción de Clerc
    velocity_limit_strategy: str = 'fraction'  # 'fraction', 'dynamic', 'clerc', 'fixed'
    
    # Tolerancia para convergencia temprana
    tolerance: float = 1e-6
    patience: int = 20


class Particle:
    """Partícula individual del enjambre"""
    
    def __init__(self, position: np.ndarray, velocity: np.ndarray,
                 var_min: np.ndarray, var_max: np.ndarray):
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.var_min = var_min
        self.var_max = var_max
        
        # Mejor personal
        self.best_position = position.copy()
        self.best_fitness = float('inf')
        
        # Fitness actual
        self.fitness = float('inf')
    
    def update_position(self):
        """Actualizar posición y mantener dentro de límites"""
        self.position += self.velocity
        self.position = np.clip(self.position, self.var_min, self.var_max)
    
    def evaluate(self, fitness_function: Callable, **kwargs):
        """Evaluar la partícula"""
        self.fitness = fitness_function(self.position, **kwargs)
        
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()
            return True  # Mejora encontrada
        return False


class VelocityLimitStrategy:
    """Estrategias para límites de velocidad en PSO"""
    
    @staticmethod
    def fraction_strategy(particle: Particle, config: PSOConfig, 
                         iteration: int) -> np.ndarray:
        """Límite fracción del rango de búsqueda"""
        vel_max = config.velocity_limit_factor * (particle.var_max - particle.var_min)
        return np.clip(particle.velocity, -vel_max, vel_max)
    
    @staticmethod
    def dynamic_strategy(particle: Particle, config: PSOConfig,
                        iteration: int) -> np.ndarray:
        """Límite dinámico que decrece con iteraciones"""
        max_iter = config.max_iterations
        factor = config.velocity_limit_factor * (1 - iteration / max_iter)
        vel_max = factor * (particle.var_max - particle.var_min)
        return np.clip(particle.velocity, -vel_max + 1e-8, vel_max - 1e-8)
    
    @staticmethod
    def clerc_strategy(particle: Particle, config: PSOConfig,
                      iteration: int) -> np.ndarray:
        """Constricción de Clerc (sin límite explícito)"""
        # Parámetros de constricción de Clerc
        phi = config.c1 + config.c2
        kappa = 1.0
        chi = 2 * kappa / abs(2 - phi - np.sqrt(phi**2 - 4*phi))
        
        # Aplicar constricción en lugar de clip
        return particle.velocity * chi
    
    @staticmethod
    def fixed_strategy(particle: Particle, config: PSOConfig,
                      iteration: int) -> np.ndarray:
        """Límite fijo (valor absoluto)"""
        vel_max = np.ones_like(particle.position) * config.velocity_limit_factor
        return np.clip(particle.velocity, -vel_max, vel_max)


class ParticleSwarmOptimizer:
    """Optimizador PSO con límites de velocidad configurables"""
    
    def __init__(self, config: PSOConfig, var_min: np.ndarray, var_max: np.ndarray):
        self.config = config
        self.var_min = var_min
        self.var_max = var_max
        
        # Seleccionar estrategia de límite
        self._strategies = {
            'fraction': VelocityLimitStrategy.fraction_strategy,
            'dynamic': VelocityLimitStrategy.dynamic_strategy,
            'clerc': VelocityLimitStrategy.clerc_strategy,
            'fixed': VelocityLimitStrategy.fixed_strategy
        }
        
        self.velocity_strategy = self._strategies.get(
            config.velocity_limit_strategy,
            VelocityLimitStrategy.fraction_strategy
        )
        
        # Estado del optimizador
        self.swarm: List[Particle] = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.convergence_history: List[float] = []
        self.velocity_history: List[float] = []  # Para análisis
        
        # Semilla para reproducibilidad
        self._rng = np.random.RandomState(42)
    
    def initialize_swarm(self):
        """Inicializar el enjambre"""
        self.swarm = []
        
        for _ in range(self.config.num_particles):
            # Posición inicial aleatoria
            position = self._rng.uniform(self.var_min, self.var_max)
            
            # Velocidad inicial aleatoria
            vel_range = 0.1 * (self.var_max - self.var_min)
            velocity = self._rng.uniform(-vel_range, vel_range)
            
            particle = Particle(position, velocity, self.var_min, self.var_max)
            self.swarm.append(particle)
    
    def update_velocities(self, iteration: int):
        """Actualizar velocidades según regla PSO"""
        for particle in self.swarm:
            # Inercia dinámica
            w = self.config.w_max - (self.config.w_max - self.config.w_min) * \
                (iteration / self.config.max_iterations)
            
            # Componentes cognitivo y social
            r1, r2 = self._rng.rand(), self._rng.rand()
            cognitive = self.config.c1 * r1 * (particle.best_position - particle.position)
            social = self.config.c2 * r2 * (self.global_best_position - particle.position)
            
            # Actualizar velocidad
            particle.velocity = w * particle.velocity + cognitive + social
            
            # Registrar velocidades para análisis
            self.velocity_history.append(np.mean(np.abs(particle.velocity)))
            
            # Aplicar estrategia de límite de velocidad
            particle.velocity = self.velocity_strategy(particle, self.config, iteration)
            
            # Actualizar posición
            particle.update_position()
    
    def optimize(self, fitness_function: Callable, 
                 fitness_kwargs: Optional[Dict] = None,
                 verbose: bool = True) -> Dict:
        """
        Ejecutar optimización PSO
        
        Args:
            fitness_function: Función de fitness a minimizar
            fitness_kwargs: Argumentos adicionales para fitness_function
            verbose: Mostrar progreso
            
        Returns:
            Diccionario con resultados
        """
        fitness_kwargs = fitness_kwargs or {}
        
        # Inicializar
        self.initialize_swarm()
        self.convergence_history = []
        self.velocity_history = []
        
        # Evaluación inicial
        for particle in self.swarm:
            particle.evaluate(fitness_function, **fitness_kwargs)
            
            if particle.fitness < self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
        
        self.convergence_history.append(self.global_best_fitness)
        
        # Bucle principal
        stagnation_counter = 0
        best_so_far = self.global_best_fitness
        
        for iteration in range(1, self.config.max_iterations + 1):
            # Actualizar velocidades y posiciones
            self.update_velocities(iteration)
            
            # Evaluar nueva generación
            improvement = False
            for particle in self.swarm:
                improved = particle.evaluate(fitness_function, **fitness_kwargs)
                
                if improved and particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
                    improvement = True
            
            self.convergence_history.append(self.global_best_fitness)
            
            # Verificar convergencia temprana
            if abs(best_so_far - self.global_best_fitness) < self.config.tolerance:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                best_so_far = self.global_best_fitness
            
            if stagnation_counter >= self.config.patience:
                if verbose:
                    print(f"Convergencia temprana en iteración {iteration}")
                break
            
            # Mostrar progreso
            if verbose and iteration % 20 == 0:
                print(f"Iteración {iteration:4d}/{self.config.max_iterations}: "
                      f"Best Fitness = {self.global_best_fitness:.6f}")
        
        return {
            'best_position': self.global_best_position,
            'best_fitness': self.global_best_fitness,
            'convergence': self.convergence_history,
            'velocity_magnitudes': self.velocity_history,
            'num_iterations': len(self.convergence_history)
        }