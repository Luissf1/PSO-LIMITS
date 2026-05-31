"""
Estudio de límites de velocidad en PSO para optimización PID de cuadrotor
Tesis Doctorado - Luis Adrián Silva Reyes
"""

import argparse
from pathlib import Path
from src.experiments.limit_study import VelocityLimitExperiment
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Estudio de límites de velocidad en PSO para PID"
    )
    parser.add_argument(
        '--runs', type=int, default=30,
        help='Número de ejecuciones por estrategia'
    )
    parser.add_argument(
        '--scenario', type=int, default=1, choices=[1,2,3,4,5],
        help='Escenario de prueba (1-5)'
    )
    parser.add_argument(
        '--output', type=str, default='results',
        help='Directorio de salida'
    )
    
    args = parser.parse_args()
    
    # Definir escenarios
    scenarios = {
        1: np.array([1.0, 0.0, 0.0, 0.0]),        # Despegue simple
        2: np.array([1.5, 0.1, -0.1, 0.0]),       # Inclinación inicial
        3: np.array([2.0, -0.2, 0.2, 0.0]),       # Mayor altura
        4: np.array([1.0, 0.0, 0.0, np.pi/4]),    # Con yaw
        5: np.array([0.5, -0.1, -0.1, -np.pi/6])  # Descenso
    }
    
    print("=" * 80)
    print("ESTUDIO DE LÍMITES DE VELOCIDAD EN PSO PARA PID")
    print("=" * 80)
    print(f"Escenario: {args.scenario} - Referencia: {scenarios[args.scenario]}")
    print(f"Ejecuciones por estrategia: {args.runs}")
    print(f"Directorio de salida: {args.output}")
    print("=" * 80)
    
    # Crear experimento
    experiment = VelocityLimitExperiment(output_dir=args.output)
    
    # Ejecutar experimento
    results = experiment.run_experiment(
        n_runs=args.runs,
        reference=scenarios[args.scenario],
        save_results=True
    )
    
    # Mostrar recomendaciones
    print("\n" + "=" * 80)
    print("RECOMENDACIONES")
    print("=" * 80)
    
    best = results.iloc[0]
    print(f"\nMejor estrategia encontrada: {best['strategy']}")
    print(f"  Fitness promedio: {best['mean_fitness']:.4f} ± {best['std_fitness']:.4f}")
    print(f"  Tasa de éxito: {best['success_rate']:.1f}%")
    
    if 'fraction' in best['strategy']:
        print(f"\n✓ Recomendación: Usar límite por fracción con factor {best['factor']}")
    elif 'dynamic' in best['strategy']:
        print(f"\n✓ Recomendación: Usar límite dinámico con factor inicial {best['factor']}")
    elif 'clerc' in best['strategy']:
        print("\n✓ Recomendación: Usar constricción de Clerc")
    
    print("\n✅ Experimento completado")
    print(f"📁 Resultados guardados en: {args.output}/")


if __name__ == "__main__":
    main()