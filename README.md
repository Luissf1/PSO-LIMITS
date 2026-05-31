# Estudio de Límites de Velocidad en PSO para Optimización PID de Cuadrotor

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Research-Tesis-red.svg)]()

## 📋 Descripción

Este repositorio contiene el código fuente para el estudio experimental del efecto de diferentes estrategias de límites de velocidad en el algoritmo de Optimización por Enjambre de Partículas (PSO) aplicado a la sintonización de controladores PID para un cuadrotor.

### 🎯 Objetivo del Estudio

Evaluar cómo diferentes estrategias de límites de velocidad afectan:
- Tasa de convergencia
- Calidad de la solución óptima
- Robustez del algoritmo
- Estabilidad del controlador resultante

### 🔬 Estrategias Implementadas

| Estrategia | Descripción | Ecuación |
|------------|-------------|----------|
| Fracción fija | Límite = α × (VarMax - VarMin) | v_max = α·Δx |
| Dinámica | Límite decreciente con iteraciones | v_max(t) = α·(1 - t/T)·Δx |
| Constricción Clerc | Factor de constricción | v ← χ·v |
| Límite absoluto | Valor máximo fijo | \|v\| ≤ V_max |

## 🚀 Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/tesis-limites-pso.git
cd tesis-limites-pso

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt