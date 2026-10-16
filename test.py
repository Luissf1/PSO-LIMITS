import numpy as np
from src.utils.fitness import QuadrotorFitness

f = QuadrotorFitness(reference=np.array([1.0, 0.0, 0.0, 0.0]))

# Gains progresivamente mejores (más agresivos pero controlados)
tests = {
    "V1_very_low": np.array([5.0, 0.1, 1.0, 5.0, 0.1, 1.0, 5.0, 0.1, 1.0, 3.0, 0.05, 0.5]),
    "V2_low":      np.array([10.0, 0.5, 3.0, 10.0, 0.5, 3.0, 10.0, 0.5, 3.0, 5.0, 0.2, 1.5]),
    "V3_medium":   np.array([15.0, 1.0, 5.0, 15.0, 1.0, 5.0, 15.0, 1.0, 5.0, 8.0, 0.4, 2.5]),
    "V4_good":     np.array([20.0, 2.0, 8.0, 20.0, 2.0, 8.0, 20.0, 2.0, 8.0, 12.0, 0.8, 4.0]),
    "V5_high":     np.array([25.0, 3.0, 10.0, 25.0, 3.0, 10.0, 25.0, 3.0, 10.0, 15.0, 1.0, 5.0]),
    "V6_very_high":np.array([30.0, 5.0, 15.0, 30.0, 5.0, 15.0, 30.0, 5.0, 15.0, 20.0, 1.5, 7.0]),
}

for name, gains in tests.items():
    score = f.evaluate(gains)
    print(f"{name:15s}: {score:.4f}")