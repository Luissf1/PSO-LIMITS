"""
Out-of-sample robustness evaluation of the best tuned PID gains per strategy
under sensor noise, wind disturbance, and +20% mass mismatch.
"""
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.controllers.pid import PIDGains, PIDController
from src.models.quadrotor import QuadrotorDynamics, QuadrotorParameters
from src.utils.fitness import QuadrotorFitness


def evaluate_perturbed(gains_arr: np.ndarray,
                       mass_scale: float = 1.0,
                       noise_std: float = 0.0,
                       wind_force: np.ndarray = None,
                       seed: int = 123) -> float:
    rng = np.random.default_rng(seed)
    params = QuadrotorParameters(mass=1.0 * mass_scale)
    model = QuadrotorDynamics(params)
    gains = PIDGains.from_array(gains_arr)
    controller = PIDController(gains, dt=0.01, mass=1.0, gravity=9.81)
    ref = np.array([1.5, 0.15, -0.15, 0.524])

    def control_law(t, state):
        meas = np.array([state[2], state[3], state[4], state[5]])
        if noise_std > 0.0:
            meas = meas + rng.normal(0.0, noise_std, size=4)
        u = controller.compute(ref, meas)
        if wind_force is not None and 3.0 <= t <= 5.0:
            u = u + wind_force
        return u

    res = model.simulate(control_law, initial_state=np.zeros(12), t_span=(0.0, 10.0), dt=0.01)
    if not res["success"] or not np.isfinite(res["state"]).all():
        return 1.0

    # Reuse QuadrotorFitness combiner on perturbed trajectory
    evaluator = QuadrotorFitness(reference=ref)
    evaluator.model = model
    # Compute tracking metrics directly on res
    t = res["t"]
    states = res["state"]
    if np.any(np.abs(states[:6, :]) > 100):
        return 1.0
    z, phi, theta, psi = states[2], states[3], states[4], states[5]
    err = np.abs(z - ref[0]) + np.abs(phi - ref[1]) + np.abs(theta - ref[2]) + np.abs(psi - ref[3])
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    from src.utils.metrics import PerformanceMetrics
    metrics = {
        "itse": float(_trapz(t * err ** 2, t)),
        "iae": float(_trapz(err, t)),
        "overshoot": max(PerformanceMetrics.overshoot(t, y, r)
                         for y, r in [(z, ref[0]), (phi, ref[1]), (theta, ref[2]), (psi, ref[3])]),
        "settling": max(PerformanceMetrics.settling_time(t, y, r, 0.05)
                        for y, r in [(z, ref[0]), (phi, ref[1]), (theta, ref[2]), (psi, ref[3])]),
    }
    return evaluator._combine(metrics)


def main():
    files = glob.glob("results/**/detailed_*.json", recursive=True)
    latest = max(files, key=lambda p: Path(p).stat().st_ctime)
    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for r in data["results"]:
        best_idx = int(np.argmin(r["all_fitness"]))
        best_gains = np.array(r["all_positions"][best_idx])
        rows.append({
            "strategy": r["name"],
            "nominal_best": r["min_fitness"],
            "mass_+20%": evaluate_perturbed(best_gains, mass_scale=1.20),
            "sensor_noise": evaluate_perturbed(best_gains, noise_std=0.01),
            "wind_gust": evaluate_perturbed(best_gains, wind_force=np.array([2.0, 0.3, -0.3, 0.1])),
        })

    df = pd.DataFrame(rows)
    out = Path(latest).parent / "robustness_validation.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved robustness table to: {out}")


if __name__ == "__main__":
    main()