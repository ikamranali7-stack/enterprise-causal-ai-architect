import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML

def run_causal_analysis():
    print("Initializing Causal AI Pipeline...")
    np.random.seed(42)
    n = 1000
    X = np.random.normal(0, 1, size=(n, 3))
    T = np.random.binomial(1, 0.5, size=(n,))
    Y = 2 * T + X[:, 0] + 1.5 * X[:, 1] + np.random.normal(0, 1, size=(n,))
    model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=42),
        model_t=RandomForestRegressor(n_estimators=100, random_state=42),
        random_state=42
    )
    model.fit(Y, T, X=X)
    effect = model.effect(X)
    print(f"Estimated Average Treatment Effect (ATE): {np.mean(effect):.4f}")

if __name__ == '__main__':
    run_causal_analysis()
