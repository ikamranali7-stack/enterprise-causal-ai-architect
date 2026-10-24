import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML

def run_pricing_causal_analysis():
    print("Initializing Dynamic Pricing & Elasticity Causal Pipeline...")
    np.random.seed(42)
    n = 2000
    competitor_price = np.random.normal(50, 5, size=n)
    seasonality = np.random.uniform(0.8, 1.2, size=n)
    customer_income = np.random.normal(70, 15, size=n)
    X = np.column_stack([competitor_price, seasonality, customer_income])
    price = 40 + 0.6 * competitor_price + 0.1 * customer_income + np.random.normal(0, 3, size=n)
    base_demand = 200 - 1.5 * price + 20 * seasonality + 0.8 * customer_income
    price_elasticity = -1.5 - 0.02 * customer_income
    demand = base_demand + price_elasticity * price + np.random.normal(0, 5, size=n)
    model = CausalForestDML(model_y=RandomForestRegressor(n_estimators=100, random_state=42), model_t=RandomForestRegressor(n_estimators=100, random_state=42), n_estimators=100, random_state=42)
    model.fit(demand, price, X=X)
    elasticity_effects = model.effect(X)
    print(f"Average Price Elasticity Effect (ATE): {np.mean(elasticity_effects):.4f}")
    print(f"Max Sensitivity Elasticity: {np.min(elasticity_effects):.4f}")
    print(f"Min Sensitivity Elasticity: {np.max(elasticity_effects):.4f}")

if __name__ == '__main__':
    run_pricing_causal_analysis()
