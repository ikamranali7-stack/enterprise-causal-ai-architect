import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML

def run_churn_causal_analysis():
    print("Initializing Causal Churn Prevention Pipeline...")
    
    # Synthetic Enterprise Churn Data
    np.random.seed(42)
    n = 2000
    
    # Confounders: Tenure, Usage Score, Support Tickets
    tenure = np.random.gamma(shape=2, scale=12, size=n)
    usage_score = np.random.beta(5, 2, size=n)
    support_tickets = np.random.poisson(lam=2, size=n)
    
    X = np.column_stack([tenure, usage_score, support_tickets])
    
    # Treatment assignment (Propensity depends on support tickets and usage)
    propensity = 1 / (1 + np.exp(-(0.5 * support_tickets - 1.0 * usage_score)))
    T = np.random.binomial(1, propensity)
    
    # Outcome: Retention probability (Intervention helps moderate users most)
    base_retention = 0.5 + 0.02 * tenure + 0.3 * usage_score - 0.05 * support_tickets
    treatment_effect = 0.25 * (usage_score < 0.7).astype(float) # More effective for mid-tier users
    Y = np.random.binomial(1, np.clip(base_retention + treatment_effect * T, 0, 1))
    
    # Fit Heterogeneous Treatment Effect Model (Causal Forest DML)
    model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=42),
        model_t=RandomForestRegressor(n_estimators=100, random_state=42),
        n_estimators=100,
        random_state=42
    )
    
    model.fit(Y, T, X=X)
    cate = model.effect(X)
    
    print(f"Average Treatment Effect (ATE) of Intervention: {np.mean(cate):.4f}")
    print(f"Max Individual CATE (High Uplift Segment): {np.max(cate):.4f}")
    print(f"Min Individual CATE (Negative/No Impact Segment): {np.min(cate):.4f}")

if __name__ == '__main__':
    run_churn_causal_analysis()
