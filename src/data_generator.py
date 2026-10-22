import numpy as np
import pandas as pd
from pathlib import Path

def generate_enterprise_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Simulates enterprise customer subscription data with confounding 
    to benchmark causal inference models (DoWhy / EconML).
    """
    np.random.seed(random_state)
    
    # Confounders (Customer Profile & Enterprise Characteristics)
    company_size = np.random.gamma(shape=2.0, scale=50.0, size=n_samples) # Number of employees
    prior_usage = np.random.normal(loc=50.0, scale=15.0, size=n_samples) # Daily active hours
    budget_index = np.random.uniform(low=0.5, high=5.0, size=n_samples)
    
    # Treatment Assignment (e.g., Targeted Enterprise Discount / Pricing Intervention)
    # Confounding: Higher budget or prior usage makes assignment more likely
    propensity_score_logit = -2.0 + 0.01 * company_size + 0.03 * prior_usage + 0.5 * budget_index
    propensity = 1 / (1 + np.exp(-propensity_score_logit))
    treatment = np.random.binomial(1, propensity)
    
    # Heterogeneous Treatment Effect (HTE): Discount increases retention value differently per segment
    base_effect = 15.0
    heterogeneous_effect = base_effect + 2.0 * budget_index - 0.05 * prior_usage
    
    # Outcome (Customer Lifetime Value / Annual Recurring Revenue)
    noise = np.random.normal(loc=0.0, scale=10.0, size=n_samples)
    arr = (
        100.0 
        + 1.5 * company_size 
        + 2.0 * prior_usage 
        + 12.0 * budget_index 
        + treatment * heterogeneous_effect 
        + noise
    )
    
    df = pd.DataFrame({
        'company_size': company_size,
        'prior_usage': prior_usage,
        'budget_index': budget_index,
        'treatment': treatment,
        'arr': arr
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic enterprise dataset...")
    df = generate_enterprise_data()
    
    # Save to data/raw/
    output_path = Path("data/raw/enterprise_subscriptions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path} with shape {df.shape}")