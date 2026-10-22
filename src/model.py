import pandas as pd
import numpy as np
from pathlib import Path
from dowhy import CausalModel
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def run_causal_pipeline():
    print("Loading enterprise subscription dataset...")
    data_path = Path("data/raw/enterprise_subscriptions.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run data_generator.py first.")
    
    df = pd.read_csv(data_path)
    
    # Define causal graph components
    treatment = "treatment"
    outcome = "arr"
    common_causes = ["company_size", "prior_usage", "budget_index"]
    
    print("\n[1] Initializing DoWhy Causal Model...")
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes
    )
    
    # Step 2: Identify causal effect under unconfoundedness
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print("Causal Estimand Identified successfully.")
    
    # Step 3: Estimate causal effect using Double Machine Learning (EconML)
    print("\n[2] Estimating Heterogeneous Treatment Effects using Double Machine Learning (EconML)...")
    X = df[common_causes]
    T = df[treatment]
    Y = df[outcome]
    
    # Double Machine Learning with discrete treatment handling
    dml_model = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=42),
        model_t=RandomForestClassifier(n_estimators=100, random_state=42),
        discrete_treatment=True,
        random_state=42
    )
    
    dml_model.fit(Y, T, X=X)
    ate = dml_model.effect(X)
    mean_ate = np.mean(ate)
    
    print(f"\n==============================================")
    print(f" Causal AI Analysis Results:")
    print(f" Estimated Average Treatment Effect (ATE): ${mean_ate:.2f}")
    print(f"==============================================")
    
    # Save processed summary or metrics if needed
    results_path = Path("models/causal_ml/ate_summary.txt")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        f.write(f"Estimated Average Treatment Effect (ATE): {mean_ate}\n")
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    run_causal_pipeline()