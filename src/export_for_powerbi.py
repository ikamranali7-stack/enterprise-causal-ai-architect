import pandas as pd
import numpy as np
from pathlib import Path
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def export_powerbi_data():
    print("Loading data and retraining model for customer-level effects...")
    df = pd.read_csv("data/raw/enterprise_subscriptions.csv")
    
    treatment = "treatment"
    outcome = "arr"
    common_causes = ["company_size", "prior_usage", "budget_index"]
    
    X = df[common_causes]
    T = df[treatment]
    Y = df[outcome]
    
    # Fit DML model
    dml_model = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=42),
        model_t=RandomForestClassifier(n_estimators=100, random_state=42),
        discrete_treatment=True,
        random_state=42
    )
    dml_model.fit(Y, T, X=X)
    
    # Compute Individual Treatment Effects (ITE) / Heterogeneous Treatment Effects
    df['predicted_treatment_effect'] = dml_model.effect(X)
    
    # Categorize enterprise segments based on budget & size
    df['segment'] = pd.qcut(df['budget_index'], q=3, labels=['SMB', 'Mid-Market', 'Enterprise'])
    
    # Save processed table for Power BI semantic layer
    output_path = Path("powerbi/causal_dashboard_feed.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nPower BI feed successfully exported to {output_path} with shape {df.shape}")

if __name__ == "__main__":
    export_powerbi_data()