# src/explainability_shap.py
import joblib
import pandas as pd
import os
import shap
import matplotlib.pyplot as plt
from src.utils import feature_engineer

MODEL_PATH = "models/model.pkl"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    # Load trained pipeline
    model = joblib.load(MODEL_PATH)
    # Load original data to get background dataset for SHAP
    df = pd.read_csv("data/train.csv")
    df = feature_engineer(df)
    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','FamilySize','IsAlone','Deck']
    X = df[features]

    # Preprocess X to the numeric array used by classifier
    preprocessor = model.named_steps['preprocessor']
    clf = model.named_steps['classifier']
    X_prep = preprocessor.transform(X)

    # Create SHAP explainer (TreeExplainer works for RandomForest/XGBoost)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_prep)

    # Summary plot
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values, X_prep, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_summary.png"), dpi=150)
    plt.close()
    print("Saved shap_summary.png")

    # Example: dependence plot for the top feature (index 0 used as example)
    try:
        shap.dependence_plot(0, shap_values, X_prep, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "shap_dependence_0.png"), dpi=150)
        plt.close()
        print("Saved shap_dependence_0.png")
    except Exception as e:
        print("Could not save dependence plot:", e)

if __name__ == "__main__":
    main()
