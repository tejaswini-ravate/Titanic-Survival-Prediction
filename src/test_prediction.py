# src/test_prediction.py
import joblib
import pandas as pd
import os
import argparse

# import your feature engineering helper
from src.utils import feature_engineer

MODEL_PATH = os.path.join("models", "model.pkl")

def build_sample(passenger: dict) -> pd.DataFrame:
    """
    Accepts a dict with passenger info and returns a single-row DataFrame
    with the raw columns expected by feature_engineer().
    """
    df = pd.DataFrame([{
        "PassengerId": passenger.get("PassengerId", 0),
        "Pclass": passenger.get("Pclass", 3),
        "Name": passenger.get("Name", "Mr Test"),
        "Sex": passenger.get("Sex", "male"),
        "Age": passenger.get("Age", 30),
        "SibSp": passenger.get("SibSp", 0),
        "Parch": passenger.get("Parch", 0),
        "Ticket": passenger.get("Ticket", None),
        "Fare": passenger.get("Fare", 7.25),
        "Cabin": passenger.get("Cabin", None),
        "Embarked": passenger.get("Embarked", "S")
    }])
    return df

def predict_single(model, df_input):
    """
    Run the pipeline model on the provided df_input (raw).
    Returns (prediction_int, probability_float).
    """
    # make sure feature engineering matches train pipeline
    df_proc = feature_engineer(df_input)
    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','FamilySize','IsAlone','Deck']
    pred = model.predict(df_proc[features])[0]
    proba = model.predict_proba(df_proc[features])[0][1]
    return pred, proba

def main(args):
    # 1) load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")

    # 2) Build sample(s)
    if args.sample_csv:
        # read multiple samples from CSV (expects at least the raw columns)
        samples = pd.read_csv(args.sample_csv)
        results = []
        for idx, row in samples.iterrows():
            raw = row.to_dict()
            df = build_sample(raw)
            pred, proba = predict_single(model, df)
            results.append({
                "index": idx,
                "prediction": int(pred),
                "probability": float(proba)
            })
            print(f"Row {idx} -> pred={pred}, prob={proba:.3f}")
        if args.output_csv:
            out_df = pd.DataFrame(results)
            out_df.to_csv(args.output_csv, index=False)
            print(f"Saved predictions to {args.output_csv}")
    else:
        # default: single hard-coded sample (you can change it or pass CLI args)
        sample = {
            "Pclass": args.pclass,
            "Name": args.name,
            "Sex": args.sex,
            "Age": args.age,
            "SibSp": args.sibsp,
            "Parch": args.parch,
            "Fare": args.fare,
            "Embarked": args.embarked
        }
        df = build_sample(sample)
        pred, proba = predict_single(model, df)
        print("=== Single sample prediction ===")
        print(f"Input: {sample}")
        print(f"Prediction (0=not survive, 1=survived): {pred}")
        print(f"Survival probability: {proba:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test predictions using saved model")
    # single-sample CLI args (optional)
    parser.add_argument("--pclass", type=int, default=3)
    parser.add_argument("--name", type=str, default="Mr Test")
    parser.add_argument("--sex", type=str, choices=["male","female"], default="male")
    parser.add_argument("--age", type=float, default=30.0)
    parser.add_argument("--sibsp", type=int, default=0)
    parser.add_argument("--parch", type=int, default=0)
    parser.add_argument("--fare", type=float, default=7.25)
    parser.add_argument("--embarked", type=str, choices=["S","C","Q"], default="S")
    # batch CSV input (optional)
    parser.add_argument("--sample-csv", dest="sample_csv", type=str,
                        help="CSV file path containing raw passenger rows to predict")
    parser.add_argument("--output-csv", dest="output_csv", type=str,
                        help="Output CSV to save predictions for batch mode")
    args = parser.parse_args()
    main(args)
