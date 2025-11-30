# src/train_model.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

# import our helper
from src.utils import feature_engineer

# Paths
DATA_PATH = "data/train.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # 1) Load data
    df = pd.read_csv(DATA_PATH)
    # 2) Feature engineering (adds Title, FamilySize, IsAlone, Deck, fills NA)
    df = feature_engineer(df)

    # 3) Choose features and target
    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','FamilySize','IsAlone','Deck']
    target = 'Survived'
    X = df[features]
    y = df[target]

    # 4) Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # 5) Preprocessing pipelines
    numeric_features = ['Age','Fare','FamilySize','SibSp','Parch']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['Pclass','Sex','Embarked','Title','Deck','IsAlone']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # 6) Full pipeline: preprocessor + classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

    # 7) Train
    print("Training model ...")
    model.fit(X_train, y_train)

    # 8) Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Evaluation on test set:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

    # 9) Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
