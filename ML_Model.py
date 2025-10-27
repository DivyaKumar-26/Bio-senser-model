import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Config
CSV_PATH = "Crop Recommendation using Soil Properties and Weather Prediction.csv"
TARGET_COL = "label"
CV_FOLDS = 10
RANDOM_STATE = 42
SAVE_PATH = "trained_crop_model.pkl"  # contains both pipeline and label encoder

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found at: {path}")
    df = pd.read_csv(path)
    return df

def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor, numeric_cols, categorical_cols

def build_models():
    return {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=50,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            multi_class="auto",
            random_state=RANDOM_STATE
        )
    }

def main():
    print("📦 Loading dataset...")
    df = load_data(CSV_PATH)
    print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataset. Available columns: {df.columns.tolist()}")

    y_raw = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
    print(f"🔢 Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"🎨 Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # Encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Train-test split (stratify to preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = build_models()

    results = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"\n🚀 Training models with {CV_FOLDS}-fold CV...\n")

    for name, estimator in models.items():
        print(f"\n🧠 Training {name} model...")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        # Cross-validation (parallel)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        print(f"📊 {name} CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Fit on full training set
        pipeline.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        results[name] = {
            "test_accuracy": test_acc,
            "pipeline": pipeline
        }

        print(f"🎯 {name} Test Accuracy: {test_acc:.3f}")
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
        print("🧩 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("-" * 60)

    # Pick best model by test accuracy
    best_model_name = max(results, key=lambda k: results[k]["test_accuracy"])
    best_pipeline = results[best_model_name]["pipeline"]
    best_test_acc = results[best_model_name]["test_accuracy"]

    print(f"\n🏆 Best Model: {best_model_name} with Test Accuracy: {best_test_acc:.3f}")

    # Re-fit best pipeline on entire dataset (optional, common practice)
    print("🔁 Re-fitting best model on entire dataset for final artifact...")
    best_pipeline.fit(X, y)

    # Save pipeline + label encoder together
    artifact = {"pipeline": best_pipeline, "label_encoder": label_encoder}
    joblib.dump(artifact, SAVE_PATH)
    print(f"💾 Saved final artifact to '{SAVE_PATH}'")

    print("\n🌿 Done! Ready for predictions.")

if __name__ == "__main__":
    main()