# ============================================================
# EXPERIMENT 34 — ONE-VS-REST (OvR) BAGGED PERCEPTRON ENSEMBLE
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier


print("============================================")
print("EXPERIMENT 34")
print("ONE-VS-REST (OvR) BAGGED PERCEPTRON")
print("============================================")


# ============================================================
# 1. DATASET PATH
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (
    CURRENT_DIR /
    "../../assets/dataset"
).resolve()


# ============================================================
# 2. LOAD DATA
# ============================================================

train_data = pd.read_csv(
    DATA_DIR / "train.csv"
)

test_data = pd.read_csv(
    DATA_DIR / "test.csv"
)


print("\n============================================")
print("DATASET LOADED")
print("============================================")

print("Training rows:", len(train_data))
print("Training columns:", len(train_data.columns))
print("Testing rows:", len(test_data))
print("Testing columns:", len(test_data.columns))


# ============================================================
# 3. CLEAN TARGET
# ============================================================

target = "employed_status"

print("\nMissing target values:", train_data[target].isna().sum())

train_clean = train_data.dropna(subset=[target]).copy()

train_clean[target] = pd.to_numeric(
    train_clean[target],
    errors="coerce"
)

train_clean = train_clean.dropna(subset=[target]).copy()
train_clean[target] = train_clean[target].astype(int)

y = train_clean[target]

print("Training rows after cleaning:", len(train_clean))
print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 4. CREATE X
# ============================================================

X = train_clean.drop(columns=[target]).copy()
X_test = test_data.copy()


# ============================================================
# 5. REMOVE ID
# ============================================================

if "anonymised_id" in X.columns:
    X = X.drop(columns=["anonymised_id"])

if "anonymised_id" in X_test.columns:
    X_test = X_test.drop(columns=["anonymised_id"])


# ============================================================
# 6. DATE FEATURES
# ============================================================

if "survey_date" in X.columns:
    X["survey_date"] = pd.to_datetime(X["survey_date"], errors="coerce")
    X_test["survey_date"] = pd.to_datetime(X_test["survey_date"], errors="coerce")

    X["survey_year"] = X["survey_date"].dt.year
    X_test["survey_year"] = X_test["survey_date"].dt.year

    X["survey_month"] = X["survey_date"].dt.month
    X_test["survey_month"] = X_test["survey_date"].dt.month

    X["survey_dayofyear"] = X["survey_date"].dt.dayofyear
    X_test["survey_dayofyear"] = X_test["survey_date"].dt.dayofyear

    X = X.drop(columns=["survey_date"])
    X_test = X_test.drop(columns=["survey_date"])


# ============================================================
# 7. IDENTIFY & FILTER HIGH-CARDINALITY CATEGORICAL FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

high_cardinality = []
for column in categorical_features:
    unique_count = X[column].nunique(dropna=True)
    if unique_count > 100:
        high_cardinality.append(column)
        print(f"Dropping high-cardinality feature: {column} ({unique_count} categories)")

if high_cardinality:
    X = X.drop(columns=high_cardinality)
    X_test = X_test.drop(columns=high_cardinality)

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=[np.number]
).columns.tolist()


print("\n============================================")
print("FINAL FEATURES")
print("============================================")
print("Total features:", len(X.columns))
print("Categorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))


# ============================================================
# 8. PREPROCESSING PIPELINES
# ============================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 9. ONE-VS-REST & BAGGED PERCEPTRON ARCHITECTURE
# ============================================================

# Base multi-layer perceptron
base_mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=0.005,
    batch_size=128,
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,
    n_iter_no_change=15,
    validation_fraction=0.15,
    random_state=42
)

# Explicit One-vs-Rest wrapper
ovr_mlp = OneVsRestClassifier(base_mlp)

# Bagging meta-estimator across 7 diverse bootstrap iterations
bagged_ovr_model = BaggingClassifier(
    estimator=ovr_mlp,
    n_estimators=7,
    max_samples=0.85,
    max_features=0.90,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", bagged_ovr_model)
])


# ============================================================
# 10. TRAIN
# ============================================================

print("\n============================================")
print("TRAINING ONE-VS-REST BAGGED PERCEPTRON")
print("============================================")
print(f"Training ensemble on {len(X)} rows...")

pipeline.fit(X, y)

print("\nTraining complete.")


# ============================================================
# 11. PREDICT
# ============================================================

print("\nGenerating test predictions...")

probabilities = pipeline.predict_proba(X_test)[:, 1]


# ============================================================
# 12. VALIDATE PREDICTIONS
# ============================================================

if len(probabilities) != len(test_data):
    raise ValueError("Prediction count does not match test data.")

if np.isnan(probabilities).any():
    raise ValueError("Predictions contain NaN values.")

if (probabilities < 0).any() or (probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1] range.")


# ============================================================
# 13. CREATE SUBMISSION FILE
# ============================================================

submission = pd.DataFrame({
    "anonymised_id": test_data["anonymised_id"],
    "employed_status": probabilities
})


# ============================================================
# 14. SAVE
# ============================================================

output_file = "submission_exp34_ovr_perceptron.csv"

submission.to_csv(output_file, index=False)


# ============================================================
# 15. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 34 COMPLETE")
print("============================================")
print(f"Saved: {output_file}")
print(f"Rows: {len(submission)}")

print("\nPrediction summary:")
print(submission["employed_status"].describe())

print("\nFirst 10 predictions:")
print(submission.head(10))

print("\n============================================")
print("BENCHMARKS")
print("============================================")
print("Current best ensemble   : 0.64516")
print("Exp 30 Logistic         : 0.59229")
print("Exp 31 ExtraTrees       : 0.61732")
print("Exp 33 Perceptron / MLP : 0.64453")
print("Exp 34 OvR Bagged MLP   : PENDING")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")