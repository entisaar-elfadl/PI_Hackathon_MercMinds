# ============================================================
# EXPERIMENT 36 — THE GOLDEN TRIFECTA
# (PERCEPTRON MLP + LOGISTIC REGRESSION + GRADIENT BOOSTING)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier


print("============================================")
print("EXPERIMENT 36")
print("TRI-HYBRID: MLP + LOGISTIC REGRESSION + HIST GB")
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

train_data = pd.read_csv(DATA_DIR / "train.csv")
test_data = pd.read_csv(DATA_DIR / "test.csv")

print("Training rows:", len(train_data))
print("Testing rows:", len(test_data))


# ============================================================
# 3. CLEAN TARGET
# ============================================================

target = "employed_status"

train_clean = train_data.dropna(subset=[target]).copy()
train_clean[target] = pd.to_numeric(train_clean[target], errors="coerce")
train_clean = train_clean.dropna(subset=[target]).copy()
train_clean[target] = train_clean[target].astype(int)

y = train_clean[target]
X = train_clean.drop(columns=[target]).copy()
X_test = test_data.copy()


# ============================================================
# 4. REMOVE ID & DATE FEATURES
# ============================================================

if "anonymised_id" in X.columns:
    X = X.drop(columns=["anonymised_id"])
if "anonymised_id" in X_test.columns:
    X_test = X_test.drop(columns=["anonymised_id"])

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
# 5. CARDINALITY HANDLING
# ============================================================

categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

high_cardinality = [c for c in categorical_features if X[c].nunique(dropna=True) > 100]
if high_cardinality:
    X = X.drop(columns=high_cardinality)
    X_test = X_test.drop(columns=high_cardinality)

categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()


# ============================================================
# 6. PREPROCESSOR & 3-WAY ENSEMBLE SETUP
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

# 1. Non-linear Neural Perceptron
mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=0.01,
    batch_size=128,
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,
    n_iter_no_change=20,
    validation_fraction=0.15,
    random_state=42
)

# 2. Linear Regularized Anchor
logistic_model = LogisticRegression(
    C=0.1,
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)

# 3. Decision-Tree Gradient Boosting
hgb_model = HistGradientBoostingClassifier(
    learning_rate=0.04,
    max_iter=400,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=25,
    random_state=42
)

tri_ensemble = VotingClassifier(
    estimators=[
        ("mlp", mlp_model),
        ("lr", logistic_model),
        ("hgb", hgb_model)
    ],
    voting="soft",
    weights=[3, 1, 2]   # Balanced weight distribution
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("ensemble", tri_ensemble)
])


# ============================================================
# 7. TRAIN & PREDICT
# ============================================================

print("\nTraining Tri-Model Ensemble...")
pipeline.fit(X, y)

print("Generating predictions...")
probabilities = pipeline.predict_proba(X_test)[:, 1]

# Validation
assert len(probabilities) == len(test_data), "Length mismatch"
assert not np.isnan(probabilities).any(), "NaN found in predictions"

submission = pd.DataFrame({
    "anonymised_id": test_data["anonymised_id"],
    "employed_status": probabilities
})

output_file = "submission_exp36_tri_hybrid_ensemble.csv"
submission.to_csv(output_file, index=False)

print(f"\nSaved successfully: {output_file}")
print(submission["employed_status"].describe())