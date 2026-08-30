# ============================================================
# EXPERIMENT 39 — MULTI-SEED 5-FOLD ENSEMBLE WITH MISSINGNESS INDICATORS
# (OPTIMIZED ON EXP 37 FOUNDATION: 15-MODEL SEED AVERAGING)
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
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score


print("============================================")
print("EXPERIMENT 39")
print("MULTI-SEED 5-FOLD HYBRID ENSEMBLE")
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

print("\n============================================")
print("DATASET LOADED")
print("============================================")
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

y = train_clean[target].reset_index(drop=True)
X = train_clean.drop(columns=[target]).reset_index(drop=True)
X_test = test_data.copy()

print(f"\nTraining rows after cleaning: {len(X)}")
print(f"Target balance:\n{y.value_counts(normalize=True)}")


# ============================================================
# 4. REMOVE ID & DATE FEATURE ENGINEERING
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
# 5. FILTER HIGH CARDINALITY CATEGORIES (>100)
# ============================================================

categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

high_cardinality = [c for c in categorical_features if X[c].nunique(dropna=True) > 100]
if high_cardinality:
    X = X.drop(columns=high_cardinality)
    X_test = X_test.drop(columns=high_cardinality)

categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nFeatures used: {len(numerical_features)} numerical, {len(categorical_features)} categorical")


# ============================================================
# 6. MODEL PIPELINE BUILDER WITH MISSINGNESS INDICATORS
# ============================================================

def build_model_pipeline(seed=42):
    # add_indicator=True captures survey omission patterns
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=350,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=seed
    )

    logistic = LogisticRegression(
        C=0.1,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed
    )

    ensemble = VotingClassifier(
        estimators=[
            ("mlp", mlp),
            ("lr", logistic)
        ],
        voting="soft",
        weights=[3.5, 1.0]
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("ensemble", ensemble)
    ])


# ============================================================
# 7. MULTI-SEED 5-FOLD CROSS-VALIDATION (3 SEEDS x 5 FOLDS = 15 MODELS)
# ============================================================

SEEDS = [42, 101, 777]
N_SPLITS = 5

all_seed_oof = []
all_test_predictions = []

print("\n============================================")
print(f"RUNNING {len(SEEDS)} SEEDS x {N_SPLITS}-FOLD CV ({len(SEEDS) * N_SPLITS} TOTAL MODELS)")
print("============================================")

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n--- Running Seed {seed} ({seed_idx + 1}/{len(SEEDS)}) ---")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(X))
    seed_test_preds = np.zeros((len(X_test), N_SPLITS))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        model = build_model_pipeline(seed=seed + fold * 10)
        model.fit(X_train_fold, y_train_fold)

        val_probs = model.predict_proba(X_val_fold)[:, 1]
        seed_oof[val_idx] = val_probs

        test_probs = model.predict_proba(X_test)[:, 1]
        seed_test_preds[:, fold] = test_probs

    seed_auc = roc_auc_score(y, seed_oof)
    print(f"Seed {seed} OOF ROC-AUC: {seed_auc:.5f}")

    all_seed_oof.append(seed_oof)
    all_test_predictions.append(seed_test_preds.mean(axis=1))


# ============================================================
# 8. OVERALL EVALUATION
# ============================================================

# Average across all seeds
final_oof_predictions = np.mean(all_seed_oof, axis=0)
final_test_probabilities = np.mean(all_test_predictions, axis=0)

overall_oof_auc = roc_auc_score(y, final_oof_predictions)
overall_oof_acc = accuracy_score(y, (final_oof_predictions >= 0.5).astype(int))

print("\n============================================")
print("FINAL MULTI-SEED CV RESULTS")
print("============================================")
print(f"Overall Multi-Seed OOF ROC-AUC: {overall_oof_auc:.5f}")
print(f"Overall Multi-Seed Accuracy   : {overall_oof_acc:.5f}")


# ============================================================
# 9. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_test_probabilities) != len(test_data):
    raise ValueError("Prediction count does not match test data.")

if np.isnan(final_test_probabilities).any():
    raise ValueError("Predictions contain NaN values.")

if (final_test_probabilities < 0).any() or (final_test_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1] range.")

submission = pd.DataFrame({
    "anonymised_id": test_data["anonymised_id"],
    "employed_status": final_test_probabilities
})

output_file = "submission_exp39_multiseed_oof_ensemble.csv"
submission.to_csv(output_file, index=False)


# ============================================================
# 10. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 39 COMPLETE")
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
print("Exp 35 Hybrid Single Fit  : 0.65089")
print("Exp 37 5-Fold Single Seed : 0.65247 (Previous Best)")
print("Exp 38 Target Encoded     : 0.64350")
print(f"Exp 39 Multi-Seed 15-Model: OOF Val = {overall_oof_auc:.5f} (Ready for submission)")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")