# ============================================================
# EXPERIMENT 37 — VALIDATION-GATED STRATIFIED 5-FOLD ENSEMBLE
# (OOF VALIDATION CHECK & MULTI-FOLD PROBABILITY BLENDING)
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score


print("============================================")
print("EXPERIMENT 37")
print("VALIDATION-GATED 5-FOLD HYBRID ENSEMBLE")
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
# 5. FILTER HIGH CARDINALITY
# ============================================================

categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

high_cardinality = [c for c in categorical_features if X[c].nunique(dropna=True) > 100]
if high_cardinality:
    X = X.drop(columns=high_cardinality)
    X_test = X_test.drop(columns=high_cardinality)

categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()


# ============================================================
# 6. MODEL FACTORY FUNCTION
# ============================================================

def build_model_pipeline(seed=42):
    """Creates a fresh hybrid pipeline with a specific random seed."""
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
# 7. STRATIFIED 5-FOLD CROSS-VALIDATION & OUT-OF-FOLD SCORING
# ============================================================

print("\n============================================")
print("RUNNING 5-FOLD STRATIFIED CROSS-VALIDATION")
print("============================================")

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(X))
test_fold_predictions = np.zeros((len(X_test), N_SPLITS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

    # Build fresh pipeline for each fold
    pipeline = build_model_pipeline(seed=42 + fold)
    pipeline.fit(X_train_fold, y_train_fold)

    # Predict on validation fold
    val_probs = pipeline.predict_proba(X_val_fold)[:, 1]
    oof_predictions[val_idx] = val_probs

    fold_auc = roc_auc_score(y_val_fold, val_probs)
    print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.5f}")

    # Predict on test set
    test_fold_predictions[:, fold] = pipeline.predict_proba(X_test)[:, 1]


# Calculate overall validation metrics
overall_oof_auc = roc_auc_score(y, oof_predictions)
overall_oof_acc = accuracy_score(y, (oof_predictions >= 0.5).astype(int))

print("\n============================================")
print("CROSS-VALIDATION RESULTS")
print("============================================")
print(f"Overall OOF ROC-AUC Score : {overall_oof_auc:.5f}")
print(f"Overall OOF Accuracy Score: {overall_oof_acc:.5f}")


# ============================================================
# 8. VALIDATION GATE (THRESHOLD CHECK)
# ============================================================

TARGET_THRESHOLD = 0.640  # Gating threshold to verify solid performance

if overall_oof_auc >= TARGET_THRESHOLD:
    print(f"\n[PASSED] Validation ROC-AUC ({overall_oof_auc:.5f}) cleared target threshold ({TARGET_THRESHOLD:.3f}).")
else:
    print(f"\n[WARNING] Validation ROC-AUC ({overall_oof_auc:.5f}) below target ({TARGET_THRESHOLD:.3f}). Proceeding with best blend.")


# ============================================================
# 9. ENSEMBLE TEST PREDICTIONS (AVERAGED OVER ALL 5 FOLDS)
# ============================================================

# Average the probabilities across all 5 fold models for maximum generalization
final_test_probabilities = test_fold_predictions.mean(axis=1)


# ============================================================
# 10. VALIDATE PREDICTIONS
# ============================================================

if len(final_test_probabilities) != len(test_data):
    raise ValueError("Prediction count does not match test data.")

if np.isnan(final_test_probabilities).any():
    raise ValueError("Predictions contain NaN values.")

if (final_test_probabilities < 0).any() or (final_test_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1] range.")


# ============================================================
# 11. CREATE & SAVE SUBMISSION FILE
# ============================================================

submission = pd.DataFrame({
    "anonymised_id": test_data["anonymised_id"],
    "employed_status": final_test_probabilities
})

output_file = "submission_exp37_gated_oof_ensemble.csv"
submission.to_csv(output_file, index=False)


# ============================================================
# 12. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 37 COMPLETE")
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
print("Exp 33 Perceptron / MLP   : 0.64453")
print("Exp 35 Hybrid Blend       : 0.65089")
print("Exp 36 Tri-Hybrid         : 0.65074")
print(f"Exp 37 5-Fold OOF Blend   : OOF Val = {overall_oof_auc:.5f} (Ready for submission)")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")