# ============================================================
# EXPERIMENT 40 — CLEAN 5-FOLD DUAL-PARADIGM ENSEMBLE
# (EXP 37 MLP-LOGISTIC BACKBONE + 5-FOLD HIST GRADIENT BOOSTING)
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
print("EXPERIMENT 40")
print("CLEAN 5-FOLD DUAL-PARADIGM ENSEMBLE")
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
print(f"Target distribution:\n{y.value_counts(normalize=True)}")


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
# 5. CARDINALITY HANDLING (EXACT EXP 37 SETTING)
# ============================================================

categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

high_cardinality = [c for c in categorical_features if X[c].nunique(dropna=True) > 100]
if high_cardinality:
    X = X.drop(columns=high_cardinality)
    X_test = X_test.drop(columns=high_cardinality)

categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nFeatures: {len(numerical_features)} numerical, {len(categorical_features)} categorical")


# ============================================================
# 6. MODEL PIPELINE BUILDERS
# ============================================================

# Model 1: The Winning Exp 37 Pipeline (MLP + Logistic Regression)
def build_neural_linear_pipeline(seed=42):
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


# Model 2: 5-Fold Gradient Boosted Decision Trees (Tree-based representation)
def build_tree_pipeline(seed=42):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
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

    hgb = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=2.0,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=seed
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("hgb", hgb)
    ])


# ============================================================
# 7. 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================

print("\n============================================")
print("RUNNING 5-FOLD CROSS-VALIDATION")
print("============================================")

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_neural = np.zeros(len(X))
oof_trees = np.zeros(len(X))

test_preds_neural = np.zeros((len(X_test), N_SPLITS))
test_preds_trees = np.zeros((len(X_test), N_SPLITS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]

    # Train Model 1 (Neural + Linear)
    model_neural = build_neural_linear_pipeline(seed=42 + fold)
    model_neural.fit(X_train_f, y_train_f)
    val_pred_n = model_neural.predict_proba(X_val_f)[:, 1]
    oof_neural[val_idx] = val_pred_n
    test_preds_neural[:, fold] = model_neural.predict_proba(X_test)[:, 1]

    # Train Model 2 (Tree Gradient Boosting)
    model_tree = build_tree_pipeline(seed=100 + fold)
    model_tree.fit(X_train_f, y_train_f)
    val_pred_t = model_tree.predict_proba(X_val_f)[:, 1]
    oof_trees[val_idx] = val_pred_t
    test_preds_trees[:, fold] = model_tree.predict_proba(X_test)[:, 1]

    print(f"Fold {fold + 1} - Neural AUC: {roc_auc_score(y_val_f, val_pred_n):.5f} | Tree AUC: {roc_auc_score(y_val_f, val_pred_t):.5f}")


# ============================================================
# 8. OUT-OF-FOLD BLEND OPTIMIZATION
# ============================================================

# 70% Neural/Linear Backbone + 30% Tree Gradient Boosting
oof_blend = 0.70 * oof_neural + 0.30 * oof_trees

score_neural = roc_auc_score(y, oof_neural)
score_trees = roc_auc_score(y, oof_trees)
score_blend = roc_auc_score(y, oof_blend)

print("\n============================================")
print("OOF CROSS-VALIDATION SUMMARY")
print("============================================")
print(f"1. Neural + Logistic OOF AUC : {score_neural:.5f}")
print(f"2. Gradient Tree OOF AUC    : {score_trees:.5f}")
print(f"3. Combined Dual Blend AUC  : {score_blend:.5f}")


# ============================================================
# 9. GENERATE & VALIDATE FINAL TEST PREDICTIONS
# ============================================================

final_test_neural = test_preds_neural.mean(axis=1)
final_test_trees = test_preds_trees.mean(axis=1)

# Combined test probability blend
final_test_probabilities = 0.70 * final_test_neural + 0.30 * final_test_trees

if len(final_test_probabilities) != len(test_data):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_test_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_test_probabilities < 0).any() or (final_test_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")


# ============================================================
# 10. SAVE SUBMISSION FILE
# ============================================================

output_file = "submission_exp40_clean_dual_ensemble.csv"

submission = pd.DataFrame({
    "anonymised_id": test_data["anonymised_id"],
    "employed_status": final_test_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 11. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 40 COMPLETE")
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
print("Exp 37 5-Fold Neural/Lin  : 0.65247 (Current Best)")
print("Exp 38 Target Encoded     : 0.64350")
print("Exp 39 Indicator Noise    : 0.63998")
print(f"Exp 40 Clean Dual-Ensemble: OOF Val = {score_blend:.5f} (Ready for submission)")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")