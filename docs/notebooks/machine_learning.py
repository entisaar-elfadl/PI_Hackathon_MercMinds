# ============================================================
# EXPERIMENT 38 — TARGET-ENCODED MULTI-SCALE 10-FOLD ENSEMBLE
# (TARGET ENCODING + DUAL MLPs + BALANCED LOGISTIC REGRESSION)
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

# Scikit-learn TargetEncoder (with robust fallback)
try:
    from sklearn.preprocessing import TargetEncoder
    HAS_TARGET_ENCODER = True
except ImportError:
    HAS_TARGET_ENCODER = False


print("============================================")
print("EXPERIMENT 38")
print("TARGET-ENCODED 10-FOLD MULTI-SCALE ENSEMBLE")
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

    # Cyclic time encoding
    X["sin_month"] = np.sin(2 * np.pi * X["survey_month"] / 12)
    X["cos_month"] = np.cos(2 * np.pi * X["survey_month"] / 12)
    X_test["sin_month"] = np.sin(2 * np.pi * X_test["survey_month"] / 12)
    X_test["cos_month"] = np.cos(2 * np.pi * X_test["survey_month"] / 12)

    X = X.drop(columns=["survey_date"])
    X_test = X_test.drop(columns=["survey_date"])


# ============================================================
# 5. FEATURE CATEGORIZATION (LOW vs HIGH CARDINALITY)
# ============================================================

raw_cat_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

low_card_features = []
high_card_features = []

for c in raw_cat_features:
    if X[c].nunique(dropna=True) <= 25:
        low_card_features.append(c)
    else:
        high_card_features.append(c)

print(f"Numerical features: {len(numerical_features)}")
print(f"Low-cardinality categorical features (One-Hot): {len(low_card_features)}")
print(f"High-cardinality categorical features (Target Encoded): {len(high_card_features)}")


# ============================================================
# 6. PIPELINE BUILDER
# ============================================================

def build_model_pipeline(seed=42):
    transformers = [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical_features),
        
        ("cat_low", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), low_card_features)
    ]

    if len(high_card_features) > 0:
        if HAS_TARGET_ENCODER:
            transformers.append((
                "cat_high",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("target_enc", TargetEncoder(smooth="auto", cv=5, random_state=seed)),
                    ("scaler", StandardScaler())
                ]),
                high_card_features
            ))
        else:
            transformers.append((
                "cat_high",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=30, sparse_output=False))
                ]),
                high_card_features
            ))

    preprocessor = ColumnTransformer(transformers=transformers)

    # 1. Wide Perceptron (MLP)
    mlp_wide = MLPClassifier(
        hidden_layer_sizes=(128, 64),
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

    # 2. Deep Perceptron (MLP)
    mlp_deep = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.015,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=350,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=seed + 100
    )

    # 3. Balanced Logistic Anchor
    logistic = LogisticRegression(
        C=0.1,
        penalty="l2",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed
    )

    # Soft voting ensemble with weighted blend
    ensemble = VotingClassifier(
        estimators=[
            ("mlp_wide", mlp_wide),
            ("mlp_deep", mlp_deep),
            ("lr", logistic)
        ],
        voting="soft",
        weights=[3.0, 2.5, 1.0]
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("ensemble", ensemble)
    ])


# ============================================================
# 7. 10-FOLD CROSS-VALIDATION & TEST PREDICTION
# ============================================================

print("\n============================================")
print("RUNNING 10-FOLD STRATIFIED CROSS-VALIDATION")
print("============================================")

N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(X))
test_fold_predictions = np.zeros((len(X_test), N_SPLITS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

    pipeline = build_model_pipeline(seed=42 + fold)
    pipeline.fit(X_train_fold, y_train_fold)

    val_probs = pipeline.predict_proba(X_val_fold)[:, 1]
    oof_predictions[val_idx] = val_probs

    fold_auc = roc_auc_score(y_val_fold, val_probs)
    print(f"Fold {fold + 1:02d}/{N_SPLITS} ROC-AUC: {fold_auc:.5f}")

    test_fold_predictions[:, fold] = pipeline.predict_proba(X_test)[:, 1]


overall_oof_auc = roc_auc_score(y, oof_predictions)
overall_oof_acc = accuracy_score(y, (oof_predictions >= 0.5).astype(int))

print("\n============================================")
print("CROSS-VALIDATION SUMMARY")
print("============================================")
print(f"Overall OOF ROC-AUC Score : {overall_oof_auc:.5f}")
print(f"Overall OOF Accuracy Score: {overall_oof_acc:.5f}")


# ============================================================
# 8. SAVE SUBMISSION FILE
# ============================================================

final_test_probabilities = test_fold_predictions.mean(axis=1)

submission = pd.DataFrame({
    "anonymised_id": test_data["anonymised_id"],
    "employed_status": final_test_probabilities
})

output_file = "submission_exp38_target_encoded_10fold.csv"
submission.to_csv(output_file, index=False)


# ============================================================
# 9. BENCHMARKS & INSPECTION
# ============================================================

print("\n============================================")
print("EXPERIMENT 38 COMPLETE")
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
print("Exp 35 Hybrid Blend       : 0.65089")
print("Exp 36 Tri-Hybrid         : 0.65074")
print("Exp 37 5-Fold OOF Blend   : 0.65247")
print(f"Exp 38 10-Fold Multi-MLP  : OOF Val = {overall_oof_auc:.5f} (Ready for submission)")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")