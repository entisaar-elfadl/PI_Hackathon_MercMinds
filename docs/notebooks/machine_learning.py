# ============================================================
# EXPERIMENT 49 — DOMAIN-ENGINEERED YOUTH LABOUR MARKET ENSEMBLE
# (EXPLICIT LAG SEMANTICS, MATRIC ORDINAL PARSER & MULTI-PARADIGM BLEND)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import re

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score


print("============================================")
print("EXPERIMENT 49")
print("DOMAIN-ENGINEERED YOUTH LABOUR MARKET ENSEMBLE")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR


# ============================================================
# 2. DOMAIN FEATURE ENGINEERING FUNCTIONS
# ============================================================

target = "employed_status"

def parse_matric_band(val):
    """Parses banded percentage strings like '50 - 59 %' into numeric midpoints."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace("%", "").strip()
    if "-" in s:
        parts = s.split("-")
        try:
            return (float(parts[0]) + float(parts[1])) / 2.0
        except Exception:
            return np.nan
    elif "<" in s:
        try:
            return float(s.replace("<", "").strip()) / 2.0
        except Exception:
            return np.nan
    elif ">" in s:
        try:
            return float(s.replace(">", "").strip()) + 5.0
        except Exception:
            return np.nan
    else:
        try:
            return float(s)
        except Exception:
            return np.nan


def engineer_domain_features(df, is_train=True):
    """Engineers labour market lag semantics, matric marks, and demographic features."""
    data = df.copy()

    # 1. Target Cleaning (if present)
    if target in data.columns:
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data.dropna(subset=[target]).copy()
        data[target] = data[target].astype(int)

    # 2. Identification
    if "anonymised_id" in data.columns:
        data = data.drop(columns=["anonymised_id"])

    # 3. Wave Timeline & Dates
    if "survey_date" in data.columns:
        data["survey_date"] = pd.to_datetime(data["survey_date"], errors="coerce")
        data["survey_year"] = data["survey_date"].dt.year
        data["survey_month"] = data["survey_date"].dt.month
        data["survey_dayofyear"] = data["survey_date"].dt.dayofyear
        data = data.drop(columns=["survey_date"])

    # 4. Lag Semantics & First-Time Observation Flag
    data["is_first_time_participant"] = (
        data["employed_lag"].isna() | 
        data["days_since_last_obs"].isna() | 
        (data["total_historical_rounds"].fillna(1) <= 1)
    ).astype(int)

    # Tri-state employment lag: -1 = Never observed before, 0 = Unemployed, 1 = Employed
    data["employed_lag_tristate"] = data["employed_lag"].fillna(-1).astype(float)

    # Log transforms for skewed duration metrics
    data["tenure_lag_log"] = np.log1p(data["tenure_lag"].fillna(0).clip(lower=0))
    data["days_since_last_obs_log"] = np.log1p(data["days_since_last_obs"].fillna(0).clip(lower=0))

    if "current_round" in data.columns and "lag_round" in data.columns:
        data["round_gap"] = (data["current_round"] - data["lag_round"].fillna(data["current_round"] - 1)).clip(lower=1)

    # 5. Matric Subject Performance Bands -> Continuous Numerical
    matric_cols = [
        "matric_englishhome", "matric_englishadd", 
        "matric_mathpure", "matric_physicalscience", "matric_mathlit"
    ]
    
    for m_col in matric_cols:
        if m_col in data.columns:
            data[f"{m_col}_num"] = data[m_col].apply(parse_matric_band)
            data = data.drop(columns=[m_col])

    # Aggregated Matric Indicators
    num_m_cols = [f"{m}_num" for m in matric_cols if f"{m}_num" in data.columns]
    data["matric_subjects_count"] = data[num_m_cols].notna().sum(axis=1)
    data["matric_avg_score"] = data[num_m_cols].mean(axis=1).fillna(-1)
    
    if "matric_mathpure_num" in data.columns and "matric_mathlit_num" in data.columns:
        data["matric_best_math"] = data[["matric_mathpure_num", "matric_mathlit_num"]].max(axis=1).fillna(-1)

    if "matric_englishhome_num" in data.columns and "matric_englishadd_num" in data.columns:
        data["matric_best_english"] = data[["matric_englishhome_num", "matric_englishadd_num"]].max(axis=1).fillna(-1)

    # 6. Education & Qualifications Hierarchy
    if "school_quintile" in data.columns:
        data["school_quintile"] = pd.to_numeric(data["school_quintile"], errors="coerce").fillna(0)

    if "work_readiness_score" in data.columns:
        data["work_readiness_score"] = pd.to_numeric(data["work_readiness_score"], errors="coerce").fillna(-1)

    # Sparse qualification indicators
    if "institution_type" in data.columns:
        data["has_tertiary_record"] = data["institution_type"].notna().astype(int)

    if "seta" in data.columns:
        data["has_seta_credential"] = data["seta"].notna().astype(int)

    return data


# ============================================================
# 3. LOAD DATA & APPLY TRANSFORMATIONS
# ============================================================

print("\n============================================")
print("LOADING DATASETS & EXTRACTING SIGNALS")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")

train_df = engineer_domain_features(train_raw, is_train=True)
test_df = engineer_domain_features(test_raw, is_train=False)

# Frequency-encode high-cardinality text fields (municipality, district, qualification name)
common_cols = [c for c in train_df.columns if c in test_df.columns and c != target]
cat_cols = train_df[common_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()

for col in cat_cols:
    if train_df[col].nunique(dropna=True) > 25:
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_freq"] = train_df[col].map(freq_map).fillna(0.0).astype(float)
        test_df[f"{col}_freq"] = test_df[col].map(freq_map).fillna(0.0).astype(float)
        train_df = train_df.drop(columns=[col])
        test_df = test_df.drop(columns=[col])

# Re-align final columns
common_cols = [c for c in train_df.columns if c in test_df.columns and c != target]

y = train_df[target].reset_index(drop=True)
X = train_df[common_cols].reset_index(drop=True)
X_test = test_df[common_cols].reset_index(drop=True)

categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"Training observations: {len(X)}")
print(f"Testing observations: {len(X_test)}")
print(f"Engineered Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")


# ============================================================
# 4. PREPROCESSOR & HYBRID ENSEMBLE ARCHITECTURE
# ============================================================

def build_model_pipeline(seed=42):
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

    # 1. Regularized Logistic Regression (Linear Log-Odds Anchor)
    logistic = LogisticRegression(
        C=0.10,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed
    )

    # 2. Gradient Boosted Decision Tree (Captures Non-Linear Lag & Matric Interactions)
    hgb = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.035,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=2.0,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=seed
    )

    # Soft Voting Hybrid (50% Linear Log-Odds + 50% Non-Linear Decision Trees)
    hybrid_ensemble = VotingClassifier(
        estimators=[
            ("lr", logistic),
            ("hgb", hgb)
        ],
        voting="soft",
        weights=[1.0, 1.0]
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("ensemble", hybrid_ensemble)
    ])


# ============================================================
# 5. 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================

print("\n============================================")
print("RUNNING 5-FOLD STRATIFIED CROSS-VALIDATION")
print("============================================")

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_probabilities = np.zeros(len(X))
test_fold_predictions = np.zeros((len(X_test), N_SPLITS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr_f, y_tr_f = X.iloc[train_idx], y.iloc[train_idx]
    X_va_f, y_va_f = X.iloc[val_idx], y.iloc[val_idx]

    pipe = build_model_pipeline(seed=42 + fold)
    pipe.fit(X_tr_f, y_tr_f)

    val_probs = pipe.predict_proba(X_va_f)[:, 1]
    oof_probabilities[val_idx] = val_probs

    fold_auc = roc_auc_score(y_va_f, val_probs)
    print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.5f}")

    test_fold_predictions[:, fold] = pipe.predict_proba(X_test)[:, 1]

overall_oof_auc = roc_auc_score(y, oof_probabilities)
print("\n============================================")
print(f"OVERALL DOMAIN OOF ROC-AUC: {overall_oof_auc:.5f}")
print("============================================")


# ============================================================
# 6. MULTI-SEED FULL DATASET ENSEMBLE INFERENCE
# ============================================================

print("\nTraining Multi-Seed Final Ensemble on Full Dataset...")
SEEDS = [42, 101, 777, 2024, 999]
full_test_preds = np.zeros((len(X_test), len(SEEDS)))

for i, seed in enumerate(SEEDS):
    full_pipe = build_model_pipeline(seed=seed)
    full_pipe.fit(X, y)
    full_test_preds[:, i] = full_pipe.predict_proba(X_test)[:, 1]

# Blend 5-Fold OOF Predictions with Full Multi-Seed Model for Maximum Stability
final_probabilities = 0.50 * test_fold_predictions.mean(axis=1) + 0.50 * full_test_preds.mean(axis=1)


# ============================================================
# 7. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp49_domain_labour_ensemble.csv"

submission = pd.DataFrame({
    "anonymised_id": test_raw["anonymised_id"],
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 49 COMPLETE")
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
print("Personal Best Benchmark          : 0.65671")
print(f"Exp 49 Domain-Engineered OOF AUC : {overall_oof_auc:.5f}")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")