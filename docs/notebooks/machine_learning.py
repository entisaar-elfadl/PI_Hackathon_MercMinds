# ============================================================
# EXPERIMENT 53 — HISTORICAL PARTICIPANT TRACK RECORD & TITAN BLEND
# (LEAK-FREE EXPANDING LONGITUDINAL HISTORY + MULTI-SEED ENSEMBLE)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


print("============================================")
print("EXPERIMENT 53")
print("HISTORICAL PARTICIPANT TRACK RECORD & TITAN BLEND")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR

ROUND_DIR = CURRENT_DIR / "round_testing"
if not ROUND_DIR.exists():
    ROUND_DIR = DATA_DIR / "round_testing"


# ============================================================
# 2. LONGITUDINAL TRACK RECORD FEATURE ENGINEERING
# ============================================================

target = "employed_status"

def extract_longitudinal_and_proven_features(train_df, test_df):
    """
    Computes expanding individual track records across historical waves
    with zero future-data leakage, and adds all proven Exp 51 features.
    """
    tr = train_df.copy()
    te = test_df.copy()

    # 1. Clean Target in Training Set
    tr[target] = pd.to_numeric(tr[target], errors="coerce")
    tr = tr.dropna(subset=[target]).copy()
    tr[target] = tr[target].astype(int)

    # 2. Parse Dates & Rounds for Chronological Sorting
    if "survey_date" in tr.columns:
        tr["survey_date_dt"] = pd.to_datetime(tr["survey_date"], errors="coerce")
    else:
        tr["survey_date_dt"] = pd.to_datetime("2020-01-01")

    if "current_round" in tr.columns:
        tr["round_num"] = pd.to_numeric(tr["current_round"], errors="coerce").fillna(1)
    else:
        tr["round_num"] = 1

    # 3. Expanding Historical Track Record in Train (Zero Future Leakage)
    if "anonymised_id" in tr.columns:
        tr = tr.sort_values(["anonymised_id", "round_num", "survey_date_dt"]).reset_index(drop=True)

        # Shift target by 1 so only past rounds are counted
        tr["user_past_sum"] = (
            tr.groupby("anonymised_id")[target]
            .transform(lambda s: s.shift(1).cumsum())
            .fillna(0.0)
        )
        tr["user_prev_obs_count"] = (
            tr.groupby("anonymised_id")[target]
            .transform(lambda s: s.shift(1).expanding().count())
            .fillna(0.0)
        )
        tr["user_past_employed_rate"] = np.where(
            tr["user_prev_obs_count"] > 0,
            tr["user_past_sum"] / tr["user_prev_obs_count"],
            np.nan
        )
        tr["user_has_past_history"] = (tr["user_prev_obs_count"] > 0).astype(float)
        tr = tr.drop(columns=["survey_date_dt", "round_num", "user_past_sum"])

        # 4. Map Total Historical Track Record to Test Set (Round 9)
        user_total_sum = tr.groupby("anonymised_id")[target].sum()
        user_total_count = tr.groupby("anonymised_id")[target].count()
        user_total_rate = user_total_sum / user_total_count

        te["user_prev_obs_count"] = te["anonymised_id"].map(user_total_count).fillna(0.0)
        te["user_past_employed_rate"] = te["anonymised_id"].map(user_total_rate)
        te["user_has_past_history"] = (te["user_prev_obs_count"] > 0).astype(float)

    # 5. Extract Proven Exp 51 Date & Lag Features
    for df in [tr, te]:
        if "survey_date" in df.columns:
            df["survey_date"] = pd.to_datetime(df["survey_date"], errors="coerce")
            df["survey_year"] = df["survey_date"].dt.year
            df["survey_month"] = df["survey_date"].dt.month
            df["survey_dayofyear"] = df["survey_date"].dt.dayofyear
            df.drop(columns=["survey_date"], inplace=True)

        if "employed_lag" in df.columns:
            df["is_first_time"] = df["employed_lag"].isna().astype(float)
        
        if "tenure_lag" in df.columns:
            df["tenure_lag_log"] = np.log1p(df["tenure_lag"].fillna(0).clip(lower=0))

        if "days_since_last_obs" in df.columns:
            df["days_since_last_obs_log"] = np.log1p(df["days_since_last_obs"].fillna(0).clip(lower=0))

        # Auto-convert numeric strings
        for col in df.columns:
            if col not in [target, "anonymised_id"] and df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > 0.6 * df[col].notna().sum():
                    df[col] = converted

    return tr, te


def build_preprocessor(numerical_cols, categorical_cols, use_quantile=False):
    """Builds standard or quantile-normalized preprocessor."""
    transformers = []

    if len(numerical_cols) > 0:
        scaler = QuantileTransformer(output_distribution="normal", random_state=42) if use_quantile else StandardScaler()
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler)
        ])
        transformers.append(("num", numeric_transformer, numerical_cols))

    if len(categorical_cols) > 0:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", categorical_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers)


# ============================================================
# 3. HIGH-SCORING MODEL SUITE
# ============================================================

def get_champion_models(seed=42):
    """Returns the proven linear champions + neural MLP."""
    return {
        "LogReg_L2_C008": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False,
            "linear"
        ),
        "LogReg_L2_C010": (
            LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False,
            "linear"
        ),
        "LogReg_ElasticNet_C010": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000, random_state=seed),
            False,
            "linear"
        ),
        "Quantile_LogReg_L2_C008": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            True,
            "linear"
        ),
        "MLP_Medium_64_32": (
            MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.01,
                batch_size=128, learning_rate_init=0.001, max_iter=350, early_stopping=True,
                n_iter_no_change=20, validation_fraction=0.15, random_state=seed
            ),
            False,
            "neural"
        )
    }


# ============================================================
# 4. LOAD & PREPARE FULL COMPETITION DATASET
# ============================================================

print("\n============================================")
print("LOADING DATA & COMPUTING LONGITUDINAL TRACK RECORDS")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")

train_df, test_df = extract_longitudinal_and_proven_features(train_raw, test_raw)

# Save test submission ID key
test_ids = test_df["anonymised_id"].copy()

# Drop ID from feature matrix
if "anonymised_id" in train_df.columns:
    train_df = train_df.drop(columns=["anonymised_id"])
if "anonymised_id" in test_df.columns:
    test_df = test_df.drop(columns=["anonymised_id"])

common_cols = [c for c in train_df.columns if c in test_df.columns and c != target]

y_full = train_df[target].reset_index(drop=True)
X_full = train_df[common_cols].reset_index(drop=True)
X_test_full = test_df[common_cols].reset_index(drop=True)

# Drop high-cardinality categorical (>100)
categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
high_cardinality = [c for c in categorical_features if X_full[c].nunique(dropna=True) > 100]
if high_cardinality:
    X_full = X_full.drop(columns=high_cardinality)
    X_test_full = X_test_full.drop(columns=high_cardinality)

categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X_full.select_dtypes(include=[np.number]).columns.tolist()

print(f"Full dataset: {len(X_full)} observations | Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")
print(f"Longitudinal features added: 'user_past_employed_rate', 'user_prev_obs_count', 'user_has_past_history'")


# ============================================================
# 5. MULTI-SEED TITAN HYBRID INFERENCE (35 MODELS)
# ============================================================

print("\n============================================")
print("TRAINING 35-MODEL MULTI-SEED ENSEMBLE (80% LINEAR + 20% NEURAL)")
print("============================================")

SEEDS = [42, 101, 777, 2024, 999, 1337, 555]

linear_logits_list = []
neural_logits_list = []

model_dict_sample = get_champion_models(seed=42)

for model_name, (_, use_quantile, model_family) in model_dict_sample.items():
    print(f" -> Training {model_name} ({model_family.upper()}) across {len(SEEDS)} seeds...")

    for s_idx, seed in enumerate(SEEDS):
        models_pool = get_champion_models(seed=seed)
        model_estimator, use_q, _ = models_pool[model_name]

        preprocessor = build_preprocessor(numerical_features, categorical_features, use_quantile=use_q)
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model_estimator)
        ])
        pipe.fit(X_full, y_full)
        probs = pipe.predict_proba(X_test_full)[:, 1]

        # Convert to log-odds (logit space)
        probs_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(probs_clipped / (1.0 - probs_clipped))

        if model_family == "linear":
            linear_logits_list.append(logits)
        else:
            neural_logits_list.append(logits)

# 80% Linear Champions + 20% Neural MLP in Logit Space
mean_linear_logits = np.mean(linear_logits_list, axis=0)
mean_neural_logits = np.mean(neural_logits_list, axis=0)

final_hybrid_logits = 0.80 * mean_linear_logits + 0.20 * mean_neural_logits
final_probabilities = 1.0 / (1.0 + np.exp(-final_hybrid_logits))


# ============================================================
# 6. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp53_historical_track_titan_ensemble.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 7. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 53 COMPLETE")
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
print("Exp 44 Top-3 Linear Blend      : 0.65630")
print("Exp 47 Pure-Logit Ensemble     : 0.65635")
print("Exp 51 Restored Titan Hybrid   : 0.65693 (Personal Best)")
print("Exp 53 Longitudinal Titan Blend: READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")