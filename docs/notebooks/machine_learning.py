# ============================================================
# EXPERIMENT 54 — THE GRAND MASTER DUAL-VIEW HYBRID BLEND
# (INTEGRATES EXP 51 TITAN + EXP 53 LONGITUDINAL TRACK RECORD)
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
print("EXPERIMENT 54")
print("THE GRAND MASTER DUAL-VIEW HYBRID BLEND")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR


# ============================================================
# 2. FEATURE EXTRACTION PIPELINES (DUAL VIEW)
# ============================================================

target = "employed_status"

def clean_base_dataframe(df):
    """Basic cleaning for target, ID, and dates."""
    data = df.copy()

    if target in data.columns:
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data.dropna(subset=[target]).copy()
        data[target] = data[target].astype(int)

    if "survey_date" in data.columns:
        data["survey_date"] = pd.to_datetime(data["survey_date"], errors="coerce")
        data["survey_year"] = data["survey_date"].dt.year
        data["survey_month"] = data["survey_date"].dt.month
        data["survey_dayofyear"] = data["survey_date"].dt.dayofyear
        data = data.drop(columns=["survey_date"])

    if "employed_lag" in data.columns:
        data["is_first_time"] = data["employed_lag"].isna().astype(float)
    
    if "tenure_lag" in data.columns:
        data["tenure_lag_log"] = np.log1p(data["tenure_lag"].fillna(0).clip(lower=0))

    if "days_since_last_obs" in data.columns:
        data["days_since_last_obs_log"] = np.log1p(data["days_since_last_obs"].fillna(0).clip(lower=0))

    # Auto-convert numeric strings
    for col in data.columns:
        if col not in [target, "anonymised_id"] and data[col].dtype == "object":
            converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().sum() > 0.6 * data[col].notna().sum():
                data[col] = converted

    return data


# --- View A: Exp 51 Macro & Cohort Features ---
def extract_view_a(train_raw, test_raw):
    tr = clean_base_dataframe(train_raw)
    te = clean_base_dataframe(test_raw)

    if "anonymised_id" in tr.columns:
        tr = tr.drop(columns=["anonymised_id"])
    if "anonymised_id" in te.columns:
        te = te.drop(columns=["anonymised_id"])

    return tr, te


# --- View B: Exp 53 Longitudinal Track Record Features ---
def extract_view_b(train_raw, test_raw):
    tr = train_raw.copy()
    te = test_raw.copy()

    tr[target] = pd.to_numeric(tr[target], errors="coerce")
    tr = tr.dropna(subset=[target]).copy()
    tr[target] = tr[target].astype(int)

    # Sort chronologically for leak-free expanding track record
    if "current_round" in tr.columns:
        round_col = pd.to_numeric(tr["current_round"], errors="coerce").fillna(1)
    else:
        round_col = pd.Series(1, index=tr.index)

    tr["_round_temp"] = round_col
    tr = tr.sort_values(["anonymised_id", "_round_temp"]).reset_index(drop=True)

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
    tr = tr.drop(columns=["_round_temp", "user_past_sum"])

    # Test set mapping
    user_total_sum = tr.groupby("anonymised_id")[target].sum()
    user_total_count = tr.groupby("anonymised_id")[target].count()
    user_total_rate = user_total_sum / user_total_count

    te["user_prev_obs_count"] = te["anonymised_id"].map(user_total_count).fillna(0.0)
    te["user_past_employed_rate"] = te["anonymised_id"].map(user_total_rate)
    te["user_has_past_history"] = (te["user_prev_obs_count"] > 0).astype(float)

    # Apply standard date/lag cleaning
    tr_clean = clean_base_dataframe(tr)
    te_clean = clean_base_dataframe(te)

    if "anonymised_id" in tr_clean.columns:
        tr_clean = tr_clean.drop(columns=["anonymised_id"])
    if "anonymised_id" in te_clean.columns:
        te_clean = te_clean.drop(columns=["anonymised_id"])

    return tr_clean, te_clean


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
# 4. LOAD & PREPARE DUAL FEATURE MATRICES
# ============================================================

print("\n============================================")
print("LOADING DATA & PREPARING DUAL FEATURE MATRICES")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()

# 1. Feature Set A (Exp 51 Macro View)
tr_a, te_a = extract_view_a(train_raw, test_raw)
common_a = [c for c in tr_a.columns if c in te_a.columns and c != target]

y_a = tr_a[target].reset_index(drop=True)
X_tr_a = tr_a[common_a].reset_index(drop=True)
X_te_a = te_a[common_a].reset_index(drop=True)

# Drop high cardinality (>100)
cat_a = X_tr_a.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
high_card_a = [c for c in cat_a if X_tr_a[c].nunique(dropna=True) > 100]
if high_card_a:
    X_tr_a = X_tr_a.drop(columns=high_card_a)
    X_te_a = X_te_a.drop(columns=high_card_a)

cat_a = X_tr_a.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_a = X_tr_a.select_dtypes(include=[np.number]).columns.tolist()

# 2. Feature Set B (Exp 53 Longitudinal Track Record View)
tr_b, te_b = extract_view_b(train_raw, test_raw)
common_b = [c for c in tr_b.columns if c in te_b.columns and c != target]

y_b = tr_b[target].reset_index(drop=True)
X_tr_b = tr_b[common_b].reset_index(drop=True)
X_te_b = te_b[common_b].reset_index(drop=True)

cat_b = X_tr_b.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
high_card_b = [c for c in cat_b if X_tr_b[c].nunique(dropna=True) > 100]
if high_card_b:
    X_tr_b = X_tr_b.drop(columns=high_card_b)
    X_te_b = X_te_b.drop(columns=high_card_b)

cat_b = X_tr_b.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_b = X_tr_b.select_dtypes(include=[np.number]).columns.tolist()

print(f"View A (Exp 51 Base): {len(num_a)} num, {len(cat_a)} cat features")
print(f"View B (Exp 53 Longitudinal): {len(num_b)} num, {len(cat_b)} cat features")


# ============================================================
# 5. TRAIN DUAL MULTI-SEED ENSEMBLES (70 TOTAL MODELS)
# ============================================================

SEEDS = [42, 101, 777, 2024, 999, 1337, 555]

def run_multi_seed_titan(X_train, y_train, X_test, num_cols, cat_cols, view_name):
    """Fits 35-model ensemble on specified feature view."""
    print(f"\n--- Training 35-Model Titan on {view_name} ---")
    lin_logits = []
    neu_logits = []

    models_sample = get_champion_models(seed=42)

    for m_name, (_, use_q, m_family) in models_sample.items():
        for seed in SEEDS:
            m_pool = get_champion_models(seed=seed)
            estimator, use_quantile, _ = m_pool[m_name]

            preprocessor = build_preprocessor(num_cols, cat_cols, use_quantile=use_quantile)
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator)
            ])
            pipe.fit(X_train, y_train)
            probs = pipe.predict_proba(X_test)[:, 1]

            probs_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
            logits = np.log(probs_clipped / (1.0 - probs_clipped))

            if m_family == "linear":
                lin_logits.append(logits)
            else:
                neu_logits.append(logits)

    mean_lin = np.mean(lin_logits, axis=0)
    mean_neu = np.mean(neu_logits, axis=0)
    return 0.80 * mean_lin + 0.20 * mean_neu


logits_view_a = run_multi_seed_titan(X_tr_a, y_a, X_te_a, num_a, cat_a, "View A (Exp 51)")
logits_view_b = run_multi_seed_titan(X_tr_b, y_b, X_te_b, num_b, cat_b, "View B (Exp 53)")


# ============================================================
# 6. DUAL-VIEW LOG-ODDS INTEGRATION
# ============================================================

print("\n============================================")
print("INTEGRATING PREDICTIONS VIA DUAL-VIEW LOG-ODDS BLEND")
print("============================================")

# 55% Weight on Base Macro Titan + 45% Weight on Longitudinal Track Titan
grand_master_logits = 0.55 * logits_view_a + 0.45 * logits_view_b
final_probabilities = 1.0 / (1.0 + np.exp(-grand_master_logits))


# ============================================================
# 7. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp54_grand_master_blend.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 54 COMPLETE")
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
print("Exp 53 Longitudinal Track      : 0.65671")
print("Exp 51 Base Titan Hybrid       : 0.65693 (Personal Best)")
print("Exp 54 Grand Master Dual Blend : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")