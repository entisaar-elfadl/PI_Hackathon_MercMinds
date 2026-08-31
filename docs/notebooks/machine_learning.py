# ============================================================
# EXPERIMENT 50 — PURE-LOGIT DOMAIN CHAMPION
# (SAFE TARGET VALIDATION + 1/0 NA INDICATORS + LOGIT ENSEMBLE)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score


print("============================================")
print("EXPERIMENT 50")
print("PURE-LOGIT DOMAIN CHAMPION ENSEMBLE")
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
# 2. SAFE DOMAIN FEATURE EXTRACTION & 1/0 NA FLAGS
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


def extract_clean_domain_features(df):
    """
    Safely extracts domain features, adds 1/0 binary missingness indicators,
    and removes Round 9 extrapolation drift features.
    """
    data = df.copy()

    # 1. Target Cleaning
    if target in data.columns:
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data.dropna(subset=[target]).copy()
        data[target] = data[target].astype(int)

    # 2. Identification
    if "anonymised_id" in data.columns:
        data = data.drop(columns=["anonymised_id"])

    # 3. Calendar Month & Day-of-Year
    if "survey_date" in data.columns:
        data["survey_date"] = pd.to_datetime(data["survey_date"], errors="coerce")
        data["survey_month"] = data["survey_date"].dt.month
        data["survey_dayofyear"] = data["survey_date"].dt.dayofyear
        data = data.drop(columns=["survey_date"])

    # Drop explicit round numbers to prevent Round 9 extrapolation drift
    for col_to_drop in ["current_round", "lag_round", "sample", "sample_first"]:
        if col_to_drop in data.columns:
            data = data.drop(columns=[col_to_drop])

    # 4. Safe Lag Features & Binary 1/0 NA Flags
    if "employed_lag" in data.columns:
        data["employed_lag_is_na"] = data["employed_lag"].isna().astype(int)
        data["employed_lag_tristate"] = data["employed_lag"].fillna(-1).astype(float)
    else:
        data["employed_lag_is_na"] = 1
        data["employed_lag_tristate"] = -1.0

    if "days_since_last_obs" in data.columns:
        data["days_since_last_obs_is_na"] = data["days_since_last_obs"].isna().astype(int)
        data["days_since_last_obs_log"] = np.log1p(data["days_since_last_obs"].fillna(0).clip(lower=0))
    else:
        data["days_since_last_obs_is_na"] = 1
        data["days_since_last_obs_log"] = 0.0

    if "tenure_lag" in data.columns:
        data["tenure_lag_is_na"] = data["tenure_lag"].isna().astype(int)
        data["tenure_lag_log"] = np.log1p(data["tenure_lag"].fillna(0).clip(lower=0))
    else:
        data["tenure_lag_is_na"] = 1
        data["tenure_lag_log"] = 0.0

    # Composite first-time observation indicator
    data["is_first_time_participant"] = (
        (data["employed_lag_is_na"] == 1) | 
        (data["days_since_last_obs_is_na"] == 1)
    ).astype(int)

    # 5. Matric Subject Performance Bands -> Continuous Numerical + NA Flags
    matric_cols = [
        "matric_englishhome", "matric_englishadd", 
        "matric_mathpure", "matric_physicalscience", "matric_mathlit"
    ]
    
    for m_col in matric_cols:
        if m_col in data.columns:
            data[f"{m_col}_is_na"] = data[m_col].isna().astype(int)
            data[f"{m_col}_num"] = data[m_col].apply(parse_matric_band)
            data = data.drop(columns=[m_col])

    num_m_cols = [f"{m}_num" for m in matric_cols if f"{m}_num" in data.columns]
    if len(num_m_cols) > 0:
        data["matric_subjects_count"] = data[num_m_cols].notna().sum(axis=1)
        data["matric_avg_score"] = data[num_m_cols].mean(axis=1).fillna(-1)
        data["has_no_matric_records"] = (data["matric_subjects_count"] == 0).astype(int)
    
    if "matric_mathpure_num" in data.columns and "matric_mathlit_num" in data.columns:
        data["matric_best_math"] = data[["matric_mathpure_num", "matric_mathlit_num"]].max(axis=1).fillna(-1)

    if "matric_englishhome_num" in data.columns and "matric_englishadd_num" in data.columns:
        data["matric_best_english"] = data[["matric_englishhome_num", "matric_englishadd_num"]].max(axis=1).fillna(-1)

    # 6. Education, SETA & Employability
    if "school_quintile" in data.columns:
        data["school_quintile"] = pd.to_numeric(data["school_quintile"], errors="coerce").fillna(0)

    if "work_readiness_score" in data.columns:
        data["work_readiness_is_na"] = data["work_readiness_score"].isna().astype(int)
        data["work_readiness_score"] = pd.to_numeric(data["work_readiness_score"], errors="coerce").fillna(-1)

    if "institution_type" in data.columns:
        data["has_tertiary_record"] = data["institution_type"].notna().astype(int)

    if "seta" in data.columns:
        data["has_seta_credential"] = data["seta"].notna().astype(int)

    # 7. Auto-convert object columns
    for col in data.columns:
        if col != target and data[col].dtype == "object":
            converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().sum() > 0.6 * data[col].notna().sum():
                data[col] = converted

    return data


def build_preprocessor(numerical_cols, categorical_cols, use_quantile=False):
    """Standard or Quantile-Gaussian preprocessor."""
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
# 3. PURE LOGIT MODEL SUITE
# ============================================================

def get_logit_models(seed=42):
    """Returns the calibrated regularized linear suite."""
    return {
        "LogReg_L2_C007": (
            LogisticRegression(C=0.07, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False
        ),
        "LogReg_L2_C010": (
            LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False
        ),
        "LogReg_ElasticNet_C008": (
            LogisticRegression(C=0.08, penalty="elasticnet", solver="saga", l1_ratio=0.12, max_iter=1000, random_state=seed),
            False
        ),
        "LogReg_ElasticNet_C010": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.18, max_iter=1000, random_state=seed),
            False
        ),
        "Quantile_LogReg_L2_C008": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            True
        ),
        "Quantile_LogReg_ElasticNet_C010": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000, random_state=seed),
            True
        )
    }


# ============================================================
# 4. LOAD AND VALIDATE ON SEQUENTIAL ROUND DATASETS
# ============================================================

print("\n============================================")
print("DISCOVERING & VALIDATING ON SEQUENTIAL ROUNDS")
print("============================================")

round_data_list = []

if ROUND_DIR.exists():
    round_folders = sorted([f for f in ROUND_DIR.glob("round_*") if f.is_dir()])
    for r_dir in round_folders:
        csv_files = list(r_dir.glob("*.csv"))
        if len(csv_files) >= 2:
            csv_files = sorted(csv_files, key=lambda f: f.stat().st_size)
            test_file, train_file = csv_files[0], csv_files[1]

            r_train_raw = pd.read_csv(train_file)
            r_test_raw = pd.read_csv(test_file)

            if target in r_train_raw.columns and target in r_test_raw.columns:
                r_train_clean = extract_clean_domain_features(r_train_raw)
                r_test_clean = extract_clean_domain_features(r_test_raw)

                common_features = [c for c in r_train_clean.columns if c in r_test_clean.columns and c != target]

                # STRICT VALIDITY CHECK: Must have >= 20 common features AND both classes (0 and 1) in train & test
                y_tr_vals = r_train_clean[target]
                y_te_vals = r_test_clean[target]

                if len(common_features) >= 20 and y_tr_vals.nunique() >= 2 and y_te_vals.nunique() >= 2:
                    round_data_list.append({
                        "name": r_dir.name,
                        "X_train": r_train_clean[common_features].reset_index(drop=True),
                        "y_train": y_tr_vals.reset_index(drop=True),
                        "X_test": r_test_clean[common_features].reset_index(drop=True),
                        "y_test": y_te_vals.reset_index(drop=True)
                    })
                    print(f" -> Loaded {r_dir.name.upper()} | Train: {len(r_train_clean)} | Test: {len(r_test_clean)} | Features: {len(common_features)}")
                else:
                    print(f" -> Skipping {r_dir.name.upper()} (Single target class or insufficient features)")


# ============================================================
# 5. RUN SAFETY-GATED ROUND TOURNAMENT
# ============================================================

print("\n============================================")
print("BENCHMARKING PURE-LOGIT MODELS ACROSS ROUNDS")
print("============================================")

candidate_dict = get_logit_models(seed=42)
tournament_scores = {name: [] for name in candidate_dict.keys()}
round_names = []

for r_data in round_data_list:
    r_name = r_data["name"]
    round_names.append(r_name.upper())
    X_tr = r_data["X_train"].copy()
    y_tr = r_data["y_train"].copy()
    X_te = r_data["X_test"].copy()
    y_te = r_data["y_test"].copy()

    cat_cols = X_tr.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    high_card = [c for c in cat_cols if X_tr[c].nunique(dropna=True) > 100]
    if high_card:
        X_tr = X_tr.drop(columns=high_card)
        X_te = X_te.drop(columns=high_card)

    cat_cols = X_tr.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()

    models_dict = get_logit_models(seed=42)

    for model_name, (model_obj, use_quantile) in models_dict.items():
        preprocessor = build_preprocessor(num_cols, cat_cols, use_quantile=use_quantile)
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model_obj)
        ])
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_te)[:, 1]
        score = roc_auc_score(y_te, probs)
        tournament_scores[model_name].append(score)

# Leaderboard
results_table = []
for model_name, scores in tournament_scores.items():
    mean_score = np.mean(scores) if scores else 0.0
    row = {"Model Architecture": model_name, "Mean Round AUC": mean_score}
    for i, s in enumerate(scores):
        row[f"{round_names[i]} AUC"] = s
    results_table.append(row)

leaderboard_df = pd.DataFrame(results_table).sort_values(by="Mean Round AUC", ascending=False).reset_index(drop=True)

print("\n============================================")
print("ROUND TOURNAMENT LEADERBOARD")
print("============================================")
print(leaderboard_df.to_string(index=False))

top_performers = leaderboard_df.head(4)["Model Architecture"].tolist()
print(f"\n🏆 TOP 4 CHAMPIONS SELECTED: {top_performers}")


# ============================================================
# 6. TRAIN MULTI-SEED LOG-ODDS ENSEMBLE ON FULL DATASET
# ============================================================

print("\n============================================")
print("TRAINING DOMAIN PURE-LOGIT ENSEMBLE ON FULL DATASET")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")

train_clean = extract_clean_domain_features(train_raw)
test_clean = extract_clean_domain_features(test_raw)

common_cols = [c for c in train_clean.columns if c in test_clean.columns and c != target]

y_full = train_clean[target].reset_index(drop=True)
X_full = train_clean[common_cols].reset_index(drop=True)
X_test_full = test_clean[common_cols].reset_index(drop=True)

categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
high_cardinality = [c for c in categorical_features if X_full[c].nunique(dropna=True) > 100]
if high_cardinality:
    X_full = X_full.drop(columns=high_cardinality)
    X_test_full = X_test_full.drop(columns=high_cardinality)

categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X_full.select_dtypes(include=[np.number]).columns.tolist()

print(f"Full dataset: {len(X_full)} rows | Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")

SEEDS = [42, 101, 777, 2024, 999, 1337, 555]
all_model_logits = []

for m_idx, m_name in enumerate(top_performers):
    print(f" -> Fitting Champion #{m_idx + 1}: {m_name} across {len(SEEDS)} seeds...")

    for s_idx, seed in enumerate(SEEDS):
        candidate_pool = get_logit_models(seed=seed)
        model_estimator, use_quantile = candidate_pool[m_name]

        preprocessor = build_preprocessor(numerical_features, categorical_features, use_quantile=use_quantile)
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model_estimator)
        ])
        pipe.fit(X_full, y_full)
        probs = pipe.predict_proba(X_test_full)[:, 1]

        # Convert to log-odds (logit space)
        probs_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(probs_clipped / (1.0 - probs_clipped))
        all_model_logits.append(logits)

# Average in Log-Odds space and convert back with Sigmoid
mean_logits = np.mean(all_model_logits, axis=0)
final_probabilities = 1.0 / (1.0 + np.exp(-mean_logits))


# ============================================================
# 7. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp50_domain_pure_logit_ensemble.csv"

submission = pd.DataFrame({
    "anonymised_id": test_raw["anonymised_id"],
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 50 COMPLETE")
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
print("Personal Best Benchmark        : 0.65671")
print("Exp 50 Domain Pure-Logit Blend : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")