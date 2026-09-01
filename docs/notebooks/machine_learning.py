# ============================================================
# EXPERIMENT 56 — HIGH-LEVERAGE LABOUR MARKET INTERACTIONS
# (JOB MOMENTUM, STEM GATEWAYS, METRO ADVANTAGE & TITAN BLEND)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


print("============================================")
print("EXPERIMENT 56")
print("HIGH-LEVERAGE LABOUR MARKET INTERACTIONS")
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
# 2. HIGH-LEVERAGE TARGETED VARIABLE ENGINEERING
# ============================================================

target = "employed_status"

def parse_matric_band(val):
    """Parses banded percentage strings into continuous numerical scores."""
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


def extract_high_leverage_features(df):
    """
    Constructs high-signal domain features focusing on employment momentum,
    STEM credentials, geographic economic hubs, and demographic interactions.
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

    # 3. Calendar Features
    if "survey_date" in data.columns:
        data["survey_date"] = pd.to_datetime(data["survey_date"], errors="coerce")
        data["survey_year"] = data["survey_date"].dt.year
        data["survey_month"] = data["survey_date"].dt.month
        data["survey_dayofyear"] = data["survey_date"].dt.dayofyear
        data = data.drop(columns=["survey_date"])

    # ------------------------------------------------------------
    # DRIVER 1: LABOUR MOMENTUM & TENURE STABILITY
    # ------------------------------------------------------------
    if "employed_lag" in data.columns:
        data["is_first_time"] = data["employed_lag"].isna().astype(float)
        data["employed_lag_val"] = data["employed_lag"].fillna(-1).astype(float)
    else:
        data["is_first_time"] = 1.0
        data["employed_lag_val"] = -1.0

    tenure_raw = pd.to_numeric(data.get("tenure_lag", 0), errors="coerce").fillna(0)
    data["tenure_lag_log"] = np.log1p(tenure_raw.clip(lower=0))
    # Stable formal employment indicator (prior job tenure > 6 months)
    data["is_long_tenure_retained"] = ((data["employed_lag_val"] == 1) & (tenure_raw > 180)).astype(float)

    days_raw = pd.to_numeric(data.get("days_since_last_obs", 0), errors="coerce").fillna(0)
    data["days_since_last_obs_log"] = np.log1p(days_raw.clip(lower=0))

    # Prior state nuances (Searching vs Discouraged vs Studying)
    if "status_broad_lag" in data.columns:
        data["was_studying_lag"] = data["status_broad_lag"].astype(str).str.contains("study", case=False, na=False).astype(float)
    else:
        data["was_studying_lag"] = 0.0

    # ------------------------------------------------------------
    # DRIVER 2: STEM CREDENTIALS & MATRIC GATEWAYS
    # ------------------------------------------------------------
    matric_cols = ["matric_englishhome", "matric_englishadd", "matric_mathpure", "matric_physicalscience", "matric_mathlit"]
    for m in matric_cols:
        if m in data.columns:
            data[f"{m}_score"] = data[m].apply(parse_matric_band)

    # Has pure math vs math literacy vs science
    data["has_pure_math"] = (data.get("matric_mathpure_score", pd.Series(np.nan, index=data.index)).notna()).astype(float)
    data["has_science"] = (data.get("matric_physicalscience_score", pd.Series(np.nan, index=data.index)).notna()).astype(float)
    data["stem_subject_count"] = data["has_pure_math"] + data["has_science"]

    # Best math score
    m_pure = data.get("matric_mathpure_score", pd.Series(np.nan, index=data.index))
    m_lit = data.get("matric_mathlit_score", pd.Series(np.nan, index=data.index))
    data["best_math_mark"] = pd.concat([m_pure, m_lit], axis=1).max(axis=1).fillna(0.0)

    # ------------------------------------------------------------
    # DRIVER 3: SPATIAL & ECONOMIC HUB ADVANTAGE
    # ------------------------------------------------------------
    if "province" in data.columns:
        prov_str = data["province"].astype(str).str.strip().str.lower()
        # Top 2 economic centers in South Africa (Gauteng + Western Cape)
        data["is_primary_metro_province"] = prov_str.isin(["gauteng", "western cape"]).astype(float)
    else:
        data["is_primary_metro_province"] = 0.0

    # ------------------------------------------------------------
    # DRIVER 4: HUMAN CAPITAL & BEHAVIOURAL READINESS
    # ------------------------------------------------------------
    age_raw = pd.to_numeric(data.get("age", 22), errors="coerce").fillna(22).clip(18, 35)
    data["age_clean"] = age_raw
    data["potential_work_years"] = (age_raw - 18).clip(lower=0)

    quintile_raw = pd.to_numeric(data.get("school_quintile", 0), errors="coerce").fillna(0)
    data["school_quintile_num"] = quintile_raw
    data["is_resourced_school"] = (quintile_raw >= 4).astype(float)

    readiness_raw = pd.to_numeric(data.get("work_readiness_score", 0.5), errors="coerce").fillna(0.5)
    data["work_readiness_num"] = readiness_raw
    
    # Interaction: High readiness + Resourced background
    data["readiness_x_quintile"] = readiness_raw * (quintile_raw / 5.0)

    # Tertiary & Vocational Flags
    data["has_tertiary"] = (data.get("institution_type", pd.Series(np.nan, index=data.index)).notna()).astype(float)
    data["has_seta"] = (data.get("seta", pd.Series(np.nan, index=data.index)).notna()).astype(float)

    # Auto-convert residual object columns
    for col in data.columns:
        if col != target and data[col].dtype == "object":
            converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().sum() > 0.6 * data[col].notna().sum():
                data[col] = converted

    return data


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
# 3. HIGH-PERFORMING MULTI-PARADIGM SUITE
# ============================================================

def get_titan_models(seed=42):
    """Returns the top linear champions, shrinkage LDA, and neural MLP."""
    return {
        "LogReg_L2_C008": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False,
            0.30
        ),
        "LogReg_L2_C010": (
            LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False,
            0.25
        ),
        "LogReg_ElasticNet_C010": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000, random_state=seed),
            False,
            0.20
        ),
        "LDA_Shrinkage_015": (
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.15),
            False,
            0.10
        ),
        "MLP_Medium_64_32": (
            MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.01,
                batch_size=128, learning_rate_init=0.001, max_iter=350, early_stopping=True,
                n_iter_no_change=20, validation_fraction=0.15, random_state=seed
            ),
            False,
            0.15
        )
    }


# ============================================================
# 4. LOAD & BENCHMARK ON SEQUENTIAL ROUND DATASETS
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
                r_train_clean = extract_high_leverage_features(r_train_raw)
                r_test_clean = extract_high_leverage_features(r_test_raw)

                common_features = [c for c in r_train_clean.columns if c in r_test_clean.columns and c != target]
                y_tr_vals = r_train_clean[target]
                y_te_vals = r_test_clean[target]

                if len(common_features) >= 15 and y_tr_vals.nunique() >= 2 and y_te_vals.nunique() >= 2:
                    round_data_list.append({
                        "name": r_dir.name,
                        "X_train": r_train_clean[common_features].reset_index(drop=True),
                        "y_train": y_tr_vals.reset_index(drop=True),
                        "X_test": r_test_clean[common_features].reset_index(drop=True),
                        "y_test": y_te_vals.reset_index(drop=True)
                    })
                    print(f" -> Loaded {r_dir.name.upper()} | Train: {len(r_train_clean)} | Test: {len(r_test_clean)} | Features: {len(common_features)}")


# ============================================================
# 5. RUN TOURNAMENT ACROSS VALID ROUNDS
# ============================================================

print("\n============================================")
print("BENCHMARKING MODELS WITH HIGH-LEVERAGE FEATURES")
print("============================================")

candidate_dict = get_titan_models(seed=42)
tournament_scores = {name: [] for name in candidate_dict.keys()}
round_names = []

for r_data in round_data_list:
    r_name = r_data["name"]
    round_names.append(r_name.upper())
    X_tr = r_data["X_train"].copy()
    y_tr = r_data["y_train"].copy()
    X_te = r_data["X_test"].copy()
    y_te = r_data["y_test"].copy()

    # Drop high-cardinality (>100)
    cat_cols = X_tr.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    high_card = [c for c in cat_cols if X_tr[c].nunique(dropna=True) > 100]
    if high_card:
        X_tr = X_tr.drop(columns=high_card)
        X_te = X_te.drop(columns=high_card)

    cat_cols = X_tr.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()

    models_dict = get_titan_models(seed=42)

    for model_name, (model_obj, use_quantile, _) in models_dict.items():
        preprocessor = build_preprocessor(num_cols, cat_cols, use_quantile=use_quantile)
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model_obj)
        ])
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_te)[:, 1]
        score = roc_auc_score(y_te, probs)
        tournament_scores[model_name].append(score)

# Leaderboard Summary
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


# ============================================================
# 6. TRAIN MULTI-SEED TITAN ENSEMBLE ON FULL DATASET
# ============================================================

print("\n============================================")
print("TRAINING MULTI-SEED ENSEMBLE ON FULL DATASET (35 MODELS)")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()

train_clean = extract_high_leverage_features(train_raw)
test_clean = extract_high_leverage_features(test_raw)

common_cols = [c for c in train_clean.columns if c in test_clean.columns and c != target]

y_full = train_clean[target].reset_index(drop=True)
X_full = train_clean[common_cols].reset_index(drop=True)
X_test_full = test_clean[common_cols].reset_index(drop=True)

# Drop high-cardinality (>100)
categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
high_cardinality = [c for c in categorical_features if X_full[c].nunique(dropna=True) > 100]
if high_cardinality:
    X_full = X_full.drop(columns=high_cardinality)
    X_test_full = X_test_full.drop(columns=high_cardinality)

categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X_full.select_dtypes(include=[np.number]).columns.tolist()

print(f"Full dataset: {len(X_full)} observations | Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")

SEEDS = [42, 101, 777, 2024, 999, 1337, 555]

model_specs = get_titan_models(seed=42)
final_accumulated_logits = np.zeros(len(X_test_full))

for m_name, (_, use_quantile, weight) in model_specs.items():
    print(f" -> Fitting {m_name:<25} (Weight = {weight*100:.0f}%) across {len(SEEDS)} seeds...")
    
    seed_logits_list = []
    for seed in SEEDS:
        models_pool = get_titan_models(seed=seed)
        model_estimator, use_q, _ = models_pool[m_name]

        preprocessor = build_preprocessor(numerical_features, categorical_features, use_quantile=use_q)
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model_estimator)
        ])
        pipe.fit(X_full, y_full)
        probs = pipe.predict_proba(X_test_full)[:, 1]

        # Logit-space conversion
        probs_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(probs_clipped / (1.0 - probs_clipped))
        seed_logits_list.append(logits)

    model_mean_logits = np.mean(seed_logits_list, axis=0)
    final_accumulated_logits += weight * model_mean_logits

final_probabilities = 1.0 / (1.0 + np.exp(-final_accumulated_logits))


# ============================================================
# 7. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp56_high_leverage_interactions.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 56 COMPLETE")
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
print("Exp 44 Top-3 Linear Blend          : 0.65630")
print("Exp 51 Base Titan Hybrid           : 0.65693 (Personal Best)")
print("Exp 54 Grand Master Dual Blend     : 0.65670")
print("Exp 56 High-Leverage Interactions  : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")