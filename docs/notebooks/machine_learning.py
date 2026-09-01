# ============================================================
# EXPERIMENT 57 — GRAND CONSENSUS & TEMPERATURE-SCALED TITAN
# (10-SEED EXP 51 ENGINE + AUTOMATED TOP-SUBMISSION CONSENSUS)
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


print("============================================")
print("EXPERIMENT 57")
print("GRAND CONSENSUS & TEMPERATURE-SCALED TITAN")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR


# ============================================================
# 2. PROVEN 0.65693 FEATURE PIPELINE
# ============================================================

target = "employed_status"

def clean_and_prepare_features(df):
    """Restores the exact, high-performing raw feature space from Exp 51."""
    data = df.copy()

    if target in data.columns:
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data.dropna(subset=[target]).copy()
        data[target] = data[target].astype(int)

    if "anonymised_id" in data.columns:
        data = data.drop(columns=["anonymised_id"])

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
# 3. WINNING EXP 51 MODEL FACTORIES
# ============================================================

def get_champion_models(seed=42):
    """Returns the top linear champions + neural MLP."""
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
# 4. LOAD & PREPARE DATASET
# ============================================================

print("\n============================================")
print("LOADING DATASET & EXTRACTING FEATURES")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()

train_clean = clean_and_prepare_features(train_raw)
test_clean = clean_and_prepare_features(test_raw)

common_cols = [c for c in train_clean.columns if c in test_clean.columns and c != target]

y_full = train_clean[target].reset_index(drop=True)
X_full = train_clean[common_cols].reset_index(drop=True)
X_test_full = test_clean[common_cols].reset_index(drop=True)

# Drop high-cardinality categorical (>100)
categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
high_cardinality = [c for c in categorical_features if X_full[c].nunique(dropna=True) > 100]
if high_cardinality:
    X_full = X_full.drop(columns=high_cardinality)
    X_test_full = X_test_full.drop(columns=high_cardinality)

categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X_full.select_dtypes(include=[np.number]).columns.tolist()

print(f"Full dataset: {len(X_full)} observations | Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")


# ============================================================
# 5. TRAIN 10-SEED TITAN ENGINE (50 TOTAL SUB-MODELS)
# ============================================================

print("\n============================================")
print("TRAINING 10-SEED TITAN ENGINE (50 SUB-MODELS)")
print("============================================")

SEEDS = [42, 101, 777, 2024, 999, 1337, 555, 888, 314, 271]

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

        probs_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(probs_clipped / (1.0 - probs_clipped))

        if model_family == "linear":
            linear_logits_list.append(logits)
        else:
            neural_logits_list.append(logits)

# 80% Linear + 20% Neural in Logit Space
mean_linear_logits = np.mean(linear_logits_list, axis=0)
mean_neural_logits = np.mean(neural_logits_list, axis=0)

titan_logits = 0.80 * mean_linear_logits + 0.20 * mean_neural_logits

# Apply Temperature Scaling (T = 1.05) to refine calibration
TEMPERATURE = 1.05
scaled_titan_logits = titan_logits / TEMPERATURE
titan_probabilities = 1.0 / (1.0 + np.exp(-scaled_titan_logits))


# ============================================================
# 6. AUTOMATED TOP-SUBMISSION CONSENSUS BLEND
# ============================================================

print("\n============================================")
print("DISCOVERING PAST TOP SUBMISSIONS FOR CONSENSUS")
print("============================================")

# Priority list of past top submissions
candidate_files = {
    "submission_exp51_restored_titan_ensemble.csv": 0.40,  # 0.65693 Personal Best
    "submission_exp53_historical_track_titan_ensemble.csv": 0.30,  # 0.65671
    "submission_exp47_pure_logit_ensemble.csv": 0.15,      # 0.65635
    "submission_exp44_optimized_linear_ensemble.csv": 0.15  # 0.65630
}

discovered_submissions = []
discovered_weights = []

for f_name, w in candidate_files.items():
    file_path = CURRENT_DIR / f_name
    if file_path.exists():
        sub_df = pd.read_csv(file_path)
        if target in sub_df.columns and len(sub_df) == len(test_raw):
            p = np.clip(sub_df[target].to_numpy(), 1e-6, 1.0 - 1e-6)
            discovered_submissions.append(np.log(p / (1.0 - p)))
            discovered_weights.append(w)
            print(f" -> Found past top submission: {f_name} (Weight: {w*100:.0f}%)")

if len(discovered_submissions) > 0:
    # Normalize weights
    total_w = sum(discovered_weights)
    norm_w = [w / total_w for w in discovered_weights]
    past_consensus_logits = sum(w * sub for w, sub in zip(norm_w, discovered_submissions))

    # 60% Fresh 10-Seed Titan + 40% Past Champions Consensus
    final_consensus_logits = 0.60 * scaled_titan_logits + 0.40 * past_consensus_logits
    final_probabilities = 1.0 / (1.0 + np.exp(-final_consensus_logits))
    print("\n✅ Successfully integrated fresh 10-Seed Titan with past top-performing champions.")
else:
    final_probabilities = titan_probabilities
    print("\nℹ️ Generating predictions purely from the fresh 10-Seed Titan engine.")


# ============================================================
# 7. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp57_grand_consensus_ensemble.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 57 COMPLETE")
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
print("Exp 57 Grand Consensus Blend   : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")