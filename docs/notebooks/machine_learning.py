# ============================================================
# EXPERIMENT 44 — PRECISION REGULARIZED LINEAR SUITE & ENSEMBLE
# (FINE-TUNED L2, ELASTICNET, SAGA & DISTRIBUTION NORMALIZATION)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


print("============================================")
print("EXPERIMENT 44")
print("PRECISION REGULARIZED LINEAR SUITE & ENSEMBLE")
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
# 2. DATA CLEANING & TYPE COERCION
# ============================================================

target = "employed_status"

def clean_data_and_extract_features(df):
    """Cleans IDs, dates, and automatically coerces numerical datatypes."""
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
# 3. CANDIDATE REGULARIZED LINEAR MODELS
# ============================================================

def get_linear_candidates(seed=42):
    """Returns a fine-grained suite of regularized linear models."""
    return {
        "LogReg L2 (C=0.05)": (
            LogisticRegression(C=0.05, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False
        ),
        "LogReg L2 (C=0.08)": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False
        ),
        "LogReg L2 (C=0.10) [Exp43 Champ]": (
            LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False
        ),
        "LogReg L2 (C=0.15)": (
            LogisticRegression(C=0.15, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False
        ),
        "LogReg ElasticNet (C=0.10, L1=0.15)": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000, random_state=seed),
            False
        ),
        "LogReg ElasticNet (C=0.08, L1=0.25)": (
            LogisticRegression(C=0.08, penalty="elasticnet", solver="saga", l1_ratio=0.25, max_iter=1000, random_state=seed),
            False
        ),
        "Quantile LogReg L2 (C=0.10)": (
            LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            True
        ),
        "Quantile LogReg L2 (C=0.08)": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            True
        ),
        "Calibrated Ridge (alpha=2.0)": (
            CalibratedClassifierCV(estimator=RidgeClassifier(alpha=2.0, random_state=seed), method="sigmoid", cv=3),
            False
        )
    }


# ============================================================
# 4. LOAD VALID ROUND DATASETS
# ============================================================

print("\n============================================")
print("DISCOVERING SEQUENTIAL ROUND DATASETS")
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
                r_train = clean_data_and_extract_features(r_train_raw)
                r_test = clean_data_and_extract_features(r_test_raw)

                common_features = [c for c in r_train.columns if c in r_test.columns and c != target]
                if len(common_features) >= 5:
                    round_data_list.append({
                        "name": r_dir.name,
                        "X_train": r_train[common_features],
                        "y_train": r_train[target],
                        "X_test": r_test[common_features],
                        "y_test": r_test[target]
                    })
                    print(f" -> Loaded {r_dir.name.upper()} | Train: {len(r_train)} | Test: {len(r_test)} | Features: {len(common_features)}")


# ============================================================
# 5. RUN LINEAR SUITE TOURNAMENT ACROSS ROUNDS
# ============================================================

print("\n============================================")
print("STARTING REGULARIZED LINEAR TOURNAMENT")
print("============================================")

candidate_dict = get_linear_candidates(seed=42)
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

    models_dict = get_linear_candidates(seed=42)

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
    row = {"Linear Architecture": model_name, "Mean Round AUC": mean_score}
    for i, s in enumerate(scores):
        row[f"{round_names[i]} AUC"] = s
    results_table.append(row)

leaderboard_df = pd.DataFrame(results_table).sort_values(by="Mean Round AUC", ascending=False).reset_index(drop=True)

print("\n============================================")
print("LINEAR TOURNAMENT LEADERBOARD")
print("============================================")
print(leaderboard_df.to_string(index=False))

# Select Top 3 Performing Models for Blending
top_3_models = leaderboard_df.head(3)["Linear Architecture"].tolist()
print(f"\n🏆 TOP 3 MODELS SELECTED FOR BLENDING: {top_3_models}")


# ============================================================
# 6. TRAIN MULTI-MODEL ENSEMBLE ON FULL DATASET
# ============================================================

print("\n============================================")
print("TRAINING TOP-3 LINEAR ENSEMBLE ON FULL DATASET")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")

train_df = clean_data_and_extract_features(train_raw)
test_df = clean_data_and_extract_features(test_raw)

common_cols = [c for c in train_df.columns if c in test_df.columns and c != target]

y_full = train_df[target].reset_index(drop=True)
X_full = train_df[common_cols].reset_index(drop=True)
X_test_full = test_df[common_cols].reset_index(drop=True)

categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
high_cardinality = [c for c in categorical_features if X_full[c].nunique(dropna=True) > 100]
if high_cardinality:
    X_full = X_full.drop(columns=high_cardinality)
    X_test_full = X_test_full.drop(columns=high_cardinality)

categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X_full.select_dtypes(include=[np.number]).columns.tolist()

print(f"Training on {len(X_full)} full dataset rows...")

# Build ensemble of the Top 3 Tournament Champions across multiple seeds
SEEDS = [42, 101, 777, 2024, 999]
all_predictions = []

for m_idx, m_name in enumerate(top_3_models):
    print(f" -> Training Model #{m_idx+1}: {m_name}...")
    model_seed_preds = np.zeros((len(X_test_full), len(SEEDS)))

    for s_idx, seed in enumerate(SEEDS):
        candidate_pool = get_linear_candidates(seed=seed)
        model_estimator, use_quantile = candidate_pool[m_name]

        preprocessor = build_preprocessor(numerical_features, categorical_features, use_quantile=use_quantile)
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model_estimator)
        ])
        pipe.fit(X_full, y_full)
        model_seed_preds[:, s_idx] = pipe.predict_proba(X_test_full)[:, 1]

    all_predictions.append(model_seed_preds.mean(axis=1))

# Final weighted soft-average of top models
final_probabilities = np.mean(all_predictions, axis=0)


# ============================================================
# 7. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp44_optimized_linear_ensemble.csv"

submission = pd.DataFrame({
    "anonymised_id": test_raw["anonymised_id"],
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 44 COMPLETE")
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
print("Exp 35 Hybrid Single Blend : 0.65089")
print("Exp 37 5-Fold Hybrid Blend : 0.65247")
print("Exp 41 Sequential Hybrid   : 0.65325")
print("Exp 43 Single LogReg Champ : 0.65502 (Previous Best)")
print("Exp 44 Top-3 Linear Blend  : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")