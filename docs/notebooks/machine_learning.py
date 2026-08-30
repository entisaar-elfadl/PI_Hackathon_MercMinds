# ============================================================
# EXPERIMENT 46 — THE GRAND MULTILINEAR & GLM TOURNAMENT
# (ELASTICNET, SPLINE GLMs, ROBUST HUBER, BAYESIAN RIDGE & L2)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer, SplineTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, BayesianRidge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score


print("============================================")
print("EXPERIMENT 46")
print("THE GRAND MULTILINEAR & GLM TOURNAMENT")
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


def build_linear_preprocessor(numerical_cols, categorical_cols, transform_type="standard"):
    """
    Builds customized preprocessors for multilinear models:
    - standard: Median Impute + Standard Scale
    - quantile: Median Impute + Quantile Gaussian Normalization
    - spline: Median Impute + Spline Expansion (Generalized Additive Model)
    """
    transformers = []

    if len(numerical_cols) > 0:
        if transform_type == "quantile":
            scaler = QuantileTransformer(output_distribution="normal", random_state=42)
            num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)])
        elif transform_type == "spline":
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
                ("scaler", StandardScaler())
            ])
        else:
            num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

        transformers.append(("num", num_pipe, numerical_cols))

    if len(categorical_cols) > 0:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers)


# ============================================================
# 3. MULTILINEAR CANDIDATE SUITE
# ============================================================

# Wrapper for BayesianRidge to provide predict_proba API
class BayesianRidgeProbabilityWrapper:
    def __init__(self, alpha_1=1e-6, lambda_1=1e-6):
        self.model = BayesianRidge(alpha_1=alpha_1, lambda_1=lambda_1)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        preds = self.model.predict(X)
        probs_1 = np.clip(preds, 0.01, 0.99)
        probs_0 = 1.0 - probs_1
        return np.vstack([probs_0, probs_1]).T


def get_multilinear_candidates(seed=42):
    """Returns a rich suite of specialized multilinear and GLM architectures."""
    return {
        # 1. Fine-tuned ElasticNet Variants
        "ElasticNet (C=0.10, L1=0.15)": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000, random_state=seed),
            "standard"
        ),
        "ElasticNet (C=0.08, L1=0.20)": (
            LogisticRegression(C=0.08, penalty="elasticnet", solver="saga", l1_ratio=0.20, max_iter=1000, random_state=seed),
            "standard"
        ),
        "ElasticNet (C=0.12, L1=0.10)": (
            LogisticRegression(C=0.12, penalty="elasticnet", solver="saga", l1_ratio=0.10, max_iter=1000, random_state=seed),
            "standard"
        ),

        # 2. L2 Regularized Linear Baselines
        "Logistic L2 (C=0.08)": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            "standard"
        ),
        "Logistic L2 (C=0.10)": (
            LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            "standard"
        ),
        "Logistic L2 (C=0.06)": (
            LogisticRegression(C=0.06, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            "standard"
        ),

        # 3. Piecewise Spline GLM (Additive Linear Model)
        "Spline GLM Logistic (C=0.10)": (
            LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            "spline"
        ),

        # 4. Quantile-Normalized Multilinear
        "Quantile ElasticNet (C=0.10)": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000, random_state=seed),
            "quantile"
        ),
        "Quantile Logistic L2 (C=0.08)": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            "quantile"
        ),

        # 5. Robust Huber Margin Linear Classifier
        "Robust Huber Linear (alpha=0.001)": (
            CalibratedClassifierCV(
                estimator=SGDClassifier(loss="modified_huber", penalty="elasticnet", l1_ratio=0.15, alpha=0.001, random_state=seed),
                method="sigmoid", cv=3
            ),
            "standard"
        ),

        # 6. Bayesian Multilinear Regression (MAP)
        "Bayesian Multilinear Regression": (
            BayesianRidgeProbabilityWrapper(),
            "standard"
        ),

        # 7. Calibrated Ridge Linear Classifier
        "Calibrated Ridge (alpha=2.0)": (
            CalibratedClassifierCV(estimator=RidgeClassifier(alpha=2.0, random_state=seed), method="sigmoid", cv=3),
            "standard"
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
# 5. RUN MULTILINEAR TOURNAMENT ACROSS ROUNDS
# ============================================================

print("\n============================================")
print("STARTING MULTILINEAR & GLM TOURNAMENT")
print("============================================")

candidate_dict = get_multilinear_candidates(seed=42)
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

    models_dict = get_multilinear_candidates(seed=42)

    for model_name, (model_obj, transform_type) in models_dict.items():
        preprocessor = build_linear_preprocessor(num_cols, cat_cols, transform_type=transform_type)
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
    row = {"Multilinear Architecture": model_name, "Mean Round AUC": mean_score}
    for i, s in enumerate(scores):
        row[f"{round_names[i]} AUC"] = s
    results_table.append(row)

leaderboard_df = pd.DataFrame(results_table).sort_values(by="Mean Round AUC", ascending=False).reset_index(drop=True)

print("\n============================================")
print("MULTILINEAR TOURNAMENT LEADERBOARD")
print("============================================")
print(leaderboard_df.to_string(index=False))

# Select Top 4 Champion Models for Ensembling
top_4_champions = leaderboard_df.head(4)["Multilinear Architecture"].tolist()
print(f"\n🏆 TOP 4 MULTILINEAR CHAMPIONS SELECTED: {top_4_champions}")


# ============================================================
# 6. TRAIN MULTI-CHAMPION ENSEMBLE ON FULL DATASET
# ============================================================

print("\n============================================")
print("TRAINING TOP-4 MULTILINEAR ENSEMBLE ON FULL DATASET")
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

SEEDS = [42, 101, 777, 2024, 999]
all_champion_predictions = []

for c_idx, c_name in enumerate(top_4_champions):
    print(f" -> Training Champion #{c_idx+1}: {c_name} across 5 seeds...")
    model_seed_preds = np.zeros((len(X_test_full), len(SEEDS)))

    for s_idx, seed in enumerate(SEEDS):
        candidate_pool = get_multilinear_candidates(seed=seed)
        model_estimator, transform_type = candidate_pool[c_name]

        preprocessor = build_linear_preprocessor(numerical_features, categorical_features, transform_type=transform_type)
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model_estimator)
        ])
        pipe.fit(X_full, y_full)
        model_seed_preds[:, s_idx] = pipe.predict_proba(X_test_full)[:, 1]

    all_champion_predictions.append(model_seed_preds.mean(axis=1))

# Smooth Soft Average of the Top 4 Multilinear Champions
final_probabilities = np.mean(all_champion_predictions, axis=0)


# ============================================================
# 7. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp46_multilinear_tournament_ensemble.csv"

submission = pd.DataFrame({
    "anonymised_id": test_raw["anonymised_id"],
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 46 COMPLETE")
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
print("Exp 41 Sequential Hybrid      : 0.65325")
print("Exp 43 Single LogReg Champ    : 0.65502")
print("Exp 44 Top-3 Linear Blend     : 0.65630 (Current Best)")
print("Exp 45 Dynamic Programming DP: 0.65613")
print("Exp 46 Multilinear Tournament : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")