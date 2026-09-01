# ============================================================
# EXPERIMENT 55 — DISCRIMINANT ANALYSIS & MULTINOMIAL TOURNAMENT
# (SHRINKAGE LDA, REGULARIZED QDA, MULTINOMIAL LOGIT & HYBRID BLEND)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


print("============================================")
print("EXPERIMENT 55")
print("DISCRIMINANT ANALYSIS & MULTINOMIAL TOURNAMENT")
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
# 2. FEATURE EXTRACTION & IMPUTATION CLEANING
# ============================================================

target = "employed_status"

def clean_and_prepare_features(df):
    """Cleans IDs, dates, and adds robust longitudinal lag signals."""
    data = df.copy()

    # 1. Target Cleaning
    if target in data.columns:
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data.dropna(subset=[target]).copy()
        data[target] = data[target].astype(int)

    # 2. ID Removal
    if "anonymised_id" in data.columns:
        data = data.drop(columns=["anonymised_id"])

    # 3. Calendar Features
    if "survey_date" in data.columns:
        data["survey_date"] = pd.to_datetime(data["survey_date"], errors="coerce")
        data["survey_year"] = data["survey_date"].dt.year
        data["survey_month"] = data["survey_date"].dt.month
        data["survey_dayofyear"] = data["survey_date"].dt.dayofyear
        data = data.drop(columns=["survey_date"])

    # 4. Lag Signals
    if "employed_lag" in data.columns:
        data["is_first_time"] = data["employed_lag"].isna().astype(float)
    
    if "tenure_lag" in data.columns:
        data["tenure_lag_log"] = np.log1p(data["tenure_lag"].fillna(0).clip(lower=0))

    if "days_since_last_obs" in data.columns:
        data["days_since_last_obs_log"] = np.log1p(data["days_since_last_obs"].fillna(0).clip(lower=0))

    # 5. Robust string-to-numeric coercion
    for col in data.columns:
        if col != target and data[col].dtype == "object":
            converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().sum() > 0.6 * data[col].notna().sum():
                data[col] = converted

    return data


def build_preprocessor(numerical_cols, categorical_cols, use_quantile=False):
    """Builds standard or quantile-normalized preprocessor with safe imputation."""
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
# 3. DISCRIMINANT & MULTINOMIAL CANDIDATE MODELS
# ============================================================

def get_candidate_models(seed=42):
    """Returns a diverse pool of generative and discriminative algorithms."""
    return {
        # 1. Linear Discriminant Analysis (Shrinkage Auto / Ledoit-Wolf)
        "LDA_Shrinkage_Auto": (
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            False
        ),
        # 2. Linear Discriminant Analysis (Shrinkage 0.15)
        "LDA_Shrinkage_015": (
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.15),
            False
        ),
        # 3. Linear Discriminant Analysis (SVD Solver)
        "LDA_SVD_Standard": (
            LinearDiscriminantAnalysis(solver="svd"),
            False
        ),
        # 4. Quadratic Discriminant Analysis (Regularized QDA)
        "QDA_Regularized_030": (
            QuadraticDiscriminantAnalysis(reg_param=0.30),
            False
        ),
        # 5. Multinomial Logistic Regression (SAGA ElasticNet)
        "Multinomial_ElasticNet": (
            LogisticRegression(multi_class="multinomial", solver="saga", penalty="elasticnet",
                               l1_ratio=0.15, C=0.10, max_iter=1000, random_state=seed),
            False
        ),
        # 6. Multinomial Logistic Regression (L-BFGS L2)
        "Multinomial_LBFGS_L2": (
            LogisticRegression(multi_class="multinomial", solver="lbfgs", C=0.08,
                               max_iter=1000, random_state=seed),
            False
        ),
        # 7. Proven Logistic L2 Baseline
        "LogReg_L2_C008": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False
        ),
        # 8. Neural MLP Classifier
        "MLP_Medium_64_32": (
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.01,
                          batch_size=128, learning_rate_init=0.001, max_iter=350, early_stopping=True,
                          n_iter_no_change=20, validation_fraction=0.15, random_state=seed),
            False
        )
    }


# ============================================================
# 4. DISCOVER & LOAD SEQUENTIAL ROUND DATASETS
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
                r_train_clean = clean_and_prepare_features(r_train_raw)
                r_test_clean = clean_and_prepare_features(r_test_raw)

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
# 5. RUN TOURNAMENT ON ROUNDS 6, 7, 8
# ============================================================

print("\n============================================")
print("BENCHMARKING DISCRIMINANT & MULTINOMIAL MODELS")
print("============================================")

candidate_dict = get_candidate_models(seed=42)
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

    models_dict = get_candidate_models(seed=42)

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
print("DISCRIMINANT & MULTINOMIAL LEADERBOARD")
print("============================================")
print(leaderboard_df.to_string(index=False))

# Select Top 4 Champions
top_performers = leaderboard_df.head(4)["Model Architecture"].tolist()
print(f"\n🏆 TOP CHAMPIONS SELECTED: {top_performers}")


# ============================================================
# 6. TRAIN MULTI-SEED GENERATIVE-DISCRIMINATIVE ENSEMBLE
# ============================================================

print("\n============================================")
print("TRAINING MULTI-SEED ENSEMBLE ON FULL DATASET")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")

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

SEEDS = [42, 101, 777, 2024, 999, 1337, 555]
all_model_logits = []

for m_idx, m_name in enumerate(top_performers):
    print(f" -> Fitting Champion #{m_idx + 1}: {m_name} across {len(SEEDS)} seeds...")

    for s_idx, seed in enumerate(SEEDS):
        candidate_pool = get_candidate_models(seed=seed)
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

output_file = "submission_exp55_discriminant_multinomial_ensemble.csv"

submission = pd.DataFrame({
    "anonymised_id": test_raw["anonymised_id"],
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 55 COMPLETE")
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
print("Exp 55 Discriminant & Multinomial  : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")