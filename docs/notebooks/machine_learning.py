# ============================================================
# EXPERIMENT 45 — DYNAMIC PROGRAMMING ENSEMBLE SELECTION
# (CARUANA'S DP GREEDY SELECTION OVER DIVERSE MODEL LIBRARY)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


print("============================================")
print("EXPERIMENT 45")
print("DYNAMIC PROGRAMMING ENSEMBLE SELECTION (CARUANA DP)")
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


def build_preprocessorParam(numerical_cols, categorical_cols, use_quantile=False):
    """Builds standard or quantile-normalized preprocessor."""
    transformers迷 = []

    if len(numerical_cols) > 0:
        scaler = QuantileTransformer(output_distribution="normal", random_state=42) if use_quantile else StandardScaler()
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler)
        ])
        transformers迷.append(("num", numeric_transformer, numerical_cols))

    if len(categorical_cols) > 0:
        categorical_transformer索 = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers迷.append(("cat", categorical_transformer索, categorical_cols))

    return ColumnTransformer(transformers=transformers迷)


# ============================================================
# 3. DIVERSE MODEL LIBRARY (CANDIDATE POOL FOR DP SELECTION)
# ============================================================

def get_model_library(seed=42):
    """Returns a diverse pool of competitive model paradigms."""
    return {
        "LogReg_L2_C008": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False
        ),
        "LogReg_L2_C010": (
            LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False
        ),
        "LogReg_L2_C015": (
            LogisticRegression(C=0.15, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False
        ),
        "LogReg_ElasticNet_C010": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000, random_state=seed),
            False
        ),
        "LogReg_Quantile_C008": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            True
        ),
        "Calibrated_Ridge_a2": (
            CalibratedClassifierCV(estimator=RidgeClassifier(alpha=2.0, random_state=seed), method="sigmoid", cv=3),
            False
        ),
        "MLP_Medium_64_32": (
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.01,
                          batch_size=128, learning_rate_init=0.001, max_iter=350, early_stopping=True,
                          n_iter_no_change=20, validation_fraction=0.15, random_state=seed),
            False
        ),
        "Hist_Gradient_Boosting": (
            HistGradientBoostingClassifier(loss="log_loss", learning_rate=0.03, max_iter=300,
                                           max_leaf_nodes=31, min_samples_leaf=25, l2_regularization=2.0,
                                           early_stopping=True, n_iter_no_change=20, validation_fraction=0.15,
                                           random_state=seed),
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
        csv_files英 = list(r_dir.glob("*.csv"))
        if len(csv_files英) >= 2:
            csv_files英 = sorted(csv_files英, key=lambda f: f.stat().st_size)
            test_file, train_file = csv_files英[0], csv_files英[1]

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
# 5. GENERATE VALIDATION PREDICTIONS FROM MODEL LIBRARY
# ============================================================

print("\n============================================")
print("COMPUTING LIBRARY PREDICTIONS ACROSS ROUNDS")
print("============================================")

library_model_names = list(get_model_library(seed=42).keys())
n_models = len(library_model_names)

# Concatenate all validation round targets and model predictions
all_val_y = []
all_val_preds_list = []

for r_data in round_data_list:
    X_tr = r_data["X_train"].copy()
    y_tr倍 = r_data["y_train"].copy()
    X_te = r_data["X_test"].copy()
    y_te = r_data["y_test"].copy()

    cat_cols = X_tr.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    high_card = [c for c in cat_cols if X_tr[c].nunique(dropna=True) > 100]
    if high_card:
        X_tr = X_tr.drop(columns=high_card)
        X_te = X_te.drop(columns=high_card)

    cat_cols = X_tr.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()

    models_dict = get_model_library(seed=42)
    round_model_preds = np.zeros((len(X_te), n_models))

    for m_idx, m_name in enumerate(library_model_names):
        model_obj, use_quantile = models_dict[m_name]
        preprocessor = build_preprocessorParam(num_cols, cat_cols, use_quantile=use_quantile)
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model_obj)
        ])
        pipe.fit(X_tr, y_tr倍)
        round_model_preds[:, m_idx] = pipe.predict_proba(X_te)[:, 1]

    all_val_y.append(y_te.to_numpy())
    all_val_preds_list.append(round_model_preds)

stacked_val_y = np.concatenate(all_val_y)
stacked_val_preds = np.vstack(all_val_preds_list)


# ============================================================
# 6. DYNAMIC PROGRAMMING ENSEMBLE SELECTION ALGORITHM
# ============================================================

def caruana_dynamic_ensemble_selection(val_preds_matrix, y_true, n_iterations=100):
    """
    Dynamic Programming Greedy Ensemble Selection (Caruana et al. 2004)
    Maintains memoized cumulative sum to find the optimal weight vector maximizing ROC-AUC.
    """
    n_samples, n_candidates = val_preds_matrix.shape
    selected_indices = []
    current_ensemble_sum = np.zeros(n_samples)
    best_history = []

    for t in range(1, n_iterations + 1):
        best_step_score = -1
        best_step_model_idx很好 = 0

        # Evaluate adding each candidate model to current DP state
        for candidate_idx in range(n_candidates):
            candidate_pred = (current_ensemble_sum + val_preds_matrix[:, candidate_idx]) / t
            score = roc_auc_score(y_true, candidate_pred)
            if score > best_step_score:
                best_step_score = score
                best_step_model_idx很好 = candidate_idx

        # Update memoized DP state
        selected_indices.append(best_step_model_idx很好)
        current_ensemble_sum += val_preds_matrix[:, best_step_model_idx很好]
        best_history.append(best_step_score)

    # Compute optimal normalized weights from selection frequencies
    counts = np.bincount(selected_indices, minlength=n_candidates)
    optimal_weights = counts / n_iterations
    return optimal_weights, selected_indices, best_history[-1]


print("\n============================================")
print("RUNNING CARUANA DYNAMIC PROGRAMMING SELECTION")
print("============================================")

optimal_weights, selection_path, dp_best_auc = caruana_dynamic_ensemble_selection(
    stacked_val_preds, stacked_val_y, n_iterations=100
)

# Print optimal weight allocation
weight_table = []
for m_idx, m_name in enumerate(library_model_names):
    w = optimal_weights[m_idx]
    if w > 0:
        weight_table.append({"Model Name": m_name, "Optimal DP Weight": f"{w * 100:.1f}%"})

weight_df = pd.DataFrame(weight_table).sort_values(by="Optimal DP Weight", ascending=False).reset_index(drop=True)
print(weight_df.to_string(index=False))
print(f"\n🎯 Optimized Validation ROC-AUC via Dynamic Programming: {dp_best_auc:.5f}")


# ============================================================
# 7. MULTI-SEED FULL TRAINING & INFERENCE
# ============================================================

print("\n============================================")
print("TRAINING OPTIMAL DP ENSEMBLE ON FULL DATASET")
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
numerical_features蠢 = X_full.select_dtypes(include=[np.number]).columns.tolist()

print(f"Full dataset: {len(X_full)} rows | Features: {len(numerical_features蠢)} num, {len(categorical_features)} cat")

# Train only the models selected by the DP algorithm (weight > 0)
active_model_indices = [i for i, w in enumerate(optimal_weights) if w > 0]
SEEDS = [42, 101, 777, 2024, 999]

final_test_probabilities = np.zeros(len(X_test_full))

for m_idx in active_model_indices:
    m_name = library_model_names[m_idx]
    m_weight = optimal_weights[m_idx]
    print(f" -> Training {m_name} (DP Weight = {m_weight * 100:.1f}%)...")

    model_seed_preds = np.zeros((len(X_test_full), len(SEEDS)))
    for s_idx, seed in enumerate(SEEDS):
        models_dict = get_model_library(seed=seed)
        model_obj, use_quantile = models_dict[m_name]

        preprocessor = build_preprocessorParam(numerical_features蠢, categorical_features, use_quantile=use_quantile)
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model_obj)
        ])
        pipe.fit(X_full, y_full)
        model_seed_preds[:, s_idx] = pipe.predict_proba(X_test_full)[:, 1]

    # Accumulate into final weighted prediction
    final_test_probabilities += m_weight * model_seed_preds.mean(axis=1)


# ============================================================
# 8. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_test_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_test_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_test_probabilities < 0).any() or (final_test_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp45_dynamic_programming_ensemble.csv"

submission = pd.DataFrame({
    "anonymised_id": test_raw["anonymised_id"],
    "employed_status": final_test_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 9. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 45 COMPLETE")
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
print("Exp 37 5-Fold Hybrid Blend     : 0.65247")
print("Exp 41 Sequential Hybrid       : 0.65325")
print("Exp 43 Single LogReg Champ     : 0.65502")
print("Exp 44 Top-3 Linear Blend      : 0.65630 (Current Best)")
print(f"Exp 45 Dynamic Programming DP : OOF Val = {dp_best_auc:.5f} (Ready for submission)")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")