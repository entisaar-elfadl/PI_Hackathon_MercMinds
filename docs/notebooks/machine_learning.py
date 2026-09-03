# ============================================================
# EXPERIMENT 78 — DYNAMIC PROGRAMMING ENSEMBLE SELECTION (CARUANA DP)
# (DISCRETE STATE MEMOIZATION + FULLY CONVERGED SEM MANIFOLD)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.decomposition import FactorAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


print("============================================")
print("EXPERIMENT 78")
print("DYNAMIC PROGRAMMING ENSEMBLE SELECTION (CARUANA DP)")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR


# ============================================================
# 2. PROVEN PURE SEM FEATURE PIPELINE
# ============================================================

target = "employed_status"

def parse_matric_band(val):
    """Converts matric percentage strings into continuous numeric marks."""
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


def extract_pure_sem_features(train_df, test_df):
    """Extracts the exact winning Exp 59/68 SEM Latent Factor representation."""
    tr = train_df.copy()
    te = test_df.copy()

    # Clean target
    tr[target] = pd.to_numeric(tr[target], errors="coerce")
    tr = tr.dropna(subset=[target]).copy()
    tr[target] = tr[target].astype(int)

    for df in [tr, te]:
        if "anonymised_id" in df.columns:
            df.drop(columns=["anonymised_id"], inplace=True)

        if "survey_date" in df.columns:
            df["survey_date"] = pd.to_datetime(df["survey_date"], errors="coerce")
            df["survey_year"] = df["survey_date"].dt.year
            df["survey_month"] = df["survey_date"].dt.month
            df["survey_dayofyear"] = df["survey_date"].dt.dayofyear
            df.drop(columns=["survey_date"], inplace=True)

        df["is_first_time"] = df["employed_lag"].isna().astype(float)
        df["employed_lag_num"] = df["employed_lag"].fillna(-1.0).astype(float)
        df["tenure_lag_log"] = np.log1p(pd.to_numeric(df.get("tenure_lag", 0), errors="coerce").fillna(0).clip(lower=0))
        df["days_since_obs_log"] = np.log1p(pd.to_numeric(df.get("days_since_last_obs", 0), errors="coerce").fillna(0).clip(lower=0))

        for m in ["matric_englishhome", "matric_englishadd", "matric_mathpure", "matric_physicalscience", "matric_mathlit"]:
            if m in df.columns:
                df[f"{m}_score"] = df[m].apply(parse_matric_band).fillna(-1.0)

        df["school_quintile_num"] = pd.to_numeric(df.get("school_quintile", 0), errors="coerce").fillna(0.0)
        df["work_readiness_num"] = pd.to_numeric(df.get("work_readiness_score", 0.5), errors="coerce").fillna(0.5)
        df["age_clean"] = pd.to_numeric(df.get("age", 22), errors="coerce").fillna(22.0).clip(18, 35)

        for col in df.columns:
            if df[col].dtype == "bool":
                df[col] = df[col].astype(float)
            elif col != target and df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if df[col].notna().sum() > 0 and converted.notna().sum() > 0.6 * df[col].notna().sum():
                    df[col] = converted

    # Latent SEM Blocks
    acad_cols = [c for c in tr.columns if "_score" in c]
    labour_cols = ["employed_lag_num", "tenure_lag_log", "days_since_obs_log", "is_first_time"]
    socio_cols = ["school_quintile_num", "work_readiness_num", "age_clean"]

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()

    if len(acad_cols) > 0:
        fa_acad = FactorAnalysis(n_components=2, random_state=42)
        X_ac_tr = scl.fit_transform(imp.fit_transform(tr[acad_cols]))
        X_ac_te = scl.transform(imp.transform(te[acad_cols]))
        tr["latent_academic_factor_1"] = fa_acad.fit_transform(X_ac_tr)[:, 0]
        tr["latent_academic_factor_2"] = fa_acad.fit_transform(X_ac_tr)[:, 1]
        te["latent_academic_factor_1"] = fa_acad.transform(X_ac_te)[:, 0]
        te["latent_academic_factor_2"] = fa_acad.transform(X_ac_te)[:, 1]

    fa_lab = FactorAnalysis(n_components=2, random_state=42)
    X_lb_tr = scl.fit_transform(imp.fit_transform(tr[labour_cols]))
    X_lb_te = scl.transform(imp.transform(te[labour_cols]))
    tr["latent_labour_momentum_1"] = fa_lab.fit_transform(X_lb_tr)[:, 0]
    tr["latent_labour_momentum_2"] = fa_lab.fit_transform(X_lb_tr)[:, 1]
    te["latent_labour_momentum_1"] = fa_lab.transform(X_lb_te)[:, 0]
    te["latent_labour_momentum_2"] = fa_lab.transform(X_lb_te)[:, 1]

    fa_soc = FactorAnalysis(n_components=2, random_state=42)
    X_sc_tr = scl.fit_transform(imp.fit_transform(tr[socio_cols]))
    X_sc_te = scl.transform(imp.transform(te[socio_cols]))
    tr["latent_socio_readiness_1"] = fa_soc.fit_transform(X_sc_tr)[:, 0]
    tr["latent_socio_readiness_2"] = fa_soc.fit_transform(X_sc_tr)[:, 1]
    te["latent_socio_readiness_1"] = fa_soc.transform(X_sc_te)[:, 0]
    te["latent_socio_readiness_2"] = fa_soc.transform(X_sc_te)[:, 1]

    all_num_block = acad_cols + labour_cols + socio_cols
    pls = PLSRegression(n_components=2)
    X_pls_tr = imp.fit_transform(tr[all_num_block])
    X_pls_te = imp.transform(te[all_num_block])
    pls.fit(X_pls_tr, tr[target])
    tr["pls_structural_latent_1"] = pls.transform(X_pls_tr)[:, 0]
    tr["pls_structural_latent_2"] = pls.transform(X_pls_tr)[:, 1]
    te["pls_structural_latent_1"] = pls.transform(X_pls_te)[:, 0]
    te["pls_structural_latent_2"] = pls.transform(X_pls_te)[:, 1]

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
# 3. HIGH-CONVERGENCE 12-MODEL CANDIDATE LIBRARY
# ============================================================

def get_candidate_library(seed=42):
    """Returns 12 fully-converged models with max_iter=2500 to prevent iteration limit wastage."""
    return {
        # 1. High-Precision L2 Models (L-BFGS)
        "LogReg_L2_C006": (
            LogisticRegression(C=0.06, penalty="l2", solver="lbfgs", max_iter=2500, tol=1e-4, random_state=seed),
            False
        ),
        "LogReg_L2_C008": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=2500, tol=1e-4, random_state=seed),
            False
        ),
        "LogReg_L2_C010": (
            LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=2500, tol=1e-4, random_state=seed),
            False
        ),
        "LogReg_L2_C015": (
            LogisticRegression(C=0.15, penalty="l2", solver="lbfgs", max_iter=2500, tol=1e-4, random_state=seed),
            False
        ),
        # 2. Fully-Converged ElasticNet Models (SAGA with max_iter=2500)
        "LogReg_ElasticNet_C008": (
            LogisticRegression(C=0.08, penalty="elasticnet", solver="saga", l1_ratio=0.10, max_iter=2500, tol=1e-4, random_state=seed),
            False
        ),
        "LogReg_ElasticNet_C010": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=seed),
            False
        ),
        # 3. Quantile-Gaussian Transformed Models
        "Quantile_LogReg_L2_C008": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=2500, tol=1e-4, random_state=seed),
            True
        ),
        "Quantile_LogReg_ElasticNet_C010": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=seed),
            True
        ),
        # 4. Multi-Scale Neural MLPs
        "MLP_Medium_64_32": (
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.010,
                          batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                          n_iter_no_change=25, validation_fraction=0.15, random_state=seed),
            False
        ),
        "MLP_Deep_128_64": (
            MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=0.015,
                          batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                          n_iter_no_change=25, validation_fraction=0.15, random_state=seed + 50),
            False
        ),
        "MLP_Wide_128_64": (
            MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=0.010,
                          batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                          n_iter_no_change=25, validation_fraction=0.15, random_state=seed + 100),
            False
        ),
        "MLP_Tanh_64_32": (
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="tanh", solver="adam", alpha=0.010,
                          batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                          n_iter_no_change=25, validation_fraction=0.15, random_state=seed + 150),
            False
        )
    }


# ============================================================
# 4. LOAD & PREPARE DATASET
# ============================================================

print("\n============================================")
print("EXTRACTING PURE EXP 59 SEM LATENT FACTORS")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()

train_sem, test_sem = extract_pure_sem_features(train_raw, test_raw)

common_cols = [c for c in train_sem.columns if c in test_sem.columns and c != target]

y = train_sem[target].reset_index(drop=True)
X = train_sem[common_cols].reset_index(drop=True)
X_test = test_sem[common_cols].reset_index(drop=True)

# Drop high-cardinality categorical (>100)
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
high_cardinality = [c for c in categorical_features if X[c].nunique(dropna=True) > 100]
if high_cardinality:
    X = X.drop(columns=high_cardinality)
    X_test = X_test.drop(columns=high_cardinality)

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"Full dataset: {len(X)} observations | Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")


# ============================================================
# 5. 5-FOLD LEAK-FREE OOF PREDICTION GENERATION
# ============================================================

print("\n============================================")
print("GENERATING 5-FOLD OOF PREDICTIONS ACROSS 12 CANDIDATE MODELS")
print("============================================")

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

model_names = list(get_candidate_library(seed=42).keys())
N_MODELS = len(model_names)

oof_probabilities = np.zeros((len(X), N_MODELS))
test_fold_predictions = np.zeros((len(X_test), N_MODELS, N_SPLITS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr_f, y_tr_f = X.iloc[train_idx], y.iloc[train_idx]
    X_va_f, y_va_f = X.iloc[val_idx], y.iloc[val_idx]

    models_dict = get_candidate_library(seed=42 + fold * 10)

    for m_idx, (m_name, (model_obj, use_quantile)) in enumerate(models_dict.items()):
        preprocessor = build_preprocessor(numerical_features, categorical_features, use_quantile=use_quantile)
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model_obj)
        ])
        pipe.fit(X_tr_f, y_tr_f)
        
        oof_probabilities[val_idx, m_idx] = pipe.predict_proba(X_va_f)[:, 1]
        test_fold_predictions[:, m_idx, fold] = pipe.predict_proba(X_test)[:, 1]

    print(f"Fold {fold + 1}/{N_SPLITS} Complete.")


# Individual OOF Scores
print("\n--- Individual Candidate Model OOF ROC-AUC Scores ---")
for m_idx, m_name in enumerate(model_names):
    auc = roc_auc_score(y, oof_probabilities[:, m_idx])
    print(f"Model {m_idx + 1:02d} ({m_name:<34}): OOF AUC = {auc:.5f}")


# ============================================================
# 6. DYNAMIC PROGRAMMING ENSEMBLE SELECTION (CARUANA DP)
# ============================================================

print("\n============================================")
print("RUNNING CARUANA DYNAMIC PROGRAMMING OPTIMIZATION (150 STEPS)")
print("============================================")

def caruana_dynamic_programming_selection(val_preds_matrix, y_true, n_iterations=150):
    """
    Dynamic Programming Ensemble Selection via State Memoization:
    At each step t, evaluates adding each candidate to the cumulative sum vector
    in O(M) operations, directly maximizing validation ROC-AUC.
    """
    n_samples, n_candidates = val_preds_matrix.shape
    selected_indices = []
    current_cumulative_sum = np.zeros(n_samples)
    best_step_scores = []

    for t in range(1, n_iterations + 1):
        best_auc = -1.0
        best_candidate_idx = 0

        # Evaluate adding each candidate to the memoized state
        for cand_idx in range(n_candidates):
            trial_pred = (current_cumulative_sum + val_preds_matrix[:, cand_idx]) / t
            score = roc_auc_score(y_true, trial_pred)
            if score > best_auc:
                best_auc = score
                best_candidate_idx = cand_idx

        # Update memoized DP state
        selected_indices.append(best_candidate_idx)
        current_cumulative_sum += val_preds_matrix[:, best_candidate_idx]
        best_step_scores.append(best_auc)

    # Compute optimal discrete weights from selection frequency
    counts = np.bincount(selected_indices, minlength=n_candidates)
    optimal_weights = counts / n_iterations
    return optimal_weights, selected_indices, best_step_scores[-1]


# Run DP selection on OOF probabilities
dp_weights, selection_path, dp_best_auc = caruana_dynamic_programming_selection(
    oof_probabilities, y.to_numpy(), n_iterations=150
)

# Display DP optimal allocation
dp_summary = []
for m_idx, m_name in enumerate(model_names):
    w = dp_weights[m_idx]
    if w > 0:
        dp_summary.append({"Model Architecture": m_name, "DP Selected Weight": f"{w * 100:.2f}%"})

dp_df = pd.DataFrame(dp_summary).sort_values(by="DP Selected Weight", ascending=False).reset_index(drop=True)

print("Learned Dynamic Programming Weights:")
print(dp_df.to_string(index=False))
print(f"\n🏆 Dynamic Programming Optimized OOF ROC-AUC: {dp_best_auc:.5f}")


# ============================================================
# 7. INFERENCE ON TEST SET (DP WEIGHTED LOGITS)
# ============================================================

print("\nGenerating final test predictions via Dynamic Programming weights...")

# Average 5-fold test probabilities per model, then convert to logit space
avg_test_probs = test_fold_predictions.mean(axis=2)
test_logits = np.zeros_like(avg_test_probs)

for m_idx in range(N_MODELS):
    p_cl = np.clip(avg_test_probs[:, m_idx], 1e-6, 1.0 - 1e-6)
    test_logits[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

# Combine logits using DP weights
final_test_logits = np.dot(test_logits, dp_weights)
final_probabilities = 1.0 / (1.0 + np.exp(-final_test_logits))


# ============================================================
# 8. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp78_dynamic_programming_sem_ensemble.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 9. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 78 COMPLETE")
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
print("Exp 59 Pure SEM Latent Titan        : 0.65832")
print("Exp 70 Repeated 5-Fold NNLS         : 0.65998")
print("Exp 68 5-Fold NNLS Super Learner    : 0.66054 (Personal Best)")
print(f"Exp 78 Dynamic Programming (Caruana): OOF Val = {dp_best_auc:.5f} (Ready for submission)")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")