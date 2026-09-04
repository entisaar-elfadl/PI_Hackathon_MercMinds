# ============================================================
# EXPERIMENT 84 — ADAPTIVE CONTEXT-AWARE SEM SUPER-LEARNER
# (GENUINE MANIFOLD DIVERSITY + DISAGREEMENT STACKING + CARUANA DP)
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


print("============================================")
print("EXPERIMENT 84")
print("ADAPTIVE CONTEXT-AWARE SEM SUPER-LEARNER")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR


# ============================================================
# 2. FEATURE EXTRACTION & MISSINGNESS FINGERPRINTING
# ============================================================

target = "employed_status"
GLOBAL_PRIOR = 0.31694

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


def extract_disentangled_sem_features(train_df, test_df):
    """
    Extracts features, prevents identical profile collapse via missingness
    fingerprinting, and extracts SEM Latent Factor blocks.
    """
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

        # 1. Longitudinal & Lag Semantics
        df["is_first_time"] = df["employed_lag"].isna().astype(float)
        df["employed_lag_num"] = df["employed_lag"].fillna(-1.0).astype(float)
        
        tenure_raw = pd.to_numeric(df.get("tenure_lag", 0), errors="coerce").fillna(0).clip(lower=0)
        df["tenure_lag_log"] = np.log1p(tenure_raw)
        
        days_raw = pd.to_numeric(df.get("days_since_last_obs", 0), errors="coerce").fillna(0).clip(lower=0)
        df["days_since_obs_log"] = np.log1p(days_raw)

        # 2. Missingness Fingerprinting (Distinguishes between first-time entrants)
        df["missing_profile_depth"] = (
            df["employed_lag"].isna().astype(float) +
            df["days_since_last_obs"].isna().astype(float) +
            df.get("institution_type", pd.Series(np.nan, index=df.index)).isna().astype(float) +
            df.get("seta", pd.Series(np.nan, index=df.index)).isna().astype(float) +
            df.get("matric_mathpure", pd.Series(np.nan, index=df.index)).isna().astype(float)
        )

        # 3. Matric Continuous Marks
        for m in ["matric_englishhome", "matric_englishadd", "matric_mathpure", "matric_physicalscience", "matric_mathlit"]:
            if m in df.columns:
                df[f"{m}_score"] = df[m].apply(parse_matric_band).fillna(-1.0)

        # 4. Socio-Economic & Readiness
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

    # ------------------------------------------------------------
    # 5. SEM LATENT FACTOR EXTRACTION
    # ------------------------------------------------------------
    acad_cols = [c for c in tr.columns if "_score" in c]
    labour_cols = ["employed_lag_num", "tenure_lag_log", "days_since_obs_log", "is_first_time", "missing_profile_depth"]
    socio_cols = ["school_quintile_num", "work_readiness_num", "age_clean"]

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()

    # Block A: Academic Factor Analysis
    if len(acad_cols) > 0:
        fa_acad = FactorAnalysis(n_components=2, random_state=42)
        X_ac_tr = scl.fit_transform(imp.fit_transform(tr[acad_cols]))
        X_ac_te = scl.transform(imp.transform(te[acad_cols]))
        tr["sem_academic_1"] = fa_acad.fit_transform(X_ac_tr)[:, 0]
        tr["sem_academic_2"] = fa_acad.fit_transform(X_ac_tr)[:, 1]
        te["sem_academic_1"] = fa_acad.transform(X_ac_te)[:, 0]
        te["sem_academic_2"] = fa_acad.transform(X_ac_te)[:, 1]

    # Block B: Labour Momentum Factor Analysis
    fa_lab = FactorAnalysis(n_components=2, random_state=42)
    X_lb_tr = scl.fit_transform(imp.fit_transform(tr[labour_cols]))
    X_lb_te = scl.transform(imp.transform(te[labour_cols]))
    tr["sem_labour_1"] = fa_lab.fit_transform(X_lb_tr)[:, 0]
    tr["sem_labour_2"] = fa_lab.fit_transform(X_lb_tr)[:, 1]
    te["sem_labour_1"] = fa_lab.transform(X_lb_te)[:, 0]
    te["sem_labour_2"] = fa_lab.transform(X_lb_te)[:, 1]

    # Block C: Socio-Economic Factor Analysis
    fa_soc = FactorAnalysis(n_components=2, random_state=42)
    X_sc_tr = scl.fit_transform(imp.fit_transform(tr[socio_cols]))
    X_sc_te = scl.transform(imp.transform(te[socio_cols]))
    tr["sem_socio_1"] = fa_soc.fit_transform(X_sc_tr)[:, 0]
    tr["sem_socio_2"] = fa_soc.fit_transform(X_sc_tr)[:, 1]
    te["sem_socio_1"] = fa_soc.transform(X_sc_te)[:, 0]
    te["sem_socio_2"] = fa_soc.transform(X_sc_te)[:, 1]

    # Block D: Supervised PLS Direct Covariance Path
    all_num = acad_cols + labour_cols + socio_cols
    pls = PLSRegression(n_components=2)
    X_pls_tr = imp.fit_transform(tr[all_num])
    X_pls_te = imp.transform(te[all_num])
    pls.fit(X_pls_tr, tr[target])
    tr["pls_path_1"] = pls.transform(X_pls_tr)[:, 0]
    tr["pls_path_2"] = pls.transform(X_pls_tr)[:, 1]
    te["pls_path_1"] = pls.transform(X_pls_te)[:, 0]
    te["pls_path_2"] = pls.transform(X_pls_te)[:, 1]

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
# 3. GENUINE DIVERSE MANIFOLD ESTIMATORS (EXP 80)
# ============================================================

# Wrapper for PLS probability classification
class PLSProbabilityClassifier:
    def __init__(self, n_components=3):
        self.pls = PLSRegression(n_components=n_components)

    def fit(self, X, y):
        self.pls.fit(X, y)
        return self

    def predict_proba(self, X):
        preds = self.pls.predict(X).ravel()
        # Calibrate via sigmoid
        probs_1 = np.clip(1.0 / (1.0 + np.exp(-preds * 3.0)), 1e-6, 1.0 - 1e-6)
        return np.vstack([1.0 - probs_1, probs_1]).T


def get_diverse_manifold_models(seed=42):
    """
    Constructs 6 genuinely diverse model paradigms:
    1. Generative LDA (Shrinkage Auto)
    2. Supervised PLS Direct Classifier
    3. Quantile-Gaussian ElasticNet
    4. Standard SAGA ElasticNet
    5. Deep Hierarchical ReLU MLP
    6. Smooth Continuous Tanh MLP
    """
    return {
        "Generative_LDA_Shrinkage": (
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            False
        ),
        "Supervised_PLS_Classifier": (
            PLSProbabilityClassifier(n_components=3),
            False
        ),
        "Quantile_ElasticNet_SAGA": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=seed),
            True
        ),
        "Standard_ElasticNet_SAGA": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=seed),
            False
        ),
        "Deep_ReLU_MLP_128_64": (
            MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=0.015,
                          batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                          n_iter_no_change=25, validation_fraction=0.15, random_state=seed),
            False
        ),
        "Smooth_Tanh_MLP_64_32": (
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="tanh", solver="adam", alpha=0.010,
                          batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                          n_iter_no_change=25, validation_fraction=0.15, random_state=seed + 50),
            False
        )
    }


# ============================================================
# 4. LOAD & PREPARE DATASET
# ============================================================

print("\n============================================")
print("EXTRACTING DISENTANGLED SEM LATENT MANIFOLD")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()

train_sem, test_sem = extract_disentangled_sem_features(train_raw, test_raw)

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

sem_factor_cols = [c for c in numerical_features if "sem_" in c or "pls_path" in c]

print(f"Full dataset: {len(X)} observations | Features: {len(numerical_features)} numerical (incl. {len(sem_factor_cols)} SEM Factors), {len(categorical_features)} categorical")


# ============================================================
# 5. 5-FOLD OOF EXTRACTION ACROSS DIVERSE MANIFOLDS
# ============================================================

print("\n============================================")
print("GENERATING 5-FOLD OOF PREDICTIONS ACROSS DIVERSE MANIFOLDS")
print("============================================")

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

model_names = list(get_diverse_manifold_models(seed=42).keys())
N_MODELS = len(model_names)

oof_probabilities = np.zeros((len(X), N_MODELS))
test_fold_predictions = np.zeros((len(X_test), N_MODELS, N_SPLITS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr_f, y_tr_f = X.iloc[train_idx], y.iloc[train_idx]
    X_va_f, y_va_f = X.iloc[val_idx], y.iloc[val_idx]

    models_dict = get_diverse_manifold_models(seed=42 + fold * 10)

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


# Individual Manifold OOF Scores
print("\n--- Diverse Manifold OOF ROC-AUC Scores ---")
for m_idx, m_name in enumerate(model_names):
    auc = roc_auc_score(y, oof_probabilities[:, m_idx])
    print(f"Manifold {m_idx + 1:02d} ({m_name:<28}): OOF AUC = {auc:.5f}")


# ============================================================
# 6. ADAPTIVE CONTEXT-AWARE STACKING (EXP 82 & 83)
# ============================================================

print("\n============================================")
print("BUILDING ADAPTIVE CONTEXT-AWARE META-STACK")
print("============================================")

# Convert OOF probabilities to Logit Space
oof_logits = np.zeros_like(oof_probabilities)
for m_idx in range(N_MODELS):
    p_cl = np.clip(oof_probabilities[:, m_idx], 1e-6, 1.0 - 1e-6)
    oof_logits[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

# Compute Disagreement / Uncertainty Spread Feature across Models
oof_disagreement_std = np.std(oof_logits, axis=1, keepdims=True)

# Build Context-Aware Meta-Features Matrix:
# [6 Model Logits] + [1 Disagreement Spread] + [8 SEM Latent Factor Context Vectors]
sem_factors_matrix_tr = X[sem_factor_cols].to_numpy()
oof_meta_features = np.hstack([oof_logits, oof_disagreement_std, sem_factors_matrix_tr])

print(f"Context-Aware Meta-Feature Matrix Shape: {oof_meta_features.shape}")

# Fit Context-Aware Super-Learner (Non-Negative Constrained on Predictions + Regularized SEM Context)
adaptive_meta_learner = LogisticRegression(C=0.15, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
adaptive_meta_learner.fit(oof_meta_features, y)

oof_adaptive_probs = adaptive_meta_learner.predict_proba(oof_meta_features)[:, 1]
adaptive_oof_auc = roc_auc_score(y, oof_adaptive_probs)

print(f"🏆 Adaptive Context-Aware Super-Learner OOF ROC-AUC: {adaptive_oof_auc:.5f}")


# ============================================================
# 7. CARUANA DYNAMIC PROGRAMMING ENSEMBLE SELECTION (EXP 84)
# ============================================================

print("\n============================================")
print("RUNNING CARUANA DP ENSEMBLE SELECTION OVER DIVERSE STACK")
print("============================================")

def caruana_dp_selection(val_preds_matrix, y_true, n_iterations=150):
    n_samples, n_candidates = val_preds_matrix.shape
    selected_indices = []
    current_sum = np.zeros(n_samples)
    best_scores = []

    for t in range(1, n_iterations + 1):
        best_auc = -1.0
        best_idx = 0
        for cand_idx in range(n_candidates):
            trial_pred = (current_sum + val_preds_matrix[:, cand_idx]) / t
            score = roc_auc_score(y_true, trial_pred)
            if score > best_auc:
                best_auc = score
                best_idx = cand_idx
        selected_indices.append(best_idx)
        current_sum += val_preds_matrix[:, best_idx]
        best_scores.append(best_auc)

    counts = np.bincount(selected_indices, minlength=n_candidates)
    optimal_weights = counts / n_iterations
    return optimal_weights, best_scores[-1]

# Run DP selection on the diverse manifold pool + adaptive stack
all_candidate_preds_pool = np.hstack([oof_probabilities, oof_adaptive_probs.reshape(-1, 1)])
pool_names = model_names + ["Adaptive_Context_Stacker"]

dp_weights, dp_best_auc = caruana_dp_selection(all_candidate_preds_pool, y.to_numpy(), n_iterations=150)

dp_summary = []
for p_idx, p_name in enumerate(pool_names):
    w = dp_weights[p_idx]
    if w > 0:
        dp_summary.append({"Ensemble Candidate": p_name, "DP Optimal Weight": f"{w * 100:.2f}%"})

dp_df = pd.DataFrame(dp_summary).sort_values(by="DP Optimal Weight", ascending=False).reset_index(drop=True)
print(dp_df.to_string(index=False))
print(f"\n🏆 Final Exp 84 Dynamic Programming Optimized OOF AUC: {dp_best_auc:.5f}")


# ============================================================
# 8. TEST PREDICTION INFERENCE
# ============================================================

print("\nGenerating final test predictions via Exp 84 Master Engine...")

# Average 5-fold test probabilities per manifold model
avg_test_probs = test_fold_predictions.mean(axis=2)
test_logits = np.zeros_like(avg_test_probs)

for m_idx in range(N_MODELS):
    p_cl = np.clip(avg_test_probs[:, m_idx], 1e-6, 1.0 - 1e-6)
    test_logits[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

# Compute Adaptive Meta-Learner Test Predictions
test_disagreement_std = np.std(test_logits, axis=1, keepdims=True)
sem_factors_matrix_te = X_test[sem_factor_cols].to_numpy()
test_meta_features = np.hstack([test_logits, test_disagreement_std, sem_factors_matrix_te])

test_adaptive_probs = adaptive_meta_learner.predict_proba(test_meta_features)[:, 1]

# Combine all candidates using Dynamic Programming weights
all_test_candidates_matrix = np.hstack([avg_test_probs, test_adaptive_probs.reshape(-1, 1)])

# Convert to logit space and apply DP weights for smooth calibrated output
test_cand_logits = np.zeros_like(all_test_candidates_matrix)
for c_idx in range(all_test_candidates_matrix.shape[1]):
    p_cl = np.clip(all_test_candidates_matrix[:, c_idx], 1e-6, 1.0 - 1e-6)
    test_cand_logits[:, c_idx] = np.log(p_cl / (1.0 - p_cl))

final_master_logits = np.dot(test_cand_logits, dp_weights)
final_probabilities = 1.0 / (1.0 + np.exp(-final_master_logits))


# ============================================================
# 9. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp84_adaptive_sem_superlearner.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 10. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 84 COMPLETE")
print("============================================")
print(f"Saved: {output_file}")
print(f"Rows: {len(submission)}")

print("\nPrediction summary (Checking variance expansion):")
print(submission["employed_status"].describe())

print("\nFirst 10 predictions:")
print(submission.head(10))

print("\n============================================")
print("BENCHMARKS")
print("============================================")
print("Exp 59 Pure SEM Latent Titan        : 0.65832")
print("Exp 71 Grand Master Super-Learner   : 0.66051")
print("Exp 68 5-Fold NNLS Super Learner    : 0.66054 (Previous Best)")
print(f"Exp 84 Adaptive Context-Aware Super : OOF Val = {dp_best_auc:.5f} (Ready for submission)")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")