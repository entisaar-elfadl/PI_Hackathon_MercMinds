# ============================================================
# EXPERIMENT 89 — LATENT CLUSTER PROFILING & BAYESIAN CALIBRATED SUPER-LEARNER
# (SEM CLUSTER ARCHETYPES + 0.66054 ELASTIC/MLP ENGINE + BAYESIAN PRIOR ALIGNMENT)
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
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


print("============================================")
print("EXPERIMENT 89")
print("LATENT CLUSTER PROFILING & BAYESIAN CALIBRATED SUPER-LEARNER")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS & DATA LOADING
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()
target = "employed_status"
GLOBAL_PRIOR = 0.31694


# ============================================================
# 2. FEATURE EXTRACTION & PURE SEM MANIFOLD
# ============================================================

def parse_matric_band(val):
    if pd.isna(val): return np.nan
    s = str(val).replace("%", "").strip()
    if "-" in s:
        parts = s.split("-")
        try: return (float(parts[0]) + float(parts[1])) / 2.0
        except: return np.nan
    elif "<" in s:
        try: return float(s.replace("<", "").strip()) / 2.0
        except: return np.nan
    elif ">" in s:
        try: return float(s.replace(">", "").strip()) + 5.0
        except: return np.nan
    else:
        try: return float(s)
        except: return np.nan


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


# ============================================================
# 3. LOAD DATASET & PREPARE MATRICES
# ============================================================

print("\n============================================")
print("EXTRACTING PURE SEM LATENT MANIFOLD")
print("============================================")

train_sem, test_sem = extract_pure_sem_features(train_raw, test_raw)

common_cols = [c for c in train_sem.columns if c in test_sem.columns and c != target]

y = train_sem[target].reset_index(drop=True)
X = train_sem[common_cols].reset_index(drop=True)
X_test = test_sem[common_cols].reset_index(drop=True)

# Drop high-cardinality categorical (>100)
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
high_cardinality = [c for c in categorical_features if X[c].nunique(dropna=True) > 100]
if high_cardinality:
    X.drop(columns=high_cardinality, inplace=True)
    X_test.drop(columns=high_cardinality, inplace=True)

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

sem_latent_cols = [c for c in numerical_features if "latent_" in c or "pls_" in c]

print(f"Full dataset: {len(X)} observations | Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")
print(f"SEM Latent Factors for Clustering: {sem_latent_cols}")


# ============================================================
# 4. PREPROCESSOR & 8-MODEL HIGH-VALUE SUITE
# ============================================================

def build_preprocessor(numerical_cols, categorical_cols, use_quantile=False):
    transformers = []
    if len(numerical_cols) > 0:
        scaler = QuantileTransformer(output_distribution="normal", random_state=42) if use_quantile else StandardScaler()
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", scaler)]), numerical_cols))
    if len(categorical_cols) > 0:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols))
    return ColumnTransformer(transformers=transformers)


def get_base_model_dict(seed=42):
    """Returns the proven 8-model suite (ElasticNet + Multi-Scale MLPs)."""
    return {
        "Quantile_ElasticNet_C010": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=seed),
            True
        ),
        "Quantile_ElasticNet_C008": (
            LogisticRegression(C=0.08, penalty="elasticnet", solver="saga", l1_ratio=0.10, max_iter=2500, tol=1e-4, random_state=seed),
            True
        ),
        "Standard_ElasticNet_C010": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=seed),
            False
        ),
        "Standard_LogReg_L2_C008": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed),
            False
        ),
        "MLP_Deep_128_64": (
            MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=0.015,
                          batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                          n_iter_no_change=25, validation_fraction=0.15, random_state=seed + 50),
            False
        ),
        "MLP_Medium_64_32": (
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.010,
                          batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                          n_iter_no_change=25, validation_fraction=0.15, random_state=seed),
            False
        ),
        "MLP_Tanh_64_32": (
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="tanh", solver="adam", alpha=0.010,
                          batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                          n_iter_no_change=25, validation_fraction=0.15, random_state=seed + 150),
            False
        ),
        "Quantile_MLP_Medium_64_32": (
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.010,
                          batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                          n_iter_no_change=25, validation_fraction=0.15, random_state=seed + 200),
            True
        )
    }


# ============================================================
# 5. 5-FOLD IN-FOLD CLUSTERING & OOF GENERATION
# ============================================================

print("\n============================================")
print("RUNNING 5-FOLD CV WITH IN-FOLD LATENT CLUSTERING (K=4)")
print("============================================")

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

model_names = list(get_base_model_dict(seed=42).keys())
N_MODELS = len(model_names)

oof_probabilities = np.zeros((len(X), N_MODELS))
test_fold_predictions = np.zeros((len(X_test), N_MODELS, N_SPLITS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr_f, y_tr_f = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
    X_va_f, y_va_f = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
    X_te_f = X_test.copy()

    # In-Fold K-Means Archetype Clustering (K=4) strictly on train fold
    kmeans = KMeans(n_clusters=4, random_state=42 + fold, n_init=10)
    
    # Extract distances to archetype centroids
    tr_cluster_dists = kmeans.fit_transform(X_tr_f[sem_latent_cols])
    va_cluster_dists = kmeans.transform(X_va_f[sem_latent_cols])
    te_cluster_dists = kmeans.transform(X_te_f[sem_latent_cols])

    for k in range(4):
        X_tr_f[f"latent_cluster_dist_{k}"] = tr_cluster_dists[:, k]
        X_va_f[f"latent_cluster_dist_{k}"] = va_cluster_dists[:, k]
        X_te_f[f"latent_cluster_dist_{k}"] = te_cluster_dists[:, k]

    num_cols_with_clusters = X_tr_f.select_dtypes(include=[np.number]).columns.tolist()

    models_dict = get_base_model_dict(seed=42 + fold * 10)

    for m_idx, (m_name, (model_obj, use_quantile)) in enumerate(models_dict.items()):
        preprocessor = build_preprocessor(num_cols_with_clusters, categorical_features, use_quantile=use_quantile)
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model_obj)
        ])
        pipe.fit(X_tr_f, y_tr_f)
        
        oof_probabilities[val_idx, m_idx] = pipe.predict_proba(X_va_f)[:, 1]
        test_fold_predictions[:, m_idx, fold] = pipe.predict_proba(X_te_f)[:, 1]

    print(f"Fold {fold + 1}/{N_SPLITS} Complete.")


# Individual OOF Scores
print("\n--- Individual Model OOF ROC-AUC Scores (With Clustering) ---")
for m_idx, m_name in enumerate(model_names):
    auc = roc_auc_score(y, oof_probabilities[:, m_idx])
    print(f"Model {m_idx + 1:02d} ({m_name:<30}): OOF AUC = {auc:.5f}")


# ============================================================
# 6. NON-NEGATIVE LEAST SQUARES (NNLS) META-OPTIMIZATION
# ============================================================

print("\n============================================")
print("FITTING NON-NEGATIVE CONSTRAINED META-LEARNER")
print("============================================")

# Convert OOF to Logit Space
oof_logits = np.zeros_like(oof_probabilities)
for m_idx in range(N_MODELS):
    p_cl = np.clip(oof_probabilities[:, m_idx], 1e-6, 1.0 - 1e-6)
    oof_logits[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

# Fit NNLS Meta-Learner (strictly non-negative w >= 0)
nnls_meta = LinearRegression(positive=True, fit_intercept=True)
nnls_meta.fit(oof_logits, y)

raw_weights = nnls_meta.coef_
sum_w = np.sum(raw_weights)
normalized_weights = raw_weights / sum_w if sum_w > 0 else np.ones(N_MODELS) / N_MODELS

print("Learned Non-Negative Meta-Weights:")
weight_summary = pd.DataFrame({
    "Base Model": model_names,
    "Raw Weight": raw_weights,
    "Normalized %": normalized_weights * 100
}).sort_values(by="Normalized %", ascending=False).reset_index(drop=True)

print(weight_summary.to_string(index=False))

stacked_oof_logits = np.dot(oof_logits, normalized_weights)
stacked_oof_probs = 1.0 / (1.0 + np.exp(-stacked_oof_logits))
stacked_oof_auc = roc_auc_score(y, stacked_oof_probs)

print(f"\n🏆 Clustered NNLS Super-Learner Stacked OOF AUC: {stacked_oof_auc:.5f}")


# ============================================================
# 7. INFERENCE ON TEST SET WITH BAYESIAN PRIOR ALIGNMENT
# ============================================================

print("\nGenerating final test predictions with Bayesian Prior Calibration...")

avg_test_probs = test_fold_predictions.mean(axis=2)
test_logits = np.zeros_like(avg_test_probs)

for m_idx in range(N_MODELS):
    p_cl = np.clip(avg_test_probs[:, m_idx], 1e-6, 1.0 - 1e-6)
    test_logits[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

final_test_logits = np.dot(test_logits, normalized_weights)

# Bayesian Log-Odds Prior Alignment (sharpens calibration around empirical 31.69% prior)
raw_test_mean_p = np.mean(1.0 / (1.0 + np.exp(-final_test_logits)))
prior_shift = np.log(GLOBAL_PRIOR / (1.0 - GLOBAL_PRIOR)) - np.log(raw_test_mean_p / (1.0 - raw_test_mean_p))

# 80% Original Stacker Logit + 20% Prior Shift (micro-adjustment)
calibrated_test_logits = final_test_logits + 0.20 * prior_shift
final_probabilities = 1.0 / (1.0 + np.exp(-calibrated_test_logits))


# ============================================================
# 8. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp89_latent_clusters_calibrated_superlearner.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 9. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 89 COMPLETE")
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
print("Exp 68 5-Fold NNLS Super Learner    : 0.66054 (Personal Best)")
print(f"Exp 89 Latent Clusters Super-Learner: OOF Val = {stacked_oof_auc:.5f} (Ready for submission)")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")