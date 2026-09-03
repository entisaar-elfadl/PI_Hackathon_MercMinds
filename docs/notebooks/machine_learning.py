# ============================================================
# EXPERIMENT 77 — DUAL-REGIME LABOUR TRANSITION SUPER-LEARNER
# (FIXED: MATCHED ARRAY MASKS + RETENTION VS JOB-FINDING NNLS)
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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


print("============================================")
print("EXPERIMENT 77")
print("DUAL-REGIME LABOUR TRANSITION SUPER-LEARNER")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR


# ============================================================
# 2. FEATURE EXTRACTION & PURE SEM MANIFOLD
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

    # ------------------------------------------------------------
    # SEM LATENT FACTOR BLOCKS
    # ------------------------------------------------------------
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
# 3. BASE MODELS SUITE
# ============================================================

def get_base_models(seed=42):
    """Returns the candidate models for NNLS stacking."""
    return {
        "LogReg_L2_C008": (LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed), False),
        "LogReg_L2_C010": (LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed), False),
        "LogReg_ElasticNet_C010": (LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000, random_state=seed), False),
        "Quantile_LogReg_ElasticNet": (LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000, random_state=seed), True),
        "MLP_Medium_64_32": (MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.01,
                                           batch_size=128, learning_rate_init=0.001, max_iter=350, early_stopping=True,
                                           n_iter_no_change=20, validation_fraction=0.15, random_state=seed), False),
        "MLP_Deep_128_64": (MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=0.015,
                                         batch_size=128, learning_rate_init=0.001, max_iter=350, early_stopping=True,
                                         n_iter_no_change=20, validation_fraction=0.15, random_state=seed + 100), False)
    }


# ============================================================
# 4. LOAD & PREPARE COMPLETE DATASET
# ============================================================

print("\n============================================")
print("EXTRACTING SEM MANIFOLD & PARTITIONING REGIMES")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()

train_sem, test_sem = extract_pure_sem_features(train_raw, test_raw)

common_cols = [c for c in train_sem.columns if c in test_sem.columns and c != target]

y_full = train_sem[target].reset_index(drop=True)
X_full = train_sem[common_cols].reset_index(drop=True)
X_test_full = test_sem[common_cols].reset_index(drop=True)

# Drop high-cardinality categorical (>100)
categorical_features = X_full.select_dtypes(include=["object", "category"]).columns.tolist()
high_cardinality = [c for c in categorical_features if X_full[c].nunique(dropna=True) > 100]
if high_cardinality:
    X_full = X_full.drop(columns=high_cardinality)
    X_test_full = X_test_full.drop(columns=high_cardinality)

categorical_features = X_full.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_features = X_full.select_dtypes(include=[np.number]).columns.tolist()

# Define Regimes directly from X_full / X_test_full (Guarantees exact length match!)
train_mask_retention = (X_full["employed_lag_num"] == 1.0).to_numpy()
train_mask_jobfinding = ~train_mask_retention

test_mask_retention = (X_test_full["employed_lag_num"] == 1.0).to_numpy()
test_mask_jobfinding = ~test_mask_retention

print(f"Total Observations: {len(X_full)} | Test Observations: {len(X_test_full)}")
print(f" -> Regime 1 (Job Retention - Employed Lag == 1): {np.sum(train_mask_retention)} Train | {np.sum(test_mask_retention)} Test")
print(f" -> Regime 2 (Job Finding  - Unemployed/First):  {np.sum(train_mask_jobfinding)} Train | {np.sum(test_mask_jobfinding)} Test")


# ============================================================
# 5. MODULAR 5-FOLD NNLS SUPER-LEARNER TRAINER
# ============================================================

def train_nnls_super_learner(X_subset, y_subset, X_target_test, regime_name):
    """Trains a leak-free 5-fold NNLS Super-Learner on a specific regime subset."""
    print(f"\n--- Training 5-Fold NNLS for {regime_name} ({len(X_subset)} rows) ---")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_sample = get_base_models(seed=42)
    model_names = list(models_sample.keys())
    n_mods = len(model_names)

    oof_probs = np.zeros((len(X_subset), n_mods))
    test_preds_folds = np.zeros((len(X_target_test), n_mods, 5))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_subset, y_subset)):
        X_tr_f, y_tr_f = X_subset.iloc[train_idx], y_subset.iloc[train_idx]
        X_va_f, y_va_f = X_subset.iloc[val_idx], y_subset.iloc[val_idx]

        models_pool = get_base_models(seed=42 + fold * 10)

        for m_idx, (m_name, (m_obj, use_q)) in enumerate(models_pool.items()):
            preproc = build_preprocessor(numerical_features, categorical_features, use_quantile=use_q)
            pipe = Pipeline([("preproc", preproc), ("model", m_obj)])
            pipe.fit(X_tr_f, y_tr_f)

            oof_probs[val_idx, m_idx] = pipe.predict_proba(X_va_f)[:, 1]
            test_preds_folds[:, m_idx, fold] = pipe.predict_proba(X_target_test)[:, 1]

    # Convert OOF to logit space
    oof_logits = np.zeros_like(oof_probs)
    for m_idx in range(n_mods):
        p_cl = np.clip(oof_probs[:, m_idx], 1e-6, 1.0 - 1e-6)
        oof_logits[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

    # NNLS Non-Negative Meta-Optimization
    nnls = LinearRegression(positive=True, fit_intercept=True)
    nnls.fit(oof_logits, y_subset)

    w_raw = nnls.coef_
    w_norm = w_raw / np.sum(w_raw) if np.sum(w_raw) > 0 else np.ones(n_mods) / n_mods

    # OOF AUC on this regime
    oof_reg_logits = np.dot(oof_logits, w_norm)
    oof_auc = roc_auc_score(y_subset, 1.0 / (1.0 + np.exp(-oof_reg_logits)))
    print(f" 🏆 {regime_name} OOF ROC-AUC: {oof_auc:.5f}")

    # Compute test logits
    avg_test_probs = test_preds_folds.mean(axis=2)
    test_logits = np.zeros_like(avg_test_probs)
    for m_idx in range(n_mods):
        p_cl = np.clip(avg_test_probs[:, m_idx], 1e-6, 1.0 - 1e-6)
        test_logits[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

    final_test_logits = np.dot(test_logits, w_norm)
    return oof_auc, final_test_logits


# ============================================================
# 6. TRAIN REGIME 1, REGIME 2 & GLOBAL ANCHOR
# ============================================================

# 1. Regime 1: Retention Specialist
auc_ret, test_logits_ret = train_nnls_super_learner(
    X_full[train_mask_retention].reset_index(drop=True),
    y_full[train_mask_retention].reset_index(drop=True),
    X_test_full,
    "Regime 1 (Job Retention Specialist)"
)

# 2. Regime 2: Job-Finding Specialist
auc_find, test_logits_find = train_nnls_super_learner(
    X_full[train_mask_jobfinding].reset_index(drop=True),
    y_full[train_mask_jobfinding].reset_index(drop=True),
    X_test_full,
    "Regime 2 (Job-Finding Specialist)"
)

# 3. Global Anchor: Proven 0.66054 Engine
auc_glob, test_logits_glob = train_nnls_super_learner(
    X_full,
    y_full,
    X_test_full,
    "Regime 3 (Global Anchor Engine)"
)


# ============================================================
# 7. CONDITIONAL DUAL-REGIME ROUTING & LOG-ODDS INTEGRATION
# ============================================================

print("\n============================================")
print("APPLYING CONDITIONAL DUAL-REGIME TEST ROUTING")
print("============================================")

final_test_logits = np.zeros(len(X_test_full))

# Route test cases with employed_lag == 1 through Retention Specialist (70% Specialist + 30% Global)
final_test_logits[test_mask_retention] = (
    0.70 * test_logits_ret[test_mask_retention] + 0.30 * test_logits_glob[test_mask_retention]
)

# Route test cases with employed_lag != 1 through Job-Finding Specialist (70% Specialist + 30% Global)
final_test_logits[test_mask_jobfinding] = (
    0.70 * test_logits_find[test_mask_jobfinding] + 0.30 * test_logits_glob[test_mask_jobfinding]
)

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

output_file = "submission_exp77_dual_regime_transition_superlearner.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 9. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 77 COMPLETE")
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
print(f"Regime 1 (Retention Model) OOF AUC : {auc_ret:.5f}")
print(f"Regime 2 (Job-Finding Model) OOF AUC: {auc_find:.5f}")
print(f"Global Anchor Engine OOF AUC       : {auc_glob:.5f}")
print("Exp 68 Benchmark                   : 0.66054")
print("Exp 77 Dual-Regime Submission      : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")