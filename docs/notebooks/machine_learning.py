# ============================================================
# EXPERIMENT 86 — THE DEFINITIVE INLINE-LONGITUDINAL SEM TITAN
# (INLINE TRANSITION DYNAMICS + STEM COMPOSITES + 15-FOLD NNLS)
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
print("EXPERIMENT 86")
print("INLINE-LONGITUDINAL SEM TITAN (0.665+ PUSH)")
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


# ============================================================
# 2. INLINE LONGITUDINAL DYNAMICS & DOMAIN FEATURE ENGINEERING
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


def extract_inline_longitudinal_sem(train_df, test_df):
    tr = train_df.copy()
    te = test_df.copy()

    # Clean target strictly in train
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
            df["survey_quarter"] = df["survey_date"].dt.quarter
            df["survey_dayofyear"] = df["survey_date"].dt.dayofyear
            df.drop(columns=["survey_date"], inplace=True)

        # ------------------------------------------------------------
        # A. INLINE LONGITUDINAL TRANSITION RATIOS
        # ------------------------------------------------------------
        curr_r = pd.to_numeric(df.get("current_round", 1), errors="coerce").fillna(1)
        lag_r = pd.to_numeric(df.get("lag_round", np.nan), errors="coerce")
        df["round_gap"] = (curr_r - lag_r).fillna(1.0).clip(lower=1.0)

        df["is_first_time"] = df["employed_lag"].isna().astype(float)
        df["employed_lag_num"] = df["employed_lag"].fillna(-1.0).astype(float)

        tenure_raw = pd.to_numeric(df.get("tenure_lag", 0), errors="coerce").fillna(0).clip(lower=0)
        df["tenure_lag_log"] = np.log1p(tenure_raw)

        days_raw = pd.to_numeric(df.get("days_since_last_obs", 0), errors="coerce").fillna(0).clip(lower=0)
        df["days_since_obs_log"] = np.log1p(days_raw)

        # Tenure retention rate vs days elapsed since previous observation
        df["tenure_retention_ratio"] = tenure_raw / (days_raw + 30.0)
        df["stable_job_momentum"] = ((df["employed_lag_num"] == 1.0) & (tenure_raw > 90.0)).astype(float)

        tot_rounds = pd.to_numeric(df.get("total_historical_rounds", 1), errors="coerce").fillna(1)
        df["historical_round_count"] = tot_rounds
        df["round_participation_density"] = tot_rounds / curr_r.clip(lower=1)

        # ------------------------------------------------------------
        # B. MATRIC CONTINUOUS MARKS & STEM COMPOSITES
        # ------------------------------------------------------------
        for m in ["matric_englishhome", "matric_englishadd", "matric_mathpure", "matric_physicalscience", "matric_mathlit"]:
            if m in df.columns:
                df[f"{m}_score"] = df[m].apply(parse_matric_band).fillna(-1.0)

        m_pure = df.get("matric_mathpure_score", pd.Series(-1.0, index=df.index))
        m_lit = df.get("matric_mathlit_score", pd.Series(-1.0, index=df.index))
        df["best_math_score"] = np.maximum(m_pure, m_lit)

        e_home = df.get("matric_englishhome_score", pd.Series(-1.0, index=df.index))
        e_add = df.get("matric_englishadd_score", pd.Series(-1.0, index=df.index))
        df["best_english_score"] = np.maximum(e_home, e_add)

        phys = df.get("matric_physicalscience_score", pd.Series(-1.0, index=df.index))
        df["stem_composite"] = np.where((m_pure > 0) & (phys > 0), (m_pure + phys) / 2.0, -1.0)

        # ------------------------------------------------------------
        # C. SOCIO-ECONOMIC & READINESS
        # ------------------------------------------------------------
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
    # D. SEM LATENT FACTOR BLOCKS (ACADEMIC, LABOUR, SOCIO, PLS)
    # ------------------------------------------------------------
    acad_cols = [c for c in tr.columns if "_score" in c or "stem_" in c or "best_" in c]
    labour_cols = [
        "employed_lag_num", "tenure_lag_log", "days_since_obs_log", "is_first_time",
        "tenure_retention_ratio", "stable_job_momentum", "round_gap", "round_participation_density"
    ]
    socio_cols = ["school_quintile_num", "work_readiness_num", "age_clean"]

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()

    # Block 1: Academic Factor Analysis
    if len(acad_cols) > 0:
        fa_acad = FactorAnalysis(n_components=2, random_state=42)
        X_ac_tr = scl.fit_transform(imp.fit_transform(tr[acad_cols]))
        X_ac_te = scl.transform(imp.transform(te[acad_cols]))
        tr["sem_academic_1"] = fa_acad.fit_transform(X_ac_tr)[:, 0]
        tr["sem_academic_2"] = fa_acad.fit_transform(X_ac_tr)[:, 1]
        te["sem_academic_1"] = fa_acad.transform(X_ac_te)[:, 0]
        te["sem_academic_2"] = fa_acad.transform(X_ac_te)[:, 1]

    # Block 2: Labour Momentum Factor Analysis
    fa_lab = FactorAnalysis(n_components=2, random_state=42)
    X_lb_tr = scl.fit_transform(imp.fit_transform(tr[labour_cols]))
    X_lb_te = scl.transform(imp.transform(te[labour_cols]))
    tr["sem_labour_1"] = fa_lab.fit_transform(X_lb_tr)[:, 0]
    tr["sem_labour_2"] = fa_lab.fit_transform(X_lb_tr)[:, 1]
    te["sem_labour_1"] = fa_lab.transform(X_lb_te)[:, 0]
    te["sem_labour_2"] = fa_lab.transform(X_lb_te)[:, 1]

    # Block 3: Socio-Economic Factor Analysis
    fa_soc = FactorAnalysis(n_components=2, random_state=42)
    X_sc_tr = scl.fit_transform(imp.fit_transform(tr[socio_cols]))
    X_sc_te = scl.transform(imp.transform(te[socio_cols]))
    tr["sem_socio_1"] = fa_soc.fit_transform(X_sc_tr)[:, 0]
    tr["sem_socio_2"] = fa_soc.fit_transform(X_sc_tr)[:, 1]
    te["sem_socio_1"] = fa_soc.transform(X_sc_te)[:, 0]
    te["sem_socio_2"] = fa_soc.transform(X_sc_te)[:, 1]

    # Block 4: Supervised PLS Direct Covariance Path (2 components)
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


# ============================================================
# 3. LOAD DATASET & PREPARE MATRICES
# ============================================================

print("\n============================================")
print("BUILDING INLINE-LONGITUDINAL SEM MANIFOLD")
print("============================================")

train_sem, test_sem = extract_inline_longitudinal_sem(train_raw, test_raw)

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

print(f"Full dataset: {len(X)} observations | Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")


# ============================================================
# 4. PREPROCESSOR & PROVEN BASE MODEL DEFINITIONS
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
    """Returns the proven 4-pillar suite (ElasticNet + Multi-Scale MLPs)."""
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
        )
    }


# ============================================================
# 5. REPEATED 5-FOLD NNLS SUPER-LEARNER (3 SEEDS = 15 FOLDS)
# ============================================================

print("\n============================================")
print("RUNNING REPEATED 5-FOLD NNLS SUPER-LEARNER (15 FOLDS)")
print("============================================")

REPEATED_SEEDS = [42, 101, 777]
N_SPLITS = 5

model_names = list(get_base_model_dict(seed=42).keys())
N_MODELS = len(model_names)

all_superlearner_test_logits = []
all_seed_oof_scores = []

for s_idx, cv_seed in enumerate(REPEATED_SEEDS):
    print(f"\n--- Running 5-Fold Partition Seed {cv_seed} ({s_idx + 1}/{len(REPEATED_SEEDS)}) ---")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=cv_seed)

    oof_probabilities = np.zeros((len(X), N_MODELS))
    test_fold_predictions = np.zeros((len(X_test), N_MODELS, N_SPLITS))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr_f, y_tr_f = X.iloc[train_idx], y.iloc[train_idx]
        X_va_f, y_va_f = X.iloc[val_idx], y.iloc[val_idx]

        models_dict = get_base_model_dict(seed=cv_seed + fold * 10)

        for m_idx, (m_name, (model_obj, use_quantile)) in enumerate(models_dict.items()):
            preprocessor = build_preprocessor(numerical_features, categorical_features, use_quantile=use_quantile)
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model_obj)
            ])
            pipe.fit(X_tr_f, y_tr_f)
            
            oof_probabilities[val_idx, m_idx] = pipe.predict_proba(X_va_f)[:, 1]
            test_fold_predictions[:, m_idx, fold] = pipe.predict_proba(X_test)[:, 1]

        print(f"  -> Fold {fold + 1}/{N_SPLITS} Complete.")

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
    norm_weights = raw_weights / sum_w if sum_w > 0 else np.ones(N_MODELS) / N_MODELS

    # Calculate this seed's OOF score
    seed_oof_logits = np.dot(oof_logits, norm_weights)
    seed_oof_auc = roc_auc_score(y, 1.0 / (1.0 + np.exp(-seed_oof_logits)))
    all_seed_oof_scores.append(seed_oof_auc)
    print(f"  🏆 Seed {cv_seed} Stacked OOF AUC = {seed_oof_auc:.5f}")

    # Generate test logits for this seed
    avg_test_probs = test_fold_predictions.mean(axis=2)
    test_logits = np.zeros_like(avg_test_probs)
    for m_idx in range(N_MODELS):
        p_cl = np.clip(avg_test_probs[:, m_idx], 1e-6, 1.0 - 1e-6)
        test_logits[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

    seed_test_logits = np.dot(test_logits, norm_weights)
    all_superlearner_test_logits.append(seed_test_logits)


# ============================================================
# 6. ENSEMBLE OF REPEATED SUPER-LEARNERS
# ============================================================

print("\n============================================")
print("INTEGRATING REPEATED SUPER-LEARNER TEST PREDICTIONS")
print("============================================")

mean_repeated_test_logits = np.mean(all_superlearner_test_logits, axis=0)
final_probabilities = 1.0 / (1.0 + np.exp(-mean_repeated_test_logits))

mean_oof_score = np.mean(all_seed_oof_scores)
print(f"Mean Repeated Stacked OOF AUC: {mean_oof_score:.5f}")


# ============================================================
# 7. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp86_inline_longitudinal_sem_titan.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 86 COMPLETE")
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
print(f"Exp 86 Inline-Longitudinal Titan    : Mean OOF = {mean_oof_score:.5f} (Ready for submission)")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")