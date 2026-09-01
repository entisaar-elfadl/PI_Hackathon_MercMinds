# ============================================================
# EXPERIMENT 61 — DUAL-MANIFOLD GRAND MASTER
# (PURE SEM LATENT MANIFOLD 0.65832 + RAW ITEM MANIFOLD 0.65693)
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


print("============================================")
print("EXPERIMENT 61")
print("DUAL-MANIFOLD GRAND MASTER (SEM + RAW TITAN)")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR


# ============================================================
# 2. FEATURE EXTRACTION PIPELINES
# ============================================================

target = "employed_status"

def parse_matric_band(val):
    """Converts matric banded percentage strings into continuous marks."""
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


# --- Manifold A: Exact Exp 59 Pure SEM Latent Features (0.65832) ---
def extract_pure_sem_manifold(train_df, test_df):
    tr = train_df.copy()
    te = test_df.copy()

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
            if col != target and df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > 0.6 * df[col].notna().sum():
                    df[col] = converted

    # Block 1: Academic FA
    acad_cols = [c for c in tr.columns if "_score" in c]
    if len(acad_cols) > 0:
        fa_acad = FactorAnalysis(n_components=2, random_state=42)
        imp_acad = SimpleImputer(strategy="median")
        scl_acad = StandardScaler()
        X_ac_tr = scl_acad.fit_transform(imp_acad.fit_transform(tr[acad_cols]))
        X_ac_te = scl_acad.transform(imp_acad.transform(te[acad_cols]))

        f_ac_tr = fa_acad.fit_transform(X_ac_tr)
        f_ac_te = fa_acad.transform(X_ac_te)
        tr["latent_academic_factor_1"] = f_ac_tr[:, 0]
        tr["latent_academic_factor_2"] = f_ac_tr[:, 1]
        te["latent_academic_factor_1"] = f_ac_te[:, 0]
        te["latent_academic_factor_2"] = f_ac_te[:, 1]

    # Block 2: Labour Momentum FA
    labour_cols = ["employed_lag_num", "tenure_lag_log", "days_since_obs_log", "is_first_time"]
    imp_lab = SimpleImputer(strategy="median")
    scl_lab = StandardScaler()
    fa_lab = FactorAnalysis(n_components=2, random_state=42)

    X_lb_tr = scl_lab.fit_transform(imp_lab.fit_transform(tr[labour_cols]))
    X_lb_te = scl_lab.transform(imp_lab.transform(te[labour_cols]))

    f_lb_tr = fa_lab.fit_transform(X_lb_tr)
    f_lb_te = fa_lab.transform(X_lb_te)
    tr["latent_labour_momentum_1"] = f_lb_tr[:, 0]
    tr["latent_labour_momentum_2"] = f_lb_tr[:, 1]
    te["latent_labour_momentum_1"] = f_lb_te[:, 0]
    te["latent_labour_momentum_2"] = f_lb_te[:, 1]

    # Block 3: Socio-Economic FA
    socio_cols = ["school_quintile_num", "work_readiness_num", "age_clean"]
    imp_soc = SimpleImputer(strategy="median")
    scl_soc = StandardScaler()
    fa_soc = FactorAnalysis(n_components=2, random_state=42)

    X_sc_tr = scl_soc.fit_transform(imp_soc.fit_transform(tr[socio_cols]))
    X_sc_te = scl_soc.transform(imp_soc.transform(te[socio_cols]))

    f_sc_tr = fa_soc.fit_transform(X_sc_tr)
    f_sc_te = fa_soc.transform(X_sc_te)
    tr["latent_socio_readiness_1"] = f_sc_tr[:, 0]
    tr["latent_socio_readiness_2"] = f_sc_tr[:, 1]
    te["latent_socio_readiness_1"] = f_sc_te[:, 0]
    te["latent_socio_readiness_2"] = f_sc_te[:, 1]

    # Block 4: Supervised PLS Path Vectors
    all_num_block = acad_cols + labour_cols + socio_cols
    pls = PLSRegression(n_components=2)
    imp_pls = SimpleImputer(strategy="median")

    X_pls_tr = imp_pls.fit_transform(tr[all_num_block])
    X_pls_te = imp_pls.transform(te[all_num_block])

    pls.fit(X_pls_tr, tr[target])
    pls_tr = pls.transform(X_pls_tr)
    pls_te = pls.transform(X_pls_te)
    tr["pls_structural_latent_1"] = pls_tr[:, 0]
    tr["pls_structural_latent_2"] = pls_tr[:, 1]
    te["pls_structural_latent_1"] = pls_te[:, 0]
    te["pls_structural_latent_2"] = pls_te[:, 1]

    return tr, te


# --- Manifold B: Exact Exp 51 Raw Item Features (0.65693) ---
def extract_raw_item_manifold(train_df, test_df):
    tr = train_df.copy()
    te = test_df.copy()

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

        if "employed_lag" in df.columns:
            df["is_first_time"] = df["employed_lag"].isna().astype(float)
        
        if "tenure_lag" in df.columns:
            df["tenure_lag_log"] = np.log1p(pd.to_numeric(df.get("tenure_lag", 0), errors="coerce").fillna(0).clip(lower=0))

        if "days_since_last_obs" in df.columns:
            df["days_since_last_obs_log"] = np.log1p(pd.to_numeric(df.get("days_since_last_obs", 0), errors="coerce").fillna(0).clip(lower=0))

        for col in df.columns:
            if col != target and df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > 0.6 * df[col].notna().sum():
                    df[col] = converted

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
# 3. HIGH-SCORING TITAN MODEL FACTORIES
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
# 4. LOAD DATA & CONSTRUCT BOTH MANIFOLD MATRICES
# ============================================================

print("\n============================================")
print("PREPARING DUAL-MANIFOLD FEATURE MATRICES")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()

# 1. Manifold A (Exp 59 SEM Latent Features)
tr_sem, te_sem = extract_pure_sem_manifold(train_raw, test_raw)
common_sem = [c for c in tr_sem.columns if c in te_sem.columns and c != target]

y_sem = tr_sem[target].reset_index(drop=True)
X_tr_sem = tr_sem[common_sem].reset_index(drop=True)
X_te_sem = te_sem[common_sem].reset_index(drop=True)

cat_sem = X_tr_sem.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
high_card_sem = [c for c in cat_sem if X_tr_sem[c].nunique(dropna=True) > 100]
if high_card_sem:
    X_tr_sem = X_tr_sem.drop(columns=high_card_sem)
    X_te_sem = X_te_sem.drop(columns=high_card_sem)

cat_sem = X_tr_sem.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_sem = X_tr_sem.select_dtypes(include=[np.number]).columns.tolist()

# 2. Manifold B (Exp 51 Raw Item Features)
tr_raw_feat, te_raw_feat = extract_raw_item_manifold(train_raw, test_raw)
common_raw = [c for c in tr_raw_feat.columns if c in te_raw_feat.columns and c != target]

y_raw = tr_raw_feat[target].reset_index(drop=True)
X_tr_raw = tr_raw_feat[common_raw].reset_index(drop=True)
X_te_raw = te_raw_feat[common_raw].reset_index(drop=True)

cat_raw = X_tr_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
high_card_raw = [c for c in cat_raw if X_tr_raw[c].nunique(dropna=True) > 100]
if high_card_raw:
    X_tr_raw = X_tr_raw.drop(columns=high_card_raw)
    X_te_raw = X_te_raw.drop(columns=high_card_raw)

cat_raw = X_tr_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_raw = X_tr_raw.select_dtypes(include=[np.number]).columns.tolist()

print(f"Manifold A (Pure SEM 0.65832): {len(num_sem)} num (incl. 8 Latent Factors), {len(cat_sem)} cat")
print(f"Manifold B (Raw Titan 0.65693): {len(num_raw)} num, {len(cat_raw)} cat")


# ============================================================
# 5. TRAIN DUAL TITAN ENGINES (7 SEEDS EACH = 70 SUB-MODELS)
# ============================================================

SEEDS = [42, 101, 777, 2024, 999, 1337, 555]

def fit_titan_engine(X_train, y_train, X_test, num_cols, cat_cols, engine_name):
    """Fits 35-model ensemble on specified feature manifold."""
    print(f"\n--- Training 35-Model Titan Engine on {engine_name} ---")
    lin_logits = []
    neu_logits = []

    models_sample = get_champion_models(seed=42)

    for m_name, (_, use_q, m_family) in models_sample.items():
        for seed in SEEDS:
            m_pool = get_champion_models(seed=seed)
            estimator, use_quantile, _ = m_pool[m_name]

            preprocessor = build_preprocessor(num_cols, cat_cols, use_quantile=use_quantile)
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator)
            ])
            pipe.fit(X_train, y_train)
            probs = pipe.predict_proba(X_test)[:, 1]

            probs_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
            logits = np.log(probs_clipped / (1.0 - probs_clipped))

            if m_family == "linear":
                lin_logits.append(logits)
            else:
                neu_logits.append(logits)

    mean_lin = np.mean(lin_logits, axis=0)
    mean_neu = np.mean(neu_logits, axis=0)
    return 0.80 * mean_lin + 0.20 * mean_neu


# Run Engine A (Pure SEM 0.65832) & Engine B (Raw Titan 0.65693)
logits_sem_engine = fit_titan_engine(X_tr_sem, y_sem, X_te_sem, num_sem, cat_sem, "Manifold A (Pure SEM)")
logits_raw_engine = fit_titan_engine(X_tr_raw, y_raw, X_te_raw, num_raw, cat_raw, "Manifold B (Raw Titan)")


# ============================================================
# 6. DUAL-MANIFOLD GRAND MASTER INTEGRATION
# ============================================================

print("\n============================================")
print("FUSING PREDICTIONS (75% PURE SEM + 25% RAW TITAN)")
print("============================================")

# 75% Weight on Peak SEM 0.65832 + 25% on Peak Raw 0.65693
final_master_logits = 0.75 * logits_sem_engine + 0.25 * logits_raw_engine
final_probabilities = 1.0 / (1.0 + np.exp(-final_master_logits))


# ============================================================
# 7. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp61_dual_manifold_sem_blend.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 61 COMPLETE")
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
print("Exp 51 Base Titan Hybrid        : 0.65693")
print("Exp 60 Deep SEM Longitudinal    : 0.65735")
print("Exp 59 Pure SEM Latent Titan    : 0.65832 (Personal Best)")
print("Exp 61 Dual-Manifold Grand Blend: READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")