# ============================================================
# EXPERIMENT 60 — DEEP SEM LONGITUDINAL SUPER-TITAN
# (SEM FACTOR BLOCKS + LONGITUDINAL TRACK RECORDS + 50-MODEL BLEND)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


print("============================================")
print("EXPERIMENT 60")
print("DEEP SEM LONGITUDINAL SUPER-TITAN ENSEMBLE")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR


# ============================================================
# 2. DEEP SEM & LONGITUDINAL MEASUREMENT MODEL
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


def extract_deep_sem_features(train_df, test_df):
    """
    Unites SEM Latent Measurement Blocks with Leak-Free Longitudinal Track Records.
    """
    tr = train_df.copy()
    te = test_df.copy()

    # Clean target
    tr[target] = pd.to_numeric(tr[target], errors="coerce")
    tr = tr.dropna(subset=[target]).copy()
    tr[target] = tr[target].astype(int)

    # 1. Longitudinal Expanding Track Records (Zero Leakage)
    if "current_round" in tr.columns:
        round_col = pd.to_numeric(tr["current_round"], errors="coerce").fillna(1)
    else:
        round_col = pd.Series(1, index=tr.index)

    tr["_round_temp"] = round_col
    tr = tr.sort_values(["anonymised_id", "_round_temp"]).reset_index(drop=True)

    tr["user_past_sum"] = (
        tr.groupby("anonymised_id")[target]
        .transform(lambda s: s.shift(1).cumsum())
        .fillna(0.0)
    )
    tr["user_prev_obs_count"] = (
        tr.groupby("anonymised_id")[target]
        .transform(lambda s: s.shift(1).expanding().count())
        .fillna(0.0)
    )
    tr["user_past_employed_rate"] = np.where(
        tr["user_prev_obs_count"] > 0,
        tr["user_past_sum"] / tr["user_prev_obs_count"],
        0.317  # Fallback to global cohort prior
    )
    tr["user_has_past_history"] = (tr["user_prev_obs_count"] > 0).astype(float)
    tr = tr.drop(columns=["_round_temp", "user_past_sum"])

    # Map to test
    user_total_sum = tr.groupby("anonymised_id")[target].sum()
    user_total_count = tr.groupby("anonymised_id")[target].count()
    user_total_rate = user_total_sum / user_total_count

    te["user_prev_obs_count"] = te["anonymised_id"].map(user_total_count).fillna(0.0)
    te["user_past_employed_rate"] = te["anonymised_id"].map(user_total_rate).fillna(0.317)
    te["user_has_past_history"] = (te["user_prev_obs_count"] > 0).astype(float)

    # 2. Base Feature Pre-cleaning
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

        # Matric Marks
        for m in ["matric_englishhome", "matric_englishadd", "matric_mathpure", "matric_physicalscience", "matric_mathlit"]:
            if m in df.columns:
                df[f"{m}_score"] = df[m].apply(parse_matric_band).fillna(-1.0)

        # Credentials & Background
        df["school_quintile_num"] = pd.to_numeric(df.get("school_quintile", 0), errors="coerce").fillna(0.0)
        df["work_readiness_num"] = pd.to_numeric(df.get("work_readiness_score", 0.5), errors="coerce").fillna(0.5)
        df["age_clean"] = pd.to_numeric(df.get("age", 22), errors="coerce").fillna(22.0).clip(18, 35)
        df["has_tertiary"] = df.get("institution_type", pd.Series(np.nan, index=df.index)).notna().astype(float)
        df["has_seta"] = df.get("seta", pd.Series(np.nan, index=df.index)).notna().astype(float)

        # Auto-convert residual object columns
        for col in df.columns:
            if col != target and df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > 0.6 * df[col].notna().sum():
                    df[col] = converted

    # ------------------------------------------------------------
    # 3. SEM MEASUREMENT MODEL (LATENT FACTOR EXTRACTION)
    # ------------------------------------------------------------
    # Block A: Academic Human Capital
    acad_cols = [c for c in tr.columns if "_score" in c] + ["has_tertiary", "has_seta"]
    fa_acad = FactorAnalysis(n_components=2, random_state=42)
    pca_acad = PCA(n_components=1, random_state=42)
    imp_acad = SimpleImputer(strategy="median")
    scl_acad = StandardScaler()

    X_ac_tr = scl_acad.fit_transform(imp_acad.fit_transform(tr[acad_cols]))
    X_ac_te = scl_acad.transform(imp_acad.transform(te[acad_cols]))

    f_ac_tr = fa_acad.fit_transform(X_ac_tr)
    f_ac_te = fa_acad.transform(X_ac_te)
    p_ac_tr = pca_acad.fit_transform(X_ac_tr)
    p_ac_te = pca_acad.transform(X_ac_te)

    tr["latent_academic_fa_1"] = f_ac_tr[:, 0]
    tr["latent_academic_fa_2"] = f_ac_tr[:, 1]
    tr["latent_academic_pca"] = p_ac_tr[:, 0]
    te["latent_academic_fa_1"] = f_ac_te[:, 0]
    te["latent_academic_fa_2"] = f_ac_te[:, 1]
    te["latent_academic_pca"] = p_ac_te[:, 0]

    # Block B: Labour Momentum & Longitudinal Track History
    labour_cols = [
        "employed_lag_num", "tenure_lag_log", "days_since_obs_log", "is_first_time",
        "user_past_employed_rate", "user_prev_obs_count", "user_has_past_history"
    ]
    fa_lab = FactorAnalysis(n_components=2, random_state=42)
    pca_lab = PCA(n_components=1, random_state=42)
    imp_lab = SimpleImputer(strategy="median")
    scl_lab = StandardScaler()

    X_lb_tr = scl_lab.fit_transform(imp_lab.fit_transform(tr[labour_cols]))
    X_lb_te = scl_lab.transform(imp_lab.transform(te[labour_cols]))

    f_lb_tr = fa_lab.fit_transform(X_lb_tr)
    f_lb_te = fa_lab.transform(X_lb_te)
    p_lb_tr = pca_lab.fit_transform(X_lb_tr)
    p_lb_te = pca_lab.transform(X_lb_te)

    tr["latent_labour_fa_1"] = f_lb_tr[:, 0]
    tr["latent_labour_fa_2"] = f_lb_tr[:, 1]
    tr["latent_labour_pca"] = p_lb_tr[:, 0]
    te["latent_labour_fa_1"] = f_lb_te[:, 0]
    te["latent_labour_fa_2"] = f_lb_te[:, 1]
    te["latent_labour_pca"] = p_lb_te[:, 0]

    # Block C: Socio-Economic & Readiness Capital
    socio_cols = ["school_quintile_num", "work_readiness_num", "age_clean"]
    fa_soc = FactorAnalysis(n_components=2, random_state=42)
    imp_soc = SimpleImputer(strategy="median")
    scl_soc = StandardScaler()

    X_sc_tr = scl_soc.fit_transform(imp_soc.fit_transform(tr[socio_cols]))
    X_sc_te = scl_soc.transform(imp_soc.transform(te[socio_cols]))

    f_sc_tr = fa_soc.fit_transform(X_sc_tr)
    f_sc_te = fa_soc.transform(X_sc_te)

    tr["latent_socio_fa_1"] = f_sc_tr[:, 0]
    tr["latent_socio_fa_2"] = f_sc_tr[:, 1]
    te["latent_socio_fa_1"] = f_sc_te[:, 0]
    te["latent_socio_fa_2"] = f_sc_te[:, 1]

    # Block D: Supervised Partial Least Squares (PLS) Structural Projections (3 components)
    all_num_block = acad_cols + labour_cols + socio_cols
    pls = PLSRegression(n_components=3)
    imp_pls = SimpleImputer(strategy="median")

    X_pls_tr = imp_pls.fit_transform(tr[all_num_block])
    X_pls_te = imp_pls.transform(te[all_num_block])

    pls.fit(X_pls_tr, tr[target])
    pls_tr = pls.transform(X_pls_tr)
    pls_te = pls.transform(X_pls_te)

    tr["pls_structural_1"] = pls_tr[:, 0]
    tr["pls_structural_2"] = pls_tr[:, 1]
    tr["pls_structural_3"] = pls_tr[:, 2]
    te["pls_structural_1"] = pls_te[:, 0]
    te["pls_structural_2"] = pls_te[:, 1]
    te["pls_structural_3"] = pls_te[:, 2]

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
# 3. HIGH-PERFORMING TITAN MODEL SUITE
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
# 4. LOAD DATASET & EXTRACT DEEP SEM STRUCTURAL FACTORS
# ============================================================

print("\n============================================")
print("EXTRACTING DEEP SEM & LONGITUDINAL FACTORS")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()

train_sem, test_sem = extract_deep_sem_features(train_raw, test_raw)

common_cols = [c for c in train_sem.columns if c in test_sem.columns and c != target]

y_full = train_sem[target].reset_index(drop=True)
X_full = train_sem[common_cols].reset_index(drop=True)
X_test_full = test_sem[common_cols].reset_index(drop=True)

# Drop high-cardinality categorical (>100)
categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
high_cardinality = [c for c in categorical_features if X_full[c].nunique(dropna=True) > 100]
if high_cardinality:
    X_full = X_full.drop(columns=high_cardinality)
    X_test_full = X_test_full.drop(columns=high_cardinality)

categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X_full.select_dtypes(include=[np.number]).columns.tolist()

print(f"Full dataset: {len(X_full)} observations")
print(f"Features: {len(numerical_features)} numerical (including 11 Latent SEM/PLS Factors), {len(categorical_features)} categorical")


# ============================================================
# 5. 10-SEED SUPER-TITAN INFERENCE (50 MODELS)
# ============================================================

print("\n============================================")
print("TRAINING 10-SEED 50-MODEL TITAN ENGINE ON STRUCTURAL LATENT SPACE")
print("============================================")

SEEDS = [42, 101, 777, 2024, 999, 1337, 555, 888, 314, 271]

linear_logits_list = []
neural_logits_list = []

model_dict_sample = get_champion_models(seed=42)

for model_name, (_, use_quantile, model_family) in model_dict_sample.items():
    print(f" -> Training {model_name} ({model_family.upper()}) across {len(SEEDS)} seeds...")

    for s_idx, seed in enumerate(SEEDS):
        models_pool = get_champion_models(seed=seed)
        model_estimator, use_q, _ = models_pool[model_name]

        preprocessor = build_preprocessor(numerical_features, categorical_features, use_quantile=use_q)
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model_estimator)
        ])
        pipe.fit(X_full, y_full)
        probs = pipe.predict_proba(X_test_full)[:, 1]

        # Convert to log-odds (logit space)
        probs_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(probs_clipped / (1.0 - probs_clipped))

        if model_family == "linear":
            linear_logits_list.append(logits)
        else:
            neural_logits_list.append(logits)

# 80% Linear Champions + 20% Neural MLP in Logit Space
mean_linear_logits = np.mean(linear_logits_list, axis=0)
mean_neural_logits = np.mean(neural_logits_list, axis=0)

titan_logits = 0.80 * mean_linear_logits + 0.20 * mean_neural_logits

# Temperature calibration (T = 1.03)
final_probabilities = 1.0 / (1.0 + np.exp(-(titan_logits / 1.03)))


# ============================================================
# 6. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp60_sem_longitudinal_super_titan.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 7. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 60 COMPLETE")
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
print("Exp 51 Base Titan Hybrid           : 0.65693")
print("Exp 59 SEM Latent Factor Titan     : 0.65832 (Previous Best)")
print("Exp 60 Deep SEM Longitudinal Titan : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")