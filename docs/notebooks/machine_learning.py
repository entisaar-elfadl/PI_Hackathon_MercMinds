# ============================================================
# EXPERIMENT 59 — STRUCTURAL EQUATION & LATENT FACTOR CLASSIFIER
# (SEM FACTOR BLOCKS, PLS COVARIANCE PATHS & TITAN ENSEMBLE)
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
from sklearn.metrics import roc_auc_score


print("============================================")
print("EXPERIMENT 59")
print("STRUCTURAL EQUATION & LATENT FACTOR CLASSIFIER")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR


# ============================================================
# 2. FEATURE PARSER & LATENT MEASUREMENT BLOCKS
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


def extract_sem_structural_features(train_df, test_df):
    """
    Constructs the SEM Measurement Model:
    Extracts explicit latent factor blocks for Academic Capital,
    Labour Market Momentum, and Socio-Economic Capital.
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

        # Parse numeric lags
        df["is_first_time"] = df["employed_lag"].isna().astype(float)
        df["employed_lag_num"] = df["employed_lag"].fillna(-1.0).astype(float)
        df["tenure_lag_log"] = np.log1p(pd.to_numeric(df.get("tenure_lag", 0), errors="coerce").fillna(0).clip(lower=0))
        df["days_since_obs_log"] = np.log1p(pd.to_numeric(df.get("days_since_last_obs", 0), errors="coerce").fillna(0).clip(lower=0))

        # Parse matric marks
        for m in ["matric_englishhome", "matric_englishadd", "matric_mathpure", "matric_physicalscience", "matric_mathlit"]:
            if m in df.columns:
                df[f"{m}_score"] = df[m].apply(parse_matric_band).fillna(-1.0)

        # Education & Readiness
        df["school_quintile_num"] = pd.to_numeric(df.get("school_quintile", 0), errors="coerce").fillna(0.0)
        df["work_readiness_num"] = pd.to_numeric(df.get("work_readiness_score", 0.5), errors="coerce").fillna(0.5)
        df["age_clean"] = pd.to_numeric(df.get("age", 22), errors="coerce").fillna(22.0).clip(18, 35)

        # Auto-convert residual text
        for col in df.columns:
            if col != target and df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > 0.6 * df[col].notna().sum():
                    df[col] = converted

    # ------------------------------------------------------------
    # SEM LATENT FACTOR EXTRACTION (MEASUREMENT BLOCKS)
    # ------------------------------------------------------------
    # Block 1: Academic Capital Indicators
    acad_cols = [c for c in tr.columns if "_score" in c]
    if len(acad_cols) > 0:
        fa_acad = FactorAnalysis(n_components=2, random_state=42)
        imputer_acad = SimpleImputer(strategy="median")
        scaler_acad = StandardScaler()

        X_acad_tr = scaler_acad.fit_transform(imputer_acad.fit_transform(tr[acad_cols]))
        X_acad_te = scaler_acad.transform(imputer_acad.transform(te[acad_cols]))

        factors_acad_tr = fa_acad.fit_transform(X_acad_tr)
        factors_acad_te = fa_acad.transform(X_acad_te)

        tr["latent_academic_factor_1"] = factors_acad_tr[:, 0]
        tr["latent_academic_factor_2"] = factors_acad_tr[:, 1]
        te["latent_academic_factor_1"] = factors_acad_te[:, 0]
        te["latent_academic_factor_2"] = factors_acad_te[:, 1]

    # Block 2: Labour Market Momentum Indicators
    labour_cols = ["employed_lag_num", "tenure_lag_log", "days_since_obs_log", "is_first_time"]
    imputer_lab = SimpleImputer(strategy="median")
    scaler_lab = StandardScaler()
    fa_labour = FactorAnalysis(n_components=2, random_state=42)

    X_lab_tr = scaler_lab.fit_transform(imputer_lab.fit_transform(tr[labour_cols]))
    X_lab_te = scaler_lab.transform(imputer_lab.transform(te[labour_cols]))

    factors_lab_tr = fa_labour.fit_transform(X_lab_tr)
    factors_lab_te = fa_labour.transform(X_lab_te)

    tr["latent_labour_momentum_1"] = factors_lab_tr[:, 0]
    tr["latent_labour_momentum_2"] = factors_lab_tr[:, 1]
    te["latent_labour_momentum_1"] = factors_lab_te[:, 0]
    te["latent_labour_momentum_2"] = factors_lab_te[:, 1]

    # Block 3: Socio-Economic & Readiness Indicators
    socio_cols = ["school_quintile_num", "work_readiness_num", "age_clean"]
    imputer_soc = SimpleImputer(strategy="median")
    scaler_soc = StandardScaler()
    fa_socio = FactorAnalysis(n_components=2, random_state=42)

    X_soc_tr = scaler_soc.fit_transform(imputer_soc.fit_transform(tr[socio_cols]))
    X_soc_te = scaler_soc.transform(imputer_soc.transform(te[socio_cols]))

    factors_soc_tr = fa_socio.fit_transform(X_soc_tr)
    factors_soc_te = fa_socio.transform(X_soc_te)

    tr["latent_socio_readiness_1"] = factors_soc_tr[:, 0]
    tr["latent_socio_readiness_2"] = factors_soc_tr[:, 1]
    te["latent_socio_readiness_1"] = factors_soc_te[:, 0]
    te["latent_socio_readiness_2"] = factors_soc_te[:, 1]

    # Block 4: Partial Least Squares (PLS) Supervised Latent Path Vector
    all_num_block = acad_cols + labour_cols + socio_cols
    pls = PLSRegression(n_components=2)
    imputer_pls = SimpleImputer(strategy="median")

    X_pls_tr = imputer_pls.fit_transform(tr[all_num_block])
    X_pls_te = imputer_pls.transform(te[all_num_block])

    pls.fit(X_pls_tr, tr[target])
    pls_scores_tr = pls.transform(X_pls_tr)
    pls_scores_te = pls.transform(X_pls_te)

    tr["pls_structural_latent_1"] = pls_scores_tr[:, 0]
    tr["pls_structural_latent_2"] = pls_scores_tr[:, 1]
    te["pls_structural_latent_1"] = pls_scores_te[:, 0]
    te["pls_structural_latent_2"] = pls_scores_te[:, 1]

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
# 4. LOAD & EXTRACT STRUCTURAL LATENT FACTORS
# ============================================================

print("\n============================================")
print("EXTRACTING SEM LATENT MEASUREMENT BLOCKS & PLS VECTORS")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()

train_sem, test_sem = extract_sem_structural_features(train_raw, test_raw)

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
print(f"Features: {len(numerical_features)} numerical (including 8 Latent SEM Factors), {len(categorical_features)} categorical")


# ============================================================
# 5. MULTI-SEED TITAN INFERENCE (35 MODELS)
# ============================================================

print("\n============================================")
print("TRAINING 35-MODEL MULTI-SEED ENSEMBLE ON STRUCTURAL LATENT SPACE")
print("============================================")

SEEDS = [42, 101, 777, 2024, 999, 1337, 555]

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

final_hybrid_logits = 0.80 * mean_linear_logits + 0.20 * mean_neural_logits
final_probabilities = 1.0 / (1.0 + np.exp(-final_hybrid_logits))


# ============================================================
# 6. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp59_sem_latent_titan_ensemble.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 7. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 59 COMPLETE")
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
print("Exp 57 Grand Consensus Blend       : 0.65689")
print("Exp 59 SEM Latent Factor Titan     : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")