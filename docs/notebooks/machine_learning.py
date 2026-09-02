# ============================================================
# EXPERIMENT 67 — MACROECONOMIC & STRUCTURAL POLICY SEM TITAN
# (POLICY INDICES, STEM GATEWAYS, REGIONAL TIERS & 120-MODEL BLEND)
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
print("EXPERIMENT 67")
print("MACROECONOMIC & STRUCTURAL POLICY SEM TITAN")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR


# ============================================================
# 2. STRUCTURAL DOMAIN & POLICY FEATURE ENGINEERING
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


def extract_macro_structural_features(train_df, test_df):
    """
    Constructs structural macroeconomic, policy, demographic,
    and technological gateway features combined with SEM latent factor blocks.
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

        # ------------------------------------------------------------
        # PILLAR 1: ECONOMIC GROWTH & TEMPORAL CYCLES
        # ------------------------------------------------------------
        if "survey_date" in df.columns:
            df["survey_date"] = pd.to_datetime(df["survey_date"], errors="coerce")
            df["survey_year"] = df["survey_date"].dt.year
            df["survey_month"] = df["survey_date"].dt.month
            df["survey_quarter"] = df["survey_date"].dt.quarter
            df["survey_dayofyear"] = df["survey_date"].dt.dayofyear
            df.drop(columns=["survey_date"], inplace=True)

        # Regional Economic Density Tier (Gauteng + WC = Tier 3; KZN = Tier 2; Rural = Tier 1)
        prov_str = df.get("province", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
        df["economic_hub_tier"] = np.where(
            prov_str.isin(["gauteng", "western cape"]), 3.0,
            np.where(prov_str.isin(["kwazulu-natal"]), 2.0, 1.0)
        )

        # ------------------------------------------------------------
        # PILLAR 2: GOVERNMENT POLICIES & FUNDING QUINTILES
        # ------------------------------------------------------------
        quintile_raw = pd.to_numeric(df.get("school_quintile", 0), errors="coerce").fillna(0.0)
        df["school_quintile_num"] = quintile_raw
        # Policy flag: No-fee school (Quintiles 1-3) vs Fee-paying (Quintiles 4-5)
        df["is_resourced_school_policy"] = (quintile_raw >= 4).astype(float)

        # SETA Vocational Skills Policy Intervention
        df["has_seta_accreditation"] = df.get("seta", pd.Series(np.nan, index=df.index)).notna().astype(float)
        # Tertiary Institution Attainment
        df["has_tertiary_qualification"] = df.get("institution_type", pd.Series(np.nan, index=df.index)).notna().astype(float)

        # ------------------------------------------------------------
        # PILLAR 3: TECHNOLOGICAL & STEM GATEWAYS
        # ------------------------------------------------------------
        for m in ["matric_englishhome", "matric_englishadd", "matric_mathpure", "matric_physicalscience", "matric_mathlit"]:
            if m in df.columns:
                df[f"{m}_score"] = df[m].apply(parse_matric_band).fillna(-1.0)

        # Pure Math is the primary formal-tech filter in SA
        has_pure_m = (df.get("matric_mathpure_score", -1.0) > 0).astype(float)
        has_phys_s = (df.get("matric_physicalscience_score", -1.0) > 0).astype(float)
        df["tech_stem_qualification_index"] = has_pure_m * 2.0 + has_phys_s * 1.5

        # Digital & Work Readiness Score
        readiness_raw = pd.to_numeric(df.get("work_readiness_score", 0.5), errors="coerce").fillna(0.5)
        df["work_readiness_num"] = readiness_raw
        # Tech Interaction: Readiness x STEM credentials
        df["stem_x_readiness"] = df["tech_stem_qualification_index"] * readiness_raw

        # ------------------------------------------------------------
        # PILLAR 4: DEMOGRAPHIC SHIFTS & CAREER PHASES
        # ------------------------------------------------------------
        age_raw = pd.to_numeric(df.get("age", 22), errors="coerce").fillna(22.0).clip(18, 35)
        df["age_clean"] = age_raw
        
        # Demographic career phase (1: 18-21, 2: 22-25, 3: 26-29, 4: 30+)
        df["career_phase_tier"] = np.where(age_raw <= 21, 1.0,
                                  np.where(age_raw <= 25, 2.0,
                                  np.where(age_raw <= 29, 3.0, 4.0)))

        # ------------------------------------------------------------
        # PILLAR 5: LABOUR MOMENTUM & NEET ATTRITION
        # ------------------------------------------------------------
        df["is_first_time"] = df["employed_lag"].isna().astype(float)
        df["employed_lag_num"] = df["employed_lag"].fillna(-1.0).astype(float)
        
        tenure_raw = pd.to_numeric(df.get("tenure_lag", 0), errors="coerce").fillna(0).clip(lower=0)
        df["tenure_lag_log"] = np.log1p(tenure_raw)
        
        days_raw = pd.to_numeric(df.get("days_since_last_obs", 0), errors="coerce").fillna(0).clip(lower=0)
        df["days_since_obs_log"] = np.log1p(days_raw)

        # NEET Attrition Risk (time away from contact while not employed)
        is_unemployed_lag = (df["employed_lag_num"] == 0).astype(float)
        df["neet_attrition_risk"] = (days_raw / 365.25) * is_unemployed_lag

        # Clean remaining text/object fields safely
        for col in df.columns:
            if df[col].dtype == "bool":
                df[col] = df[col].astype(float)
            elif col != target and df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if df[col].notna().sum() > 0 and converted.notna().sum() > 0.6 * df[col].notna().sum():
                    df[col] = converted

    # ------------------------------------------------------------
    # 6. STRUCTURAL EQUATION MODELING (SEM FACTOR INTEGRATION)
    # ------------------------------------------------------------
    acad_cols = [c for c in tr.columns if "_score" in c] + ["tech_stem_qualification_index", "stem_x_readiness"]
    labour_cols = ["employed_lag_num", "tenure_lag_log", "days_since_obs_log", "is_first_time", "neet_attrition_risk"]
    socio_cols = ["school_quintile_num", "work_readiness_num", "age_clean", "economic_hub_tier", "career_phase_tier"]

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()

    # Block A: Academic & STEM Latent Factor
    fa_acad = FactorAnalysis(n_components=2, random_state=42)
    X_ac_tr = scl.fit_transform(imp.fit_transform(tr[acad_cols]))
    X_ac_te = scl.transform(imp.transform(te[acad_cols]))
    tr["latent_stem_academic_1"] = fa_acad.fit_transform(X_ac_tr)[:, 0]
    tr["latent_stem_academic_2"] = fa_acad.fit_transform(X_ac_tr)[:, 1]
    te["latent_stem_academic_1"] = fa_acad.transform(X_ac_te)[:, 0]
    te["latent_stem_academic_2"] = fa_acad.transform(X_ac_te)[:, 1]

    # Block B: Labour Momentum & Job Retention Latent Factor
    fa_lab = FactorAnalysis(n_components=2, random_state=42)
    X_lb_tr = scl.fit_transform(imp.fit_transform(tr[labour_cols]))
    X_lb_te = scl.transform(imp.transform(te[labour_cols]))
    tr["latent_labour_momentum_1"] = fa_lab.fit_transform(X_lb_tr)[:, 0]
    tr["latent_labour_momentum_2"] = fa_lab.fit_transform(X_lb_tr)[:, 1]
    te["latent_labour_momentum_1"] = fa_lab.transform(X_lb_te)[:, 0]
    te["latent_labour_momentum_2"] = fa_lab.transform(X_lb_te)[:, 1]

    # Block C: Socio-Economic & Regional Capital Latent Factor
    fa_soc = FactorAnalysis(n_components=2, random_state=42)
    X_sc_tr = scl.fit_transform(imp.fit_transform(tr[socio_cols]))
    X_sc_te = scl.transform(imp.transform(te[socio_cols]))
    tr["latent_macro_socio_1"] = fa_soc.fit_transform(X_sc_tr)[:, 0]
    tr["latent_macro_socio_2"] = fa_soc.fit_transform(X_sc_tr)[:, 1]
    te["latent_macro_socio_1"] = fa_soc.transform(X_sc_te)[:, 0]
    te["latent_macro_socio_2"] = fa_soc.transform(X_sc_te)[:, 1]

    # Block D: Supervised Partial Least Squares (PLS) Path Vectors
    all_num = acad_cols + labour_cols + socio_cols
    pls = PLSRegression(n_components=2)
    X_pls_tr = imp.fit_transform(tr[all_num])
    X_pls_te = imp.transform(te[all_num])
    pls.fit(X_pls_tr, tr[target])
    tr["pls_structural_path_1"] = pls.transform(X_pls_tr)[:, 0]
    tr["pls_structural_path_2"] = pls.transform(X_pls_tr)[:, 1]
    te["pls_structural_path_1"] = pls.transform(X_pls_te)[:, 0]
    te["pls_structural_path_2"] = pls.transform(X_pls_te)[:, 1]

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
# 3. 10-MODEL TITAN REGULARIZATION SUITE
# ============================================================

def get_titan_suite(seed=42):
    """Returns the precision 10-model regularized linear and neural suite."""
    return {
        "LogReg_L2_C006": (
            LogisticRegression(C=0.06, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False, "linear"
        ),
        "LogReg_L2_C008": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False, "linear"
        ),
        "LogReg_L2_C010": (
            LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False, "linear"
        ),
        "LogReg_L2_C012": (
            LogisticRegression(C=0.12, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            False, "linear"
        ),
        "LogReg_ElasticNet_C008_L1_010": (
            LogisticRegression(C=0.08, penalty="elasticnet", solver="saga", l1_ratio=0.10, max_iter=1000, random_state=seed),
            False, "linear"
        ),
        "LogReg_ElasticNet_C010_L1_015": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000, random_state=seed),
            False, "linear"
        ),
        "Quantile_LogReg_L2_C008": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed),
            True, "linear"
        ),
        "Quantile_LogReg_ElasticNet_C010": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000, random_state=seed),
            True, "linear"
        ),
        "MLP_Medium_64_32": (
            MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.01,
                batch_size=128, learning_rate_init=0.001, max_iter=350, early_stopping=True,
                n_iter_no_change=20, validation_fraction=0.15, random_state=seed
            ),
            False, "neural"
        ),
        "MLP_Deep_128_64": (
            MLPClassifier(
                hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=0.015,
                batch_size=128, learning_rate_init=0.001, max_iter=350, early_stopping=True,
                n_iter_no_change=20, validation_fraction=0.15, random_state=seed + 100
            ),
            False, "neural"
        )
    }


# ============================================================
# 4. LOAD & EXTRACT MACRO STRUCTURAL FACTORS
# ============================================================

print("\n============================================")
print("EXTRACTING STRUCTURAL POLICY & MACROECONOMIC INDICES")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()

train_macro, test_macro = extract_macro_structural_features(train_raw, test_raw)

common_cols = [c for c in train_macro.columns if c in test_macro.columns and c != target]

y_full = train_macro[target].reset_index(drop=True)
X_full = train_macro[common_cols].reset_index(drop=True)
X_test_full = test_macro[common_cols].reset_index(drop=True)

# Drop high-cardinality categorical (>100)
categorical_features = X_full.select_dtypes(include=["object", "category"]).columns.tolist()
high_cardinality = [c for c in categorical_features if X_full[c].nunique(dropna=True) > 100]
if high_cardinality:
    X_full = X_full.drop(columns=high_cardinality)
    X_test_full = X_test_full.drop(columns=high_cardinality)

categorical_features = X_full.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_features = X_full.select_dtypes(include=[np.number]).columns.tolist()

print(f"Full dataset: {len(X_full)} observations")
print(f"Features: {len(numerical_features)} numerical (incl. Policy/SEM Factors), {len(categorical_features)} categorical")


# ============================================================
# 5. TRAIN 120-MODEL TITAN ENGINE (12 SEEDS x 10 MODELS)
# ============================================================

print("\n============================================")
print("TRAINING 120-MODEL TITAN ENGINE (12 SEEDS x 10 MODELS)")
print("============================================")

SEEDS = [42, 101, 777, 2024, 999, 1337, 555, 888, 314, 271, 1618, 404]

linear_logits_list = []
neural_logits_list = []

model_dict_sample = get_titan_suite(seed=42)

for model_name, (_, use_quantile, model_family) in model_dict_sample.items():
    print(f" -> Training {model_name:<32} ({model_family.upper()}) across {len(SEEDS)} seeds...")

    for s_idx, seed in enumerate(SEEDS):
        models_pool = get_titan_suite(seed=seed)
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

# 82% Linear Champions + 18% Neural Diversity in Pure Logit Space
mean_linear_logits = np.mean(linear_logits_list, axis=0)
mean_neural_logits = np.mean(neural_logits_list, axis=0)

final_titan_logits = 0.82 * mean_linear_logits + 0.18 * mean_neural_logits

# Pure Logistic Sigmoid
final_probabilities = 1.0 / (1.0 + np.exp(-final_titan_logits))


# ============================================================
# 6. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp67_macro_policy_sem_titan.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 7. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 67 COMPLETE")
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
print("Exp 51 Base Titan Hybrid            : 0.65693")
print("Exp 66 Precision SEM Titan-120      : 0.65809")
print("Exp 59 Pure SEM Latent Titan        : 0.65832 (Current Best)")
print("Exp 67 Macro Policy SEM Titan       : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")