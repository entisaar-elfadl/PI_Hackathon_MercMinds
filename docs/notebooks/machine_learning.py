# ============================================================
# EXPERIMENT 95 — PREBUILT MULTI-LAYER STACKING (AUTOGLUON TABULAR)
# (PREBUILT MULTI-LAYER ENSEMBLE: LIGHTGBM, CATBOOST, XGBOOST & NEURAL NETS)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# 1. Check for Prebuilt AutoGluon Framework
try:
    from autogluon.tabular import TabularDataset, TabularPredictor
    HAS_AUTOGLUON = True
except ImportError:
    HAS_AUTOGLUON = False

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
print("EXPERIMENT 95")
print("PREBUILT MULTI-LAYER AUTO-STACKING ENGINE")
print(f"AutoGluon Installed: {HAS_AUTOGLUON}")
print("============================================")


# ============================================================
# 2. DIRECTORY PATHS & DATA LOADING
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
# 3. DOMAIN FEATURE CLEANING
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


def prepare_tabular_data(train_df, test_df):
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
            df["survey_quarter"] = df["survey_date"].dt.quarter
            df["survey_dayofyear"] = df["survey_date"].dt.dayofyear
            df.drop(columns=["survey_date"], inplace=True)

        df["is_first_time"] = df["employed_lag"].isna().astype(float)
        df["employed_lag_num"] = df["employed_lag"].fillna(-1.0).astype(float)
        
        tenure_raw = pd.to_numeric(df.get("tenure_lag", 0), errors="coerce").fillna(0).clip(lower=0)
        df["tenure_lag_log"] = np.log1p(tenure_raw)
        
        days_raw = pd.to_numeric(df.get("days_since_last_obs", 0), errors="coerce").fillna(0).clip(lower=0)
        df["days_since_obs_log"] = np.log1p(days_raw)

        # Parse matric continuous marks
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

    return tr, te


train_clean, test_clean = prepare_tabular_data(train_raw, test_raw)


# ============================================================
# 4. PREBUILT AUTOGLUON EXECUTION OR PROVEN NNLS FALLBACK
# ============================================================

if HAS_AUTOGLUON:
    print("\n============================================")
    print("RUNNING PREBUILT AUTOGLUON MULTI-LAYER STACKING (BEST QUALITY)")
    print("============================================")

    # Convert to AutoGluon Tabular Dataset format
    ag_train = TabularDataset(train_clean)
    ag_test = TabularDataset(test_clean)

    # Initialize Prebuilt Multi-Layer Tabular Predictor
    predictor = TabularPredictor(
        label=target,
        eval_metric="roc_auc",
        problem_type="binary",
        path="autogluon_merc_minds_model"
    ).fit(
        train_data=ag_train,
        presets="best_quality",  # Automatically trains LightGBM, CatBoost, XGBoost, MLPs and multi-layer stacks them
        time_limit=600,          # 10 minutes maximum training budget
        auto_stack=True,         # Enables 2-layer stacking and bagging
        verbosity=2
    )

    print("\n--- AutoGluon Leaderboard of Trained Prebuilt Models ---")
    leaderboard = predictor.leaderboard(silent=True)
    print(leaderboard.head(10).to_string(index=False))

    # Predict positive class probabilities
    pred_probabilities = predictor.predict_proba(ag_test)[1].to_numpy()

else:
    print("\nℹ️ AutoGluon not found. Running Proven 0.66054 NNLS SEM Super-Learner directly...")
    print("(To run AutoGluon later, install it via: pip install autogluon)")

    # Latent SEM Blocks Extraction
    acad_cols = [c for c in train_clean.columns if "_score" in c]
    labour_cols = ["employed_lag_num", "tenure_lag_log", "days_since_obs_log", "is_first_time"]
    socio_cols = ["school_quintile_num", "work_readiness_num", "age_clean"]

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()

    if len(acad_cols) > 0:
        fa_acad = FactorAnalysis(n_components=2, random_state=42)
        X_ac_tr = scl.fit_transform(imp.fit_transform(train_clean[acad_cols]))
        X_ac_te = scl.transform(imp.transform(test_clean[acad_cols]))
        train_clean["latent_academic_factor_1"] = fa_acad.fit_transform(X_ac_tr)[:, 0]
        train_clean["latent_academic_factor_2"] = fa_acad.fit_transform(X_ac_tr)[:, 1]
        test_clean["latent_academic_factor_1"] = fa_acad.transform(X_ac_te)[:, 0]
        test_clean["latent_academic_factor_2"] = fa_acad.transform(X_ac_te)[:, 1]

    fa_lab = FactorAnalysis(n_components=2, random_state=42)
    X_lb_tr = scl.fit_transform(imp.fit_transform(train_clean[labour_cols]))
    X_lb_te = scl.transform(imp.transform(test_clean[labour_cols]))
    train_clean["latent_labour_momentum_1"] = fa_lab.fit_transform(X_lb_tr)[:, 0]
    train_clean["latent_labour_momentum_2"] = fa_lab.fit_transform(X_lb_tr)[:, 1]
    test_clean["latent_labour_momentum_1"] = fa_lab.transform(X_lb_te)[:, 0]
    test_clean["latent_labour_momentum_2"] = fa_lab.transform(X_lb_te)[:, 1]

    fa_soc = FactorAnalysis(n_components=2, random_state=42)
    X_sc_tr = scl.fit_transform(imp.fit_transform(train_clean[socio_cols]))
    X_sc_te = scl.transform(imp.transform(test_clean[socio_cols]))
    train_clean["latent_socio_readiness_1"] = fa_soc.fit_transform(X_sc_tr)[:, 0]
    train_clean["latent_socio_readiness_2"] = fa_soc.fit_transform(X_sc_tr)[:, 1]
    test_clean["latent_socio_readiness_1"] = fa_soc.transform(X_sc_te)[:, 0]
    test_clean["latent_socio_readiness_2"] = fa_soc.transform(X_sc_te)[:, 1]

    all_num_block = acad_cols + labour_cols + socio_cols
    pls = PLSRegression(n_components=2)
    X_pls_tr = imp.fit_transform(train_clean[all_num_block])
    X_pls_te = imp.transform(test_clean[all_num_block])
    pls.fit(X_pls_tr, train_clean[target])
    train_clean["pls_structural_latent_1"] = pls.transform(X_pls_tr)[:, 0]
    train_clean["pls_structural_latent_2"] = pls.transform(X_pls_tr)[:, 1]
    test_clean["pls_structural_latent_1"] = pls.transform(X_pls_te)[:, 0]
    test_clean["pls_structural_latent_2"] = pls.transform(X_pls_te)[:, 1]

    common_cols = [c for c in train_clean.columns if c in test_clean.columns and c != target]

    y = train_clean[target].reset_index(drop=True)
    X = train_clean[common_cols].reset_index(drop=True)
    X_test = test_clean[common_cols].reset_index(drop=True)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    high_c = [c for c in cat_cols if X[c].nunique(dropna=True) > 100]
    if high_c:
        X.drop(columns=high_c, inplace=True)
        X_test.drop(columns=high_c, inplace=True)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    def build_preprocessor(numerical_cols, categorical_cols, use_quantile=False):
        transformers = []
        if len(numerical_cols) > 0:
            scaler = QuantileTransformer(output_distribution="normal", random_state=42) if use_quantile else StandardScaler()
            transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", scaler)]), numerical_cols))
        if len(categorical_cols) > 0:
            transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols))
        return ColumnTransformer(transformers=transformers)

    models = {
        "Quantile_ElasticNet": (LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=42), True),
        "Standard_ElasticNet": (LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=42), False),
        "MLP_Deep_128_64": (MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=0.015, batch_size=128, max_iter=400, early_stopping=True, random_state=42), False),
        "MLP_Medium_64_32": (MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.010, batch_size=128, max_iter=400, early_stopping=True, random_state=42), False)
    }

    m_names = list(models.keys())
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    oof_probs = np.zeros((len(X), len(m_names)))
    test_fold_preds = np.zeros((len(X_test), len(m_names), N_SPLITS))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr_f, y_tr_f = X.iloc[train_idx], y.iloc[train_idx]
        X_va_f, y_va_f = X.iloc[val_idx], y.iloc[val_idx]

        for m_idx, (m_name, (m_obj, use_q)) in enumerate(models.items()):
            preproc = build_preprocessor(num_cols, cat_cols, use_quantile=use_q)
            pipe = Pipeline([("preproc", preproc), ("model", m_obj)])
            pipe.fit(X_tr_f, y_tr_f)
            oof_probs[val_idx, m_idx] = pipe.predict_proba(X_va_f)[:, 1]
            test_fold_preds[:, m_idx, fold] = pipe.predict_proba(X_test)[:, 1]

    oof_logits = np.zeros_like(oof_probs)
    for m in range(len(m_names)):
        p_cl = np.clip(oof_probs[:, m], 1e-6, 1.0 - 1e-6)
        oof_logits[:, m] = np.log(p_cl / (1.0 - p_cl))

    nnls = LinearRegression(positive=True, fit_intercept=True).fit(oof_logits, y)
    w_norm = nnls.coef_ / np.sum(nnls.coef_) if np.sum(nnls.coef_) > 0 else np.ones(len(m_names)) / len(m_names)

    avg_test_probs = test_fold_preds.mean(axis=2)
    test_logits = np.zeros_like(avg_test_probs)
    for m in range(len(m_names)):
        p_cl = np.clip(avg_test_probs[:, m], 1e-6, 1.0 - 1e-6)
        test_logits[:, m] = np.log(p_cl / (1.0 - p_cl))

    final_test_logits = np.dot(test_logits, w_norm)
    pred_probabilities = 1.0 / (1.0 + np.exp(-final_test_logits))


# ============================================================
# 5. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

final_probabilities = np.clip(pred_probabilities, 1e-6, 1.0 - 1e-6)

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp95_prebuilt_autogluon_stacker.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 6. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 95 COMPLETE")
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
print("Target Leaderboard Benchmark (Excel-lent Minds): 0.66372")
print("Exp 68 NNLS Super-Learner Baseline            : 0.66054")
print("Exp 95 Prebuilt AutoGluon Multi-Layer Stacker : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")