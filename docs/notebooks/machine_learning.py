# ============================================================
# EXPERIMENT 100 — THE CENTURION GRAND MASTER FINALE
# (CURATED AUTOGLUON ON SEM MANIFOLD + ELASTICNET + DEEP MLPs + SIMPLEX NNLS)
# ============================================================

import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# 1. Check for AutoGluon availability
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
from sklearn.metrics import roc_auc_score, log_loss


print("============================================")
print("EXPERIMENT 100 — THE CENTURION FINALE")
print(f"AutoGluon Engine Available: {HAS_AUTOGLUON}")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS & CLEAN LOCAL MODEL DIRECTORY
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR

# Local temp directory to prevent Windows OneDrive file locks
MODEL_SAVE_PATH = str(Path(tempfile.gettempdir()) / "autogluon_exp100_centurion")

if Path(MODEL_SAVE_PATH).exists():
    try:
        shutil.rmtree(MODEL_SAVE_PATH, ignore_errors=True)
    except Exception:
        pass

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()
target = "employed_status"


# ============================================================
# 2. EXACT PROVEN 0.66054 SEM FEATURE PIPELINE
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


print("\nExtracting Pure SEM Latent Features...")
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

print(f"Full Dataset: {len(X)} observations | {len(numerical_features)} numerical, {len(categorical_features)} categorical")


# ============================================================
# 3. PHASE 1: CURATED AUTOGLUON GBDT & PYTORCH ENGINE
# ============================================================

ag_oof_probs = None
ag_test_probs = None

if HAS_AUTOGLUON:
    print("\n============================================")
    print("PHASE 1: TRAINING CURATED AUTOGLUON ENGINE (LIGHTGBM + CATBOOST + PYTORCH)")
    print("============================================")

    # Clean dataset without text n-gram columns
    ag_train = TabularDataset(pd.concat([X, y], axis=1))
    ag_test = TabularDataset(X_test)

    # Curated model hyperparameters: strictly regularized trees & PyTorch NN (no FastAI, no unregularized RF)
    curated_hyperparameters = {
        'GBM': [{'learning_rate': 0.03, 'max_depth': 5, 'num_leaves': 31, 'ag_args': {'name_suffix': '_Reg'}}],
        'CAT': [{'depth': 5, 'learning_rate': 0.04, 'l2_leaf_reg': 3.0, 'ag_args': {'name_suffix': '_Reg'}}],
        'NN_TORCH': [{'num_layers': 3, 'hidden_size': 128, 'dropout_prob': 0.15, 'learning_rate': 0.001}]
    }

    try:
        predictor = TabularPredictor(
            label=target,
            eval_metric="roc_auc",
            problem_type="binary",
            path=MODEL_SAVE_PATH
        ).fit(
            train_data=ag_train,
            hyperparameters=curated_hyperparameters,
            time_limit=300,                  # 5 minutes
            num_bag_folds=5,                 # 5-fold OOF CV
            dynamic_stacking=False,          # Prevents Windows file locks
            verbosity=1
        )

        ag_oof_probs = predictor.predict_proba_oof()[1].to_numpy()
        ag_test_probs = predictor.predict_proba(ag_test)[1].to_numpy()
        ag_oof_auc = roc_auc_score(y, ag_oof_probs)
        print(f"✅ Curated AutoGluon Engine OOF ROC-AUC: {ag_oof_auc:.5f}")
    except Exception as e:
        print(f"⚠️ AutoGluon run skipped due to: {e}. Proceeding with pure SEM Super-Learner.")
        HAS_AUTOGLUON = False


# ============================================================
# 4. PHASE 2: PROVEN 0.66054 NNLS SUPER-LEARNER ENGINE
# ============================================================

print("\n============================================")
print("PHASE 2: TRAINING 5-FOLD EXP 68/93 PROVEN MODEL POOL")
print("============================================")

def build_preprocessor(numerical_cols, categorical_cols, use_quantile=False):
    transformers = []
    if len(numerical_cols) > 0:
        scaler = QuantileTransformer(output_distribution="normal", random_state=42) if use_quantile else StandardScaler()
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", scaler)]), numerical_cols))
    if len(categorical_cols) > 0:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols))
    return ColumnTransformer(transformers=transformers)


proven_base_models = {
    "Quantile_ElasticNet_C010": (LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=42), True),
    "Quantile_ElasticNet_C008": (LogisticRegression(C=0.08, penalty="elasticnet", solver="saga", l1_ratio=0.10, max_iter=2500, tol=1e-4, random_state=42), True),
    "Standard_ElasticNet_C010": (LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=42), False),
    "Standard_LogReg_L2_C008": (LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=2000, random_state=42), False),
    "MLP_Deep_128_64": (MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=0.015, batch_size=128, max_iter=400, early_stopping=True, random_state=42), False),
    "MLP_Medium_64_32": (MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.010, batch_size=128, max_iter=400, early_stopping=True, random_state=42), False)
}

m_names = list(proven_base_models.keys())
N_MODELS = len(m_names)
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

sem_oof_probs = np.zeros((len(X), N_MODELS))
sem_test_fold_preds = np.zeros((len(X_test), N_MODELS, N_SPLITS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr_f, y_tr_f = X.iloc[train_idx], y.iloc[train_idx]
    X_va_f, y_va_f = X.iloc[val_idx], y.iloc[val_idx]

    for m_idx, (m_name, (m_obj, use_q)) in enumerate(proven_base_models.items()):
        preproc = build_preprocessor(numerical_features, categorical_features, use_quantile=use_q)
        pipe = Pipeline([("preproc", preproc), ("model", m_obj)])
        pipe.fit(X_tr_f, y_tr_f)
        
        sem_oof_probs[val_idx, m_idx] = pipe.predict_proba(X_va_f)[:, 1]
        sem_test_fold_preds[:, m_idx, fold] = pipe.predict_proba(X_test)[:, 1]

sem_avg_test_probs = sem_test_fold_preds.mean(axis=2)


# ============================================================
# 5. PHASE 3: SIMPLEX CONVEX META-OPTIMIZATION (SLSQP LOG-LOSS)
# ============================================================

print("\n============================================")
print("PHASE 3: SIMPLEX CONVEX OPTIMIZATION (SLSQP CROSS-ENTROPY)")
print("============================================")

# Combine candidate pool: Base SEM Models (+ AutoGluon if available)
if HAS_AUTOGLUON and ag_oof_probs is not None:
    all_oof_probs = np.hstack([sem_oof_probs, ag_oof_probs.reshape(-1, 1)])
    all_test_probs = np.hstack([sem_avg_test_probs, ag_test_probs.reshape(-1, 1)])
    all_model_names = m_names + ["Curated_AutoGluon_Engine"]
else:
    all_oof_probs = sem_oof_probs
    all_test_probs = sem_avg_test_probs
    all_model_names = m_names

n_total_candidates = len(all_model_names)

# Convert OOF & Test Probabilities to Logit Space
oof_logits = np.zeros_like(all_oof_probs)
test_logits = np.zeros_like(all_test_probs)

for m_idx in range(n_total_candidates):
    p_cl_oof = np.clip(all_oof_probs[:, m_idx], 1e-6, 1.0 - 1e-6)
    oof_logits[:, m_idx] = np.log(p_cl_oof / (1.0 - p_cl_oof))
    p_cl_te = np.clip(all_test_probs[:, m_idx], 1e-6, 1.0 - 1e-6)
    test_logits[:, m_idx] = np.log(p_cl_te / (1.0 - p_cl_te))

y_true_arr = y.to_numpy()

# SLSQP Simplex Log-Loss Objective: min -sum(y*log(p) + (1-y)*log(1-p)) s.t. w >= 0, sum(w) = 1
def simplex_logloss_objective(weights, z_matrix, y_true):
    z_blend = np.dot(z_matrix, weights)
    p_blend = np.clip(1.0 / (1.0 + np.exp(-z_blend)), 1e-7, 1.0 - 1e-7)
    return log_loss(y_true, p_blend)

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
bounds = [(0.0, 1.0) for _ in range(n_total_candidates)]
w_init = np.ones(n_total_candidates) / n_total_candidates

opt_res = minimize(
    simplex_logloss_objective,
    w_init,
    args=(oof_logits, y_true_arr),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 500, 'ftol': 1e-9}
)

optimal_simplex_weights = opt_res.x / np.sum(opt_res.x)

# Display Learned Weights
weight_table = pd.DataFrame({
    "Candidate Model": all_model_names,
    "Simplex Weight": optimal_simplex_weights,
    "Allocation %": optimal_simplex_weights * 100
}).sort_values(by="Allocation %", ascending=False).reset_index(drop=True)

print("Learned Simplex Optimal Weights:")
print(weight_table.to_string(index=False))

stacked_oof_logits = np.dot(oof_logits, optimal_simplex_weights)
stacked_oof_probs = 1.0 / (1.0 + np.exp(-stacked_oof_logits))
final_oof_auc = roc_auc_score(y_true_arr, stacked_oof_probs)

print(f"\n🏆 Centurion Masterpiece Stacked OOF ROC-AUC: {final_oof_auc:.5f}")


# ============================================================
# 6. INFERENCE ON TEST SET
# ============================================================

print("\nGenerating final test predictions via Centurion Simplex Stacker...")

final_test_logits = np.dot(test_logits, optimal_simplex_weights)
final_probabilities = 1.0 / (1.0 + np.exp(-final_test_logits))
final_probabilities = np.clip(final_probabilities, 1e-6, 1.0 - 1e-6)


# ============================================================
# 7. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp100_centurion_masterpiece.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & INSPECTION
# ============================================================

print("\n============================================")
print("EXPERIMENT 100 COMPLETE — THE CENTURION MILESTONE")
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
print("Exp 30 Baseline Logistic            : 0.59229")
print("Exp 59 Pure SEM Latent Titan        : 0.65832")
print("Exp 68 / 93 Dual Peak Baseline      : 0.66054")
print(f"Exp 100 Centurion Masterpiece       : OOF Val = {final_oof_auc:.5f} (Ready for submission)")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")