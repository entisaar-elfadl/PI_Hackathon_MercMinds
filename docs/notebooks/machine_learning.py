# ============================================================
# EXPERIMENT 93 — SIMPLEX CONVEX OPTIMIZATION & LOSS REDESIGN
# (SLSQP LOG-LOSS SIMPLEX + DIRECT AUC MAXIMIZER + EXP 68 CORE)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

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
print("EXPERIMENT 93")
print("SIMPLEX CONVEX OPTIMIZATION & LOSS REDESIGN")
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
# 2. EXACT WINNING 0.66054 SEM FEATURE PIPELINE
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


def build_preprocessor(numerical_cols, categorical_cols, use_quantile=False):
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
# 3. EXACT WINNING 9-MODEL BASE SUITE (EXP 68)
# ============================================================

def get_base_model_dict(seed=42):
    return {
        "LogReg_L2_C006": (LogisticRegression(C=0.06, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed), False),
        "LogReg_L2_C008": (LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed), False),
        "LogReg_L2_C010": (LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed), False),
        "LogReg_ElasticNet_C008": (LogisticRegression(C=0.08, penalty="elasticnet", solver="saga", l1_ratio=0.10, max_iter=2500, tol=1e-4, random_state=seed), False),
        "LogReg_ElasticNet_C010": (LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=seed), False),
        "Quantile_LogReg_L2_C008": (LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed), True),
        "Quantile_LogReg_ElasticNet": (LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=seed), True),
        "MLP_Medium_64_32": (MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.01,
                                           batch_size=128, learning_rate_init=0.001, max_iter=350, early_stopping=True,
                                           n_iter_no_change=20, validation_fraction=0.15, random_state=seed), False),
        "MLP_Deep_128_64": (MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=0.015,
                                         batch_size=128, learning_rate_init=0.001, max_iter=350, early_stopping=True,
                                         n_iter_no_change=20, validation_fraction=0.15, random_state=seed + 100), False)
    }


# ============================================================
# 4. LOAD DATASET & RUN 5-FOLD OOF EXTRACTION
# ============================================================

print("\n============================================")
print("EXTRACTING PURE SEM LATENT MANIFOLD")
print("============================================")

train_sem, test_sem = extract_pure_sem_features(train_raw, test_raw)
common_cols = [c for c in train_sem.columns if c in test_sem.columns and c != target]

y = train_sem[target].reset_index(drop=True)
X = train_sem[common_cols].reset_index(drop=True)
X_test = test_sem[common_cols].reset_index(drop=True)

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
high_cardinality = [c for c in categorical_features if X[c].nunique(dropna=True) > 100]
if high_cardinality:
    X.drop(columns=high_cardinality, inplace=True)
    X_test.drop(columns=high_cardinality, inplace=True)

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"Dataset: {len(X)} observations | {len(numerical_features)} numerical, {len(categorical_features)} categorical")

# 5-Fold Stratified CV
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

model_names = list(get_base_model_dict(seed=42).keys())
N_MODELS = len(model_names)

oof_probabilities = np.zeros((len(X), N_MODELS))
test_fold_predictions = np.zeros((len(X_test), N_MODELS, N_SPLITS))

print("\nGenerating 5-Fold OOF Predictions...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr_f, y_tr_f = X.iloc[train_idx], y.iloc[train_idx]
    X_va_f, y_va_f = X.iloc[val_idx], y.iloc[val_idx]

    models_dict = get_base_model_dict(seed=42 + fold * 10)

    for m_idx, (m_name, (model_obj, use_quantile)) in enumerate(models_dict.items()):
        preprocessor = build_preprocessor(numerical_features, categorical_features, use_quantile=use_quantile)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model_obj)])
        pipe.fit(X_tr_f, y_tr_f)
        
        oof_probabilities[val_idx, m_idx] = pipe.predict_proba(X_va_f)[:, 1]
        test_fold_predictions[:, m_idx, fold] = pipe.predict_proba(X_test)[:, 1]

    print(f"  -> Fold {fold + 1}/{N_SPLITS} Complete.")


# Convert OOF & Test Probabilities to Logit Space
oof_logits = np.zeros_like(oof_probabilities)
for m_idx in range(N_MODELS):
    p_cl = np.clip(oof_probabilities[:, m_idx], 1e-6, 1.0 - 1e-6)
    oof_logits[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

avg_test_probs = test_fold_predictions.mean(axis=2)
test_logits = np.zeros_like(avg_test_probs)
for m_idx in range(N_MODELS):
    p_cl = np.clip(avg_test_probs[:, m_idx], 1e-6, 1.0 - 1e-6)
    test_logits[:, m_idx] = np.log(p_cl / (1.0 - p_cl))


# ============================================================
# 5. BENCHMARKING ADVANCED CONVEX OPTIMIZERS
# ============================================================

print("\n============================================")
print("BENCHMARKING CONVEX OPTIMIZATION FORMULATIONS")
print("============================================")

y_true_vals = y.to_numpy()

# ------------------------------------------------------------
# 1. BASELINE: Classic OLS-NNLS (Exp 68 Baseline)
# ------------------------------------------------------------
nnls_classic = LinearRegression(positive=True, fit_intercept=True).fit(oof_logits, y_true_vals)
w_classic = nnls_classic.coef_ / np.sum(nnls_classic.coef_) if np.sum(nnls_classic.coef_) > 0 else np.ones(N_MODELS)/N_MODELS
auc_classic = roc_auc_score(y_true_vals, 1.0 / (1.0 + np.exp(-np.dot(oof_logits, w_classic))))
loss_classic = log_loss(y_true_vals, 1.0 / (1.0 + np.exp(-np.dot(oof_logits, w_classic))))
print(f"1. Classic OLS-NNLS Baseline   : OOF AUC = {auc_classic:.5f} | Log-Loss = {loss_classic:.5f}")


# ------------------------------------------------------------
# 2. CONVEX OPTIMIZER A: Simplex-Constrained Log-Loss (SLSQP)
# Directly minimizes cross-entropy: min -sum(y*log(p) + (1-y)*log(1-p)) s.t. w >= 0, sum(w) = 1
# ------------------------------------------------------------
def simplex_logloss_objective(weights, z_matrix, y_true):
    z_blend = np.dot(z_matrix, weights)
    p_blend = np.clip(1.0 / (1.0 + np.exp(-z_blend)), 1e-7, 1.0 - 1e-7)
    return log_loss(y_true, p_blend)

constraints_simplex = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
bounds_simplex = [(0.0, 1.0) for _ in range(N_MODELS)]
w_init = np.ones(N_MODELS) / N_MODELS

opt_logloss = minimize(
    simplex_logloss_objective,
    w_init,
    args=(oof_logits, y_true_vals),
    method='SLSQP',
    bounds=bounds_simplex,
    constraints=constraints_simplex,
    options={'maxiter': 500, 'ftol': 1e-9}
)
w_slsqp = opt_logloss.x / np.sum(opt_logloss.x)
auc_slsqp = roc_auc_score(y_true_vals, 1.0 / (1.0 + np.exp(-np.dot(oof_logits, w_slsqp))))
loss_slsqp = log_loss(y_true_vals, 1.0 / (1.0 + np.exp(-np.dot(oof_logits, w_slsqp))))
print(f"2. Simplex Log-Loss (SLSQP)    : OOF AUC = {auc_slsqp:.5f} | Log-Loss = {loss_slsqp:.5f}")


# ------------------------------------------------------------
# 3. CONVEX OPTIMIZER B: Direct Simplex ROC-AUC Maximizer (Nelder-Mead)
# Directly maximizes the rank-ordering competition metric on the unit simplex
# ------------------------------------------------------------
def direct_auc_objective(raw_w, z_matrix, y_true):
    # Softmax parameterization guarantees w_i >= 0 and sum(w) = 1
    w_norm = np.exp(raw_w - np.max(raw_w))
    w_norm = w_norm / np.sum(w_norm)
    z_blend = np.dot(z_matrix, w_norm)
    return -roc_auc_score(y_true, z_blend)

opt_auc = minimize(
    direct_auc_objective,
    np.zeros(N_MODELS),
    args=(oof_logits, y_true_vals),
    method='Nelder-Mead',
    options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-6}
)
w_raw_auc = np.exp(opt_auc.x - np.max(opt_auc.x))
w_direct_auc = w_raw_auc / np.sum(w_raw_auc)
auc_direct = roc_auc_score(y_true_vals, 1.0 / (1.0 + np.exp(-np.dot(oof_logits, w_direct_auc))))
loss_direct = log_loss(y_true_vals, 1.0 / (1.0 + np.exp(-np.dot(oof_logits, w_direct_auc))))
print(f"3. Direct AUC Maximizer        : OOF AUC = {auc_direct:.5f} | Log-Loss = {loss_direct:.5f}")


# ------------------------------------------------------------
# 4. CONVEX OPTIMIZER C: Entropy-Regularized Simplex Blending
# Adds maximum entropy penalty -tau * sum(w * log(w)) to prevent single-model overfitting
# ------------------------------------------------------------
def entropy_regularized_objective(weights, z_matrix, y_true, tau=0.005):
    z_blend = np.dot(z_matrix, weights)
    p_blend = np.clip(1.0 / (1.0 + np.exp(-z_blend)), 1e-7, 1.0 - 1e-7)
    entropy = -np.sum(weights * np.log(weights + 1e-12))
    return log_loss(y_true, p_blend) - tau * entropy

opt_entropy = minimize(
    entropy_regularized_objective,
    w_init,
    args=(oof_logits, y_true_vals, 0.005),
    method='SLSQP',
    bounds=bounds_simplex,
    constraints=constraints_simplex,
    options={'maxiter': 500, 'ftol': 1e-9}
)
w_entropy = opt_entropy.x / np.sum(opt_entropy.x)
auc_entropy = roc_auc_score(y_true_vals, 1.0 / (1.0 + np.exp(-np.dot(oof_logits, w_entropy))))
loss_entropy = log_loss(y_true_vals, 1.0 / (1.0 + np.exp(-np.dot(oof_logits, w_entropy))))
print(f"4. Entropy-Regularized Simplex : OOF AUC = {auc_entropy:.5f} | Log-Loss = {loss_entropy:.5f}")


# ============================================================
# 6. LEADERBOARD OF OPTIMIZATION FORMULATIONS
# ============================================================

optimizer_comparison = pd.DataFrame([
    {"Optimizer Formulation": "Direct Simplex AUC Maximizer", "OOF ROC-AUC": auc_direct, "Log-Loss": loss_direct, "Weights": w_direct_auc},
    {"Optimizer Formulation": "Simplex Log-Loss (SLSQP)", "OOF ROC-AUC": auc_slsqp, "Log-Loss": loss_slsqp, "Weights": w_slsqp},
    {"Optimizer Formulation": "Entropy-Regularized Simplex", "OOF ROC-AUC": auc_entropy, "Log-Loss": loss_entropy, "Weights": w_entropy},
    {"Optimizer Formulation": "Classic OLS-NNLS Baseline", "OOF ROC-AUC": auc_classic, "Log-Loss": loss_classic, "Weights": w_classic}
]).sort_values(by="OOF ROC-AUC", ascending=False).reset_index(drop=True)

print("\n============================================")
print("CONVEX OPTIMIZATION COMPARISON")
print("============================================")
print(optimizer_comparison[["Optimizer Formulation", "OOF ROC-AUC", "Log-Loss"]].to_string(index=False))

# Automatically select the winning convex optimizer
winning_optimizer_name = optimizer_comparison.iloc[0]["Optimizer Formulation"]
winning_weights = optimizer_comparison.iloc[0]["Weights"]
winning_auc = optimizer_comparison.iloc[0]["OOF ROC-AUC"]

print(f"\n🏆 WINNING CONVEX FORMULATION: '{winning_optimizer_name}' (OOF AUC: {winning_auc:.5f})")

# Print learned weights allocation
weights_df = pd.DataFrame({
    "Base Model": model_names,
    "Optimal Simplex Weight": winning_weights,
    "Allocation %": winning_weights * 100
}).sort_values(by="Allocation %", ascending=False).reset_index(drop=True)

print("\nOptimal Weight Allocation on Unit Simplex:")
print(weights_df.to_string(index=False))


# ============================================================
# 7. INFERENCE ON TEST SET
# ============================================================

print("\nGenerating final test predictions via winning convex optimization...")

final_test_logits = np.dot(test_logits, winning_weights)
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

output_file = "submission_exp93_simplex_convex_optimization.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 9. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 93 COMPLETE")
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
print("Exp 68 Classic NNLS Baseline       : 0.66054")
print(f"Exp 93 Simplex Convex Optimization : OOF Val = {winning_auc:.5f} ({winning_optimizer_name})")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")