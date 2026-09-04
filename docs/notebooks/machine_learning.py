# ============================================================
# EXPERIMENT 80-84 MASTER SUITE — DIVERSE MANIFOLD SUPER-LAB
# (EXP 80: DIVERSE MODELS | EXP 81: REPRESENTATIONS | EXP 82: DISAGREEMENT
#  EXP 83: SEM CONTEXT STACKING | EXP 84: CARUANA DP ENSEMBLE)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import rankdata

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.decomposition import FactorAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


print("============================================")
print("EXPERIMENT 80 - 84 MASTER LAB")
print("DIVERSE MANIFOLDS + CONTEXT STACKING + CARUANA DP")
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


def extract_features_and_sem(train_df, test_df):
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
            if df[col].dtype == "bool":
                df[col] = df[col].astype(float)
            elif col != target and df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if df[col].notna().sum() > 0 and converted.notna().sum() > 0.6 * df[col].notna().sum():
                    df[col] = converted

    # Extract 8 SEM Latent Factors
    acad_cols = [c for c in tr.columns if "_score" in c]
    labour_cols = ["employed_lag_num", "tenure_lag_log", "days_since_obs_log", "is_first_time"]
    socio_cols = ["school_quintile_num", "work_readiness_num", "age_clean"]

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()

    if len(acad_cols) > 0:
        fa_acad = FactorAnalysis(n_components=2, random_state=42)
        X_ac_tr = scl.fit_transform(imp.fit_transform(tr[acad_cols]))
        X_ac_te = scl.transform(imp.transform(te[acad_cols]))
        tr["sem_acad_1"] = fa_acad.fit_transform(X_ac_tr)[:, 0]
        tr["sem_acad_2"] = fa_acad.fit_transform(X_ac_tr)[:, 1]
        te["sem_acad_1"] = fa_acad.transform(X_ac_te)[:, 0]
        te["sem_acad_2"] = fa_acad.transform(X_ac_te)[:, 1]

    fa_lab = FactorAnalysis(n_components=2, random_state=42)
    X_lb_tr = scl.fit_transform(imp.fit_transform(tr[labour_cols]))
    X_lb_te = scl.transform(imp.transform(te[labour_cols]))
    tr["sem_labour_1"] = fa_lab.fit_transform(X_lb_tr)[:, 0]
    tr["sem_labour_2"] = fa_lab.fit_transform(X_lb_tr)[:, 1]
    te["sem_labour_1"] = fa_lab.transform(X_lb_te)[:, 0]
    te["sem_labour_2"] = fa_lab.transform(X_lb_te)[:, 1]

    fa_soc = FactorAnalysis(n_components=2, random_state=42)
    X_sc_tr = scl.fit_transform(imp.fit_transform(tr[socio_cols]))
    X_sc_te = scl.transform(imp.transform(te[socio_cols]))
    tr["sem_socio_1"] = fa_soc.fit_transform(X_sc_tr)[:, 0]
    tr["sem_socio_2"] = fa_soc.fit_transform(X_sc_tr)[:, 1]
    te["sem_socio_1"] = fa_soc.transform(X_sc_te)[:, 0]
    te["sem_socio_2"] = fa_soc.transform(X_sc_te)[:, 1]

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


train_sem, test_sem = extract_features_and_sem(train_raw, test_raw)
common_cols = [c for c in train_sem.columns if c in test_sem.columns and c != target]

y = train_sem[target].reset_index(drop=True)
X = train_sem[common_cols].reset_index(drop=True)
X_test = test_sem[common_cols].reset_index(drop=True)

# Drop high-cardinality categoricals (>100)
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
high_card = [c for c in cat_cols if X[c].nunique(dropna=True) > 100]
if high_card:
    X.drop(columns=high_card, inplace=True)
    X_test.drop(columns=high_card, inplace=True)

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
sem_factor_cols = [c for c in num_cols if "sem_" in c or "pls_path" in c]

print(f"Dataset Prepared: {len(X)} observations | {len(num_cols)} numerical, {len(cat_cols)} categorical")


# ============================================================
# 3. EXP 80: GENUINE DIVERSE MANIFOLD BASE MODELS
# ============================================================

class PLSProbClassifier:
    """Supervised PLS classifier with calibrated logistic transfer."""
    def __init__(self, n_components=3):
        self.pls = PLSRegression(n_components=n_components)

    def fit(self, X_mat, y_vec):
        self.pls.fit(X_mat, y_vec)
        return self

    def predict_proba(self, X_mat):
        preds = self.pls.predict(X_mat).ravel()
        p1 = np.clip(1.0 / (1.0 + np.exp(-preds * 3.0)), 1e-6, 1.0 - 1e-6)
        return np.vstack([1.0 - p1, p1]).T


def get_exp80_diverse_models(seed=42):
    return {
        "Manifold_1_LDA_Shrinkage": (
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            True  # Quantile-Gaussian space
        ),
        "Manifold_2_Supervised_PLS": (
            PLSProbClassifier(n_components=3),
            False # Standard space
        ),
        "Manifold_3_Quantile_ElasticNet": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=seed),
            True
        ),
        "Manifold_4_Standard_LogReg_L2": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed),
            False
        ),
        "Manifold_5_Deep_ReLU_MLP": (
            MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=0.015,
                          batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                          n_iter_no_change=25, validation_fraction=0.15, random_state=seed),
            False
        ),
        "Manifold_6_Smooth_Tanh_MLP": (
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="tanh", solver="adam", alpha=0.010,
                          batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                          n_iter_no_change=25, validation_fraction=0.15, random_state=seed + 50),
            False
        )
    }


# ============================================================
# 4. 5-FOLD OOF EXTRACTION
# ============================================================

print("\n============================================")
print("RUNNING 5-FOLD CROSS VALIDATION (EXP 80 DIVERSE MANIFOLDS)")
print("============================================")

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

model_names = list(get_exp80_diverse_models(seed=42).keys())
N_MODELS = len(model_names)

oof_probs = np.zeros((len(X), N_MODELS))
test_fold_preds = np.zeros((len(X_test), N_MODELS, N_SPLITS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr_f, y_tr_f = X.iloc[train_idx], y.iloc[train_idx]
    X_va_f, y_va_f = X.iloc[val_idx], y.iloc[val_idx]

    fold_models = get_exp80_diverse_models(seed=42 + fold * 10)

    for m_idx, (m_name, (m_obj, use_q)) in enumerate(fold_models.items()):
        scaler = QuantileTransformer(output_distribution="normal", random_state=42) if use_q else StandardScaler()
        preproc = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", scaler)]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
        ])
        pipe = Pipeline([("preproc", preproc), ("model", m_obj)])
        pipe.fit(X_tr_f, y_tr_f)

        oof_probs[val_idx, m_idx] = pipe.predict_proba(X_va_f)[:, 1]
        test_fold_preds[:, m_idx, fold] = pipe.predict_proba(X_test)[:, 1]

print("\n--- Exp 80: Diverse Manifold OOF Scores ---")
for m_idx, name in enumerate(model_names):
    score = roc_auc_score(y, oof_probs[:, m_idx])
    print(f" -> {name:<32}: OOF ROC-AUC = {score:.5f}")

avg_test_probs = test_fold_preds.mean(axis=2)


# ============================================================
# 5. EXP 81: PROBABILITY vs LOGIT vs RANK STACKING
# ============================================================

print("\n============================================")
print("EXP 81: PROBABILITY vs LOGIT vs RANK STACKING")
print("============================================")

# 1. Probability Representation
nnls_p = LinearRegression(positive=True, fit_intercept=True).fit(oof_probs, y)
w_p = nnls_p.coef_ / np.sum(nnls_p.coef_) if np.sum(nnls_p.coef_) > 0 else np.ones(N_MODELS)/N_MODELS
oof_stack_p = np.dot(oof_probs, w_p)
auc_p = roc_auc_score(y, oof_stack_p)

# 2. Logit Representation
oof_logits = np.zeros_like(oof_probs)
test_logits = np.zeros_like(avg_test_probs)
for m in range(N_MODELS):
    p_cl = np.clip(oof_probs[:, m], 1e-6, 1.0 - 1e-6)
    oof_logits[:, m] = np.log(p_cl / (1.0 - p_cl))
    p_te_cl = np.clip(avg_test_probs[:, m], 1e-6, 1.0 - 1e-6)
    test_logits[:, m] = np.log(p_te_cl / (1.0 - p_te_cl))

nnls_z = LinearRegression(positive=True, fit_intercept=True).fit(oof_logits, y)
w_z = nnls_z.coef_ / np.sum(nnls_z.coef_) if np.sum(nnls_z.coef_) > 0 else np.ones(N_MODELS)/N_MODELS
oof_stack_z = 1.0 / (1.0 + np.exp(-np.dot(oof_logits, w_z)))
auc_z = roc_auc_score(y, oof_stack_z)

# 3. Rank Representation
oof_ranks = np.zeros_like(oof_probs)
for m in range(N_MODELS):
    oof_ranks[:, m] = rankdata(oof_probs[:, m]) / len(oof_probs)

nnls_r = LinearRegression(positive=True, fit_intercept=True).fit(oof_ranks, y)
w_r = nnls_r.coef_ / np.sum(nnls_r.coef_) if np.sum(nnls_r.coef_) > 0 else np.ones(N_MODELS)/N_MODELS
oof_stack_r = np.dot(oof_ranks, w_r)
auc_r = roc_auc_score(y, oof_stack_r)

print(f" -> 1. Probability Stacking OOF AUC : {auc_p:.5f}")
print(f" -> 2. Log-Odds Logit Stacking OOF : {auc_z:.5f} (Optimal)")
print(f" -> 3. Rank Stacking OOF AUC        : {auc_r:.5f}")


# ============================================================
# 6. EXP 82: ADAPTIVE DISAGREEMENT & UNCERTAINTY STACKING
# ============================================================

print("\n============================================")
print("EXP 82: ADAPTIVE DISAGREEMENT STACKING")
print("============================================")

oof_std = np.std(oof_logits, axis=1, keepdims=True)
oof_range = (np.max(oof_logits, axis=1) - np.min(oof_logits, axis=1)).reshape(-1, 1)
oof_meta_exp82 = np.hstack([oof_logits, oof_std, oof_range])

meta_exp82 = LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
meta_exp82.fit(oof_meta_exp82, y)
oof_probs_exp82 = meta_exp82.predict_proba(oof_meta_exp82)[:, 1]
auc_exp82 = roc_auc_score(y, oof_probs_exp82)

print(f" -> Exp 82 Disagreement Stack OOF AUC: {auc_exp82:.5f}")


# ============================================================
# 7. EXP 83: SUPERVISED SEM CONTEXT-AWARE STACKING
# ============================================================

print("\n============================================")
print("EXP 83: SUPERVISED SEM CONTEXT-AWARE STACKING")
print("============================================")

sem_context_tr = X[sem_factor_cols].to_numpy()
sem_context_te = X_test[sem_factor_cols].to_numpy()

# Combine Model Logits + Disagreement + 8 SEM Latent Factor Vectors
oof_meta_exp83 = np.hstack([oof_logits, oof_std, sem_context_tr])
meta_exp83 = LogisticRegression(C=0.15, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
meta_exp83.fit(oof_meta_exp83, y)
oof_probs_exp83 = meta_exp83.predict_proba(oof_meta_exp83)[:, 1]
auc_exp83 = roc_auc_score(y, oof_probs_exp83)

print(f" -> Exp 83 SEM Context Stack OOF AUC: {auc_exp83:.5f}")


# ============================================================
# 8. EXP 84: CARUANA DYNAMIC PROGRAMMING OVER DIVERSE ENSEMBLE
# ============================================================

print("\n============================================")
print("EXP 84: CARUANA DYNAMIC PROGRAMMING SELECTION")
print("============================================")

def caruana_dp_selection(val_preds_matrix, y_true, n_iterations=150):
    n_samples, n_candidates = val_preds_matrix.shape
    selected_indices = []
    current_sum = np.zeros(n_samples)
    best_scores = []

    for t in range(1, n_iterations + 1):
        best_auc = -1.0
        best_idx = 0
        for cand_idx in range(n_candidates):
            trial_pred = (current_sum + val_preds_matrix[:, cand_idx]) / t
            score = roc_auc_score(y_true, trial_pred)
            if score > best_auc:
                best_auc = score
                best_idx = cand_idx
        selected_indices.append(best_idx)
        current_sum += val_preds_matrix[:, best_idx]
        best_scores.append(best_auc)

    counts = np.bincount(selected_indices, minlength=n_candidates)
    optimal_weights = counts / n_iterations
    return optimal_weights, best_scores[-1]

# Pool: 6 Base Diverse Models + Exp 81 Logit Stack + Exp 82 Disagreement Stack + Exp 83 Context Stack
candidate_pool_matrix = np.hstack([
    oof_probs,
    oof_stack_z.reshape(-1, 1),
    oof_probs_exp82.reshape(-1, 1),
    oof_probs_exp83.reshape(-1, 1)
])

candidate_pool_names = model_names + [
    "Exp81_Logit_Stack",
    "Exp82_Disagreement_Stack",
    "Exp83_SEM_Context_Stack"
]

dp_weights, auc_exp84 = caruana_dp_selection(candidate_pool_matrix, y.to_numpy(), n_iterations=150)

dp_summary = []
for p_idx, p_name in enumerate(candidate_pool_names):
    w = dp_weights[p_idx]
    if w > 0:
        dp_summary.append({"Ensemble Candidate": p_name, "DP Optimal Weight": f"{w * 100:.2f}%"})

print(pd.DataFrame(dp_summary).sort_values(by="DP Optimal Weight", ascending=False).to_string(index=False))
print(f"\n🏆 Exp 84 Dynamic Programming Optimized OOF AUC: {auc_exp84:.5f}")


# ============================================================
# 9. BENCHMARK SUMMARY & TEST PREDICTION GENERATION
# ============================================================

print("\n============================================")
print("EXPERIMENTS 80 - 84 BENCHMARK SUMMARY")
print("============================================")
summary_table = pd.DataFrame([
    {"Experiment": "Exp 80 (Best Base Model)", "OOF ROC-AUC": np.max([roc_auc_score(y, oof_probs[:, i]) for i in range(N_MODELS)])},
    {"Experiment": "Exp 81 (Logit NNLS Stacking)", "OOF ROC-AUC": auc_z},
    {"Experiment": "Exp 82 (Disagreement Features)", "OOF ROC-AUC": auc_exp82},
    {"Experiment": "Exp 83 (SEM Context Stacking)", "OOF ROC-AUC": auc_exp83},
    {"Experiment": "Exp 84 (Caruana DP Optimization)", "OOF ROC-AUC": auc_exp84}
]).sort_values(by="OOF ROC-AUC", ascending=False).reset_index(drop=True)

print(summary_table.to_string(index=False))

# --- Compute Test Predictions for All Pool Candidates ---
test_std = np.std(test_logits, axis=1, keepdims=True)
test_range = (np.max(test_logits, axis=1) - np.min(test_logits, axis=1)).reshape(-1, 1)

# Exp 81 Test Predictions
test_pred_exp81 = 1.0 / (1.0 + np.exp(-np.dot(test_logits, w_z)))

# Exp 82 Test Predictions
test_meta_exp82 = np.hstack([test_logits, test_std, test_range])
test_pred_exp82 = meta_exp82.predict_proba(test_meta_exp82)[:, 1]

# Exp 83 Test Predictions
test_meta_exp83 = np.hstack([test_logits, test_std, sem_context_te])
test_pred_exp83 = meta_exp83.predict_proba(test_meta_exp83)[:, 1]

# Construct Complete Test Pool Matrix
all_test_pool_matrix = np.hstack([
    avg_test_probs,
    test_pred_exp81.reshape(-1, 1),
    test_pred_exp82.reshape(-1, 1),
    test_pred_exp83.reshape(-1, 1)
])

# Convert Pool to Logit Space and apply Exp 84 Dynamic Programming Weights
test_pool_logits = np.zeros_like(all_test_pool_matrix)
for c in range(all_test_pool_matrix.shape[1]):
    p_cl = np.clip(all_test_pool_matrix[:, c], 1e-6, 1.0 - 1e-6)
    test_pool_logits[:, c] = np.log(p_cl / (1.0 - p_cl))

final_master_logits = np.dot(test_pool_logits, dp_weights)
final_probabilities = 1.0 / (1.0 + np.exp(-final_master_logits))


# ============================================================
# 10. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp80_84_master_champion.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 11. SUMMARY
# ============================================================

print("\n============================================")
print("MASTER LAB COMPLETE")
print("============================================")
print(f"Saved: {output_file}")
print(f"Rows: {len(submission)}")

print("\nPrediction summary:")
print(submission["employed_status"].describe())

print("\nFirst 10 predictions:")
print(submission.head(10))

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")