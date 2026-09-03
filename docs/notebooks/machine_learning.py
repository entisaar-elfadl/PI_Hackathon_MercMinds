# ============================================================
# EXPERIMENT 73 — FULL-PANEL TRAJECTORY & NATIVE CATEGORICAL GBDT
# (FIXED: ORDINAL ENCODER max_categories=250 + FULL-PANEL RECONSTRUCTION)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, QuantileTransformer
from sklearn.decomposition import FactorAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


print("============================================")
print("EXPERIMENT 73")
print("FULL-PANEL TRAJECTORY & NATIVE CATEGORICAL GBDT")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR


# ============================================================
# 2. PARTICIPANT LIFETIME TRAJECTORY & DOMAIN FEATURE ENGINEERING
# ============================================================

target = "employed_status"
GLOBAL_PRIOR = 0.31694

def parse_matric_band(val):
    """Parses matric percentage strings into continuous numeric marks."""
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


def build_full_panel_datasets(train_raw, test_raw):
    """
    Constructs participant lifetime trajectories across survey rounds (1-8 -> 9)
    with zero future leakage in train, and exact lifetime history mapped to test.
    """
    tr = train_raw.copy()
    te = test_raw.copy()

    # Clean target in train
    tr[target] = pd.to_numeric(tr[target], errors="coerce")
    tr = tr.dropna(subset=[target]).copy()
    tr[target] = tr[target].astype(int)

    # ------------------------------------------------------------
    # 1. PARTICIPANT LIFETIME PANEL RECONSTRUCTION
    # ------------------------------------------------------------
    if "current_round" in tr.columns:
        tr["_round_order"] = pd.to_numeric(tr["current_round"], errors="coerce").fillna(1)
    else:
        tr["_round_order"] = 1

    # Sort strictly chronologically
    tr = tr.sort_values(["anonymised_id", "_round_order"]).reset_index(drop=True)

    # In Train: Expanding past history (only rounds strictly PRIOR to current row)
    tr["user_past_employed_sum"] = (
        tr.groupby("anonymised_id")[target]
        .transform(lambda s: s.shift(1).cumsum())
        .fillna(0.0)
    )
    tr["user_past_obs_count"] = (
        tr.groupby("anonymised_id")[target]
        .transform(lambda s: s.shift(1).expanding().count())
        .fillna(0.0)
    )
    tr["user_lifetime_emp_rate"] = np.where(
        tr["user_past_obs_count"] > 0,
        tr["user_past_employed_sum"] / tr["user_past_obs_count"],
        GLOBAL_PRIOR
    )
    tr["user_last_known_status"] = (
        tr.groupby("anonymised_id")[target]
        .transform(lambda s: s.shift(1))
        .fillna(-1.0)
    )
    tr["user_has_past_history"] = (tr["user_past_obs_count"] > 0).astype(float)
    tr = tr.drop(columns=["_round_order", "user_past_employed_sum"])

    # In Test (Round 9): Full lifetime history from all training rounds (1-8)
    user_total_sum = tr.groupby("anonymised_id")[target].sum()
    user_total_count = tr.groupby("anonymised_id")[target].count()
    user_lifetime_rate = user_total_sum / user_total_count
    user_latest_status = tr.groupby("anonymised_id")[target].last()

    te["user_past_obs_count"] = te["anonymised_id"].map(user_total_count).fillna(0.0)
    te["user_lifetime_emp_rate"] = te["anonymised_id"].map(user_lifetime_rate).fillna(GLOBAL_PRIOR)
    te["user_last_known_status"] = te["anonymised_id"].map(user_latest_status).fillna(-1.0)
    te["user_has_past_history"] = (te["user_past_obs_count"] > 0).astype(float)

    # ------------------------------------------------------------
    # 2. CALENDAR & LAG TRANSFORMATIONS
    # ------------------------------------------------------------
    for df in [tr, te]:
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

        for m in ["matric_englishhome", "matric_englishadd", "matric_mathpure", "matric_physicalscience", "matric_mathlit"]:
            if m in df.columns:
                df[f"{m}_score"] = df[m].apply(parse_matric_band).fillna(-1.0)

        df["school_quintile_num"] = pd.to_numeric(df.get("school_quintile", 0), errors="coerce").fillna(0.0)
        df["work_readiness_num"] = pd.to_numeric(df.get("work_readiness_score", 0.5), errors="coerce").fillna(0.5)
        df["age_clean"] = pd.to_numeric(df.get("age", 22), errors="coerce").fillna(22.0).clip(18, 35)

        for col in df.columns:
            if df[col].dtype == "bool":
                df[col] = df[col].astype(float)
            elif col not in [target, "anonymised_id"] and df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if df[col].notna().sum() > 0 and converted.notna().sum() > 0.6 * df[col].notna().sum():
                    df[col] = converted

    return tr, te


# ============================================================
# 3. SEM FACTOR EXTRACTION FOR LINEAR ENGINE
# ============================================================

def attach_sem_latent_factors(tr, te):
    """Computes Factor Analysis & PLS factors on numeric columns."""
    tr_sem = tr.copy()
    te_sem = te.copy()

    acad_cols = [c for c in tr_sem.columns if "_score" in c]
    labour_cols = ["employed_lag_num", "tenure_lag_log", "days_since_obs_log", "is_first_time", "user_has_past_history"]
    socio_cols = ["school_quintile_num", "work_readiness_num", "age_clean"]

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()

    if len(acad_cols) > 0:
        fa_acad = FactorAnalysis(n_components=2, random_state=42)
        X_ac_tr = scl.fit_transform(imp.fit_transform(tr_sem[acad_cols]))
        X_ac_te = scl.transform(imp.transform(te_sem[acad_cols]))
        tr_sem["latent_acad_1"] = fa_acad.fit_transform(X_ac_tr)[:, 0]
        tr_sem["latent_acad_2"] = fa_acad.fit_transform(X_ac_tr)[:, 1]
        te_sem["latent_acad_1"] = fa_acad.transform(X_ac_te)[:, 0]
        te_sem["latent_acad_2"] = fa_acad.transform(X_ac_te)[:, 1]

    fa_lab = FactorAnalysis(n_components=2, random_state=42)
    X_lb_tr = scl.fit_transform(imp.fit_transform(tr_sem[labour_cols]))
    X_lb_te = scl.transform(imp.transform(te_sem[labour_cols]))
    tr_sem["latent_labour_1"] = fa_lab.fit_transform(X_lb_tr)[:, 0]
    tr_sem["latent_labour_2"] = fa_lab.fit_transform(X_lb_tr)[:, 1]
    te_sem["latent_labour_1"] = fa_lab.transform(X_lb_te)[:, 0]
    te_sem["latent_labour_2"] = fa_lab.transform(X_lb_te)[:, 1]

    all_num = acad_cols + labour_cols + socio_cols
    pls = PLSRegression(n_components=2)
    X_pls_tr = imp.fit_transform(tr_sem[all_num])
    X_pls_te = imp.transform(te_sem[all_num])
    pls.fit(X_pls_tr, tr_sem[target])
    tr_sem["pls_latent_1"] = pls.transform(X_pls_tr)[:, 0]
    tr_sem["pls_latent_2"] = pls.transform(X_pls_tr)[:, 1]
    te_sem["pls_latent_1"] = pls.transform(X_pls_te)[:, 0]
    te_sem["pls_latent_2"] = pls.transform(X_pls_te)[:, 1]

    return tr_sem, te_sem


# ============================================================
# 4. LOAD & CONSTRUCT COMPLETE FEATURE MATRICES
# ============================================================

print("\n============================================")
print("RECONSTRUCTING PARTICIPANT PANEL TRAJECTORIES")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()

tr_panel, te_panel = build_full_panel_datasets(train_raw, test_raw)
tr_sem, te_sem = attach_sem_latent_factors(tr_panel, te_panel)

# Remove ID from modeling matrix
tr_clean = tr_sem.drop(columns=["anonymised_id"])
te_clean = te_sem.drop(columns=["anonymised_id"])

common_cols = [c for c in tr_clean.columns if c in te_clean.columns and c != target]

y = tr_clean[target].reset_index(drop=True)
X = tr_clean[common_cols].reset_index(drop=True)
X_test = te_clean[common_cols].reset_index(drop=True)

# ------------------------------------------------------------
# Separate Feature Sets for Trees vs Linear Models
# ------------------------------------------------------------
tree_cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
tree_num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Linear Models drop categoricals > 100 for OneHot stability
linear_high_card = [c for c in tree_cat_cols if X[c].nunique(dropna=True) > 100]
X_linear = X.drop(columns=linear_high_card)
X_test_linear = X_test.drop(columns=linear_high_card)

linear_cat_cols = X_linear.select_dtypes(include=["object", "category"]).columns.tolist()
linear_num_cols = X_linear.select_dtypes(include=[np.number]).columns.tolist()

print(f"Total Observations: {len(X)}")
print(f"Tree Engine: {len(tree_num_cols)} numerical, {len(tree_cat_cols)} categorical (ALL municipalities & qualifications retained!)")
print(f"Linear Engine: {len(linear_num_cols)} numerical, {len(linear_cat_cols)} categorical")


# ============================================================
# 5. ENGINE 1: NATIVE CATEGORICAL GRADIENT BOOSTED TREES (5-FOLD)
# ============================================================

print("\n============================================")
print("TRAINING ENGINE 1: NATIVE CATEGORICAL GRADIENT BOOSTING")
print("============================================")

# max_categories=250 automatically groups long-tail rare categories into index 249,
# strictly satisfying HistGradientBoosting's <= 255 cardinality requirement!
tree_preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), tree_num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(
                max_categories=250,
                handle_unknown="use_encoded_value",
                unknown_value=np.nan
            ))
        ]), tree_cat_cols)
    ]
)

# Fit-transform features for trees
X_tree_proc = tree_preprocessor.fit_transform(X)
X_test_tree_proc = tree_preprocessor.transform(X_test)

# Compute exact boolean categorical mask matching the transformed columns
n_num_proc = len(tree_num_cols)
n_cat_proc = len(tree_cat_cols)
cat_mask = np.array([False] * n_num_proc + [True] * n_cat_proc)

tree_models_specs = [
    ("HistGB_Standard", HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.03, max_iter=350, max_leaf_nodes=31,
        min_samples_leaf=25, l2_regularization=2.0, categorical_features=cat_mask,
        early_stopping=True, n_iter_no_change=20, validation_fraction=0.15, random_state=42
    )),
    ("HistGB_Deep", HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.025, max_iter=400, max_leaf_nodes=45,
        min_samples_leaf=20, l2_regularization=3.0, categorical_features=cat_mask,
        early_stopping=True, n_iter_no_change=20, validation_fraction=0.15, random_state=101
    )),
    ("HistGB_Compact", HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.035, max_iter=300, max_leaf_nodes=21,
        min_samples_leaf=35, l2_regularization=1.0, categorical_features=cat_mask,
        early_stopping=True, n_iter_no_change=20, validation_fraction=0.15, random_state=777
    ))
]

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

tree_oof_preds = np.zeros(len(X))
tree_test_fold_preds = np.zeros((len(X_test), N_SPLITS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr_f, y_tr_f = X_tree_proc[train_idx], y.iloc[train_idx]
    X_va_f, y_va_f = X_tree_proc[val_idx], y.iloc[val_idx]

    fold_val_acc = np.zeros(len(val_idx))
    fold_test_acc = np.zeros(len(X_test))

    for _, t_model in tree_models_specs:
        t_model.fit(X_tr_f, y_tr_f)
        fold_val_acc += t_model.predict_proba(X_va_f)[:, 1] / len(tree_models_specs)
        fold_test_acc += t_model.predict_proba(X_test_tree_proc)[:, 1] / len(tree_models_specs)

    tree_oof_preds[val_idx] = fold_val_acc
    tree_test_fold_preds[:, fold] = fold_test_acc
    print(f"  -> Tree Fold {fold + 1} ROC-AUC: {roc_auc_score(y_va_f, fold_val_acc):.5f}")

tree_total_oof_auc = roc_auc_score(y, tree_oof_preds)
print(f"🏆 Engine 1 (Native Categorical GBDT) OOF ROC-AUC: {tree_total_oof_auc:.5f}")


# ============================================================
# 6. ENGINE 2: 0.66054 NNLS SUPER-LEARNER (5-FOLD)
# ============================================================

print("\n============================================")
print("TRAINING ENGINE 2: PROVEN 0.66054 NNLS SUPER-LEARNER")
print("============================================")

def build_linear_preprocessor(numerical_cols, categorical_cols, use_quantile=False):
    transformers = []
    if len(numerical_cols) > 0:
        scaler = QuantileTransformer(output_distribution="normal", random_state=42) if use_quantile else StandardScaler()
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", scaler)]), numerical_cols))
    if len(categorical_cols) > 0:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols))
    return ColumnTransformer(transformers=transformers)

linear_base_models = {
    "Quantile_ElasticNet": (LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000, random_state=42), True),
    "Standard_ElasticNet": (LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000, random_state=42), False),
    "MLP_Deep_128_64": (MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=0.015, batch_size=128, max_iter=350, early_stopping=True, random_state=42), False),
    "MLP_Medium_64_32": (MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.010, batch_size=128, max_iter=350, early_stopping=True, random_state=42), False)
}

m_names = list(linear_base_models.keys())
lin_oof_probs = np.zeros((len(X), len(m_names)))
lin_test_fold_preds = np.zeros((len(X_test), len(m_names), N_SPLITS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_linear, y)):
    X_tr_f, y_tr_f = X_linear.iloc[train_idx], y.iloc[train_idx]
    X_va_f, y_va_f = X_linear.iloc[val_idx], y.iloc[val_idx]

    for m_idx, (m_name, (m_obj, use_q)) in enumerate(linear_base_models.items()):
        preproc = build_linear_preprocessor(linear_num_cols, linear_cat_cols, use_quantile=use_q)
        pipe = Pipeline([("preproc", preproc), ("model", m_obj)])
        pipe.fit(X_tr_f, y_tr_f)
        lin_oof_probs[val_idx, m_idx] = pipe.predict_proba(X_va_f)[:, 1]
        lin_test_fold_preds[:, m_idx, fold] = pipe.predict_proba(X_test_linear)[:, 1]

# NNLS Meta-Optimization for Linear Engine
lin_oof_logits = np.zeros_like(lin_oof_probs)
for m_idx in range(len(m_names)):
    p_cl = np.clip(lin_oof_probs[:, m_idx], 1e-6, 1.0 - 1e-6)
    lin_oof_logits[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

nnls_meta = LinearRegression(positive=True, fit_intercept=True)
nnls_meta.fit(lin_oof_logits, y)

w_raw = nnls_meta.coef_
w_norm = w_raw / np.sum(w_raw) if np.sum(w_raw) > 0 else np.ones(len(m_names)) / len(m_names)

lin_stacked_oof_logits = np.dot(lin_oof_logits, w_norm)
lin_total_oof_auc = roc_auc_score(y, 1.0 / (1.0 + np.exp(-lin_stacked_oof_logits)))
print(f"🏆 Engine 2 (0.66054 NNLS Super-Learner) OOF ROC-AUC: {lin_total_oof_auc:.5f}")


# ============================================================
# 7. DUAL-ENGINE PROBABILITY INTEGRATION (50% GBDT + 50% NNLS)
# ============================================================

print("\n============================================")
print("FUSING GBDT TREE ENGINE WITH NNLS SUPER-LEARNER")
print("============================================")

# Engine 1 Test Logits
tree_final_test_probs = tree_test_fold_preds.mean(axis=1)
p_tree_cl = np.clip(tree_final_test_probs, 1e-6, 1.0 - 1e-6)
z_tree = np.log(p_tree_cl / (1.0 - p_tree_cl))

# Engine 2 Test Logits
avg_lin_test_probs = lin_test_fold_preds.mean(axis=2)
z_lin_models = np.zeros_like(avg_lin_test_probs)
for m_idx in range(len(m_names)):
    p_cl = np.clip(avg_lin_test_probs[:, m_idx], 1e-6, 1.0 - 1e-6)
    z_lin_models[:, m_idx] = np.log(p_cl / (1.0 - p_cl))
z_linear = np.dot(z_lin_models, w_norm)

# 50% Tree Engine + 50% Linear/Neural Super-Learner
final_integrated_logits = 0.50 * z_tree + 0.50 * z_linear
final_probabilities = 1.0 / (1.0 + np.exp(-final_integrated_logits))


# ============================================================
# 8. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp73_panel_gbdt_superlearner.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 9. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 73 COMPLETE")
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
print(f"Engine 1 (Native GBDT Trees) OOF AUC : {tree_total_oof_auc:.5f}")
print(f"Engine 2 (0.66054 NNLS Stack) OOF AUC: {lin_total_oof_auc:.5f}")
print("Exp 68 Baseline Benchmark            : 0.66054")
print("Exp 73 Full-Panel Hybrid Submission  : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")