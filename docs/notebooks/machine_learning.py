# ============================================================
# EXPERIMENT 92 — SEMI-SUPERVISED PSEUDO-LABELED NNLS SUPER-LEARNER
# (ROUND 9 TRANSDUCTIVE DOMAIN ADAPTATION + SEM LATENT MANIFOLD + NNLS)
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
print("EXPERIMENT 92")
print("SEMI-SUPERVISED PSEUDO-LABELED NNLS SUPER-LEARNER")
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
# 2. FEATURE PARSERS & SEM FACTOR EXTRACTION
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
    """Extracts the exact winning Exp 59/68 SEM Latent Factor representation."""
    tr = train_df.copy()
    te = test_df.copy()

    # Clean target
    if target in tr.columns:
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


def get_base_model_dict(seed=42):
    return {
        "LogReg_L2_C006": (LogisticRegression(C=0.06, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed), False),
        "LogReg_L2_C008": (LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed), False),
        "LogReg_L2_C010": (LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed), False),
        "LogReg_ElasticNet_C008": (LogisticRegression(C=0.08, penalty="elasticnet", solver="saga", l1_ratio=0.10, max_iter=2500, tol=1e-4, random_state=seed), False),
        "LogReg_ElasticNet_C010": (LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=seed), False),
        "Quantile_LogReg_L2_C008": (LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed), True),
        "Quantile_LogReg_ElasticNet": (LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=seed), True),
        "MLP_Medium_64_32": (MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.010,
                                           batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                                           n_iter_no_change=25, validation_fraction=0.15, random_state=seed), False),
        "MLP_Deep_128_64": (MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=0.015,
                                         batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                                         n_iter_no_change=25, validation_fraction=0.15, random_state=seed + 100), False)
    }


# ============================================================
# 3. PHASE 1: FIT TEACHER MODEL (0.66054 NNLS ENGINE)
# ============================================================

print("\n============================================")
print("PHASE 1: TRAINING TEACHER MODEL (EXP 68 NNLS ENGINE)")
print("============================================")

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

print(f"Teacher Train observations: {len(X)} | Features: {len(numerical_features)} num, {len(categorical_features)} cat")

# 5-Fold Stratified CV for Teacher
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

model_names = list(get_base_model_dict(seed=42).keys())
N_MODELS = len(model_names)

oof_probs_teacher = np.zeros((len(X), N_MODELS))
test_fold_preds_teacher = np.zeros((len(X_test), N_MODELS, N_SPLITS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr_f, y_tr_f = X.iloc[train_idx], y.iloc[train_idx]
    X_va_f, y_va_f = X.iloc[val_idx], y.iloc[val_idx]

    models_dict = get_base_model_dict(seed=42 + fold * 10)

    for m_idx, (m_name, (model_obj, use_quantile)) in enumerate(models_dict.items()):
        preprocessor = build_preprocessor(numerical_features, categorical_features, use_quantile=use_quantile)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model_obj)])
        pipe.fit(X_tr_f, y_tr_f)
        
        oof_probs_teacher[val_idx, m_idx] = pipe.predict_proba(X_va_f)[:, 1]
        test_fold_preds_teacher[:, m_idx, fold] = pipe.predict_proba(X_test)[:, 1]

# Teacher NNLS Optimization
oof_logits_teacher = np.zeros_like(oof_probs_teacher)
for m_idx in range(N_MODELS):
    p_cl = np.clip(oof_probs_teacher[:, m_idx], 1e-6, 1.0 - 1e-6)
    oof_logits_teacher[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

nnls_teacher = LinearRegression(positive=True, fit_intercept=True).fit(oof_logits_teacher, y)
norm_weights_teacher = nnls_teacher.coef_ / np.sum(nnls_teacher.coef_) if np.sum(nnls_teacher.coef_) > 0 else np.ones(N_MODELS)/N_MODELS

# Teacher Predictions on Round 9 Test Set
avg_test_probs_teacher = test_fold_preds_teacher.mean(axis=2)
test_logits_teacher = np.zeros_like(avg_test_probs_teacher)
for m_idx in range(N_MODELS):
    p_cl = np.clip(avg_test_probs_teacher[:, m_idx], 1e-6, 1.0 - 1e-6)
    test_logits_teacher[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

teacher_test_logits = np.dot(test_logits_teacher, norm_weights_teacher)
teacher_test_probabilities = 1.0 / (1.0 + np.exp(-teacher_test_logits))


# ============================================================
# 4. PHASE 2: HIGH-CONFIDENCE PSEUDO-LABELING
# ============================================================

print("\n============================================")
print("PHASE 2: EXTRACTING HIGH-CONFIDENCE ROUND 9 PSEUDO-LABELS")
print("============================================")

# Selection criteria for confident Round 9 predictions
POS_THRESHOLD = 0.76
NEG_THRESHOLD = 0.14

pseudo_pos_mask = teacher_test_probabilities >= POS_THRESHOLD
pseudo_neg_mask = teacher_test_probabilities <= NEG_THRESHOLD
pseudo_mask = pseudo_pos_mask | pseudo_neg_mask

n_pos = np.sum(pseudo_pos_mask)
n_neg = np.sum(pseudo_neg_mask)
n_total_pseudo = np.sum(pseudo_mask)

print(f" -> Confident Positive Pseudo-Labels (P >= {POS_THRESHOLD}): {n_pos}")
print(f" -> Confident Negative Pseudo-Labels (P <= {NEG_THRESHOLD}): {n_neg}")
print(f" -> Total Round 9 Pseudo-Labels Added: {n_total_pseudo} ({n_total_pseudo/len(test_raw)*100:.1f}% of Test Set)")

# Construct Pseudo-Labeled Dataset
X_pseudo = X_test[pseudo_mask].copy()
y_pseudo = np.where(pseudo_pos_mask[pseudo_mask], 1, 0)

# Augment Training Dataset
X_augmented = pd.concat([X, X_pseudo], axis=0).reset_index(drop=True)
y_augmented = pd.concat([pd.Series(y), pd.Series(y_pseudo)], axis=0).reset_index(drop=True)

print(f"Augmented Training Size: {len(X_augmented)} rows (16,653 Train + {n_total_pseudo} Round 9 Pseudo-Labels)")


# ============================================================
# 5. PHASE 3: FIT DOMAIN-ADAPTED STUDENT SUPER-LEARNER
# ============================================================

print("\n============================================")
print("PHASE 3: TRAINING DOMAIN-ADAPTED STUDENT SUPER-LEARNER")
print("============================================")

skf_aug = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_probs_student = np.zeros((len(X_augmented), N_MODELS))
test_fold_preds_student = np.zeros((len(X_test), N_MODELS, N_SPLITS))

for fold, (train_idx, val_idx) in enumerate(skf_aug.split(X_augmented, y_augmented)):
    X_tr_f, y_tr_f = X_augmented.iloc[train_idx], y_augmented.iloc[train_idx]
    X_va_f, y_va_f = X_augmented.iloc[val_idx], y_augmented.iloc[val_idx]

    models_dict = get_base_model_dict(seed=42 + fold * 10)

    for m_idx, (m_name, (model_obj, use_quantile)) in enumerate(models_dict.items()):
        preprocessor = build_preprocessor(numerical_features, categorical_features, use_quantile=use_quantile)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model_obj)])
        pipe.fit(X_tr_f, y_tr_f)
        
        oof_probs_student[val_idx, m_idx] = pipe.predict_proba(X_va_f)[:, 1]
        test_fold_preds_student[:, m_idx, fold] = pipe.predict_proba(X_test)[:, 1]

    print(f"Student Fold {fold + 1}/{N_SPLITS} Complete.")

# Student NNLS Optimization
oof_logits_student = np.zeros_like(oof_probs_student)
for m_idx in range(N_MODELS):
    p_cl = np.clip(oof_probs_student[:, m_idx], 1e-6, 1.0 - 1e-6)
    oof_logits_student[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

nnls_student = LinearRegression(positive=True, fit_intercept=True).fit(oof_logits_student, y_augmented)
norm_weights_student = nnls_student.coef_ / np.sum(nnls_student.coef_) if np.sum(nnls_student.coef_) > 0 else np.ones(N_MODELS)/N_MODELS

print("\nLearned Student Meta-Weights:")
weight_summary = pd.DataFrame({
    "Base Model": model_names,
    "Raw Weight": nnls_student.coef_,
    "Normalized %": norm_weights_student * 100
}).sort_values(by="Normalized %", ascending=False).reset_index(drop=True)
print(weight_summary.to_string(index=False))

# Student Test Logits
avg_test_probs_student = test_fold_preds_student.mean(axis=2)
test_logits_student = np.zeros_like(avg_test_probs_student)
for m_idx in range(N_MODELS):
    p_cl = np.clip(avg_test_probs_student[:, m_idx], 1e-6, 1.0 - 1e-6)
    test_logits_student[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

student_test_logits = np.dot(test_logits_student, norm_weights_student)

# ------------------------------------------------------------
# 6. TEACHER-STUDENT LOG-ODDS BLEND (80% Student + 20% Teacher Anchor)
# ------------------------------------------------------------
final_master_logits = 0.80 * student_test_logits + 0.20 * teacher_test_logits
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

output_file = "submission_exp92_pseudolabeled_sem_superlearner.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 92 COMPLETE")
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
print("Exp 91 50-Fold Multi-Seed Ultra     : 0.65969")
print("Exp 68 5-Fold NNLS Super Learner    : 0.66054 (Personal Best)")
print(f"Exp 92 Semi-Supervised Super-Learner: Domain Adapted ({n_total_pseudo} Pseudo-Labels)")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")