# ============================================================
# EXPERIMENT 62 — MULTI-VIEW LEAK-FREE OOF STACKING
# (RAW, IN-FOLD SEM, IN-FOLD PLS, COMBINED, INTERACTIONS & MLP)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import FactorAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score


print("============================================")
print("EXPERIMENT 62")
print("MULTI-VIEW LEAK-FREE OOF STACKING")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR


# ============================================================
# 2. FEATURE EXTRACTION & DOMAIN PARSERS
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


def prepare_base_features(df):
    """Cleans IDs, dates, matric marks, and numeric lags."""
    data = df.copy()

    if target in data.columns:
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data.dropna(subset=[target]).copy()
        data[target] = data[target].astype(int)

    if "anonymised_id" in data.columns:
        data = data.drop(columns=["anonymised_id"])

    if "survey_date" in data.columns:
        data["survey_date"] = pd.to_datetime(data["survey_date"], errors="coerce")
        data["survey_year"] = data["survey_date"].dt.year
        data["survey_month"] = data["survey_date"].dt.month
        data["survey_dayofyear"] = data["survey_date"].dt.dayofyear
        data = data.drop(columns=["survey_date"])

    data["is_first_time"] = data["employed_lag"].isna().astype(float)
    data["employed_lag_num"] = data["employed_lag"].fillna(-1.0).astype(float)
    data["tenure_lag_log"] = np.log1p(pd.to_numeric(data.get("tenure_lag", 0), errors="coerce").fillna(0).clip(lower=0))
    data["days_since_obs_log"] = np.log1p(pd.to_numeric(data.get("days_since_last_obs", 0), errors="coerce").fillna(0).clip(lower=0))

    # Matric parsing
    for m in ["matric_englishhome", "matric_englishadd", "matric_mathpure", "matric_physicalscience", "matric_mathlit"]:
        if m in data.columns:
            data[f"{m}_score"] = data[m].apply(parse_matric_band).fillna(-1.0)

    data["school_quintile_num"] = pd.to_numeric(data.get("school_quintile", 0), errors="coerce").fillna(0.0)
    data["work_readiness_num"] = pd.to_numeric(data.get("work_readiness_score", 0.5), errors="coerce").fillna(0.5)
    data["age_clean"] = pd.to_numeric(data.get("age", 22), errors="coerce").fillna(22.0).clip(18, 35)

    # Auto-convert numeric strings
    for col in data.columns:
        if col != target and data[col].dtype == "object":
            converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().sum() > 0.6 * data[col].notna().sum():
                data[col] = converted

    return data


# ============================================================
# 3. IN-FOLD MULTI-VIEW TRANSFORMER (ZERO TARGET LEAKAGE)
# ============================================================

class InFoldMultiViewTransformer:
    """
    Fits SEM Factor Analysis and Supervised PLS strictly on the training fold,
    then transforms train, validation, and test subsets to create 6 distinct views.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit_transform_views(self, X_train, y_train, X_val, X_test, num_cols, cat_cols):
        # 1. Base Preprocessors
        preproc_std = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
        ])

        X_tr_raw = preproc_std.fit_transform(X_train)
        X_va_raw = preproc_std.transform(X_val)
        X_te_raw = preproc_std.transform(X_test)

        # ------------------------------------------------------------
        # View 1: Raw Representation
        # ------------------------------------------------------------
        view_raw = (X_tr_raw, X_va_raw, X_te_raw)

        # ------------------------------------------------------------
        # View 2: SEM / Unsupervised Factor Analysis (Fitted in-fold)
        # ------------------------------------------------------------
        acad_cols = [c for c in num_cols if "_score" in c]
        lab_cols = ["employed_lag_num", "tenure_lag_log", "days_since_obs_log", "is_first_time"]
        soc_cols = ["school_quintile_num", "work_readiness_num", "age_clean"]

        imp_num = SimpleImputer(strategy="median")
        scl_num = StandardScaler()
        X_tr_num = scl_num.fit_transform(imp_num.fit_transform(X_train[num_cols]))
        X_va_num = scl_num.transform(imp_num.transform(X_val[num_cols]))
        X_te_num = scl_num.transform(imp_num.transform(X_test[num_cols]))

        fa_acad = FactorAnalysis(n_components=2, random_state=self.random_state)
        fa_lab = FactorAnalysis(n_components=2, random_state=self.random_state)
        fa_soc = FactorAnalysis(n_components=2, random_state=self.random_state)

        # Extract in-fold factor indices
        acad_idx = [num_cols.index(c) for c in acad_cols if c in num_cols]
        lab_idx = [num_cols.index(c) for c in lab_cols if c in num_cols]
        soc_idx = [num_cols.index(c) for c in soc_cols if c in num_cols]

        f_ac_tr = fa_acad.fit_transform(X_tr_num[:, acad_idx])
        f_ac_va = fa_acad.transform(X_va_num[:, acad_idx])
        f_ac_te = fa_acad.transform(X_te_num[:, acad_idx])

        f_lb_tr = fa_lab.fit_transform(X_tr_num[:, lab_idx])
        f_lb_va = fa_lab.transform(X_va_num[:, lab_idx])
        f_lb_te = fa_lab.transform(X_te_num[:, lab_idx])

        f_sc_tr = fa_soc.fit_transform(X_tr_num[:, soc_idx])
        f_sc_va = fa_soc.transform(X_va_num[:, soc_idx])
        f_sc_te = fa_soc.transform(X_te_num[:, soc_idx])

        X_tr_sem = np.hstack([f_ac_tr, f_lb_tr, f_sc_tr])
        X_va_sem = np.hstack([f_ac_va, f_lb_va, f_sc_va])
        X_te_sem = np.hstack([f_ac_te, f_lb_te, f_sc_te])
        view_sem = (X_tr_sem, X_va_sem, X_te_sem)

        # ------------------------------------------------------------
        # View 3: Supervised PLS Latent Projections (Fitted on train target only)
        # ------------------------------------------------------------
        pls = PLSRegression(n_components=3)
        pls.fit(X_tr_num, y_train)

        X_tr_pls = pls.transform(X_tr_num)
        X_va_pls = pls.transform(X_va_num)
        X_te_pls = pls.transform(X_te_num)
        view_pls = (X_tr_pls, X_va_pls, X_te_pls)

        # ------------------------------------------------------------
        # View 4: Combined Raw + SEM Latent Representation
        # ------------------------------------------------------------
        X_tr_comb = np.hstack([X_tr_raw, X_tr_sem, X_tr_pls])
        X_va_comb = np.hstack([X_va_raw, X_va_sem, X_va_pls])
        X_te_comb = np.hstack([X_te_raw, X_te_sem, X_te_pls])
        view_comb = (X_tr_comb, X_va_comb, X_te_comb)

        # ------------------------------------------------------------
        # View 5: High-Leverage Interaction Features
        # ------------------------------------------------------------
        def make_interactions(df_orig):
            d = pd.DataFrame(index=df_orig.index)
            r = pd.to_numeric(df_orig.get("work_readiness_score", 0.5), errors="coerce").fillna(0.5)
            q = pd.to_numeric(df_orig.get("school_quintile", 0), errors="coerce").fillna(0)
            d["inter_readiness_quintile"] = r * (q / 5.0)
            emp = df_orig["employed_lag"].fillna(-1).astype(float)
            ten = np.log1p(pd.to_numeric(df_orig.get("tenure_lag", 0), errors="coerce").fillna(0).clip(lower=0))
            d["inter_emp_tenure"] = emp * ten
            d["inter_stable_job"] = ((emp == 1) & (ten > 5.0)).astype(float)
            return d.to_numpy()

        inter_tr = make_interactions(X_train)
        inter_va = make_interactions(X_val)
        inter_te = make_interactions(X_test)

        X_tr_inter = np.hstack([X_tr_raw, inter_tr])
        X_va_inter = np.hstack([X_va_raw, inter_va])
        X_te_inter = np.hstack([X_te_raw, inter_te])
        view_inter = (X_tr_inter, X_va_inter, X_te_inter)

        # ------------------------------------------------------------
        # View 6: Neural MLP Input (Standard Scaled Raw)
        # ------------------------------------------------------------
        view_mlp = view_raw

        return {
            "view_raw": view_raw,
            "view_sem": view_sem,
            "view_pls": view_pls,
            "view_comb": view_comb,
            "view_inter": view_inter,
            "view_mlp": view_mlp
        }


# ============================================================
# 4. LOAD & PREPARE DATASET
# ============================================================

print("\n============================================")
print("LOADING DATASET")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()

train_clean = prepare_base_features(train_raw)
test_clean = prepare_base_features(test_raw)

common_cols = [c for c in train_clean.columns if c in test_clean.columns and c != target]

y = train_clean[target].reset_index(drop=True)
X = train_clean[common_cols].reset_index(drop=True)
X_test = test_clean[common_cols].reset_index(drop=True)

# Drop high-cardinality categorical (>100)
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
high_card = [c for c in cat_cols if X[c].nunique(dropna=True) > 100]
if high_card:
    X = X.drop(columns=high_card)
    X_test = X_test.drop(columns=high_card)

cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"Training observations: {len(X)} | Testing observations: {len(X_test)}")
print(f"Features: {len(num_cols)} numerical, {len(cat_cols)} categorical")


# ============================================================
# 5. 5-FOLD LEAK-FREE OOF MULTI-VIEW STACKING
# ============================================================

print("\n============================================")
print("RUNNING 5-FOLD LEAK-FREE MULTI-VIEW STACKING")
print("============================================")

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# 6 Views: [Raw, SEM, PLS, Combined, Interaction, MLP]
view_names = ["Raw Linear", "SEM Factor", "PLS Latent", "Raw+SEM Comb", "Interactions", "Neural MLP"]
N_VIEWS = len(view_names)

oof_matrix = np.zeros((len(X), N_VIEWS))
test_fold_preds = np.zeros((len(X_test), N_VIEWS, N_SPLITS))

transformer = InFoldMultiViewTransformer(random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr_f, y_tr_f = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
    X_va_f, y_va_f = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()

    # In-fold transformation (zero target leakage)
    views = transformer.fit_transform_views(X_tr_f, y_tr_f, X_va_f, X_test, num_cols, cat_cols)

    # 1. Model for View 1 (Raw)
    m1 = LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42 + fold)
    m1.fit(views["view_raw"][0], y_tr_f)
    p1_va = m1.predict_proba(views["view_raw"][1])[:, 1]
    p1_te = m1.predict_proba(views["view_raw"][2])[:, 1]

    # 2. Model for View 2 (SEM)
    m2 = LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42 + fold)
    m2.fit(views["view_sem"][0], y_tr_f)
    p2_va = m2.predict_proba(views["view_sem"][1])[:, 1]
    p2_te = m2.predict_proba(views["view_sem"][2])[:, 1]

    # 3. Model for View 3 (PLS)
    m3 = LogisticRegression(C=0.15, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42 + fold)
    m3.fit(views["view_pls"][0], y_tr_f)
    p3_va = m3.predict_proba(views["view_pls"][1])[:, 1]
    p3_te = m3.predict_proba(views["view_pls"][2])[:, 1]

    # 4. Model for View 4 (Combined)
    m4 = LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000, random_state=42 + fold)
    m4.fit(views["view_comb"][0], y_tr_f)
    p4_va = m4.predict_proba(views["view_comb"][1])[:, 1]
    p4_te = m4.predict_proba(views["view_comb"][2])[:, 1]

    # 5. Model for View 5 (Interactions)
    m5 = LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42 + fold)
    m5.fit(views["view_inter"][0], y_tr_f)
    p5_va = m5.predict_proba(views["view_inter"][1])[:, 1]
    p5_te = m5.predict_proba(views["view_inter"][2])[:, 1]

    # 6. Model for View 6 (Neural MLP)
    m6 = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.01,
                       batch_size=128, learning_rate_init=0.001, max_iter=350, early_stopping=True,
                       n_iter_no_change=20, validation_fraction=0.15, random_state=42 + fold)
    m6.fit(views["view_mlp"][0], y_tr_f)
    p6_va = m6.predict_proba(views["view_mlp"][1])[:, 1]
    p6_te = m6.predict_proba(views["view_mlp"][2])[:, 1]

    # Store OOF predictions
    fold_val_preds = [p1_va, p2_va, p3_va, p4_va, p5_va, p6_va]
    fold_test_preds = [p1_te, p2_te, p3_te, p4_te, p5_te, p6_te]

    for v_idx in range(N_VIEWS):
        oof_matrix[val_idx, v_idx] = fold_val_preds[v_idx]
        test_fold_preds[:, v_idx, fold] = fold_test_preds[v_idx]

    fold_scores = [roc_auc_score(y_va_f, fold_val_preds[v]) for v in range(N_VIEWS)]
    print(f"Fold {fold + 1} AUCs: Raw={fold_scores[0]:.4f} | SEM={fold_scores[1]:.4f} | PLS={fold_scores[2]:.4f} | Comb={fold_scores[3]:.4f} | Inter={fold_scores[4]:.4f} | MLP={fold_scores[5]:.4f}")


# ============================================================
# 6. OOF EVALUATION OF INDIVIDUAL VIEWS
# ============================================================

print("\n============================================")
print("INDIVIDUAL VIEW OUT-OF-FOLD (OOF) SCORES")
print("============================================")

for v_idx, name in enumerate(view_names):
    score = roc_auc_score(y, oof_matrix[:, v_idx])
    print(f"View {v_idx + 1} ({name:<16}): OOF ROC-AUC = {score:.5f}")


# ============================================================
# 7. TRAIN META-LEARNER ON UNBIASED OOF FEATURES
# ============================================================

print("\n============================================")
print("TRAINING REGULARIZED META-LEARNER ON OOF STACK")
print("============================================")

# Convert OOF probabilities to log-odds (logit space) for linear meta-learner
oof_logits = np.zeros_like(oof_matrix)
for v in range(N_VIEWS):
    p_cl = np.clip(oof_matrix[:, v], 1e-6, 1.0 - 1e-6)
    oof_logits[:, v] = np.log(p_cl / (1.0 - p_cl))

meta_learner = LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
meta_learner.fit(oof_logits, y)

# Meta-learner evaluation on OOF
oof_stack_probs = meta_learner.predict_proba(oof_logits)[:, 1]
stacked_oof_auc = roc_auc_score(y, oof_stack_probs)

print(f"🏆 Meta-Learner Stacked OOF ROC-AUC : {stacked_oof_auc:.5f}")
print(f"Learned View Weights (Coefficients):\n{pd.Series(meta_learner.coef_[0], index=view_names).to_string()}")


# ============================================================
# 8. TEST PREDICTION INFERENCE
# ============================================================

# Average fold test predictions per view, then convert to logit space
avg_test_preds = test_fold_preds.mean(axis=2)
test_logits = np.zeros_like(avg_test_preds)

for v in range(N_VIEWS):
    p_cl = np.clip(avg_test_preds[:, v], 1e-6, 1.0 - 1e-6)
    test_logits[:, v] = np.log(p_cl / (1.0 - p_cl))

final_probabilities = meta_learner.predict_proba(test_logits)[:, 1]


# ============================================================
# 9. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp62_multi_view_leak_free_stacking.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 10. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 62 COMPLETE")
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
print("Exp 59 Pure SEM Latent Titan    : 0.65832 (Personal Best)")
print(f"Exp 62 Multi-View OOF Stacking  : OOF Val = {stacked_oof_auc:.5f} (Ready for submission)")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")