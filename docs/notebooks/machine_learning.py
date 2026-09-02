# ============================================================
# EXPERIMENT 63 — SUPPORT VECTOR MACHINES (SVM) & LATENT ENSEMBLE
# (CALIBRATED LINEAR SVM, RBF KERNEL SVM, SEM FACTORS & TITAN BLEND)
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
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


print("============================================")
print("EXPERIMENT 63")
print("SUPPORT VECTOR MACHINES (SVM) & LATENT ENSEMBLE")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR

ROUND_DIR = CURRENT_DIR / "round_testing"
if not ROUND_DIR.exists():
    ROUND_DIR = DATA_DIR / "round_testing"


# ============================================================
# 2. FEATURE PARSERS & SEM LATENT FACTORS
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


def extract_features_with_sem(train_df, test_df):
    """Extracts proven date/lag features and computes SEM latent factors."""
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
            if col != target and df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > 0.6 * data[col].notna().sum() if "data" in locals() else True:
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
        tr["latent_academic_1"] = fa_acad.fit_transform(X_ac_tr)[:, 0]
        tr["latent_academic_2"] = fa_acad.fit_transform(X_ac_tr)[:, 1]
        te["latent_academic_1"] = fa_acad.transform(X_ac_te)[:, 0]
        te["latent_academic_2"] = fa_acad.transform(X_ac_te)[:, 1]

    fa_lab = FactorAnalysis(n_components=2, random_state=42)
    X_lb_tr = scl.fit_transform(imp.fit_transform(tr[labour_cols]))
    X_lb_te = scl.transform(imp.transform(te[labour_cols]))
    tr["latent_labour_1"] = fa_lab.fit_transform(X_lb_tr)[:, 0]
    tr["latent_labour_2"] = fa_lab.fit_transform(X_lb_tr)[:, 1]
    te["latent_labour_1"] = fa_lab.transform(X_lb_te)[:, 0]
    te["latent_labour_2"] = fa_lab.transform(X_lb_te)[:, 1]

    # PLS Path
    all_num = acad_cols + labour_cols + socio_cols
    pls = PLSRegression(n_components=2)
    X_pls_tr = imp.fit_transform(tr[all_num])
    X_pls_te = imp.transform(te[all_num])
    pls.fit(X_pls_tr, tr[target])
    tr["pls_latent_1"] = pls.transform(X_pls_tr)[:, 0]
    tr["pls_latent_2"] = pls.transform(X_pls_tr)[:, 1]
    te["pls_latent_1"] = pls.transform(X_pls_te)[:, 0]
    te["pls_latent_2"] = pls.transform(X_pls_te)[:, 1]

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
# 3. SVM & DISCRIMINATIVE CANDIDATE SUITE
# ============================================================

def get_svm_candidate_models(seed=42):
    """Returns a rich suite of Support Vector Machines and Logistic Anchors."""
    
    # 1. Linear SVM with Platt Sigmoid Calibration (C=0.08)
    lsvc_008 = LinearSVC(C=0.08, dual="auto", max_iter=2000, random_state=seed)
    svm_linear_008 = CalibratedClassifierCV(estimator=lsvc_008, method="sigmoid", cv=3)

    # 2. Linear SVM with Platt Sigmoid Calibration (C=0.15)
    lsvc_015 = LinearSVC(C=0.15, dual="auto", max_iter=2000, random_state=seed)
    svm_linear_015 = CalibratedClassifierCV(estimator=lsvc_015, method="sigmoid", cv=3)

    # 3. Non-Linear RBF Kernel SVM (C=1.0)
    svm_rbf_10 = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)

    # 4. Non-Linear RBF Kernel SVM (C=0.5, Higher Regularization)
    svm_rbf_05 = SVC(C=0.5, kernel="rbf", gamma="scale", probability=True, random_state=seed)

    # 5. Proven Logistic Regression Champions
    logreg_008 = LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed)
    logreg_010 = LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed)
    logreg_elastic = LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000, random_state=seed)

    # 6. Neural MLP
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.01,
                        batch_size=128, learning_rate_init=0.001, max_iter=350, early_stopping=True,
                        n_iter_no_change=20, validation_fraction=0.15, random_state=seed)

    return {
        "Calibrated_Linear_SVM_C008": (svm_linear_008, False),
        "Calibrated_Linear_SVM_C015": (svm_linear_015, False),
        "RBF_Kernel_SVM_C10": (svm_rbf_10, False),
        "RBF_Kernel_SVM_C05": (svm_rbf_05, False),
        "LogReg_L2_C008": (logreg_008, False),
        "LogReg_L2_C010": (logreg_010, False),
        "LogReg_ElasticNet_C010": (logreg_elastic, False),
        "MLP_Medium_64_32": (mlp, False)
    }


# ============================================================
# 4. LOAD AND VALIDATE ON SEQUENTIAL ROUND DATASETS
# ============================================================

print("\n============================================")
print("DISCOVERING & VALIDATING ON SEQUENTIAL ROUNDS")
print("============================================")

round_data_list = []

if ROUND_DIR.exists():
    round_folders = sorted([f for f in ROUND_DIR.glob("round_*") if f.is_dir()])
    for r_dir in round_folders:
        csv_files = list(r_dir.glob("*.csv"))
        if len(csv_files) >= 2:
            csv_files = sorted(csv_files, key=lambda f: f.stat().st_size)
            test_file, train_file = csv_files[0], csv_files[1]

            r_train_raw = pd.read_csv(train_file)
            r_test_raw = pd.read_csv(test_file)

            if target in r_train_raw.columns and target in r_test_raw.columns:
                r_train_clean, r_test_clean = extract_features_with_sem(r_train_raw, r_test_raw)

                common_features = [c for c in r_train_clean.columns if c in r_test_clean.columns and c != target]
                y_tr_vals = r_train_clean[target]
                y_te_vals = r_test_clean[target]

                if len(common_features) >= 15 and y_tr_vals.nunique() >= 2 and y_te_vals.nunique() >= 2:
                    round_data_list.append({
                        "name": r_dir.name,
                        "X_train": r_train_clean[common_features].reset_index(drop=True),
                        "y_train": y_tr_vals.reset_index(drop=True),
                        "X_test": r_test_clean[common_features].reset_index(drop=True),
                        "y_test": y_te_vals.reset_index(drop=True)
                    })
                    print(f" -> Loaded {r_dir.name.upper()} | Train: {len(r_train_clean)} | Test: {len(r_test_clean)} | Features: {len(common_features)}")


# ============================================================
# 5. RUN SVM TOURNAMENT ON ROUNDS 6, 7, 8
# ============================================================

print("\n============================================")
print("BENCHMARKING SUPPORT VECTOR MACHINES ACROSS ROUNDS")
print("============================================")

candidate_dict = get_svm_candidate_models(seed=42)
tournament_scores = {name: [] for name in candidate_dict.keys()}
round_names = []

for r_data in round_data_list:
    r_name = r_data["name"]
    round_names.append(r_name.upper())
    X_tr = r_data["X_train"].copy()
    y_tr = r_data["y_train"].copy()
    X_te = r_data["X_test"].copy()
    y_te = r_data["y_test"].copy()

    cat_cols = X_tr.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    high_card = [c for c in cat_cols if X_tr[c].nunique(dropna=True) > 100]
    if high_card:
        X_tr = X_tr.drop(columns=high_card)
        X_te = X_te.drop(columns=high_card)

    cat_cols = X_tr.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()

    models_dict = get_svm_candidate_models(seed=42)

    for model_name, (model_obj, use_quantile) in models_dict.items():
        preprocessor = build_preprocessor(num_cols, cat_cols, use_quantile=use_quantile)
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model_obj)
        ])
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_te)[:, 1]
        score = roc_auc_score(y_te, probs)
        tournament_scores[model_name].append(score)

# Leaderboard Summary
results_table = []
for model_name, scores in tournament_scores.items():
    mean_score = np.mean(scores) if scores else 0.0
    row = {"Model Architecture": model_name, "Mean Round AUC": mean_score}
    for i, s in enumerate(scores):
        row[f"{round_names[i]} AUC"] = s
    results_table.append(row)

leaderboard_df = pd.DataFrame(results_table).sort_values(by="Mean Round AUC", ascending=False).reset_index(drop=True)

print("\n============================================")
print("SVM TOURNAMENT LEADERBOARD")
print("============================================")
print(leaderboard_df.to_string(index=False))

top_performers = leaderboard_df.head(4)["Model Architecture"].tolist()
print(f"\n🏆 TOP CHAMPIONS SELECTED: {top_performers}")


# ============================================================
# 6. MULTI-SEED MAXIMUM-MARGIN TITAN INFERENCE (35 MODELS)
# ============================================================

print("\n============================================")
print("TRAINING MULTI-SEED SVM-TITAN ENSEMBLE ON FULL DATASET")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()

train_clean, test_clean = extract_features_with_sem(train_raw, test_raw)

common_cols = [c for c in train_clean.columns if c in test_clean.columns and c != target]

y_full = train_clean[target].reset_index(drop=True)
X_full = train_clean[common_cols].reset_index(drop=True)
X_test_full = test_clean[common_cols].reset_index(drop=True)

categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
high_cardinality = [c for c in categorical_features if X_full[c].nunique(dropna=True) > 100]
if high_cardinality:
    X_full = X_full.drop(columns=high_cardinality)
    X_test_full = X_test_full.drop(columns=high_cardinality)

categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X_full.select_dtypes(include=[np.number]).columns.tolist()

print(f"Full dataset: {len(X_full)} observations | Features: {len(numerical_features)} numerical (incl. SEM), {len(categorical_features)} categorical")

SEEDS = [42, 101, 777, 2024, 999, 1337, 555]
all_model_logits = []

for m_idx, m_name in enumerate(top_performers):
    print(f" -> Fitting Champion #{m_idx + 1}: {m_name} across {len(SEEDS)} seeds...")

    for s_idx, seed in enumerate(SEEDS):
        candidate_pool = get_svm_candidate_models(seed=seed)
        model_estimator, use_quantile = candidate_pool[m_name]

        preprocessor = build_preprocessor(numerical_features, categorical_features, use_quantile=use_quantile)
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model_estimator)
        ])
        pipe.fit(X_full, y_full)
        probs = pipe.predict_proba(X_test_full)[:, 1]

        # Convert to log-odds (logit space)
        probs_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(probs_clipped / (1.0 - probs_clipped))
        all_model_logits.append(logits)

# Logit-Space Average & Sigmoid Conversion
mean_logits = np.mean(all_model_logits, axis=0)
final_probabilities = 1.0 / (1.0 + np.exp(-mean_logits))


# ============================================================
# 7. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp63_svm_maximum_margin_ensemble.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 63 COMPLETE")
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
print("Exp 59 Pure SEM Latent Titan       : 0.65832 (Personal Best)")
print("Exp 63 SVM Maximum-Margin Ensemble : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")