# ============================================================
# EXPERIMENT 100 — THE CENTURION GRAND MASTER FINALE
# (THE DEFINITIVE 100TH MILESTONE: 4-PILLAR CORE + SIMPLEX SLSQP + 25-FOLD FUSION)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.decomposition import FactorAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss


print("============================================")
print("EXPERIMENT 100 — THE CENTURION FINALE")
print("THE DEFINITIVE 100TH MILESTONE MASTER ENGINE")
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
    """Extracts the exact winning Exp 59/68/93 SEM Latent Factor representation."""
    tr = train_df.copy()
    te = test_df.copy()

    # Clean target strictly in train
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
# 3. PREPROCESSOR & THE 4 CORE CHAMPION PILLARS
# ============================================================

def build_preprocessor(numerical_cols, categorical_cols, use_quantile=False):
    transformers = []
    if len(numerical_cols) > 0:
        scaler = QuantileTransformer(output_distribution="normal", random_state=42) if use_quantile else StandardScaler()
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", scaler)]), numerical_cols))
    if len(categorical_cols) > 0:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols))
    return ColumnTransformer(transformers=transformers)


def get_champion_pillars(seed=42):
    """The exact 4-pillar model suite that achieved 0.66054."""
    return {
        "Quantile_ElasticNet_C010": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=seed),
            True
        ),
        "Standard_ElasticNet_C010": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=seed),
            False
        ),
        "MLP_Deep_128_64": (
            MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=0.015,
                          batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                          n_iter_no_change=25, validation_fraction=0.15, random_state=seed + 50),
            False
        ),
        "MLP_Medium_64_32": (
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.010,
                          batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                          n_iter_no_change=25, validation_fraction=0.15, random_state=seed),
            False
        )
    }


# ============================================================
# 4. 25-FOLD MULTI-SEED ENGINE & SIMPLEX SLSQP OPTIMIZATION
# ============================================================

print("\n============================================")
print("TRAINING 25-FOLD MULTI-SEED CENTURION ENGINE (5 SEEDS x 5 FOLDS)")
print("============================================")

SEEDS = [42, 101, 777, 2024, 999]
N_SPLITS = 5

pillar_names = list(get_champion_pillars(seed=42).keys())
N_PILLARS = len(pillar_names)

all_seed_test_logits = []
all_seed_oof_scores = []

y_true_arr = y.to_numpy()

# Simplex SLSQP Log-Loss Objective (min cross-entropy s.t. w >= 0, sum(w) = 1)
def simplex_logloss_objective(weights, z_matrix, y_true):
    z_blend = np.dot(z_matrix, weights)
    p_blend = np.clip(1.0 / (1.0 + np.exp(-z_blend)), 1e-7, 1.0 - 1e-7)
    return log_loss(y_true, p_blend)

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
bounds = [(0.0, 1.0) for _ in range(N_PILLARS)]
w_init = np.ones(N_PILLARS) / N_PILLARS

for s_idx, cv_seed in enumerate(SEEDS):
    print(f"\n--- Running Seed {cv_seed:<4} ({s_idx + 1}/{len(SEEDS)}) ---")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=cv_seed)

    oof_probs_seed = np.zeros((len(X), N_PILLARS))
    test_fold_preds_seed = np.zeros((len(X_test), N_PILLARS, N_SPLITS))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr_f, y_tr_f = X.iloc[train_idx], y.iloc[train_idx]
        X_va_f, y_va_f = X.iloc[val_idx], y.iloc[val_idx]

        models_dict = get_champion_pillars(seed=cv_seed + fold * 10)

        for m_idx, (m_name, (m_obj, use_q)) in enumerate(models_dict.items()):
            preproc = build_preprocessor(numerical_features, categorical_features, use_quantile=use_q)
            pipe = Pipeline([("preproc", preproc), ("model", m_obj)])
            pipe.fit(X_tr_f, y_tr_f)

            oof_probs_seed[val_idx, m_idx] = pipe.predict_proba(X_va_f)[:, 1]
            test_fold_preds_seed[:, m_idx, fold] = pipe.predict_proba(X_test)[:, 1]

    # Convert OOF to Logit Space
    oof_logits_seed = np.zeros_like(oof_probs_seed)
    for m_idx in range(N_PILLARS):
        p_cl = np.clip(oof_probs_seed[:, m_idx], 1e-6, 1.0 - 1e-6)
        oof_logits_seed[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

    # Exact Simplex SLSQP Log-Loss Optimization
    opt_res = minimize(
        simplex_logloss_objective,
        w_init,
        args=(oof_logits_seed, y_true_arr),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-9}
    )

    w_opt_seed = opt_res.x / np.sum(opt_res.x)
    seed_oof_logits = np.dot(oof_logits_seed, w_opt_seed)
    seed_oof_auc = roc_auc_score(y_true_arr, 1.0 / (1.0 + np.exp(-seed_oof_logits)))
    all_seed_oof_scores.append(seed_oof_auc)
    print(f"  🏆 Seed {cv_seed} Simplex OOF AUC: {seed_oof_auc:.5f}")

    # Generate test logits for this seed
    avg_test_probs = test_fold_preds_seed.mean(axis=2)
    test_logits_seed = np.zeros_like(avg_test_probs)
    for m_idx in range(N_PILLARS):
        p_cl = np.clip(avg_test_probs[:, m_idx], 1e-6, 1.0 - 1e-6)
        test_logits_seed[:, m_idx] = np.log(p_cl / (1.0 - p_cl))

    seed_test_logits = np.dot(test_logits_seed, w_opt_seed)
    all_seed_test_logits.append(seed_test_logits)

# Centurion Multi-Seed Test Logits
fresh_centurion_logits = np.mean(all_seed_test_logits, axis=0)
mean_25fold_oof_auc = np.mean(all_seed_oof_scores)
print(f"\nMean 25-Fold Simplex OOF ROC-AUC: {mean_25fold_oof_auc:.5f}")


# ============================================================
# 5. MULTI-CHAMPION CONSENSUS INTEGRATION
# ============================================================

print("\n============================================")
print("FUSING 25-FOLD ENGINE WITH VERIFIED 0.66054 CHAMPIONS")
print("============================================")

champion_files = {
    "submission_exp93_simplex_convex_optimization.csv": 0.40,  # 0.66054 SLSQP Peak
    "submission_exp68_nnls_sem_superlearner.csv": 0.40,        # 0.66054 NNLS Peak
    "submission_exp71_grand_master_calibrated_superlearner.csv": 0.20  # 0.66051
}

discovered_logits = []
discovered_weights = []

for filename, weight in champion_files.items():
    f_path = CURRENT_DIR / filename
    if f_path.exists():
        sub_df = pd.read_csv(f_path)
        if target in sub_df.columns and len(sub_df) == len(test_raw):
            p = np.clip(sub_df[target].to_numpy(), 1e-6, 1.0 - 1e-6)
            z = np.log(p / (1.0 - p))
            discovered_logits.append(z)
            discovered_weights.append(weight)
            print(f" -> [LOADED] {filename:<60} (Weight: {weight*100:.0f}%)")

if len(discovered_logits) > 0:
    total_w = sum(discovered_weights)
    norm_w = [w / total_w for w in discovered_weights]
    past_champions_logits = sum(w * z for w, z in zip(norm_w, discovered_logits))

    # 50% Fresh 25-Fold Engine + 50% Verified 0.66054 Consensus
    final_master_logits = 0.50 * fresh_centurion_logits + 0.50 * past_champions_logits
    print("✅ Successfully fused fresh 25-Fold engine with verified 0.66054 champions.")
else:
    final_master_logits = fresh_centurion_logits
    print("ℹ️ Using fresh 25-Fold Centurion engine predictions directly.")

final_probabilities = 1.0 / (1.0 + np.exp(-final_master_logits))
final_probabilities = np.clip(final_probabilities, 1e-6, 1.0 - 1e-6)


# ============================================================
# 6. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp100_centurion_grand_master.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 7. SUMMARY & INSPECTION
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
print("THE 100-EXPERIMENT BENCHMARK PROGRESSION")
print("============================================")
print("Exp 30 Baseline Logistic            : 0.59229")
print("Exp 33 Multi-Layer Perceptron       : 0.64453")
print("Exp 44 Top-3 Regularized Linear     : 0.65630")
print("Exp 51 Titan Multi-Seed Hybrid      : 0.65693")
print("Exp 59 SEM Latent Factor Manifold   : 0.65832")
print("Exp 68 / 93 Dual Peak Baseline      : 0.66054 (All-Time Peak)")
print(f"Exp 100 Centurion Grand Master      : 25-Fold Fusion (Ready for submission)")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")