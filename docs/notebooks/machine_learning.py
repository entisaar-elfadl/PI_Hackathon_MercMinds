# ============================================================
# EXPERIMENT 65 — THE OLYMPIAN MULTI-CHAMPION CONSENSUS
# (PRECISION LOG-ODDS & RANK BLEND OF ALL >= 0.657 CHAMPIONS)
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
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


print("============================================")
print("EXPERIMENT 65")
print("THE OLYMPIAN MULTI-CHAMPION CONSENSUS")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR


# ============================================================
# 2. LOAD TEST IDs & DATA
# ============================================================

test_raw = pd.read_csv(DATA_DIR / "test.csv")
train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_ids = test_raw["anonymised_id"].copy()
target = "employed_status"

print(f"Test cohort observations: {len(test_raw)}")


# ============================================================
# 3. DISCOVER ALL TOP-TIER SUBMISSION FILES
# ============================================================

print("\n============================================")
print("DISCOVERING TOP-TIER CHAMPION SUBMISSIONS")
print("============================================")

# Priority leaderboard weights based on empirical public scores
champion_registry = {
    "submission_exp59_sem_latent_titan_ensemble.csv": 0.35,  # 0.65832 Peak
    "submission_exp61_dual_manifold_sem_blend.csv": 0.25,    # 0.65787
    "submission_exp64_varimax_sem_super_champion.csv": 0.20, # 0.65768
    "submission_exp60_sem_longitudinal_super_titan.csv": 0.10, # 0.65735
    "submission_exp51_restored_titan_ensemble.csv": 0.10     # 0.65693
}

found_logits = []
found_ranks = []
normalized_weights = []

for filename, weight in champion_registry.items():
    f_path = CURRENT_DIR / filename
    if f_path.exists():
        sub_df = pd.read_csv(f_path)
        if target in sub_df.columns and len(sub_df) == len(test_raw):
            p = np.clip(sub_df[target].to_numpy(), 1e-6, 1.0 - 1e-6)
            
            # Logit representation
            z = np.log(p / (1.0 - p))
            # Percentile Rank representation
            r = rankdata(p) / len(p)
            
            found_logits.append(z)
            found_ranks.append(r)
            normalized_weights.append(weight)
            print(f" -> [FOUND] {filename:<50} | Target Weight: {weight*100:.0f}%")
    else:
        print(f" -> [MISSING] {filename}")


# ============================================================
# 4. STANDALONE ENGINE FALLBACK (IF < 2 FILES FOUND)
# ============================================================

if len(found_logits) < 2:
    print("\nℹ️ Fewer than 2 past files found. Generating fresh Exp 59 Pure SEM Engine on the fly...")

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

    tr = train_raw.copy()
    te = test_raw.copy()
    tr[target] = pd.to_numeric(tr[target], errors="coerce")
    tr = tr.dropna(subset=[target]).copy()
    tr[target] = tr[target].astype(int)

    for df in [tr, te]:
        if "anonymised_id" in df.columns: df.drop(columns=["anonymised_id"], inplace=True)
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
            if m in df.columns: df[f"{m}_score"] = df[m].apply(parse_matric_band).fillna(-1.0)
        df["school_quintile_num"] = pd.to_numeric(df.get("school_quintile", 0), errors="coerce").fillna(0.0)
        df["work_readiness_num"] = pd.to_numeric(df.get("work_readiness_score", 0.5), errors="coerce").fillna(0.5)
        df["age_clean"] = pd.to_numeric(df.get("age", 22), errors="coerce").fillna(22.0).clip(18, 35)
        for col in df.columns:
            if col != target and df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > 0.6 * df[col].notna().sum(): df[col] = converted

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

    common_cols = [c for c in tr.columns if c in te.columns and c != target]
    y_full = tr[target].reset_index(drop=True)
    X_full = tr[common_cols].reset_index(drop=True)
    X_test_full = te[common_cols].reset_index(drop=True)

    cat_cols = X_full.select_dtypes(include=["object", "category"]).columns.tolist()
    high_c = [c for c in cat_cols if X_full[c].nunique(dropna=True) > 100]
    if high_c:
        X_full = X_full.drop(columns=high_c)
        X_test_full = X_test_full.drop(columns=high_c)

    cat_cols = X_full.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()

    models = {
        "LogReg_L2_C008": (LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000), False, "linear"),
        "LogReg_L2_C010": (LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000), False, "linear"),
        "LogReg_ElasticNet_C010": (LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=1000), False, "linear"),
        "Quantile_LogReg_L2_C008": (LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=1000), True, "linear"),
        "MLP_Medium_64_32": (MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.01, max_iter=350, early_stopping=True), False, "neural")
    }

    SEEDS = [42, 101, 777, 2024, 999, 1337, 555]
    lin_l = []
    neu_l = []

    for m_name, (estimator, use_q, fam) in models.items():
        for s in SEEDS:
            scaler = QuantileTransformer(output_distribution="normal", random_state=s) if use_q else StandardScaler()
            preproc = ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", scaler)]), num_cols),
                ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
            ])
            est_clone = LogisticRegression(C=getattr(estimator, "C", 0.1), penalty=getattr(estimator, "penalty", "l2"), solver=getattr(estimator, "solver", "lbfgs"), l1_ratio=getattr(estimator, "l1_ratio", None), max_iter=1000, random_state=s) if fam == "linear" else MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=350, random_state=s)
            pipe = Pipeline([("preproc", preproc), ("model", est_clone)])
            pipe.fit(X_full, y_full)
            p = np.clip(pipe.predict_proba(X_test_full)[:, 1], 1e-6, 1.0 - 1e-6)
            z_val = np.log(p / (1.0 - p))
            if fam == "linear": lin_l.append(z_val)
            else: neu_l.append(z_val)

    mean_z = 0.80 * np.mean(lin_l, axis=0) + 0.20 * np.mean(neu_l, axis=0)
    found_logits.append(mean_z)
    found_ranks.append(rankdata(mean_z) / len(mean_z))
    normalized_weights.append(1.0)


# ============================================================
# 5. OLYMPIAN CONSENSUS FUSION (LOGIT + RANK RE-MAPPING)
# ============================================================

print("\n============================================")
print("COMPUTING OLYMPIAN CONSENSUS INTEGRATION")
print("============================================")

total_w = sum(normalized_weights)
weights_arr = np.array([w / total_w for w in normalized_weights])

# 1. Precision Log-Odds Blend
weighted_logits = sum(w * z for w, z in zip(weights_arr, found_logits))
prob_from_logits = 1.0 / (1.0 + np.exp(-weighted_logits))

# 2. Monotonic Rank Consensus
weighted_ranks = sum(w * r for w, r in zip(weights_arr, found_ranks))

# 3. Optimal Hybrid Fusion (70% Logit Probability + 30% Rank Re-projection)
# Smoothly interpolates calibrated probabilities with rank density
final_probabilities = 0.70 * prob_from_logits + 0.30 * (weighted_ranks * (prob_from_logits.max() - prob_from_logits.min()) + prob_from_logits.min())


# ============================================================
# 6. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp65_olympian_consensus.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 7. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 65 COMPLETE")
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
print("Exp 64 Varimax-SEM Super-Champion   : 0.65768")
print("Exp 61 Dual-Manifold SEM Blend      : 0.65787")
print("Exp 59 Pure SEM Latent Titan        : 0.65832 (Personal Best)")
print("Exp 65 Olympian Consensus Ensemble  : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")