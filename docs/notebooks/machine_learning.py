# ============================================================
# EXPERIMENT 96 — WINDOWS-SAFE AUTOGLUON MULTI-LAYER STACKER
# (LIGHTGBM + CATBOOST + XGBOOST + PYTORCH NN + MULTI-LAYER ENSEMBLE)
# ============================================================

import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from autogluon.tabular import TabularDataset, TabularPredictor


print("============================================")
print("EXPERIMENT 96")
print("WINDOWS-SAFE AUTOGLUON MULTI-LAYER STACKER")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS & CLEAN LOCAL MODEL PATH
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR

# Save model locally in system TEMP directory to bypass OneDrive file locks
MODEL_SAVE_PATH = str(Path(tempfile.gettempdir()) / "autogluon_merc_minds_exp96")

if Path(MODEL_SAVE_PATH).exists():
    try:
        shutil.rmtree(MODEL_SAVE_PATH, ignore_errors=True)
    except Exception:
        pass


# ============================================================
# 2. FEATURE PARSING & DOMAIN CLEANING
# ============================================================

target = "employed_status"

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


print("\nLoading datasets and parsing features...")
train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()

train_clean, test_clean = prepare_tabular_data(train_raw, test_raw)


# ============================================================
# 3. FIT WINDOWS-SAFE AUTOGLUON MULTI-LAYER PREDICTOR
# ============================================================

print("\n============================================")
print("TRAINING AUTOGLUON BEST_QUALITY PREDICTOR (700s TIME BUDGET)")
print("============================================")

ag_train = TabularDataset(train_clean)
ag_test = TabularDataset(test_clean)

# Initialize AutoGluon Tabular Predictor targeting ROC-AUC
predictor = TabularPredictor(
    label=target,
    eval_metric="roc_auc",
    problem_type="binary",
    path=MODEL_SAVE_PATH
)

predictor.fit(
    train_data=ag_train,
    presets="best_quality",               # Multi-layer bagging & stacking out-of-the-box
    time_limit=700,                       # 11.5 minutes training budget
    excluded_model_types=["FASTAI"],       # Bypasses Python 3.13 __doc__ incompatibility
    dynamic_stacking=False,               # Bypasses Windows OneDrive file-deletion locks
    num_bag_folds=5,                      # 5-Fold out-of-fold cross-validation bagging
    num_stack_levels=1,                   # Level 1 Base Models -> Level 2 Stacking Ensemble
    verbosity=2
)


# ============================================================
# 4. LEADERBOARD & INFERENCE
# ============================================================

print("\n============================================")
print("AUTOGLUON TRAINED MODELS LEADERBOARD")
print("============================================")
leaderboard = predictor.leaderboard(silent=True)
print(leaderboard[["model", "score_val", "pred_time_val", "fit_time"]].head(12).to_string(index=False))

best_val_score = leaderboard.iloc[0]["score_val"]
best_model_name = leaderboard.iloc[0]["model"]
print(f"\n🏆 Champion Model: '{best_model_name}' (Validation ROC-AUC: {best_val_score:.5f})")

print("\nGenerating test predictions on Round 9...")
# Predict positive class probabilities (class 1)
test_prob_df = predictor.predict_proba(ag_test)
final_probabilities = test_prob_df[1].to_numpy()


# ============================================================
# 5. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

final_probabilities = np.clip(final_probabilities, 1e-6, 1.0 - 1e-6)

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp96_autogluon_best_quality.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 6. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 96 COMPLETE")
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
print(f"AutoGluon Best Validation ROC-AUC   : {best_val_score:.5f}")
print("Target Leaderboard (Excel-lent Minds): 0.66372")
print("Exp 68 Baseline Benchmark            : 0.66054")
print("Exp 96 AutoGluon Best Quality        : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")