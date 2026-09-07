"""
================================================================================
PI HACKATHON — MERCMINDS WINNING SUBMISSION PIPELINE
================================================================================
Model Architecture : Structural Equation Modeling (SEM) Latent Manifold
                     + Non-Negative Least Squares (NNLS) Super-Learner
Public Benchmark   : 0.66054 ROC-AUC (Leaderboard Peak)
Environment Target : Python >= 3.10 | scikit-learn >= 1.3 | pandas >= 2.0
Reproducibility    : Deterministic 5-Fold Stratified Cross-Validation (Seed 42)
Output Artifact    : final_winning_submission.csv
================================================================================
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import FactorAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# 1. CONFIGURATION & ENVIRONMENT SETUP
# ==============================================================================

class PipelineConfig:
    """Global execution parameters and reproducibility seeds."""
    RANDOM_STATE: int = 42
    N_SPLITS: int = 5
    TARGET_COL: str = "employed_status"
    ID_COL: str = "anonymised_id"
    GLOBAL_PRIOR: float = 0.31694
    MAX_CATEGORICAL_CARDINALITY: int = 100
    OUTPUT_FILE: str = "final_winning_submission.csv"


def resolve_dataset_directory() -> Path:
    """Robustly discovers the dataset directory across execution environments."""
    current_path = Path.cwd()
    candidate_paths = [
        (current_path / "../assets/dataset").resolve(),
        current_path / "dataset",
        current_path.parent / "dataset",
        current_path,
    ]
    for path in candidate_paths:
        if (path / "train.csv").exists() and (path / "test.csv").exists():
            return path
    raise FileNotFoundError(
        "Could not locate train.csv and test.csv in default directory structures."
    )


# ==============================================================================
# 2. DOMAIN FEATURE ENGINEERING & ORDINAL INTERVAL PARSERS
# ==============================================================================

def parse_matric_percentage_band(band_value: object) -> float:
    """
    Parses South African National Senior Certificate banded percentage strings
    (e.g., '50 - 59 %', '80 - 100 %', '< 30 %') into continuous interval midpoints.
    """
    if pd.isna(band_value):
        return np.nan

    cleaned_str = str(band_value).replace("%", "").strip()

    if "-" in cleaned_str:
        tokens = cleaned_str.split("-")
        try:
            return (float(tokens[0]) + float(tokens[1])) / 2.0
        except Exception:
            return np.nan
    elif "<" in cleaned_str:
        try:
            return float(cleaned_str.replace("<", "").strip()) / 2.0
        except Exception:
            return np.nan
    elif ">" in cleaned_str:
        try:
            return float(cleaned_str.replace(">", "").strip()) + 5.0
        except Exception:
            return np.nan
    else:
        try:
            return float(cleaned_str)
        except Exception:
            return np.nan


def extract_sem_latent_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs the Structural Equation Modeling (SEM) Latent Subspace:
    1. Extracts domain lag semantics and continuous matric score bands.
    2. Builds orthogonal Factor Analysis blocks for Academic Capital,
       Labour Market Momentum, and Socio-Economic Readiness.
    3. Projects supervised covariance via 2-component Partial Least Squares (PLS).
    """
    tr = train_df.copy()
    te = test_df.copy()

    # Target Cleaning (Training partition only)
    if PipelineConfig.TARGET_COL in tr.columns:
        tr[PipelineConfig.TARGET_COL] = pd.to_numeric(
            tr[PipelineConfig.TARGET_COL], errors="coerce"
        )
        tr = tr.dropna(subset=[PipelineConfig.TARGET_COL]).copy()
        tr[PipelineConfig.TARGET_COL] = tr[PipelineConfig.TARGET_COL].astype(int)

    # Base Transformations across Train & Test partitions
    for df in [tr, te]:
        if PipelineConfig.ID_COL in df.columns:
            df.drop(columns=[PipelineConfig.ID_COL], inplace=True)

        # Date Component Extraction
        if "survey_date" in df.columns:
            df["survey_date"] = pd.to_datetime(df["survey_date"], errors="coerce")
            df["survey_year"] = df["survey_date"].dt.year
            df["survey_month"] = df["survey_date"].dt.month
            df["survey_dayofyear"] = df["survey_date"].dt.dayofyear
            df.drop(columns=["survey_date"], inplace=True)

        # Labour Market Transition Signals
        df["is_first_time"] = df["employed_lag"].isna().astype(float)
        df["employed_lag_num"] = df["employed_lag"].fillna(-1.0).astype(float)
        
        raw_tenure = pd.to_numeric(df.get("tenure_lag", 0), errors="coerce").fillna(0).clip(lower=0)
        df["tenure_lag_log"] = np.log1p(raw_tenure)
        
        raw_days = pd.to_numeric(df.get("days_since_last_obs", 0), errors="coerce").fillna(0).clip(lower=0)
        df["days_since_obs_log"] = np.log1p(raw_days)

        # Continuous Matric Marks
        matric_columns = [
            "matric_englishhome", "matric_englishadd",
            "matric_mathpure", "matric_physicalscience", "matric_mathlit"
        ]
        for m_col in matric_columns:
            if m_col in df.columns:
                df[f"{m_col}_score"] = df[m_col].apply(parse_matric_percentage_band).fillna(-1.0)

        # Socio-Economic & Readiness Standardization
        df["school_quintile_num"] = pd.to_numeric(df.get("school_quintile", 0), errors="coerce").fillna(0.0)
        df["work_readiness_num"] = pd.to_numeric(df.get("work_readiness_score", 0.5), errors="coerce").fillna(0.5)
        df["age_clean"] = pd.to_numeric(df.get("age", 22), errors="coerce").fillna(22.0).clip(18, 35)

        # Data Type Sanitization (Converts boolean flags and numeric strings)
        for col in df.columns:
            if df[col].dtype == "bool":
                df[col] = df[col].astype(float)
            elif col != PipelineConfig.TARGET_COL and df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if df[col].notna().sum() > 0 and converted.notna().sum() > 0.6 * df[col].notna().sum():
                    df[col] = converted

    # --------------------------------------------------------------------------
    # SEM LATENT MEASUREMENT BLOCKS (ORTHOGONAL FACTOR DECOMPOSITION)
    # --------------------------------------------------------------------------
    academic_cols = [c for c in tr.columns if "_score" in c]
    labour_cols = ["employed_lag_num", "tenure_lag_log", "days_since_obs_log", "is_first_time"]
    socio_cols = ["school_quintile_num", "work_readiness_num", "age_clean"]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    # Block A: Academic Human Capital Latent Factors
    if len(academic_cols) > 0:
        fa_acad = FactorAnalysis(n_components=2, random_state=PipelineConfig.RANDOM_STATE)
        X_acad_tr = scaler.fit_transform(imputer.fit_transform(tr[academic_cols]))
        X_acad_te = scaler.transform(imputer.transform(te[academic_cols]))
        tr["latent_academic_factor_1"] = fa_acad.fit_transform(X_acad_tr)[:, 0]
        tr["latent_academic_factor_2"] = fa_acad.fit_transform(X_acad_tr)[:, 1]
        te["latent_academic_factor_1"] = fa_acad.transform(X_acad_te)[:, 0]
        te["latent_academic_factor_2"] = fa_acad.transform(X_acad_te)[:, 1]

    # Block B: Labour Market Momentum Latent Factors
    fa_labour = FactorAnalysis(n_components=2, random_state=PipelineConfig.RANDOM_STATE)
    X_lab_tr = scaler.fit_transform(imputer.fit_transform(tr[labour_cols]))
    X_lab_te = scaler.transform(imputer.transform(te[labour_cols]))
    tr["latent_labour_momentum_1"] = fa_labour.fit_transform(X_lab_tr)[:, 0]
    tr["latent_labour_momentum_2"] = fa_labour.fit_transform(X_lab_tr)[:, 1]
    te["latent_labour_momentum_1"] = fa_labour.transform(X_lab_te)[:, 0]
    te["latent_labour_momentum_2"] = fa_labour.transform(X_lab_te)[:, 1]

    # Block C: Socio-Economic Capital Latent Factors
    fa_socio = FactorAnalysis(n_components=2, random_state=PipelineConfig.RANDOM_STATE)
    X_soc_tr = scaler.fit_transform(imputer.fit_transform(tr[socio_cols]))
    X_soc_te = scaler.transform(imputer.transform(te[socio_cols]))
    tr["latent_socio_readiness_1"] = fa_socio.fit_transform(X_soc_tr)[:, 0]
    tr["latent_socio_readiness_2"] = fa_socio.fit_transform(X_soc_tr)[:, 1]
    te["latent_socio_readiness_1"] = fa_socio.transform(X_soc_te)[:, 0]
    te["latent_socio_readiness_2"] = fa_socio.transform(X_soc_te)[:, 1]

    # Block D: Supervised Partial Least Squares (PLS) Covariance Projections
    all_numeric_block = academic_cols + labour_cols + socio_cols
    pls = PLSRegression(n_components=2)
    X_pls_tr = imputer.fit_transform(tr[all_numeric_block])
    X_pls_te = imputer.transform(te[all_num_block if "all_num_block" in locals() else all_numeric_block])
    pls.fit(X_pls_tr, tr[PipelineConfig.TARGET_COL])
    
    tr["pls_structural_latent_1"] = pls.transform(X_pls_tr)[:, 0]
    tr["pls_structural_latent_2"] = pls.transform(X_pls_tr)[:, 1]
    te["pls_structural_latent_1"] = pls.transform(X_pls_te)[:, 0]
    te["pls_structural_latent_2"] = pls.transform(X_pls_te)[:, 1]

    return tr, te


# ==============================================================================
# 3. PIPELINE PREPROCESSORS & MODEL FACTORIES
# ==============================================================================

def create_column_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
    use_quantile_scaling: bool = False
) -> ColumnTransformer:
    """Builds standard or Gaussian-Quantile transformed preprocessors."""
    transformers = []

    if len(numerical_features) > 0:
        scaler = (
            QuantileTransformer(output_distribution="normal", random_state=PipelineConfig.RANDOM_STATE)
            if use_quantile_scaling else StandardScaler()
        )
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler)
        ])
        transformers.append(("num", numeric_pipeline, numerical_features))

    if len(categorical_features) > 0:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)


def get_champion_base_models(seed: int) -> Dict[str, Tuple[object, bool]]:
    """
    Returns the exact 9-model multi-paradigm candidate suite:
    - L2 Regularized Logistic Regressors (C grid)
    - ElasticNet SAGA Regressors (L1/L2 shrinkage)
    - Quantile-Gaussian Projected Models
    - Deep and Medium Multi-Layer Perceptrons (MLPs)
    """
    return {
        "LogReg_L2_C006": (
            LogisticRegression(C=0.06, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed),
            False
        ),
        "LogReg_L2_C008": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed),
            False
        ),
        "LogReg_L2_C010": (
            LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed),
            False
        ),
        "LogReg_ElasticNet_C008": (
            LogisticRegression(C=0.08, penalty="elasticnet", solver="saga", l1_ratio=0.10, max_iter=2500, tol=1e-4, random_state=seed),
            False
        ),
        "LogReg_ElasticNet_C010": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=seed),
            False
        ),
        "Quantile_LogReg_L2_C008": (
            LogisticRegression(C=0.08, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed),
            True
        ),
        "Quantile_LogReg_ElasticNet": (
            LogisticRegression(C=0.10, penalty="elasticnet", solver="saga", l1_ratio=0.15, max_iter=2500, tol=1e-4, random_state=seed),
            True
        ),
        "MLP_Medium_64_32": (
            MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.010,
                batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                n_iter_no_change=25, validation_fraction=0.15, random_state=seed
            ),
            False
        ),
        "MLP_Deep_128_64": (
            MLPClassifier(
                hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=0.015,
                batch_size=128, learning_rate_init=0.001, max_iter=400, early_stopping=True,
                n_iter_no_change=25, validation_fraction=0.15, random_state=seed + 100
            ),
            False
        )
    }


# ==============================================================================
# 4. MAIN PIPELINE EXECUTION & CROSS-VALIDATION
# ==============================================================================

def main():
    print("=" * 80)
    print("STARTING PI HACKATHON REPRODUCIBLE WINNING PIPELINE")
    print("=" * 80)

    # 1. Load Data
    data_dir = resolve_dataset_directory()
    print(f"[1/6] Ingesting datasets from: {data_dir}")
    train_raw = pd.read_csv(data_dir / "train.csv")
    test_raw = pd.read_csv(data_dir / "test.csv")
    test_ids = test_raw[PipelineConfig.ID_COL].copy()

    # 2. Extract SEM Latent Representation
    print("[2/6] Extracting Structural Equation Modeling (SEM) Latent Factor Space...")
    train_sem, test_sem = extract_sem_latent_features(train_raw, test_raw)

    common_columns = [
        c for c in train_sem.columns
        if c in test_sem.columns and c != PipelineConfig.TARGET_COL
    ]

    y_train = train_sem[PipelineConfig.TARGET_COL].reset_index(drop=True)
    X_train = train_sem[common_columns].reset_index(drop=True)
    X_test = test_sem[common_columns].reset_index(drop=True)

    # 3. Categorical Filtering (Prunes sparse high-cardinality noise > 100)
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    high_cardinality = [c for c in categorical_cols if X_train[c].nunique(dropna=True) > PipelineConfig.MAX_CATEGORICAL_CARDINALITY]
    if high_cardinality:
        X_train.drop(columns=high_cardinality, inplace=True)
        X_test.drop(columns=high_cardinality, inplace=True)

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    print(f"      -> Processed Observations : {len(X_train)} Train | {len(X_test)} Test")
    print(f"      -> Feature Space Breakdown: {len(numerical_cols)} Numerical (incl. 8 SEM factors), {len(categorical_cols)} Categorical")

    # 4. Stratified 5-Fold Cross-Validation
    print(f"\n[3/6] Generating Out-Of-Fold (OOF) Predictions via {PipelineConfig.N_SPLITS}-Fold Stratified CV...")
    skf = StratifiedKFold(
        n_splits=PipelineConfig.N_SPLITS,
        shuffle=True,
        random_state=PipelineConfig.RANDOM_STATE
    )

    base_models_zoo = get_champion_base_models(seed=PipelineConfig.RANDOM_STATE)
    model_names = list(base_models_zoo.keys())
    n_models = len(model_names)

    oof_probabilities = np.zeros((len(X_train), n_models))
    test_fold_predictions = np.zeros((len(X_test), n_models, PipelineConfig.N_SPLITS))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr_fold, y_tr_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va_fold, y_va_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]

        models_pool = get_champion_base_models(seed=PipelineConfig.RANDOM_STATE + fold * 10)

        for m_idx, (m_name, (model_obj, use_quantile)) in enumerate(models_pool.items()):
            preprocessor = create_column_preprocessor(
                numerical_cols, categorical_cols, use_quantile_scaling=use_quantile
            )
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model_obj)
            ])
            pipeline.fit(X_tr_fold, y_tr_fold)

            oof_probabilities[val_idx, m_idx] = pipeline.predict_proba(X_va_fold)[:, 1]
            test_fold_predictions[:, m_idx, fold] = pipeline.predict_proba(X_test)[:, 1]

        print(f"      -> Fold {fold + 1}/{PipelineConfig.N_SPLITS} Completed.")

    # 5. Model Validation & OOF Scoring
    print("\n[4/6] Evaluating Base Estimators on Out-Of-Fold Cross-Validation:")
    print("-" * 65)
    for m_idx, m_name in enumerate(model_names):
        individual_auc = roc_auc_score(y_train, oof_probabilities[:, m_idx])
        print(f"      Model {m_idx + 1:02d} | {m_name:<30} | OOF ROC-AUC: {individual_auc:.5f}")
    print("-" * 65)

    # 6. Non-Negative Least Squares (NNLS) Meta-Optimization
    print("\n[5/6] Solving Non-Negative Least Squares (NNLS) Convex Meta-Weights...")
    oof_logits = np.zeros_like(oof_probabilities)
    for m_idx in range(n_models):
        p_clipped = np.clip(oof_probabilities[:, m_idx], 1e-6, 1.0 - 1e-6)
        oof_logits[:, m_idx] = np.log(p_clipped / (1.0 - p_clipped))

    # Fit Non-Negative Meta-Estimator (w_i >= 0 strictly guaranteed)
    nnls_meta = LinearRegression(positive=True, fit_intercept=True)
    nnls_meta.fit(oof_logits, y_train)

    raw_weights = nnls_meta.coef_
    total_weight = np.sum(raw_weights)
    normalized_weights = raw_weights / total_weight if total_weight > 0 else np.ones(n_models) / n_models

    weight_table = pd.DataFrame({
        "Base Model Architecture": model_names,
        "Raw Coefficient": raw_weights,
        "Optimal Weight (%)": normalized_weights * 100
    }).sort_values(by="Optimal Weight (%)", ascending=False).reset_index(drop=True)

    print("\nLearned Convex Meta-Learner Weight Distribution:")
    print(weight_table.to_string(index=False))

    # Compute Final Stacked Out-of-Fold Score
    stacked_oof_logits = np.dot(oof_logits, normalized_weights)
    stacked_oof_probs = 1.0 / (1.0 + np.exp(-stacked_oof_logits))
    final_oof_auc = roc_auc_score(y_train, stacked_oof_probs)
    final_oof_loss = log_loss(y_train, stacked_oof_probs)

    print("\n" + "=" * 65)
    print(f"🏆 FINAL STACKED OOF ROC-AUC SCORE : {final_oof_auc:.5f}")
    print(f"   FINAL STACKED OOF LOG-LOSS      : {final_oof_loss:.5f}")
    print("=" * 65)

    # 7. Test Set Inference
    print("\n[6/6] Generating Final Test Set Predictions on Round 9 Cohort...")
    avg_test_probabilities = test_fold_predictions.mean(axis=2)
    test_logits = np.zeros_like(avg_test_probabilities)

    for m_idx in range(n_models):
        p_clipped = np.clip(avg_test_probabilities[:, m_idx], 1e-6, 1.0 - 1e-6)
        test_logits[:, m_idx] = np.log(p_clipped / (1.0 - p_clipped))

    final_test_logits = np.dot(test_logits, normalized_weights)
    final_test_probabilities = 1.0 / (1.0 + np.exp(-final_test_logits))

    # 8. Robustness & Integrity Assertions
    assert len(final_test_probabilities) == len(test_raw), "Prediction count mismatch with test dataset."
    assert not np.isnan(final_test_probabilities).any(), "Found NaN in final prediction outputs."
    assert ((final_test_probabilities >= 0.0) & (final_test_probabilities <= 1.0)).all(), "Probabilities fall outside [0, 1]."

    # 9. Save Winning Submission
    submission_df = pd.DataFrame({
        PipelineConfig.ID_COL: test_ids,
        PipelineConfig.TARGET_COL: final_test_probabilities
    })
    
    submission_df.to_csv(PipelineConfig.OUTPUT_FILE, index=False)
    print(f"\n✅ Winning submission artifact successfully exported to: {PipelineConfig.OUTPUT_FILE}")
    print(f"   Total Prediction Rows: {len(submission_df)}")

    print("\n--- Final Prediction Summary Statistics ---")
    print(submission_df[PipelineConfig.TARGET_COL].describe())
    print("\n--- First 10 Predicted Probabilities ---")
    print(submission_df.head(10))
    print("\n================================================================================")
    print("REPRODUCIBILITY PIPELINE COMPLETED SUCCESSFULLY.")
    print("================================================================================")


if __name__ == "__main__":
    main()