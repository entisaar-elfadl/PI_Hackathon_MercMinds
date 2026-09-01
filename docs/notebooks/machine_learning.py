# ============================================================
# EXPERIMENT 58 — STACKING PERCEPTRONS & CLASSIFICATION TREES
# (META-LEARNER OVER MULTI-LAYER PERCEPTRON, RANDOM FOREST, EXTRA TREES & HIST GB)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier
)
from sklearn.metrics import roc_auc_score


print("============================================")
print("EXPERIMENT 58")
print("STACKING PERCEPTRONS & CLASSIFICATION TREES")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR


# ============================================================
# 2. PROVEN FEATURE EXTRACTION
# ============================================================

target = "employed_status"

def clean_and_prepare_features(df):
    """Restores the proven high-scoring raw feature space."""
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

    if "employed_lag" in data.columns:
        data["is_first_time"] = data["employed_lag"].isna().astype(float)
    
    if "tenure_lag" in data.columns:
        data["tenure_lag_log"] = np.log1p(data["tenure_lag"].fillna(0).clip(lower=0))

    if "days_since_last_obs" in data.columns:
        data["days_since_last_obs_log"] = np.log1p(data["days_since_last_obs"].fillna(0).clip(lower=0))

    for col in data.columns:
        if col != target and data[col].dtype == "object":
            converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().sum() > 0.6 * data[col].notna().sum():
                data[col] = converted

    return data


def build_preprocessor(numerical_cols, categorical_cols):
    """Builds standard scaler + one-hot preprocessor."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )


# ============================================================
# 3. BASE ESTIMATORS (PERCEPTRONS + CLASSIFICATION TREES)
# ============================================================

def build_stacked_pipeline(numerical_cols, categorical_cols, seed=42):
    """
    Constructs a meta-learning StackingClassifier:
    1. Multi-Layer Perceptron (Neural network representation)
    2. Random Forest (Deep orthogonal tree splits)
    3. Extra Trees (Randomized variance-reducing tree splits)
    4. HistGradientBoosting (Gradient boosted sequential trees)
    5. Logistic Regression (Linear regularized anchor)
    Meta-Learner: Logistic Regression (Learns how to weight them on OOF predictions)
    """
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    # 1. Perceptron (Multi-Layer Neural Net)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=seed
    )

    # 2. Random Forest Classification Trees
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=15,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1
    )

    # 3. Extra Trees (Extremely Randomized Classification Trees)
    et = ExtraTreesClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1
    )

    # 4. HistGradientBoosting (Boosted Decision Trees)
    hgb = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.035,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=2.0,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=seed
    )

    # 5. Regularized Linear Anchor
    lr = LogisticRegression(
        C=0.10,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed
    )

    base_estimators = [
        ("perceptron_mlp", mlp),
        ("random_forest", rf),
        ("extra_trees", et),
        ("gradient_trees", hgb),
        ("logistic_anchor", lr)
    ]

    # The Meta-Learner that trains to figure out which base model to trust
    meta_learner = LogisticRegression(
        C=0.10,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed
    )

    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,                     # 5-fold cross-validation internally for meta-features
        stack_method="predict_proba",
        n_jobs=-1
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("stacker", stacker)
    ])


# ============================================================
# 4. LOAD & PREPARE DATASET
# ============================================================

print("\n============================================")
print("LOADING DATASET & EXTRACTING FEATURES")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()

train_clean = clean_and_prepare_features(train_raw)
test_clean = clean_and_prepare_features(test_raw)

common_cols = [c for c in train_clean.columns if c in test_clean.columns and c != target]

y_full = train_clean[target].reset_index(drop=True)
X_full = train_clean[common_cols].reset_index(drop=True)
X_test_full = test_clean[common_cols].reset_index(drop=True)

# Drop high-cardinality categorical (>100)
categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
high_cardinality = [c for c in categorical_features if X_full[c].nunique(dropna=True) > 100]
if high_cardinality:
    X_full = X_full.drop(columns=high_cardinality)
    X_test_full = X_test_full.drop(columns=high_cardinality)

categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X_full.select_dtypes(include=[np.number]).columns.tolist()

print(f"Full dataset: {len(X_full)} observations")
print(f"Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")


# ============================================================
# 5. TRAIN MULTI-SEED STACKED META-LEARNER
# ============================================================

print("\n============================================")
print("TRAINING STACKED META-LEARNER (5-FOLD CV INTERNAL)")
print("============================================")

SEEDS = [42, 101, 777, 2024, 999]
seed_predictions = []

for s_idx, seed in enumerate(SEEDS):
    print(f"\n[Seed {seed} ({s_idx + 1}/{len(SEEDS)})] Training Perceptrons + Classification Trees Stacker...")
    pipeline = build_stacked_pipeline(numerical_features, categorical_features, seed=seed)
    pipeline.fit(X_full, y_full)

    # Inspect the learned meta-weights from the final estimator
    meta_model = pipeline.named_steps["stacker"].final_estimator_
    if hasattr(meta_model, "coef_"):
        print(f" -> Learned Meta-Model Coefficients:\n    {meta_model.coef_[0]}")

    probs = pipeline.predict_proba(X_test_full)[:, 1]
    seed_predictions.append(probs)

# Average predictions across seeds in Logit (Log-Odds) Space
logits_list = []
for p in seed_predictions:
    p_clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    logits_list.append(np.log(p_clipped / (1.0 - p_clipped)))

mean_logits = np.mean(logits_list, axis=0)
final_probabilities = 1.0 / (1.0 + np.exp(-mean_logits))


# ============================================================
# 6. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp58_stacked_perceptrons_trees.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 7. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 58 COMPLETE")
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
print("Exp 44 Top-3 Linear Blend          : 0.65630")
print("Exp 51 Base Titan Hybrid           : 0.65693 (Personal Best)")
print("Exp 57 Grand Consensus Blend       : 0.65689")
print("Exp 58 Stacked Perceptrons + Trees : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")