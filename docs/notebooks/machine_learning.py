# ============================================================
# EXPERIMENT 42 — 21-MODEL MULTI-ARCHITECTURE NEURAL COMMITTEE
# (3 DIVERSE MLP ARCHITECTURES + DUAL REGULARIZED LINEAR ANCHORS)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


print("============================================")
print("EXPERIMENT 42")
print("21-MODEL MULTI-ARCHITECTURE NEURAL COMMITTEE")
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
# 2. FEATURE PIPELINE BUILDERS
# ============================================================

target = "employed_status"

def clean_data_and_extract_features(df):
    """Applies clean date extraction and ID handling."""
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

    return data


def create_committee_pipeline(numerical_cols, categorical_cols, seed=42):
    """Builds a multi-architecture committee of 3 MLPs + 2 Logistic Regressors."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # Architecture A: Proven Medium 2-Layer MLP
    mlp_medium = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=350,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=seed
    )

    # Architecture B: Deep 3-Layer MLP
    mlp_deep = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.02,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=350,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=seed + 100
    )

    # Architecture C: Compact Regularized MLP
    mlp_compact = MLPClassifier(
        hidden_layer_sizes=(48, 24),
        activation="relu",
        solver="adam",
        alpha=0.005,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=350,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=seed + 200
    )

    # Linear Anchor 1 (C=0.1)
    lr_1 = LogisticRegression(
        C=0.1,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed
    )

    # Linear Anchor 2 (C=0.05, higher regularization)
    lr_2 = LogisticRegression(
        C=0.05,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed + 50
    )

    committee = VotingClassifier(
        estimators=[
            ("mlp_med", mlp_medium),
            ("mlp_deep", mlp_deep),
            ("mlp_comp", mlp_compact),
            ("lr_1", lr_1),
            ("lr_2", lr_2)
        ],
        voting="soft",
        weights=[3.0, 2.5, 2.0, 1.0, 0.5]
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("committee", committee)
    ])


# ============================================================
# 3. RUN ROUND TESTING (EXPANDING TEMPORAL VALIDATION)
# ============================================================

print("\n============================================")
print("EVALUATING TEMPORAL ROUND PERFORMANCE")
print("============================================")

round_scores = {}

if ROUND_DIR.exists():
    round_folders = sorted([f for f in ROUND_DIR.glob("round_*") if f.is_dir()])
    print(f"Found {len(round_folders)} round folders in: {ROUND_DIR.name}/")

    for r_dir in round_folders:
        round_name = r_dir.name
        csv_files = list(r_dir.glob("*.csv"))

        train_files = [f for f in csv_files if "round_1_" in f.name or "train" in f.name]
        test_files = [f for f in csv_files if f not in train_files]

        if not train_files or not test_files:
            continue

        r_train_raw = pd.read_csv(train_files[0])
        r_test_raw = pd.read_csv(test_files[0])

        if target not in r_train_raw.columns or target not in r_test_raw.columns:
            continue

        r_train = clean_data_and_extract_features(r_train_raw)
        r_test = clean_data_and_extract_features(r_test_raw)

        y_r_train = r_train[target]
        X_r_train = r_train.drop(columns=[target])
        y_r_test = r_test[target]
        X_r_test = r_test.drop(columns=[target])

        # Drop cardinality > 100
        cat_cols = X_r_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        high_card = [c for c in cat_cols if X_r_train[c].nunique(dropna=True) > 100]
        if high_card:
            X_r_train = X_r_train.drop(columns=high_card)
            X_r_test = X_r_test.drop(columns=high_card)

        cat_cols = X_r_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        num_cols = X_r_train.select_dtypes(include=[np.number]).columns.tolist()

        model = create_committee_pipeline(num_cols, cat_cols, seed=42)
        model.fit(X_r_train, y_r_train)

        probs = model.predict_proba(X_r_test)[:, 1]
        score = roc_auc_score(y_r_test, probs)
        acc = accuracy_score(y_r_test, (probs >= 0.5).astype(int))

        round_scores[round_name] = score
        print(f" -> {round_name.upper():<10} | Training Rows: {len(X_r_train):<6} | Test Rows: {len(X_r_test):<5} | ROC-AUC: {score:.5f} | Acc: {acc:.4f}")

    if round_scores:
        avg_round_auc = np.mean(list(round_scores.values()))
        print(f"\nAverage Sequential Round ROC-AUC: {avg_round_auc:.5f}")


# ============================================================
# 4. LOAD FULL COMPETITION DATA
# ============================================================

print("\n============================================")
print("TRAINING 21-MODEL COMMITTEE ON FULL DATASET")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")

train_df = clean_data_and_extract_features(train_raw)
test_df = clean_data_and_extract_features(test_raw)

y_full = train_df[target].reset_index(drop=True)
X_full = train_df.drop(columns=[target]).reset_index(drop=True)
X_test_full = test_df.reset_index(drop=True)

# Drop high-cardinality (>100)
categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
high_cardinality = [c for c in categorical_features if X_full[c].nunique(dropna=True) > 100]
if high_cardinality:
    X_full = X_full.drop(columns=high_cardinality)
    X_test_full = X_test_full.drop(columns=high_cardinality)

categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X_full.select_dtypes(include=[np.number]).columns.tolist()

print(f"Training on {len(X_full)} full dataset rows...")
print(f"Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")


# ============================================================
# 5. MULTI-SEED INFERENCE ACROSS 7 SEEDS (21 DIVERSE MODELS)
# ============================================================

SEEDS = [42, 101, 777, 2024, 999, 1337, 555]
full_test_predictions = np.zeros((len(X_test_full), len(SEEDS)))

print(f"\nTraining committee across {len(SEEDS)} distinct seeds...")
for i, seed in enumerate(SEEDS):
    print(f" -> Fitting Seed {seed} ({i + 1}/{len(SEEDS)})...")
    committee_pipe = create_committee_pipeline(numerical_features, categorical_features, seed=seed)
    committee_pipe.fit(X_full, y_full)
    full_test_predictions[:, i] = committee_pipe.predict_proba(X_test_full)[:, 1]

# Final smooth committee average
final_probabilities = full_test_predictions.mean(axis=1)


# ============================================================
# 6. VALIDATE PREDICTIONS
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")


# ============================================================
# 7. SAVE SUBMISSION FILE
# ============================================================

output_file = "submission_exp42_neural_committee_ensemble.csv"

submission = pd.DataFrame({
    "anonymised_id": test_raw["anonymised_id"],
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 42 COMPLETE")
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
print("Exp 35 Single Hybrid Blend    : 0.65089")
print("Exp 37 5-Fold Hybrid Blend    : 0.65247")
print("Exp 41 Sequential Multi-Seed  : 0.65325 (Current Best)")
if round_scores:
    for r_name, score in round_scores.items():
        print(f"Exp 42 {r_name.upper()} Holdout AUC   : {score:.5f}")
    print(f"Exp 42 Mean Round AUC         : {avg_round_auc:.5f}")
print("Exp 42 21-Model Full Ensemble : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")