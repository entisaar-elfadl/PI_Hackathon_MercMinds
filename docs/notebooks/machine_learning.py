# ============================================================
# EXPERIMENT 41 — SEQUENTIAL ROUND-TESTING & TEMPORAL ENSEMBLE
# (EXPANDING WINDOW BACKTESTING ACROSS ROUNDS 6, 7, 8)
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
print("EXPERIMENT 41")
print("SEQUENTIAL ROUND-TESTING & TEMPORAL ENSEMBLE")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS
# ============================================================

CURRENT_DIR = Path.cwd()

# 1. Main assets dataset directory
DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR

# 2. Sequential round testing directory
ROUND_DIR = CURRENT_DIR / "round_testing"
if not ROUND_DIR.exists():
    ROUND_DIR = DATA_DIR / "round_testing"


# ============================================================
# 2. FEATURE PIPELINE & MODEL FACTORY
# ============================================================

target = "employed_status"

def clean_data_and_extract_features(df):
    """Applies clean date extraction and ID handling."""
    data = df.copy()

    # Clean target if present
    if target in data.columns:
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data.dropna(subset=[target]).copy()
        data[target] = data[target].astype(int)

    # Drop ID
    if "anonymised_id" in data.columns:
        data = data.drop(columns=["anonymised_id"])

    # Date feature engineering
    if "survey_date" in data.columns:
        data["survey_date"] = pd.to_datetime(data["survey_date"], errors="coerce")
        data["survey_year"] = data["survey_date"].dt.year
        data["survey_month"] = data["survey_date"].dt.month
        data["survey_dayofyear"] = data["survey_date"].dt.dayofyear
        data = data.drop(columns=["survey_date"])

    return data


def create_pipeline(numerical_cols, categorical_cols, seed=42):
    """Builds the high-performing Exp 37 Neural-Linear hybrid pipeline."""
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

    mlp = MLPClassifier(
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

    logistic = LogisticRegression(
        C=0.1,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed
    )

    ensemble = VotingClassifier(
        estimators=[
            ("mlp", mlp),
            ("lr", logistic)
        ],
        voting="soft",
        weights=[3.5, 1.0]
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("ensemble", ensemble)
    ])


# ============================================================
# 3. RUN ROUND TESTING (EXPANDING TEMPORAL VALIDATION)
# ============================================================

print("\n============================================")
print("EVALUATING TEMPORAL ROUND PERFORMANCE")
print("============================================")

round_scores = {}

if ROUND_DIR.exists():
    # Detect round folders (e.g. round_6, round_7, round_8)
    round_folders = sorted([f for f in ROUND_DIR.glob("round_*") if f.is_dir()])
    print(f"Found {len(round_folders)} round folders in: {ROUND_DIR.name}/")

    for r_dir in round_folders:
        round_name = r_dir.name
        csv_files = list(r_dir.glob("*.csv"))

        # Identify train (e.g. round_1_n-1.csv) and test (round_n.csv)
        train_files = [f for f in csv_files if "round_1_" in f.name or "train" in f.name]
        test_files = [f for f in csv_files if f not in train_files]

        if not train_files or not test_files:
            continue

        r_train_raw = pd.read_csv(train_files[0])
        r_test_raw = pd.read_csv(test_files[0])

        if target not in r_train_raw.columns or target not in r_test_raw.columns:
            continue

        # Clean datasets
        r_train = clean_data_and_extract_features(r_train_raw)
        r_test = clean_data_and_extract_features(r_test_raw)

        # Separate X and y
        y_r_train = r_train[target]
        X_r_train = r_train.drop(columns=[target])
        y_r_test = r_test[target]
        X_r_test = r_test.drop(columns=[target])

        # Filter high-cardinality (>100)
        cat_cols = X_r_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        high_card = [c for c in cat_cols if X_r_train[c].nunique(dropna=True) > 100]
        if high_card:
            X_r_train = X_r_train.drop(columns=high_card)
            X_r_test = X_r_test.drop(columns=high_card)

        cat_cols = X_r_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        num_cols = X_r_train.select_dtypes(include=[np.number]).columns.tolist()

        # Train on past rounds, test on the unseen upcoming round
        model = create_pipeline(num_cols, cat_cols, seed=42)
        model.fit(X_r_train, y_r_train)

        probs = model.predict_proba(X_r_test)[:, 1]
        score = roc_auc_score(y_r_test, probs)
        acc = accuracy_score(y_r_test, (probs >= 0.5).astype(int))

        round_scores[round_name] = score
        print(f" -> {round_name.upper():<10} | Training Rows: {len(X_r_train):<6} | Test Rows: {len(X_r_test):<5} | ROC-AUC: {score:.5f} | Acc: {acc:.4f}")

    if round_scores:
        avg_round_auc = np.mean(list(round_scores.values()))
        print(f"\nAverage Sequential Round ROC-AUC: {avg_round_auc:.5f}")
else:
    print(f"Directory {ROUND_DIR} not found. Proceeding directly to full training.")


# ============================================================
# 4. LOAD FULL COMPETITION DATA
# ============================================================

print("\n============================================")
print("TRAINING ON FULL DATASET FOR FINAL SUBMISSION")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")

train_df = clean_data_and_extract_features(train_raw)
test_df = clean_data_and_extract_features(test_raw)

y_full = train_df[target].reset_index(drop=True)
X_full = train_df.drop(columns=[target]).reset_index(drop=True)
X_test_full = test_df.reset_index(drop=True)

# Filter high-cardinality (>100)
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
# 5. MULTI-SEED INFERENCE ON FULL DATA
# ============================================================

# Average 5 distinct random seed initializations on full data
seeds = [42, 101, 777, 2024, 999]
full_test_predictions = np.zeros((len(X_test_full), len(seeds)))

for i, seed in enumerate(seeds):
    final_model = create_pipeline(numerical_features, categorical_features, seed=seed)
    final_model.fit(X_full, y_full)
    full_test_predictions[:, i] = final_model.predict_proba(X_test_full)[:, 1]

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

output_file = "submission_exp41_sequential_round_ensemble.csv"

submission = pd.DataFrame({
    "anonymised_id": test_raw["anonymised_id"],
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 41 COMPLETE")
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
print("Exp 35 Single Hybrid Blend : 0.65089")
print("Exp 37 5-Fold Hybrid Blend : 0.65247 (Current Best)")
if round_scores:
    for r_name, score in round_scores.items():
        print(f"Exp 41 {r_name.upper()} Holdout AUC: {score:.5f}")
    print(f"Exp 41 Mean Round AUC      : {avg_round_auc:.5f}")
print("Exp 41 Full Submission     : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")