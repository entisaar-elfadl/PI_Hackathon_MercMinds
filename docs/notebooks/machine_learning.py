# ============================================================
# EXPERIMENT 43 — AUTOMATED ROUND TOURNAMENT & MODEL SELECTION
# (FIXED: AUTO-VALIDATES FEATURE OVERLAP & BENCHMARKS ON VALID ROUNDS)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


print("============================================")
print("EXPERIMENT 43")
print("AUTOMATED ROUND TOURNAMENT & MODEL SELECTION")
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
# 2. ROBUST DATA CLEANING & TYPE COERCION
# ============================================================

target = "employed_status"

def clean_data_and_extract_features(df):
    """Cleans IDs, dates, and automatically coerces numerical datatypes."""
    data = df.copy()

    # Clean target if present
    if target in data.columns:
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data.dropna(subset=[target]).copy()
        data[target] = data[target].astype(int)

    # Drop ID
    if "anonymised_id" in data.columns:
        data = data.drop(columns=["anonymised_id"])

    # Date extraction
    if "survey_date" in data.columns:
        data["survey_date"] = pd.to_datetime(data["survey_date"], errors="coerce")
        data["survey_year"] = data["survey_date"].dt.year
        data["survey_month"] = data["survey_date"].dt.month
        data["survey_dayofyear"] = data["survey_date"].dt.dayofyear
        data = data.drop(columns=["survey_date"])

    # Auto-convert numeric columns that were parsed as object
    for col in data.columns:
        if col != target and data[col].dtype == "object":
            converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().sum() > 0.6 * data[col].notna().sum():
                data[col] = converted

    return data


def build_preprocessor(numerical_cols, categorical_cols):
    """Safely builds ColumnTransformer handling empty subsets."""
    transformers = []

    if len(numerical_cols) > 0:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
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
# 3. CANDIDATE MODEL FACTORIES
# ============================================================

def get_candidate_models(seed=42):
    """Returns a dictionary of diverse competitive model candidates."""
    
    # 1. Logistic Regression
    lr = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed)
    
    # 2. Calibrated Ridge Classifier
    ridge_base = RidgeClassifier(alpha=1.0, random_state=seed)
    ridge_cal = CalibratedClassifierCV(estimator=ridge_base, method="sigmoid", cv=3)
    
    # 3. MLP Medium (64, 32)
    mlp_med = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=0.01, batch_size=128, learning_rate_init=0.001,
        max_iter=350, early_stopping=True, n_iter_no_change=20,
        validation_fraction=0.15, random_state=seed
    )
    
    # 4. Deep MLP (128, 64)
    mlp_deep = MLPClassifier(
        hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
        alpha=0.02, batch_size=128, learning_rate_init=0.001,
        max_iter=350, early_stopping=True, n_iter_no_change=20,
        validation_fraction=0.15, random_state=seed + 10
    )
    
    # 5. HistGradientBoosting Trees
    hgb = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.03, max_iter=300,
        max_leaf_nodes=31, min_samples_leaf=25, l2_regularization=2.0,
        early_stopping=True, n_iter_no_change=20, validation_fraction=0.15,
        random_state=seed
    )
    
    # 6. Winning Exp 41 Hybrid (MLP + Logistic)
    hybrid_exp41 = VotingClassifier(
        estimators=[("mlp", mlp_med), ("lr", lr)],
        voting="soft", weights=[3.5, 1.0]
    )
    
    # 7. Tri-Hybrid (MLP + Logistic + Trees)
    tri_hybrid = VotingClassifier(
        estimators=[("mlp", mlp_med), ("lr", lr), ("hgb", hgb)],
        voting="soft", weights=[3.0, 1.0, 1.5]
    )

    # 8. Multi-Linear Neural Blend (MLP + Ridge + Logistic)
    linear_neural_blend = VotingClassifier(
        estimators=[("mlp", mlp_med), ("ridge", ridge_cal), ("lr", lr)],
        voting="soft", weights=[4.0, 1.0, 1.0]
    )

    return {
        "Logistic Regression": lr,
        "Calibrated Ridge": ridge_cal,
        "MLP Medium (64-32)": mlp_med,
        "MLP Deep (128-64)": mlp_deep,
        "HistGradientBoosting": hgb,
        "Exp41 MLP+Logistic Hybrid": hybrid_exp41,
        "Tri-Hybrid (MLP+LR+HGB)": tri_hybrid,
        "Multi-Linear Neural Blend": linear_neural_blend
    }


# ============================================================
# 4. DISCOVER & LOAD VALID ROUND DATASETS
# ============================================================

print("\n============================================")
print("DISCOVERING SEQUENTIAL ROUND DATASETS")
print("============================================")

round_data_list = []

if ROUND_DIR.exists():
    round_folders = sorted([f for f in ROUND_DIR.glob("round_*") if f.is_dir()])
    print(f"Scanning {len(round_folders)} round folders in: {ROUND_DIR.name}/")

    for r_dir in round_folders:
        csv_files = list(r_dir.glob("*.csv"))
        if len(csv_files) >= 2:
            csv_files = sorted(csv_files, key=lambda f: f.stat().st_size)
            test_file = csv_files[0]   # smaller CSV
            train_file = csv_files[1]  # larger CSV

            r_train_raw = pd.read_csv(train_file)
            r_test_raw = pd.read_csv(test_file)

            if target in r_train_raw.columns and target in r_test_raw.columns:
                r_train = clean_data_and_extract_features(r_train_raw)
                r_test = clean_data_and_extract_features(r_test_raw)

                # Keep only common features between train and test
                common_features = [c for c in r_train.columns if c in r_test.columns and c != target]

                if len(common_features) >= 5:
                    round_data_list.append({
                        "name": r_dir.name,
                        "X_train": r_train[common_features],
                        "y_train": r_train[target],
                        "X_test": r_test[common_features],
                        "y_test": r_test[target]
                    })
                    print(f" -> Loaded {r_dir.name.upper()} | Train: {len(r_train)} rows | Test: {len(r_test)} rows | Common Features: {len(common_features)}")
                else:
                    print(f" -> Skipping {r_dir.name.upper()} (Only {len(common_features)} common feature(s) found)")


# ============================================================
# 5. RUN AUTOMATED ROUND TOURNAMENT
# ============================================================

print("\n============================================")
print("STARTING TOURNAMENT ACROSS VALID ROUNDS")
print("============================================")

candidate_names = list(get_candidate_models(seed=42).keys())
tournament_scores = {name: [] for name in candidate_names}
round_names = []

for r_data in round_data_list:
    r_name = r_data["name"]
    round_names.append(r_name.upper())
    X_tr = r_data["X_train"].copy()
    y_tr = r_data["y_train"].copy()
    X_te = r_data["X_test"].copy()
    y_te = r_data["y_test"].copy()

    # Drop high cardinality (>100) on true categorical columns only
    cat_cols = X_tr.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    high_card = [c for c in cat_cols if X_tr[c].nunique(dropna=True) > 100]
    if high_card:
        X_tr = X_tr.drop(columns=high_card)
        X_te = X_te.drop(columns=high_card)

    cat_cols = X_tr.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()

    models_dict = get_candidate_models(seed=42)

    for model_name, model_obj in models_dict.items():
        preprocessor = build_preprocessor(num_cols, cat_cols)
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
print("TOURNAMENT LEADERBOARD")
print("============================================")
print(leaderboard_df.to_string(index=False))

# Automatically select the winning model
best_model_name = leaderboard_df.iloc[0]["Model Architecture"]
best_model_score = leaderboard_df.iloc[0]["Mean Round AUC"]
print(f"\n🏆 CHAMPION MODEL SELECTED: '{best_model_name}' (Mean Round AUC: {best_model_score:.5f})")


# ============================================================
# 6. TRAIN CHAMPION ON FULL DATASET (MULTI-SEED)
# ============================================================

print("\n============================================")
print(f"TRAINING CHAMPION ('{best_model_name}') ON FULL DATASET")
print("============================================")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")

train_df = clean_data_and_extract_features(train_raw)
test_df = clean_data_and_extract_features(test_raw)

# Keep common columns
common_cols = [c for c in train_df.columns if c in test_df.columns and c != target]

y_full = train_df[target].reset_index(drop=True)
X_full = train_df[common_cols].reset_index(drop=True)
X_test_full = test_df[common_cols].reset_index(drop=True)

# Drop high-cardinality categorical (>100)
categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
high_cardinality = [c for c in categorical_features if X_full[c].nunique(dropna=True) > 100]
if high_cardinality:
    X_full = X_full.drop(columns=high_cardinality)
    X_test_full = X_test_full.drop(columns=high_cardinality)

categorical_features = X_full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_features = X_full.select_dtypes(include=[np.number]).columns.tolist()

print(f"Training on {len(X_full)} full dataset rows...")
print(f"Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")

# Multi-seed training across 5 diverse seeds
SEEDS = [42, 101, 777, 2024, 999]
full_test_predictions = np.zeros((len(X_test_full), len(SEEDS)))

for i, seed in enumerate(SEEDS):
    champion_dict = get_candidate_models(seed=seed)
    champion_estimator = champion_dict[best_model_name]

    preprocessor = build_preprocessor(numerical_features, categorical_features)
    champion_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("champion", champion_estimator)
    ])
    champion_pipe.fit(X_full, y_full)
    full_test_predictions[:, i] = champion_pipe.predict_proba(X_test_full)[:, 1]

final_probabilities = full_test_predictions.mean(axis=1)


# ============================================================
# 7. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp43_best_round_selected_model.csv"

submission = pd.DataFrame({
    "anonymised_id": test_raw["anonymised_id"],
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 8. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 43 COMPLETE")
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
print("Exp 35 Hybrid Single Blend : 0.65089")
print("Exp 37 5-Fold Hybrid Blend : 0.65247")
print("Exp 41 Sequential Hybrid   : 0.65325 (Previous Best)")
print(f"Exp 43 Round Tournament    : Selected '{best_model_name}' ({best_model_score:.5f} Round AUC)")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")