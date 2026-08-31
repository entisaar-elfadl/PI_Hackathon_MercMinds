# ============================================================
# EXPERIMENT 52 — 68%+ VALIDATION-GATED ROUND OPTIMIZER
# (AUTOMATED CANDIDATE SEARCH & TARGET THRESHOLD EXECUTION)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import itertools

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


print("============================================")
print("EXPERIMENT 52")
print("68%+ VALIDATION-GATED ROUND OPTIMIZER")
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
# 2. PROVEN 0.65693 FEATURE PREPARATION
# ============================================================

target = "employed_status"

def clean_and_prepare_features(df):
    """Restores the high-performing feature set with non-destructive lag signals."""
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
# 3. CANDIDATE SEARCH SPACE DEFINITION
# ============================================================

def generate_candidate_configs():
    """Generates a search grid of regularized linear & hybrid configurations."""
    configs = []

    # 1. Fine L2 Regularization Grid
    for c_val in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
        configs.append({
            "name": f"LogReg_L2_C{c_val:.2f}",
            "type": "pure_linear",
            "model": LogisticRegression(C=c_val, penalty="l2", solver="lbfgs", max_iter=1000),
            "use_quantile": False,
            "linear_weight": 1.0,
            "neural_weight": 0.0
        })

    # 2. ElasticNet SAGA Grid
    for c_val, l1_val in [(0.06, 0.10), (0.08, 0.12), (0.10, 0.15), (0.12, 0.10)]:
        configs.append({
            "name": f"LogReg_ElasticNet_C{c_val:.2f}_L1_{l1_val:.2f}",
            "type": "pure_linear",
            "model": LogisticRegression(C=c_val, penalty="elasticnet", solver="saga", l1_ratio=l1_val, max_iter=1000),
            "use_quantile": False,
            "linear_weight": 1.0,
            "neural_weight": 0.0
        })

    # 3. Quantile-Gaussian Linear Models
    for c_val in [0.06, 0.08, 0.10]:
        configs.append({
            "name": f"Quantile_LogReg_L2_C{c_val:.2f}",
            "type": "pure_linear",
            "model": LogisticRegression(C=c_val, penalty="l2", solver="lbfgs", max_iter=1000),
            "use_quantile": True,
            "linear_weight": 1.0,
            "neural_weight": 0.0
        })

    # 4. Precision 80/20 & 85/15 Linear-Neural Hybrids
    for l_weight, n_weight in [(0.85, 0.15), (0.80, 0.20), (0.75, 0.25)]:
        for c_val in [0.08, 0.10]:
            configs.append({
                "name": f"Titan_Hybrid_C{c_val:.2f}_Ratio_{int(l_weight*100)}_{int(n_weight*100)}",
                "type": "hybrid",
                "linear_model": LogisticRegression(C=c_val, penalty="l2", solver="lbfgs", max_iter=1000),
                "neural_model": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                                              alpha=0.01, batch_size=128, learning_rate_init=0.001, max_iter=350,
                                              early_stopping=True, n_iter_no_change=20, validation_fraction=0.15),
                "use_quantile": False,
                "linear_weight": l_weight,
                "neural_weight": n_weight
            })

    return configs


# ============================================================
# 4. DISCOVER & LOAD SEQUENTIAL ROUND DATASETS
# ============================================================

print("\n============================================")
print("DISCOVERING SEQUENTIAL ROUND DATASETS")
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
                r_train_clean = clean_and_prepare_features(r_train_raw)
                r_test_clean = clean_and_prepare_features(r_test_raw)

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
# 5. 68%+ VALIDATION-GATED ROUND SEARCH
# ============================================================

TARGET_GATE_AUC = 0.680  # Target threshold on Round 8 validation

print("\n============================================")
print(f"SEARCHING CANDIDATES WITH TARGET GATE: {TARGET_GATE_AUC * 100:.1f}%")
print("============================================")

candidate_configs = generate_candidate_configs()
evaluation_results = []

for idx, config in enumerate(candidate_configs):
    r_scores = {}
    
    for r_data in round_data_list:
        r_name = r_data["name"].upper()
        X_tr = r_data["X_train"].copy()
        y_tr = r_data["y_train"].copy()
        X_te = r_data["X_test"].copy()
        y_te = r_data["y_test"].copy()

        # Drop high-cardinality (>100)
        cat_cols = X_tr.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        high_card = [c for c in cat_cols if X_tr[c].nunique(dropna=True) > 100]
        if high_card:
            X_tr = X_tr.drop(columns=high_card)
            X_te = X_te.drop(columns=high_card)

        cat_cols = X_tr.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()

        preprocessor = build_preprocessor(num_cols, cat_cols, use_quantile=config["use_quantile"])

        if config["type"] == "pure_linear":
            pipe = Pipeline([("preprocessor", preprocessor), ("model", config["model"])])
            pipe.fit(X_tr, y_tr)
            probs = pipe.predict_proba(X_te)[:, 1]
        else:
            # Hybrid Linear + Neural
            pipe_lin = Pipeline([("preprocessor", preprocessor), ("model", config["linear_model"])])
            pipe_neu = Pipeline([("preprocessor", preprocessor), ("model", config["neural_model"])])
            pipe_lin.fit(X_tr, y_tr)
            pipe_neu.fit(X_tr, y_tr)

            p_lin = np.clip(pipe_lin.predict_proba(X_te)[:, 1], 1e-6, 1 - 1e-6)
            p_neu = np.clip(pipe_neu.predict_proba(X_te)[:, 1], 1e-6, 1 - 1e-6)

            z_lin = np.log(p_lin / (1 - p_lin))
            z_neu = np.log(p_neu / (1 - p_neu))
            z_blend = config["linear_weight"] * z_lin + config["neural_weight"] * z_neu
            probs = 1.0 / (1.0 + np.exp(-z_blend))

        score = roc_auc_score(y_te, probs)
        r_scores[r_name] = score

    mean_round_score = np.mean(list(r_scores.values()))
    r8_score = r_scores.get("ROUND_8", mean_round_score)

    evaluation_results.append({
        "Config": config,
        "Name": config["name"],
        "Mean Round AUC": mean_round_score,
        "Round 8 AUC": r8_score,
        **{f"{k} AUC": v for k, v in r_scores.items()}
    })

    status_icon = "🔥 [GATE PASSED]" if r8_score >= TARGET_GATE_AUC else "⏳"
    print(f"Candidate #{idx+1:02d}: {config['name']:<48} | Mean: {mean_round_score:.5f} | R8: {r8_score:.5f} {status_icon}")


# ============================================================
# 6. LEADERBOARD & WINNER SELECTION
# ============================================================

leaderboard_df = pd.DataFrame(evaluation_results).sort_values(by="Round 8 AUC", ascending=False).reset_index(drop=True)

display_df = leaderboard_df.drop(columns=["Config"])
print("\n============================================")
print("ROUND OPTIMIZATION LEADERBOARD")
print("============================================")
print(display_df.head(10).to_string(index=False))

winning_entry = leaderboard_df.iloc[0]
champion_config = winning_entry["Config"]
champion_r8_score = winning_entry["Round 8 AUC"]
champion_mean_score = winning_entry["Mean Round AUC"]

print("\n============================================")
if champion_r8_score >= TARGET_GATE_AUC:
    print(f"🏆 GATE UNLOCKED: Champion '{champion_config['name']}' cleared target gate with {champion_r8_score*100:.2f}% Round 8 AUC!")
else:
    print(f"🏆 BEST CHAMPION SELECTED: '{champion_config['name']}' reached highest benchmark: {champion_r8_score*100:.2f}% Round 8 AUC (Mean: {champion_mean_score*100:.2f}%).")
print("============================================")


# ============================================================
# 7. TRAIN CHAMPION ON FULL DATASET (7-SEED ENSEMBLE)
# ============================================================

print(f"\nTraining Champion '{champion_config['name']}' on full dataset across 7 seeds...")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")

train_clean = clean_and_prepare_features(train_raw)
test_clean = clean_and_prepare_features(test_raw)

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

print(f"Full dataset: {len(X_full)} observations | Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")

SEEDS = [42, 101, 777, 2024, 999, 1337, 555]
seed_predictions = []

for s_idx, seed in enumerate(SEEDS):
    preprocessor = build_preprocessor(numerical_features, categorical_features, use_quantile=champion_config["use_quantile"])

    if champion_config["type"] == "pure_linear":
        # Clone linear model with fresh random state
        linear_model = LogisticRegression(
            C=champion_config["model"].C,
            penalty=champion_config["model"].penalty,
            solver=champion_config["model"].solver,
            l1_ratio=getattr(champion_config["model"], "l1_ratio", None),
            max_iter=1000,
            random_state=seed
        )
        pipe = Pipeline([("preprocessor", preprocessor), ("model", linear_model)])
        pipe.fit(X_full, y_full)
        seed_predictions.append(pipe.predict_proba(X_test_full)[:, 1])
    else:
        # Clone hybrid components with fresh random state
        linear_model = LogisticRegression(
            C=champion_config["linear_model"].C,
            penalty=champion_config["linear_model"].penalty,
            solver=champion_config["linear_model"].solver,
            max_iter=1000,
            random_state=seed
        )
        neural_model = MLPClassifier(
            hidden_layer_sizes=champion_config["neural_model"].hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=champion_config["neural_model"].alpha,
            batch_size=128,
            learning_rate_init=0.001,
            max_iter=350,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.15,
            random_state=seed
        )

        pipe_lin = Pipeline([("preprocessor", preprocessor), ("model", linear_model)])
        pipe_neu = Pipeline([("preprocessor", preprocessor), ("model", neural_model)])
        pipe_lin.fit(X_full, y_full)
        pipe_neu.fit(X_full, y_full)

        p_lin = np.clip(pipe_lin.predict_proba(X_test_full)[:, 1], 1e-6, 1 - 1e-6)
        p_neu = np.clip(pipe_neu.predict_proba(X_test_full)[:, 1], 1e-6, 1 - 1e-6)

        z_lin = np.log(p_lin / (1 - p_lin))
        z_neu = np.log(p_neu / (1 - p_neu))
        z_blend = champion_config["linear_weight"] * z_lin + champion_config["neural_weight"] * z_neu
        seed_predictions.append(1.0 / (1.0 + np.exp(-z_blend)))

final_probabilities = np.mean(seed_predictions, axis=0)


# ============================================================
# 8. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp52_validation_gated_champion.csv"

submission = pd.DataFrame({
    "anonymised_id": test_raw["anonymised_id"],
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 9. SUMMARY & BENCHMARKS
# ============================================================

print("\n============================================")
print("EXPERIMENT 52 COMPLETE")
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
print("Personal Best Benchmark          : 0.65693")
print(f"Exp 52 Champion ({champion_config['name']}): Round 8 AUC = {champion_r8_score:.5f}")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")