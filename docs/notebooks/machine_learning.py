# ============================================================
# EXPERIMENT 51 — MODEL DIVERSITY TOURNAMENT
#
# Goal:
#   Find complementary MODEL ARCHITECTURES rather than
#   creating more train/validation splits.
#
#   Models tested:
#   1. Logistic Regression L2
#   2. Logistic Regression ElasticNet
#   3. Logistic Regression stronger L1
#   4. HistGradientBoosting
#   5. ExtraTrees
#   6. RandomForest
#   7. RandomForest balanced
#   8. SGD Logistic
#   9. Ridge calibrated
#
#   Round 6 / Round 7 are ONLY used to identify models that
#   generalise well across time.
#
#   Final models are retrained on ALL available training data.
#
#   No XGBoost required.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)

from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    RidgeClassifier
)

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENT = 51
TARGET = "employed_status"

print("=" * 60)
print(f"EXPERIMENT {EXPERIMENT}")
print("MODEL DIVERSITY TOURNAMENT")
print("=" * 60)


# ============================================================
# 1. DIRECTORY
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()

if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR

ROUND_DIR = CURRENT_DIR / "round_testing"

if not ROUND_DIR.exists():
    ROUND_DIR = DATA_DIR / "round_testing"

print("\nDATA DIRECTORY:")
print(DATA_DIR)

print("\nROUND DIRECTORY:")
print(ROUND_DIR)


# ============================================================
# 2. LOAD MAIN DATA
# ============================================================

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")

print("\n" + "=" * 60)
print("DATASET")
print("=" * 60)

print("Training rows :", len(train_raw))
print("Testing rows  :", len(test_raw))
print("Training cols :", len(train_raw.columns))
print("Testing cols  :", len(test_raw.columns))


# ============================================================
# 3. CLEAN TARGET
# ============================================================

train_raw[TARGET] = pd.to_numeric(
    train_raw[TARGET],
    errors="coerce"
)

missing_target = train_raw[TARGET].isna().sum()

print("\nMissing target values:", missing_target)

train_raw = train_raw.dropna(
    subset=[TARGET]
).copy()

train_raw[TARGET] = train_raw[TARGET].astype(int)

print("Training rows after cleaning:", len(train_raw))

print("\nTarget distribution:")
print(train_raw[TARGET].value_counts())


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    data = df.copy()

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    if "anonymised_id" in data.columns:
        data = data.drop(columns=["anonymised_id"])

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    if "survey_date" in data.columns:

        date = pd.to_datetime(
            data["survey_date"],
            errors="coerce"
        )

        data["survey_year"] = date.dt.year
        data["survey_month"] = date.dt.month
        data["survey_day"] = date.dt.day
        data["survey_dayofyear"] = date.dt.dayofyear

        data = data.drop(columns=["survey_date"])

    # --------------------------------------------------------
    # MATRIC SCORE PARSER
    # --------------------------------------------------------

    def parse_score(value):

        if pd.isna(value):
            return np.nan

        s = str(value).strip()

        s = s.replace("%", "")

        # "50 - 59"
        if "-" in s:

            try:

                parts = s.split("-")

                low = float(parts[0].strip())
                high = float(parts[1].strip())

                return (low + high) / 2

            except:
                return np.nan

        # "< 30"
        if "<" in s:

            try:

                number = float(
                    s.replace("<", "").strip()
                )

                return number / 2

            except:
                return np.nan

        # "> 70"
        if ">" in s:

            try:

                number = float(
                    s.replace(">", "").strip()
                )

                return number + 5

            except:
                return np.nan

        try:
            return float(s)
        except:
            return np.nan

    matric_columns = [
        "matric_englishhome",
        "matric_englishadd",
        "matric_mathpure",
        "matric_physicalscience",
        "matric_mathlit"
    ]

    for col in matric_columns:

        if col in data.columns:

            # Missing indicator
            data[f"{col}_missing"] = (
                data[col].isna().astype(int)
            )

            # Numeric representation
            data[f"{col}_numeric"] = (
                data[col].apply(parse_score)
            )

    # --------------------------------------------------------
    # MATRIC AGGREGATES
    # --------------------------------------------------------

    numeric_matric = [
        f"{c}_numeric"
        for c in matric_columns
        if f"{c}_numeric" in data.columns
    ]

    if numeric_matric:

        data["matric_count"] = (
            data[numeric_matric]
            .notna()
            .sum(axis=1)
        )

        data["matric_average"] = (
            data[numeric_matric]
            .mean(axis=1)
        )

        data["matric_max"] = (
            data[numeric_matric]
            .max(axis=1)
        )

        data["matric_min"] = (
            data[numeric_matric]
            .min(axis=1)
        )

    # --------------------------------------------------------
    # EMPLOYMENT LAG
    # --------------------------------------------------------

    if "employed_lag" in data.columns:

        data["employed_lag_missing"] = (
            data["employed_lag"]
            .isna()
            .astype(int)
        )

        data["employed_lag_known"] = (
            data["employed_lag"]
            .fillna(-1)
        )

    # --------------------------------------------------------
    # DAYS SINCE LAST OBSERVATION
    # --------------------------------------------------------

    if "days_since_last_obs" in data.columns:

        data["days_since_missing"] = (
            data["days_since_last_obs"]
            .isna()
            .astype(int)
        )

        data["days_since_log"] = np.log1p(
            data["days_since_last_obs"]
            .fillna(0)
            .clip(lower=0)
        )

    # --------------------------------------------------------
    # TENURE
    # --------------------------------------------------------

    if "tenure_lag" in data.columns:

        data["tenure_missing"] = (
            data["tenure_lag"]
            .isna()
            .astype(int)
        )

        data["tenure_log"] = np.log1p(
            data["tenure_lag"]
            .fillna(0)
            .clip(lower=0)
        )

    # --------------------------------------------------------
    # WORK READINESS
    # --------------------------------------------------------

    if "work_readiness_score" in data.columns:

        data["work_readiness_missing"] = (
            data["work_readiness_score"]
            .isna()
            .astype(int)
        )

    # --------------------------------------------------------
    # TERTIARY INDICATOR
    # --------------------------------------------------------

    if "institution_type" in data.columns:

        data["has_institution"] = (
            data["institution_type"]
            .notna()
            .astype(int)
        )

    # --------------------------------------------------------
    # SETA INDICATOR
    # --------------------------------------------------------

    if "seta" in data.columns:

        data["has_seta"] = (
            data["seta"]
            .notna()
            .astype(int)
        )

    # --------------------------------------------------------
    # AUTO NUMERIC CONVERSION
    # --------------------------------------------------------

    for col in data.columns:

        if col == TARGET:
            continue

        if data[col].dtype == "object":

            converted = pd.to_numeric(
                data[col],
                errors="coerce"
            )

            non_missing = data[col].notna().sum()

            if non_missing > 0:

                conversion_rate = (
                    converted.notna().sum()
                    / non_missing
                )

                if conversion_rate > 0.60:

                    data[col] = converted

    return data


# ============================================================
# FEATURE ENGINEERING
# ============================================================

train_df = engineer_features(train_raw)
test_df = engineer_features(test_raw)

common_features = [
    c for c in train_df.columns
    if c in test_df.columns
    and c != TARGET
]

X_full = train_df[common_features].copy()
X_test = test_df[common_features].copy()

y_full = train_df[TARGET].copy()

print("\n" + "=" * 60)
print("FEATURES")
print("=" * 60)

print("Total features:", len(common_features))


# ============================================================
# 5. REMOVE EXTREME HIGH CARDINALITY
# ============================================================

categorical = X_full.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

high_cardinality = [
    c for c in categorical
    if X_full[c].nunique(dropna=True) > 250
]

print("\nHigh-cardinality features:")

if high_cardinality:

    for c in high_cardinality:
        print(" -", c)

    X_full = X_full.drop(
        columns=high_cardinality
    )

    X_test = X_test.drop(
        columns=high_cardinality
    )

else:

    print("None")


categorical = X_full.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical = X_full.select_dtypes(
    include=[np.number]
).columns.tolist()

print("\nNumerical:", len(numerical))
print("Categorical:", len(categorical))


# ============================================================
# 6. PREPROCESSORS
# ============================================================

def make_linear_preprocessor():

    transformers = []

    if numerical:

        num_pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scale",
                StandardScaler()
            )
        ])

        transformers.append(
            ("num", num_pipe, numerical)
        )

    if categorical:

        cat_pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ])

        transformers.append(
            ("cat", cat_pipe, categorical)
        )

    return ColumnTransformer(
        transformers=transformers
    )


def make_tree_preprocessor():

    transformers = []

    if numerical:

        num_pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ])

        transformers.append(
            ("num", num_pipe, numerical)
        )

    if categorical:

        cat_pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )
            )
        ])

        transformers.append(
            ("cat", cat_pipe, categorical)
        )

    return ColumnTransformer(
        transformers=transformers
    )


# ============================================================
# 7. MODEL FACTORY
# ============================================================

def get_models(seed=42):

    models = {}

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    models["LogReg_L2_C005"] = (
        LogisticRegression(
            C=0.05,
            penalty="l2",
            solver="lbfgs",
            max_iter=1500,
            random_state=seed
        ),
        "linear"
    )

    models["LogReg_L2_C008"] = (
        LogisticRegression(
            C=0.08,
            penalty="l2",
            solver="lbfgs",
            max_iter=1500,
            random_state=seed
        ),
        "linear"
    )

    models["LogReg_L2_C010"] = (
        LogisticRegression(
            C=0.10,
            penalty="l2",
            solver="lbfgs",
            max_iter=1500,
            random_state=seed
        ),
        "linear"
    )

    # --------------------------------------------------------
    # ELASTIC NET
    # --------------------------------------------------------

    models["ElasticNet_12"] = (
        LogisticRegression(
            C=0.08,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.12,
            max_iter=2000,
            random_state=seed
        ),
        "linear"
    )

    models["ElasticNet_20"] = (
        LogisticRegression(
            C=0.08,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.20,
            max_iter=2000,
            random_state=seed
        ),
        "linear"
    )

    models["ElasticNet_30"] = (
        LogisticRegression(
            C=0.08,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.30,
            max_iter=2000,
            random_state=seed
        ),
        "linear"
    )

    # --------------------------------------------------------
    # EXTRA TREES
    # --------------------------------------------------------

    models["ExtraTrees_500"] = (
        ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=5,
            max_depth=None,
            class_weight=None,
            random_state=seed,
            n_jobs=-1
        ),
        "tree"
    )

    models["ExtraTrees_800"] = (
        ExtraTreesClassifier(
            n_estimators=800,
            max_features=0.7,
            min_samples_leaf=5,
            max_depth=None,
            class_weight=None,
            random_state=seed,
            n_jobs=-1
        ),
        "tree"
    )

    models["ExtraTrees_Leaf10"] = (
        ExtraTreesClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=10,
            max_depth=None,
            class_weight=None,
            random_state=seed,
            n_jobs=-1
        ),
        "tree"
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    models["RandomForest_500"] = (
        RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=5,
            max_depth=None,
            class_weight=None,
            random_state=seed,
            n_jobs=-1
        ),
        "tree"
    )

    models["RandomForest_800"] = (
        RandomForestClassifier(
            n_estimators=800,
            max_features=0.7,
            min_samples_leaf=5,
            max_depth=None,
            class_weight=None,
            random_state=seed,
            n_jobs=-1
        ),
        "tree"
    )

    # --------------------------------------------------------
    # HISTOGRAM GRADIENT BOOSTING
    # --------------------------------------------------------

    models["HistGB_200"] = (
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=seed
        ),
        "tree"
    )

    models["HistGB_300"] = (
        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=2.0,
            random_state=seed
        ),
        "tree"
    )

    # --------------------------------------------------------
    # SGD
    # --------------------------------------------------------

    models["SGD_Logistic"] = (
        CalibratedClassifierCV(
            estimator=SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=0.0005,
                l1_ratio=0.15,
                max_iter=1500,
                random_state=seed
            ),
            method="sigmoid",
            cv=3
        ),
        "linear"
    )

    # --------------------------------------------------------
    # RIDGE
    # --------------------------------------------------------

    models["Ridge_Calibrated"] = (
        CalibratedClassifierCV(
            estimator=RidgeClassifier(
                alpha=2.0
            ),
            method="sigmoid",
            cv=3
        ),
        "linear"
    )

    return models


# ============================================================
# 8. LOAD ROUND DATASETS
# ============================================================

print("\n" + "=" * 60)
print("ROUND VALIDATION")
print("=" * 60)

round_data = []

if ROUND_DIR.exists():

    round_folders = sorted(
        [
            f
            for f in ROUND_DIR.glob("round_*")
            if f.is_dir()
        ]
    )

    for folder in round_folders:

        csvs = list(folder.glob("*.csv"))

        if len(csvs) < 2:
            continue

        # Identify train/test by target column
        train_file = None
        test_file = None

        for csv in csvs:

            temp = pd.read_csv(csv, nrows=5)

            if TARGET in temp.columns:
                train_file = csv
            else:
                test_file = csv

        if train_file is None or test_file is None:
            continue

        r_train_raw = pd.read_csv(train_file)
        r_test_raw = pd.read_csv(test_file)

        r_train_raw[TARGET] = pd.to_numeric(
            r_train_raw[TARGET],
            errors="coerce"
        )

        r_train_raw = r_train_raw.dropna(
            subset=[TARGET]
        ).copy()

        r_train_raw[TARGET] = (
            r_train_raw[TARGET]
            .astype(int)
        )

        # Test round should contain target according to
        # your round_testing structure.
        r_test_raw[TARGET] = pd.to_numeric(
            r_test_raw[TARGET],
            errors="coerce"
        )

        r_test_raw = r_test_raw.dropna(
            subset=[TARGET]
        ).copy()

        r_test_raw[TARGET] = (
            r_test_raw[TARGET]
            .astype(int)
        )

        r_train = engineer_features(
            r_train_raw
        )

        r_test = engineer_features(
            r_test_raw
        )

        features = [
            c for c in r_train.columns
            if c in r_test.columns
            and c != TARGET
        ]

        if len(features) < 5:
            continue

        Xr_train = r_train[features].copy()
        Xr_test = r_test[features].copy()

        # Same high-cardinality rule
        cats = Xr_train.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        high = [
            c for c in cats
            if Xr_train[c].nunique(
                dropna=True
            ) > 250
        ]

        if high:

            Xr_train = Xr_train.drop(
                columns=high
            )

            Xr_test = Xr_test.drop(
                columns=high
            )

        round_data.append({
            "name": folder.name,
            "X_train": Xr_train,
            "y_train": r_train[TARGET].reset_index(drop=True),
            "X_test": Xr_test,
            "y_test": r_test[TARGET].reset_index(drop=True)
        })

        print(
            f"Loaded {folder.name.upper()} | "
            f"Train: {len(Xr_train)} | "
            f"Test: {len(Xr_test)}"
        )


# ============================================================
# 9. MODEL TOURNAMENT
# ============================================================

print("\n" + "=" * 60)
print("STARTING MODEL TOURNAMENT")
print("=" * 60)

base_models = get_models(seed=42)

scores = {
    name: []
    for name in base_models
}

round_names = []

for r in round_data:

    print(
        f"\nTesting models on "
        f"{r['name'].upper()}..."
    )

    Xtr = r["X_train"].copy()
    ytr = r["y_train"].copy()

    Xte = r["X_test"].copy()
    yte = r["y_test"].copy()

    cats = Xtr.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    nums = Xtr.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    round_names.append(
        r["name"].upper()
    )

    for model_name, (estimator, model_type) in base_models.items():

        try:

            if model_type == "linear":
                preprocessor = make_linear_preprocessor()

            else:
                preprocessor = make_tree_preprocessor()

            pipeline = Pipeline([
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    estimator
                )
            ])

            pipeline.fit(
                Xtr,
                ytr
            )

            probabilities = (
                pipeline
                .predict_proba(Xte)[:, 1]
            )

            auc = roc_auc_score(
                yte,
                probabilities
            )

            scores[model_name].append(
                auc
            )

            print(
                f"  {model_name:<25} "
                f"AUC = {auc:.6f}"
            )

        except Exception as e:

            print(
                f"  {model_name:<25} "
                f"FAILED: {str(e)[:100]}"
            )

            scores[model_name].append(
                np.nan
            )


# ============================================================
# 10. LEADERBOARD
# ============================================================

rows = []

for model_name, model_scores in scores.items():

    valid = [
        s for s in model_scores
        if not np.isnan(s)
    ]

    if not valid:
        mean_auc = 0
    else:
        mean_auc = np.mean(valid)

    row = {
        "Model": model_name,
        "Mean_AUC": mean_auc
    }

    for i, score in enumerate(model_scores):

        if i < len(round_names):

            row[
                f"{round_names[i]}_AUC"
            ] = score

    rows.append(row)


leaderboard = (
    pd.DataFrame(rows)
    .sort_values(
        "Mean_AUC",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 60)
print("MODEL TOURNAMENT LEADERBOARD")
print("=" * 60)

print(
    leaderboard.to_string(
        index=False
    )
)


# ============================================================
# 11. SELECT DIVERSE CHAMPIONS
# ============================================================

# Instead of blindly selecting the four highest,
# select strong models from different architectures.

ordered_models = (
    leaderboard["Model"]
    .tolist()
)

champions = []

architecture_used = set()

for model_name in ordered_models:

    if len(champions) >= 6:
        break

    if model_name.startswith("LogReg"):
        architecture = "logistic"

    elif model_name.startswith("ElasticNet"):
        architecture = "elastic"

    elif model_name.startswith("ExtraTrees"):
        architecture = "extratrees"

    elif model_name.startswith("RandomForest"):
        architecture = "forest"

    elif model_name.startswith("HistGB"):
        architecture = "histgb"

    elif model_name.startswith("SGD"):
        architecture = "sgd"

    elif model_name.startswith("Ridge"):
        architecture = "ridge"

    else:
        architecture = model_name

    if architecture not in architecture_used:

        champions.append(
            model_name
        )

        architecture_used.add(
            architecture
        )


# Add additional strongest models if needed

for model_name in ordered_models:

    if len(champions) >= 6:
        break

    if model_name not in champions:
        champions.append(
            model_name
        )


print("\n" + "=" * 60)
print("SELECTED CHAMPIONS")
print("=" * 60)

for i, champion in enumerate(champions, 1):

    score = leaderboard.loc[
        leaderboard["Model"] == champion,
        "Mean_AUC"
    ].iloc[0]

    print(
        f"{i}. {champion:<30} "
        f"Mean AUC: {score:.6f}"
    )


# ============================================================
# 12. CORRELATION-BASED DIVERSITY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING CHAMPION PREDICTIONS")
print("=" * 60)

champion_predictions = {}

for model_name in champions:

    print(
        f"Training {model_name}..."
    )

    estimator, model_type = (
        get_models(seed=42)[model_name]
    )

    if model_type == "linear":
        preprocessor = (
            make_linear_preprocessor()
        )
    else:
        preprocessor = (
            make_tree_preprocessor()
        )

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            estimator
        )
    ])

    pipeline.fit(
        X_full,
        y_full
    )

    pred = (
        pipeline
        .predict_proba(X_test)[:, 1]
    )

    champion_predictions[
        model_name
    ] = pred


# ============================================================
# 13. PREDICTION CORRELATION
# ============================================================

prediction_df = pd.DataFrame(
    champion_predictions
)

print("\nPrediction correlation:")
print(
    prediction_df.corr()
    .round(3)
    .to_string()
)


# ============================================================
# 14. ENSEMBLE WEIGHT SEARCH
# ============================================================

print("\n" + "=" * 60)
print("ENSEMBLE WEIGHT SEARCH")
print("=" * 60)

# We cannot directly evaluate the final test set,
# so use sequential round performance to determine
# model reliability.

mean_scores = {}

for model_name in champions:

    row = leaderboard[
        leaderboard["Model"] == model_name
    ]

    mean_scores[model_name] = (
        row["Mean_AUC"].iloc[0]
    )


# ------------------------------------------------------------
# Weight scheme 1:
# Performance-weighted
# ------------------------------------------------------------

raw_weights = np.array([
    mean_scores[m]
    for m in champions
])

raw_weights = (
    raw_weights - raw_weights.min()
)

if raw_weights.sum() == 0:

    performance_weights = (
        np.ones(len(champions))
        / len(champions)
    )

else:

    performance_weights = (
        raw_weights
        / raw_weights.sum()
    )


# ------------------------------------------------------------
# Weight scheme 2:
# Soft performance weighting
# ------------------------------------------------------------

temperature = 0.02

exp_scores = np.exp(
    raw_weights / temperature
)

soft_weights = (
    exp_scores
    / exp_scores.sum()
)


# ------------------------------------------------------------
# Weight scheme 3:
# Top-model focused
# ------------------------------------------------------------

rank_weights = np.array([
    1.0 / (i + 1)
    for i in range(len(champions))
])

rank_weights = (
    rank_weights
    / rank_weights.sum()
)


print("\nPerformance weights:")

for model, weight in zip(
    champions,
    performance_weights
):

    print(
        f"{model:<30} "
        f"{weight:.4f}"
    )


# ============================================================
# 15. CREATE ENSEMBLES
# ============================================================

prediction_matrix = np.column_stack([
    champion_predictions[m]
    for m in champions
])


ensemble_1 = (
    prediction_matrix
    @ performance_weights
)

ensemble_2 = (
    prediction_matrix
    @ soft_weights
)

ensemble_3 = (
    prediction_matrix
    @ rank_weights
)

# Simple average
ensemble_4 = (
    prediction_matrix
    .mean(axis=1)
)


# ============================================================
# 16. BLEND DIVERSE ENSEMBLES
# ============================================================

# Main ensemble:
#
# 40% performance-weighted
# 30% soft-weighted
# 20% rank-weighted
# 10% equal-weight
#
# This prevents one model from completely dominating.

final_probabilities = (
    0.40 * ensemble_1
    + 0.30 * ensemble_2
    + 0.20 * ensemble_3
    + 0.10 * ensemble_4
)


# ============================================================
# 17. SANITY CHECK
# ============================================================

if len(final_probabilities) != len(test_raw):

    raise ValueError(
        "Prediction count does not match test data."
    )

if np.isnan(final_probabilities).any():

    raise ValueError(
        "Predictions contain NaN."
    )

final_probabilities = np.clip(
    final_probabilities,
    1e-6,
    1 - 1e-6
)


# ============================================================
# 18. SAVE MAIN SUBMISSION
# ============================================================

output_file = (
    "submission_exp51_model_diversity_ensemble.csv"
)

submission = pd.DataFrame({

    "anonymised_id":
        test_raw["anonymised_id"],

    "employed_status":
        final_probabilities
})


submission.to_csv(
    output_file,
    index=False
)


# ============================================================
# 19. SAVE INDIVIDUAL CHAMPION PREDICTIONS
# ============================================================

for model_name in champions:

    safe_name = (
        model_name
        .replace(" ", "_")
        .replace("/", "_")
        .replace("=", "")
    )

    individual_submission = pd.DataFrame({

        "anonymised_id":
            test_raw["anonymised_id"],

        "employed_status":
            champion_predictions[
                model_name
            ]
    })

    individual_file = (
        f"submission_exp51_{safe_name}.csv"
    )

    individual_submission.to_csv(
        individual_file,
        index=False
    )


# ============================================================
# 20. ALSO SAVE THE DIFFERENT ENSEMBLES
# ============================================================

ensemble_outputs = {

    "performance":
        ensemble_1,

    "soft":
        ensemble_2,

    "rank":
        ensemble_3,

    "average":
        ensemble_4,

    "final":
        final_probabilities
}


for name, probabilities in (
    ensemble_outputs.items()
):

    filename = (
        f"submission_exp51_ensemble_{name}.csv"
    )

    pd.DataFrame({

        "anonymised_id":
            test_raw["anonymised_id"],

        "employed_status":
            probabilities

    }).to_csv(
        filename,
        index=False
    )


# ============================================================
# 21. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 51 COMPLETE")
print("=" * 60)

print(
    f"Saved main submission:\n"
    f"{output_file}"
)

print(
    f"\nRows: {len(submission)}"
)

print("\nPrediction summary:")
print(
    submission[
        "employed_status"
    ].describe()
)


print("\nFirst 10 predictions:")
print(
    submission.head(10)
)


print("\n" + "=" * 60)
print("BENCHMARKS")
print("=" * 60)

print(
    "Exp 44 Top-3 Linear Blend     : 0.65630"
)

print(
    "Exp 47 Pure-Logit Ensemble    : 0.65635"
)

print(
    "Exp 46 Multilinear Ensemble   : 0.65701"
)

print(
    "Exp 51 Model Diversity        : READY"
)

print("\n" + "=" * 60)
print("FILES CREATED")
print("=" * 60)

print(
    f"\nMain:\n"
    f"  {output_file}"
)

print("\nIndividual champions:")

for model_name in champions:

    safe_name = (
        model_name
        .replace(" ", "_")
        .replace("/", "_")
        .replace("=", "")
    )

    print(
        f"  submission_exp51_{safe_name}.csv"
    )

print("\nEnsembles:")

for name in ensemble_outputs:

    print(
        f"  submission_exp51_ensemble_{name}.csv"
    )

print("\n" + "=" * 60)
print("READY FOR KAGGLE")
print("=" * 60)