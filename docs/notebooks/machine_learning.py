# ============================================================
# EXPERIMENT 52 — FULL MODEL DIVERSITY TOURNAMENT
#
# Goal:
#   Find a stronger ensemble than Exp 46 (0.65701)
#
# Models:
#   1. Logistic Regression L2
#   2. Logistic Regression ElasticNet
#   3. ExtraTrees
#   4. Random Forest
#   5. HistGradientBoosting
#   6. RandomForest with class weighting
#   7. CatBoost (OPTIONAL if installed)
#
# Ensemble:
#   - Probability averaging
#   - Rank averaging
#   - Automatic blend search
#
# IMPORTANT:
#   No XGBoost required.
#   No sequential round splits.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import roc_auc_score


# ============================================================
# OPTIONAL CATBOOST
# ============================================================

CATBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
    print("CatBoost detected.")
except ImportError:
    print("CatBoost not installed.")
    print("Continuing without CatBoost.")


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT 52")
print("FULL MODEL DIVERSITY TOURNAMENT")
print("=" * 70)


# ============================================================
# 1. DIRECTORY
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()

if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

print()
print("DATA DIRECTORY")
print("-" * 70)
print(DATA_DIR)

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Could not find: {TRAIN_FILE}")

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Could not find: {TEST_FILE}")


# ============================================================
# 2. LOAD DATA
# ============================================================

train_raw = pd.read_csv(TRAIN_FILE)
test_raw = pd.read_csv(TEST_FILE)

print()
print("=" * 70)
print("DATASET LOADED")
print("=" * 70)

print(f"Training rows   : {len(train_raw)}")
print(f"Training columns: {train_raw.shape[1]}")
print(f"Testing rows    : {len(test_raw)}")
print(f"Testing columns : {test_raw.shape[1]}")


# ============================================================
# 3. TARGET CLEANING
# ============================================================

TARGET = "employed_status"
ID_COL = "anonymised_id"

train_raw[TARGET] = pd.to_numeric(
    train_raw[TARGET],
    errors="coerce"
)

missing_target = train_raw[TARGET].isna().sum()

print()
print("TARGET")
print("-" * 70)
print(f"Missing target values: {missing_target}")

train_raw = train_raw.dropna(subset=[TARGET]).copy()

train_raw[TARGET] = train_raw[TARGET].astype(int)

print(f"Rows after cleaning  : {len(train_raw)}")

print()
print("Target distribution:")
print(train_raw[TARGET].value_counts())

print()
print("Target proportions:")
print(train_raw[TARGET].value_counts(normalize=True))


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def parse_matric_band(value):

    if pd.isna(value):
        return np.nan

    s = str(value).strip().replace("%", "")

    try:

        if "-" in s:

            parts = s.split("-")

            if len(parts) >= 2:
                a = float(parts[0].strip())
                b = float(parts[1].strip())

                return (a + b) / 2

        if "<" in s:
            number = float(
                s.replace("<", "").strip()
            )

            return number / 2

        if ">" in s:
            number = float(
                s.replace(">", "").strip()
            )

            return number + 5

        return float(s)

    except:
        return np.nan


def engineer_features(df):

    data = df.copy()

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    if ID_COL in data.columns:
        data = data.drop(columns=[ID_COL])


    # --------------------------------------------------------
    # DATE
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

        data["survey_quarter"] = date.dt.quarter

        data = data.drop(columns=["survey_date"])


    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    if TARGET in data.columns:

        data[TARGET] = pd.to_numeric(
            data[TARGET],
            errors="coerce"
        )


    # --------------------------------------------------------
    # EMPLOYMENT HISTORY
    # --------------------------------------------------------

    if "employed_lag" in data.columns:

        data["employed_lag_missing"] = (
            data["employed_lag"].isna().astype(int)
        )

        data["employed_lag_value"] = (
            pd.to_numeric(
                data["employed_lag"],
                errors="coerce"
            )
            .fillna(-1)
        )


    if "days_since_last_obs" in data.columns:

        data["days_since_missing"] = (
            data["days_since_last_obs"]
            .isna()
            .astype(int)
        )

        days = pd.to_numeric(
            data["days_since_last_obs"],
            errors="coerce"
        )

        data["days_since_log"] = np.log1p(
            days.clip(lower=0).fillna(0)
        )


    if "tenure_lag" in data.columns:

        data["tenure_missing"] = (
            data["tenure_lag"]
            .isna()
            .astype(int)
        )

        tenure = pd.to_numeric(
            data["tenure_lag"],
            errors="coerce"
        )

        data["tenure_log"] = np.log1p(
            tenure.clip(lower=0).fillna(0)
        )


    # --------------------------------------------------------
    # FIRST-TIME PARTICIPANT
    # --------------------------------------------------------

    missing_flags = []

    if "employed_lag" in data.columns:
        missing_flags.append(
            data["employed_lag"].isna()
        )

    if "days_since_last_obs" in data.columns:
        missing_flags.append(
            data["days_since_last_obs"].isna()
        )

    if missing_flags:

        first_time = missing_flags[0]

        for flag in missing_flags[1:]:
            first_time = first_time | flag

        data["is_first_time_participant"] = (
            first_time.astype(int)
        )


    # --------------------------------------------------------
    # MATRIC FEATURES
    # --------------------------------------------------------

    matric_columns = [
        "matric_englishhome",
        "matric_englishadd",
        "matric_mathpure",
        "matric_physicalscience",
        "matric_mathlit"
    ]

    matric_numeric = []

    for col in matric_columns:

        if col in data.columns:

            data[f"{col}_missing"] = (
                data[col]
                .isna()
                .astype(int)
            )

            new_col = f"{col}_numeric"

            data[new_col] = data[col].apply(
                parse_matric_band
            )

            matric_numeric.append(new_col)

            data = data.drop(
                columns=[col]
            )


    if matric_numeric:

        data["matric_subject_count"] = (
            data[matric_numeric]
            .notna()
            .sum(axis=1)
        )

        data["matric_average"] = (
            data[matric_numeric]
            .mean(axis=1)
        )

        data["matric_max"] = (
            data[matric_numeric]
            .max(axis=1)
        )

        data["matric_min"] = (
            data[matric_numeric]
            .min(axis=1)
        )

        math_cols = [
            c for c in [
                "matric_mathpure_numeric",
                "matric_mathlit_numeric"
            ]
            if c in data.columns
        ]

        if math_cols:

            data["best_math"] = (
                data[math_cols]
                .max(axis=1)
            )

        english_cols = [
            c for c in [
                "matric_englishhome_numeric",
                "matric_englishadd_numeric"
            ]
            if c in data.columns
        ]

        if english_cols:

            data["best_english"] = (
                data[english_cols]
                .max(axis=1)
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
    # EDUCATION / TERTIARY
    # --------------------------------------------------------

    if "institution_type" in data.columns:

        data["has_tertiary_record"] = (
            data["institution_type"]
            .notna()
            .astype(int)
        )


    if "seta" in data.columns:

        data["has_seta_record"] = (
            data["seta"]
            .notna()
            .astype(int)
        )


    # --------------------------------------------------------
    # NUMERIC CONVERSION
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
# 5. ENGINEER
# ============================================================

train = engineer_features(train_raw)
test = engineer_features(test_raw)

feature_columns = [
    c for c in train.columns
    if c != TARGET and c in test.columns
]

X = train[feature_columns].copy()
y = train[TARGET].copy()

X_test = test[feature_columns].copy()

print()
print("=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

print(f"Total features: {len(feature_columns)}")

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=[np.number]
).columns.tolist()

print(f"Numerical features : {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")


# ============================================================
# 6. REMOVE VERY HIGH CARDINALITY CATEGORIES
# ============================================================

high_cardinality = []

for col in categorical_features:

    unique_count = X[col].nunique(
        dropna=True
    )

    if unique_count > 100:

        high_cardinality.append(col)

if high_cardinality:

    print()
    print("Removing high-cardinality categorical features:")

    for col in high_cardinality:
        print(
            f" - {col}: "
            f"{X[col].nunique(dropna=True)} unique"
        )

    X = X.drop(
        columns=high_cardinality
    )

    X_test = X_test.drop(
        columns=high_cardinality
    )

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()


# ============================================================
# 7. FIX CATEGORICAL DATA
# ============================================================

for col in categorical_features:

    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

    X[col] = X[col].replace(
        "nan",
        "MISSING"
    )

    X_test[col] = X_test[col].replace(
        "nan",
        "MISSING"
    )


# ============================================================
# 8. VALIDATION SPLIT
#
# This is ONLY used to compare models.
# Final training later uses ALL rows.
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()
print("=" * 70)
print("VALIDATION SET")
print("=" * 70)

print(f"Training rows  : {len(X_train)}")
print(f"Validation rows: {len(X_valid)}")


# ============================================================
# 9. PREPROCESSOR FOR LINEAR MODELS
# ============================================================

linear_transformers = []

if numerical_features:

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    linear_transformers.append(
        (
            "num",
            numeric_pipeline,
            numerical_features
        )
    )


if categorical_features:

    categorical_pipeline = Pipeline([
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
                sparse_output=True
            )
        )
    ])

    linear_transformers.append(
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    )


linear_preprocessor = ColumnTransformer(
    transformers=linear_transformers
)


# ============================================================
# 10. MODEL DEFINITIONS
# ============================================================

models = {}


# ------------------------------------------------------------
# MODEL 1 — LOGISTIC L2
# ------------------------------------------------------------

models["LogReg_L2"] = Pipeline([
    (
        "preprocessor",
        linear_preprocessor
    ),
    (
        "model",
        LogisticRegression(
            C=0.07,
            penalty="l2",
            solver="lbfgs",
            max_iter=1500,
            random_state=42
        )
    )
])


# ------------------------------------------------------------
# MODEL 2 — LOGISTIC ELASTICNET
# ------------------------------------------------------------

models["LogReg_ElasticNet"] = Pipeline([
    (
        "preprocessor",
        linear_preprocessor
    ),
    (
        "model",
        LogisticRegression(
            C=0.08,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.20,
            max_iter=1500,
            random_state=42
        )
    )
])


# ------------------------------------------------------------
# MODEL 3 — EXTRA TREES
# ------------------------------------------------------------

tree_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(
                strategy="median"
            ),
            numerical_features
        ),
        (
            "cat",
            Pipeline([
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
            ]),
            categorical_features
        )
    ],
    sparse_threshold=0
)


models["ExtraTrees"] = Pipeline([
    (
        "preprocessor",
        tree_preprocessor
    ),
    (
        "model",
        ExtraTreesClassifier(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
])


# ------------------------------------------------------------
# MODEL 4 — EXTRA TREES MORE CONSERVATIVE
# ------------------------------------------------------------

models["ExtraTrees_Deep"] = Pipeline([
    (
        "preprocessor",
        tree_preprocessor
    ),
    (
        "model",
        ExtraTreesClassifier(
            n_estimators=700,
            max_depth=18,
            min_samples_leaf=4,
            max_features=0.70,
            class_weight="balanced",
            random_state=123,
            n_jobs=-1
        )
    )
])


# ------------------------------------------------------------
# MODEL 5 — RANDOM FOREST
# ------------------------------------------------------------

models["RandomForest"] = Pipeline([
    (
        "preprocessor",
        tree_preprocessor
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=600,
            max_depth=16,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
])


# ------------------------------------------------------------
# MODEL 6 — RANDOM FOREST WITHOUT BALANCING
# ------------------------------------------------------------

models["RandomForest_Unbalanced"] = Pipeline([
    (
        "preprocessor",
        tree_preprocessor
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=600,
            max_depth=18,
            min_samples_leaf=4,
            max_features=0.70,
            class_weight=None,
            random_state=777,
            n_jobs=-1
        )
    )
])


# ------------------------------------------------------------
# MODEL 7 — HISTOGRAM GRADIENT BOOSTING
#
# Convert one-hot data to dense first.
# ------------------------------------------------------------

models["HistGradientBoosting"] = Pipeline([
    (
        "preprocessor",
        tree_preprocessor
    ),
    (
        "model",
        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42
        )
    )
])


# ============================================================
# 11. OPTIONAL CATBOOST
# ============================================================

if CATBOOST_AVAILABLE:

    # CatBoost needs categorical columns as strings
    catboost_features = categorical_features.copy()

    catboost_model = CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.035,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )

    # CatBoost can handle categorical variables directly.
    models["CatBoost"] = (
        catboost_model
    )


# ============================================================
# 12. TRAIN MODELS
# ============================================================

print()
print("=" * 70)
print("MODEL TOURNAMENT")
print("=" * 70)

validation_predictions = {}
test_predictions = {}

model_scores = []


for model_name, model in models.items():

    print()
    print("-" * 70)
    print(f"TRAINING: {model_name}")
    print("-" * 70)

    # --------------------------------------------------------
    # CatBoost
    # --------------------------------------------------------

    if (
        CATBOOST_AVAILABLE
        and model_name == "CatBoost"
    ):

        cat_indices = [
            X_train.columns.get_loc(c)
            for c in categorical_features
        ]

        X_train_cb = X_train.copy()
        X_valid_cb = X_valid.copy()
        X_test_cb = X_test.copy()

        for col in categorical_features:

            X_train_cb[col] = (
                X_train_cb[col]
                .fillna("MISSING")
                .astype(str)
            )

            X_valid_cb[col] = (
                X_valid_cb[col]
                .fillna("MISSING")
                .astype(str)
            )

            X_test_cb[col] = (
                X_test_cb[col]
                .fillna("MISSING")
                .astype(str)
            )

        model.fit(
            X_train_cb,
            y_train,
            cat_features=cat_indices
        )

        valid_prob = model.predict_proba(
            X_valid_cb
        )[:, 1]

        test_prob = model.predict_proba(
            X_test_cb
        )[:, 1]

    else:

        model.fit(
            X_train,
            y_train
        )

        valid_prob = model.predict_proba(
            X_valid
        )[:, 1]

        test_prob = model.predict_proba(
            X_test
        )[:, 1]


    score = roc_auc_score(
        y_valid,
        valid_prob
    )

    validation_predictions[
        model_name
    ] = valid_prob

    test_predictions[
        model_name
    ] = test_prob

    model_scores.append(
        {
            "Model": model_name,
            "Validation_AUC": score
        }
    )

    print(
        f"{model_name} validation AUC: "
        f"{score:.6f}"
    )


# ============================================================
# 13. MODEL LEADERBOARD
# ============================================================

scores_df = pd.DataFrame(
    model_scores
).sort_values(
    "Validation_AUC",
    ascending=False
).reset_index(drop=True)

print()
print("=" * 70)
print("MODEL LEADERBOARD")
print("=" * 70)

print(
    scores_df.to_string(
        index=False
    )
)


# ============================================================
# 14. RANK TRANSFORMATION
# ============================================================

def rank_predictions(values):

    return (
        pd.Series(values)
        .rank(
            method="average",
            pct=True
        )
        .values
    )


validation_ranks = {}
test_ranks = {}

for name in validation_predictions:

    validation_ranks[name] = rank_predictions(
        validation_predictions[name]
    )

    test_ranks[name] = rank_predictions(
        test_predictions[name]
    )


# ============================================================
# 15. AUTOMATIC TWO-MODEL RANK BLEND SEARCH
#
# Test every pair of models.
# Search weights from 0.00 to 1.00.
# ============================================================

print()
print("=" * 70)
print("PAIRWISE RANK BLEND SEARCH")
print("=" * 70)

blend_results = []

model_names = list(
    validation_predictions.keys()
)

for i in range(len(model_names)):

    for j in range(i + 1, len(model_names)):

        model_a = model_names[i]
        model_b = model_names[j]

        rank_a = validation_ranks[model_a]
        rank_b = validation_ranks[model_b]

        for weight in np.arange(
            0.0,
            1.01,
            0.025
        ):

            blended = (
                weight * rank_a
                +
                (1.0 - weight) * rank_b
            )

            auc = roc_auc_score(
                y_valid,
                blended
            )

            blend_results.append(
                {
                    "Model_A": model_a,
                    "Model_B": model_b,
                    "Weight_A": weight,
                    "Weight_B": 1.0 - weight,
                    "AUC": auc
                }
            )


blend_df = pd.DataFrame(
    blend_results
).sort_values(
    "AUC",
    ascending=False
).reset_index(drop=True)


print()
print("TOP 20 PAIRWISE RANK BLENDS")
print()

print(
    blend_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 16. TOP-3 MODEL RANK BLEND SEARCH
#
# Use the best 6 models from individual validation.
# Search coarse weights.
# ============================================================

top_models = scores_df.head(
    min(6, len(scores_df))
)["Model"].tolist()

print()
print("=" * 70)
print("TOP-3 RANK BLEND SEARCH")
print("=" * 70)

triple_results = []

for i in range(len(top_models)):

    for j in range(i + 1, len(top_models)):

        for k in range(j + 1, len(top_models)):

            a = top_models[i]
            b = top_models[j]
            c = top_models[k]

            rank_a = validation_ranks[a]
            rank_b = validation_ranks[b]
            rank_c = validation_ranks[c]

            # Coarse grid.
            for wa in np.arange(
                0.0,
                1.01,
                0.05
            ):

                for wb in np.arange(
                    0.0,
                    1.01 - wa,
                    0.05
                ):

                    wc = 1.0 - wa - wb

                    if wc < 0:
                        continue

                    blended = (
                        wa * rank_a
                        +
                        wb * rank_b
                        +
                        wc * rank_c
                    )

                    auc = roc_auc_score(
                        y_valid,
                        blended
                    )

                    triple_results.append(
                        {
                            "Model_A": a,
                            "Model_B": b,
                            "Model_C": c,
                            "Weight_A": wa,
                            "Weight_B": wb,
                            "Weight_C": wc,
                            "AUC": auc
                        }
                    )


triple_df = pd.DataFrame(
    triple_results
).sort_values(
    "AUC",
    ascending=False
).reset_index(drop=True)


print()
print("TOP 20 THREE-MODEL RANK BLENDS")
print()

print(
    triple_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 17. CHOOSE BEST ENSEMBLE
# ============================================================

best_pair = blend_df.iloc[0]

best_triple = triple_df.iloc[0]

print()
print("=" * 70)
print("ENSEMBLE CHAMPIONS")
print("=" * 70)

print()
print("BEST PAIR:")
print(best_pair)

print()
print("BEST TRIPLE:")
print(best_triple)


if best_triple["AUC"] >= best_pair["AUC"]:

    ensemble_type = "TRIPLE"

    champion_models = [
        best_triple["Model_A"],
        best_triple["Model_B"],
        best_triple["Model_C"]
    ]

    champion_weights = [
        best_triple["Weight_A"],
        best_triple["Weight_B"],
        best_triple["Weight_C"]
    ]

    champion_auc = best_triple["AUC"]

else:

    ensemble_type = "PAIR"

    champion_models = [
        best_pair["Model_A"],
        best_pair["Model_B"]
    ]

    champion_weights = [
        best_pair["Weight_A"],
        best_pair["Weight_B"]
    ]

    champion_auc = best_pair["AUC"]


print()
print("=" * 70)
print("SELECTED CHAMPION")
print("=" * 70)

print(f"Type: {ensemble_type}")

for name, weight in zip(
    champion_models,
    champion_weights
):

    print(
        f" - {name}: "
        f"{weight:.3f}"
    )

print(
    f"Validation AUC: "
    f"{champion_auc:.6f}"
)


# ============================================================
# 18. OPTIONAL COMPARE WITH SIMPLE ALL-MODEL BLEND
# ============================================================

print()
print("=" * 70)
print("TESTING ALL-MODEL RANK BLEND")
print("=" * 70)

all_rank_validation = np.mean(
    [
        validation_ranks[name]
        for name in model_names
    ],
    axis=0
)

all_rank_auc = roc_auc_score(
    y_valid,
    all_rank_validation
)

print(
    f"All-model rank blend AUC: "
    f"{all_rank_auc:.6f}"
)


# ============================================================
# 19. BUILD VALIDATION CHAMPION PREDICTION
# ============================================================

champion_validation = np.zeros(
    len(X_valid)
)

for name, weight in zip(
    champion_models,
    champion_weights
):

    champion_validation += (
        weight
        *
        validation_ranks[name]
    )


champion_validation_auc = roc_auc_score(
    y_valid,
    champion_validation
)

print()
print(
    f"Selected champion validation AUC: "
    f"{champion_validation_auc:.6f}"
)


# ============================================================
# 20. RETRAIN CHAMPION MODELS ON FULL DATA
# ============================================================

print()
print("=" * 70)
print("RETRAINING CHAMPION MODELS ON FULL DATA")
print("=" * 70)

full_predictions = []

for name, weight in zip(
    champion_models,
    champion_weights
):

    print()
    print(
        f"Training final model: "
        f"{name}"
    )

    # --------------------------------------------------------
    # Recreate model fresh
    # --------------------------------------------------------

    if name == "LogReg_L2":

        final_model = Pipeline([
            (
                "preprocessor",
                linear_preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    C=0.07,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1500,
                    random_state=42
                )
            )
        ])

    elif name == "LogReg_ElasticNet":

        final_model = Pipeline([
            (
                "preprocessor",
                linear_preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    C=0.08,
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.20,
                    max_iter=1500,
                    random_state=42
                )
            )
        ])

    elif name == "ExtraTrees":

        final_model = Pipeline([
            (
                "preprocessor",
                tree_preprocessor
            ),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=1000,
                    max_depth=None,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ])

    elif name == "ExtraTrees_Deep":

        final_model = Pipeline([
            (
                "preprocessor",
                tree_preprocessor
            ),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=1000,
                    max_depth=18,
                    min_samples_leaf=4,
                    max_features=0.70,
                    class_weight="balanced",
                    random_state=123,
                    n_jobs=-1
                )
            )
        ])

    elif name == "RandomForest":

        final_model = Pipeline([
            (
                "preprocessor",
                tree_preprocessor
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=900,
                    max_depth=16,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ])

    elif name == "RandomForest_Unbalanced":

        final_model = Pipeline([
            (
                "preprocessor",
                tree_preprocessor
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=900,
                    max_depth=18,
                    min_samples_leaf=4,
                    max_features=0.70,
                    class_weight=None,
                    random_state=777,
                    n_jobs=-1
                )
            )
        ])

    elif name == "HistGradientBoosting":

        final_model = Pipeline([
            (
                "preprocessor",
                tree_preprocessor
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=400,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    random_state=42
                )
            )
        ])

    elif name == "CatBoost":

        final_model = CatBoostClassifier(
            iterations=800,
            depth=6,
            learning_rate=0.035,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=5,
            random_seed=42,
            verbose=False,
            allow_writing_files=False
        )

    else:

        raise ValueError(
            f"Unknown model: {name}"
        )


    # --------------------------------------------------------
    # CatBoost
    # --------------------------------------------------------

    if (
        CATBOOST_AVAILABLE
        and name == "CatBoost"
    ):

        X_full_cb = X.copy()
        X_test_cb = X_test.copy()

        cat_indices = [
            X.columns.get_loc(c)
            for c in categorical_features
        ]

        for col in categorical_features:

            X_full_cb[col] = (
                X_full_cb[col]
                .fillna("MISSING")
                .astype(str)
            )

            X_test_cb[col] = (
                X_test_cb[col]
                .fillna("MISSING")
                .astype(str)
            )

        final_model.fit(
            X_full_cb,
            y,
            cat_features=cat_indices
        )

        prediction = final_model.predict_proba(
            X_test_cb
        )[:, 1]

    else:

        final_model.fit(
            X,
            y
        )

        prediction = final_model.predict_proba(
            X_test
        )[:, 1]


    # Rank the final predictions
    ranked = rank_predictions(
        prediction
    )

    full_predictions.append(
        weight * ranked
    )

    print(
        f"Finished {name}"
    )


# ============================================================
# 21. FINAL RANK ENSEMBLE
# ============================================================

final_probabilities = np.sum(
    full_predictions,
    axis=0
)


# ============================================================
# 22. OPTIONAL CALIBRATION
#
# Keep rank blend as the primary submission.
# Convert rank scores to a smooth probability-like value.
# ============================================================

final_probabilities = np.clip(
    final_probabilities,
    0.000001,
    0.999999
)


# ============================================================
# 23. VALIDATE OUTPUT
# ============================================================

if len(final_probabilities) != len(test_raw):

    raise ValueError(
        "Prediction count does not match test rows."
    )

if np.isnan(final_probabilities).any():

    raise ValueError(
        "Final predictions contain NaN."
    )

if np.isinf(final_probabilities).any():

    raise ValueError(
        "Final predictions contain infinity."
    )


# ============================================================
# 24. SAVE SUBMISSION
# ============================================================

OUTPUT_FILE = (
    "submission_exp52_model_tournament.csv"
)

submission = pd.DataFrame({

    ID_COL:
        test_raw[ID_COL],

    TARGET:
        final_probabilities
})

submission.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 25. SAVE TOURNAMENT RESULTS
# ============================================================

scores_df.to_csv(
    "exp52_model_scores.csv",
    index=False
)

blend_df.head(100).to_csv(
    "exp52_pairwise_blends.csv",
    index=False
)

triple_df.head(100).to_csv(
    "exp52_triple_blends.csv",
    index=False
)


# ============================================================
# 26. FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT 52 COMPLETE")
print("=" * 70)

print()
print(f"Saved submission:")
print(f"  {OUTPUT_FILE}")

print()
print(f"Rows    : {len(submission)}")
print(f"Columns : {submission.shape[1]}")

print()
print("Selected ensemble:")
for name, weight in zip(
    champion_models,
    champion_weights
):
    print(
        f"  {name}: "
        f"{weight:.3f}"
    )

print()
print("Validation AUC:")
print(
    f"  Champion: "
    f"{champion_auc:.6f}"
)

print()
print("Prediction summary:")
print(
    submission[TARGET].describe()
)

print()
print("First 10 predictions:")
print(
    submission.head(10)
)

print()
print("=" * 70)
print("BENCHMARKS")
print("=" * 70)

print(
    "Exp 44 Top-3 Linear Blend : 0.65630"
)

print(
    "Exp 47 Pure-Logit Ensemble: 0.65635"
)

print(
    "Personal Best             : 0.65671"
)

print(
    "Exp 46 Multilinear        : 0.65701"
)

print(
    "Exp 51 Ensemble           : 0.65355"
)

print(
    f"Exp 52 Local Champion     : "
    f"{champion_auc:.6f}"
)

print()
print("=" * 70)
print("READY FOR KAGGLE SUBMISSION")
print("=" * 70)