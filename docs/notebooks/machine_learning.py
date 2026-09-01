
# ============================================================
# EXPERIMENT 62 — SEM × SUPERVISED STACKED CLASSIFIER
#
# HARD ROUND 8 GATE
#
# NOTHING IS TRAINED ON THE FULL DATASET UNLESS:
#
#       ROUND 8 ROC-AUC >= 0.6800
#
# Strategy:
#   1. Load Round 8 correctly:
#        rounds_1_7.csv -> training
#        round_8.csv    -> validation
#
#   2. Build multiple DISTINCT model manifolds:
#
#        A. Raw Logistic Regression
#        B. Raw ElasticNet Logistic
#        C. SEM Factor Analysis + Logistic
#        D. SEM PLS + Logistic
#        E. Raw + SEM combined Logistic
#        F. Raw + SEM interaction Logistic
#        G. Combined MLP
#
#   3. Benchmark ALL models on Round 8.
#
#   4. HARD GATE:
#        Only models >= 0.6800 survive.
#
#   5. Create OOF predictions for surviving models.
#
#   6. Train a LogisticRegression meta-classifier on OOF
#      predictions.
#
#   7. Validate the stack on Round 8.
#
#   8. The stack itself must also reach >= 0.6800.
#
#   9. ONLY THEN train on the COMPLETE dataset.
#
#  10. Produce Kaggle submission.
#
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    QuantileTransformer,
    PolynomialFeatures
)

from sklearn.decomposition import FactorAnalysis
from sklearn.cross_decomposition import PLSRegression

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ============================================================
# 0. EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "EXP 62 — SEM × SUPERVISED STACKED CLASSIFIER"

ROUND_NUMBER = 8

REQUIRED_ROUND_AUC = 0.6800

RANDOM_STATE = 42

SEEDS = [
    42,
    101,
    777,
    2024,
    999
]

OOF_FOLDS = 5


print("=" * 60)
print(EXPERIMENT_NAME)
print("=" * 60)

print()
print(f"Required Round {ROUND_NUMBER} ROC-AUC: >= {REQUIRED_ROUND_AUC:.4f}")
print()


# ============================================================
# 1. DIRECTORY DISCOVERY
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()

if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR

ROUND_DIR = CURRENT_DIR / "round_testing"

if not ROUND_DIR.exists():
    ROUND_DIR = DATA_DIR / "round_testing"


print("=" * 60)
print("DIRECTORIES")
print("=" * 60)

print(f"Current directory : {CURRENT_DIR}")
print(f"Dataset directory: {DATA_DIR}")
print(f"Round directory  : {ROUND_DIR}")

print()


# ============================================================
# 2. BASIC CONFIGURATION
# ============================================================

TARGET = "employed_status"
ID_COLUMN = "anonymised_id"


# ============================================================
# 3. MATRIC PARSER
# ============================================================

def parse_matric_band(value):
    """
    Convert strings such as:

        50 - 59 %
        70 - 79 %
        < 40 %
        > 80 %

    into approximate continuous values.
    """

    if pd.isna(value):
        return np.nan

    text = str(value).replace("%", "").strip()

    try:

        if "-" in text:

            parts = text.split("-")

            if len(parts) == 2:

                low = float(parts[0].strip())
                high = float(parts[1].strip())

                return (low + high) / 2.0

        if "<" in text:

            number = float(
                text.replace("<", "").strip()
            )

            return number / 2.0

        if ">" in text:

            number = float(
                text.replace(">", "").strip()
            )

            return number + 5.0

        return float(text)

    except Exception:

        return np.nan


# ============================================================
# 4. GENERAL FEATURE CLEANING
# ============================================================

def base_feature_engineering(df):
    """
    Shared feature engineering.

    This function intentionally does NOT use the target
    except for cleaning the target column itself.
    """

    data = df.copy()

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if TARGET in data.columns:

        data[TARGET] = pd.to_numeric(
            data[TARGET],
            errors="coerce"
        )

        data = data.dropna(
            subset=[TARGET]
        ).copy()

        data[TARGET] = data[TARGET].astype(int)

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    if ID_COLUMN in data.columns:

        data = data.drop(
            columns=[ID_COLUMN]
        )

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

    if "survey_date" in data.columns:

        date_values = pd.to_datetime(
            data["survey_date"],
            errors="coerce"
        )

        data["survey_year"] = date_values.dt.year
        data["survey_month"] = date_values.dt.month
        data["survey_dayofyear"] = date_values.dt.dayofyear
        data["survey_quarter"] = date_values.dt.quarter

        data = data.drop(
            columns=["survey_date"]
        )

    # --------------------------------------------------------
    # Labour-history features
    # --------------------------------------------------------

    if "employed_lag" in data.columns:

        data["is_first_time"] = (
            data["employed_lag"]
            .isna()
            .astype(float)
        )

        data["employed_lag_num"] = pd.to_numeric(
            data["employed_lag"],
            errors="coerce"
        )

    else:

        data["is_first_time"] = 1.0
        data["employed_lag_num"] = -1.0

    if "tenure_lag" in data.columns:

        tenure = pd.to_numeric(
            data["tenure_lag"],
            errors="coerce"
        )

        data["tenure_lag_log"] = np.log1p(
            tenure.fillna(0).clip(lower=0)
        )

        data["tenure_lag_is_na"] = (
            tenure.isna()
            .astype(float)
        )

    else:

        data["tenure_lag_log"] = 0.0
        data["tenure_lag_is_na"] = 1.0

    if "days_since_last_obs" in data.columns:

        days = pd.to_numeric(
            data["days_since_last_obs"],
            errors="coerce"
        )

        data["days_since_obs_log"] = np.log1p(
            days.fillna(0).clip(lower=0)
        )

        data["days_since_obs_is_na"] = (
            days.isna()
            .astype(float)
        )

    else:

        data["days_since_obs_log"] = 0.0
        data["days_since_obs_is_na"] = 1.0

    # --------------------------------------------------------
    # Matric variables
    # --------------------------------------------------------

    matric_columns = [
        "matric_englishhome",
        "matric_englishadd",
        "matric_mathpure",
        "matric_physicalscience",
        "matric_mathlit"
    ]

    for column in matric_columns:

        if column in data.columns:

            data[f"{column}_score"] = (
                data[column]
                .apply(parse_matric_band)
            )

            data[f"{column}_is_na"] = (
                data[column]
                .isna()
                .astype(float)
            )

    score_columns = [
        f"{c}_score"
        for c in matric_columns
        if f"{c}_score" in data.columns
    ]

    if score_columns:

        data["matric_subject_count"] = (
            data[score_columns]
            .notna()
            .sum(axis=1)
        )

        data["matric_average"] = (
            data[score_columns]
            .mean(axis=1)
        )

        data["matric_average_is_na"] = (
            data["matric_average"]
            .isna()
            .astype(float)
        )

        data["matric_best_score"] = (
            data[score_columns]
            .max(axis=1)
        )

    # --------------------------------------------------------
    # School quintile
    # --------------------------------------------------------

    if "school_quintile" in data.columns:

        data["school_quintile"] = pd.to_numeric(
            data["school_quintile"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Work readiness
    # --------------------------------------------------------

    if "work_readiness_score" in data.columns:

        work_ready = pd.to_numeric(
            data["work_readiness_score"],
            errors="coerce"
        )

        data["work_readiness_is_na"] = (
            work_ready.isna()
            .astype(float)
        )

        data["work_readiness_score"] = work_ready

    # --------------------------------------------------------
    # Tertiary record
    # --------------------------------------------------------

    if "institution_type" in data.columns:

        data["has_tertiary_record"] = (
            data["institution_type"]
            .notna()
            .astype(float)
        )

    # --------------------------------------------------------
    # SETA
    # --------------------------------------------------------

    if "seta" in data.columns:

        data["has_seta_record"] = (
            data["seta"]
            .notna()
            .astype(float)
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in data.columns:

        if column == TARGET:
            continue

        if data[column].dtype == "object":

            converted = pd.to_numeric(
                data[column],
                errors="coerce"
            )

            non_missing_original = (
                data[column]
                .notna()
                .sum()
            )

            non_missing_converted = (
                converted
                .notna()
                .sum()
            )

            if (
                non_missing_original > 0
                and
                non_missing_converted
                >= 0.60 * non_missing_original
            ):

                data[column] = converted

    return data


# ============================================================
# 5. SAFE COMMON FEATURE ALIGNMENT
# ============================================================

def align_features(train_df, test_df):

    common = [
        column
        for column in train_df.columns
        if column in test_df.columns
        and column != TARGET
    ]

    train_x = train_df[common].copy()
    test_x = test_df[common].copy()

    # --------------------------------------------------------
    # Remove very high-cardinality categorical columns.
    # --------------------------------------------------------

    categorical = train_x.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    high_cardinality = [
        column
        for column in categorical
        if train_x[column]
        .nunique(dropna=True) > 100
    ]

    if high_cardinality:

        print(
            "Dropping high-cardinality columns:",
            high_cardinality
        )

        train_x = train_x.drop(
            columns=high_cardinality
        )

        test_x = test_x.drop(
            columns=high_cardinality
        )

    return train_x, test_x


# ============================================================
# 6. PREPROCESSOR
# ============================================================

def build_preprocessor(
    numerical_columns,
    categorical_columns,
    quantile=False
):

    transformers = []

    # --------------------------------------------------------
    # Numerical
    # --------------------------------------------------------

    if numerical_columns:

        if quantile:

            scaler = QuantileTransformer(
                output_distribution="normal",
                random_state=RANDOM_STATE
            )

        else:

            scaler = StandardScaler()

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    scaler
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numerical_columns
            )
        )

    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[
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
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    return ColumnTransformer(
        transformers=transformers
    )


# ============================================================
# 7. CLASSIFICATION MODEL FACTORIES
# ============================================================

def raw_logistic():

    return LogisticRegression(
        C=0.08,
        penalty="l2",
        solver="lbfgs",
        max_iter=1500,
        random_state=RANDOM_STATE
    )


def raw_elastic():

    return LogisticRegression(
        C=0.10,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.15,
        max_iter=1500,
        random_state=RANDOM_STATE
    )


def sem_logistic():

    return LogisticRegression(
        C=0.10,
        penalty="l2",
        solver="lbfgs",
        max_iter=1500,
        random_state=RANDOM_STATE
    )


def combined_logistic():

    return LogisticRegression(
        C=0.07,
        penalty="l2",
        solver="lbfgs",
        max_iter=1500,
        random_state=RANDOM_STATE
    )


def interaction_logistic():

    return LogisticRegression(
        C=0.05,
        penalty="l2",
        solver="lbfgs",
        max_iter=1500,
        random_state=RANDOM_STATE
    )


def mlp_model(seed):

    return MLPClassifier(
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


# ============================================================
# 8. SEM FEATURE GENERATION
# ============================================================

def build_sem_features(
    train_df,
    test_df
):

    train = train_df.copy()
    test = test_df.copy()

    # --------------------------------------------------------
    # Academic block
    # --------------------------------------------------------

    academic_columns = [
        c for c in train.columns
        if "_score" in c
    ]

    if len(academic_columns) >= 2:

        imputer = SimpleImputer(
            strategy="median"
        )

        scaler = StandardScaler()

        train_academic = scaler.fit_transform(
            imputer.fit_transform(
                train[academic_columns]
            )
        )

        test_academic = scaler.transform(
            imputer.transform(
                test[academic_columns]
            )
        )

        n_components = min(
            2,
            len(academic_columns)
        )

        fa = FactorAnalysis(
            n_components=n_components,
            random_state=RANDOM_STATE
        )

        train_factor = fa.fit_transform(
            train_academic
        )

        test_factor = fa.transform(
            test_academic
        )

        for i in range(n_components):

            train[
                f"latent_academic_{i+1}"
            ] = train_factor[:, i]

            test[
                f"latent_academic_{i+1}"
            ] = test_factor[:, i]

    # --------------------------------------------------------
    # Labour block
    # --------------------------------------------------------

    labour_columns = [
        c
        for c in [
            "employed_lag_num",
            "tenure_lag_log",
            "days_since_obs_log",
            "is_first_time"
        ]
        if c in train.columns
    ]

    if len(labour_columns) >= 2:

        imputer = SimpleImputer(
            strategy="median"
        )

        scaler = StandardScaler()

        train_labour = scaler.fit_transform(
            imputer.fit_transform(
                train[labour_columns]
            )
        )

        test_labour = scaler.transform(
            imputer.transform(
                test[labour_columns]
            )
        )

        n_components = min(
            2,
            len(labour_columns)
        )

        fa = FactorAnalysis(
            n_components=n_components,
            random_state=RANDOM_STATE
        )

        train_factor = fa.fit_transform(
            train_labour
        )

        test_factor = fa.transform(
            test_labour
        )

        for i in range(n_components):

            train[
                f"latent_labour_{i+1}"
            ] = train_factor[:, i]

            test[
                f"latent_labour_{i+1}"
            ] = test_factor[:, i]

    # --------------------------------------------------------
    # Socioeconomic block
    # --------------------------------------------------------

    socio_columns = [
        c
        for c in [
            "school_quintile",
            "work_readiness_score",
            "age"
        ]
        if c in train.columns
    ]

    if len(socio_columns) >= 2:

        imputer = SimpleImputer(
            strategy="median"
        )

        scaler = StandardScaler()

        train_socio = scaler.fit_transform(
            imputer.fit_transform(
                train[socio_columns]
            )
        )

        test_socio = scaler.transform(
            imputer.transform(
                test[socio_columns]
            )
        )

        n_components = min(
            2,
            len(socio_columns)
        )

        fa = FactorAnalysis(
            n_components=n_components,
            random_state=RANDOM_STATE
        )

        train_factor = fa.fit_transform(
            train_socio
        )

        test_factor = fa.transform(
            test_socio
        )

        for i in range(n_components):

            train[
                f"latent_socio_{i+1}"
            ] = train_factor[:, i]

            test[
                f"latent_socio_{i+1}"
            ] = test_factor[:, i]

    # --------------------------------------------------------
    # Supervised PLS block
    #
    # IMPORTANT:
    # This is fitted using TRAIN ONLY.
    # --------------------------------------------------------

    pls_columns = []

    for column in [
        *academic_columns,
        *labour_columns,
        *socio_columns
    ]:

        if column in train.columns:
            pls_columns.append(column)

    pls_columns = list(dict.fromkeys(pls_columns))

    if (
        len(pls_columns) >= 3
        and TARGET in train.columns
    ):

        imputer = SimpleImputer(
            strategy="median"
        )

        train_pls = imputer.fit_transform(
            train[pls_columns]
        )

        test_pls = imputer.transform(
            test[pls_columns]
        )

        n_components = min(
            2,
            len(pls_columns) - 1,
            len(train) - 1
        )

        if n_components >= 1:

            pls = PLSRegression(
                n_components=n_components,
                scale=True,
                max_iter=500
            )

            pls.fit(
                train_pls,
                train[TARGET].values
            )

            train_latent = pls.transform(
                train_pls
            )

            test_latent = pls.transform(
                test_pls
            )

            for i in range(n_components):

                train[
                    f"pls_latent_{i+1}"
                ] = train_latent[:, i]

                test[
                    f"pls_latent_{i+1}"
                ] = test_latent[:, i]

    return train, test


# ============================================================
# 9. BUILD SEM DATASET
# ============================================================

def prepare_sem_dataset(
    train_df,
    test_df
):

    train_sem, test_sem = build_sem_features(
        train_df,
        test_df
    )

    train_x, test_x = align_features(
        train_sem,
        test_sem
    )

    return (
        train_x,
        test_x,
        train_sem[TARGET].reset_index(drop=True)
    )


# ============================================================
# 10. PREPARE RAW DATA
# ============================================================

print("=" * 60)
print("LOADING ROUND 8")
print("=" * 60)


ROUND_8_DIR = ROUND_DIR / "round_8"


if not ROUND_8_DIR.exists():

    raise RuntimeError(
        f"Round 8 directory does not exist:\n"
        f"{ROUND_8_DIR}"
    )


# ------------------------------------------------------------
# IMPORTANT:
#
# Your actual files are:
#
#     rounds_1_7.csv
#     round_8.csv
#
# We explicitly use those names.
# ------------------------------------------------------------

round_train_file = (
    ROUND_8_DIR / "rounds_1_7.csv"
)

round_test_file = (
    ROUND_8_DIR / "round_8.csv"
)


if not round_train_file.exists():

    raise RuntimeError(
        f"Missing Round 8 training file:\n"
        f"{round_train_file}"
    )


if not round_test_file.exists():

    raise RuntimeError(
        f"Missing Round 8 validation file:\n"
        f"{round_test_file}"
    )


print(
    f"Round training file: {round_train_file}"
)

print(
    f"Round validation file: {round_test_file}"
)

print()


round_train_raw = pd.read_csv(
    round_train_file
)

round_test_raw = pd.read_csv(
    round_test_file
)


print(
    f"Round train rows: {len(round_train_raw)}"
)

print(
    f"Round 8 rows     : {len(round_test_raw)}"
)

print()


if TARGET not in round_train_raw.columns:

    raise RuntimeError(
        f"{TARGET} is missing from Round training data."
    )


if TARGET not in round_test_raw.columns:

    raise RuntimeError(
        f"{TARGET} is missing from Round 8 validation data."
    )


# ============================================================
# 11. CLEAN ROUND DATA
# ============================================================

round_train_clean = base_feature_engineering(
    round_train_raw
)

round_test_clean = base_feature_engineering(
    round_test_raw
)


# ============================================================
# 12. ALIGN RAW MANIFOLD
# ============================================================

X_round_raw, X_round8_raw = align_features(
    round_train_clean,
    round_test_clean
)

y_round = (
    round_train_clean[TARGET]
    .reset_index(drop=True)
)

y_round8 = (
    round_test_clean[TARGET]
    .reset_index(drop=True)
)


print(
    f"Raw manifold: {X_round_raw.shape[1]} features"
)

print()


# ============================================================
# 13. ALIGN SEM MANIFOLD
# ============================================================

print("=" * 60)
print("CONSTRUCTING SEM MANIFOLD")
print("=" * 60)


X_round_sem, X_round8_sem, y_sem = (
    prepare_sem_dataset(
        round_train_clean,
        round_test_clean
    )
)


print(
    f"SEM manifold: {X_round_sem.shape[1]} features"
)

print()


# ============================================================
# 14. FEATURE TYPE DISCOVERY
# ============================================================

def get_feature_types(X):

    categorical = X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    numerical = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    return numerical, categorical


# ============================================================
# 15. FIT RAW LOGISTIC
# ============================================================

def fit_raw_logistic(
    X_train,
    y_train,
    X_test
):

    num_cols, cat_cols = get_feature_types(
        X_train
    )

    preprocessor = build_preprocessor(
        num_cols,
        cat_cols,
        quantile=False
    )

    pipe = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                raw_logistic()
            )
        ]
    )

    pipe.fit(
        X_train,
        y_train
    )

    return pipe.predict_proba(
        X_test
    )[:, 1]


# ============================================================
# 16. FIT RAW ELASTICNET
# ============================================================

def fit_raw_elastic(
    X_train,
    y_train,
    X_test
):

    num_cols, cat_cols = get_feature_types(
        X_train
    )

    preprocessor = build_preprocessor(
        num_cols,
        cat_cols,
        quantile=False
    )

    pipe = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                raw_elastic()
            )
        ]
    )

    pipe.fit(
        X_train,
        y_train
    )

    return pipe.predict_proba(
        X_test
    )[:, 1]


# ============================================================
# 17. FIT SEM LOGISTIC
# ============================================================

def fit_sem_logistic(
    X_train,
    y_train,
    X_test
):

    num_cols, cat_cols = get_feature_types(
        X_train
    )

    preprocessor = build_preprocessor(
        num_cols,
        cat_cols,
        quantile=True
    )

    pipe = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                sem_logistic()
            )
        ]
    )

    pipe.fit(
        X_train,
        y_train
    )

    return pipe.predict_proba(
        X_test
    )[:, 1]


# ============================================================
# 18. CREATE COMBINED MANIFOLD
# ============================================================

def create_combined_manifold(
    raw_train,
    raw_test,
    sem_train,
    sem_test
):

    combined_train = raw_train.copy()
    combined_test = raw_test.copy()

    sem_only_columns = [
        c
        for c in sem_train.columns
        if (
            c.startswith("latent_")
            or c.startswith("pls_latent_")
        )
    ]

    for column in sem_only_columns:

        if column in sem_train.columns:

            combined_train[column] = (
                sem_train[column].values
            )

            combined_test[column] = (
                sem_test[column].values
            )

    return combined_train, combined_test


X_round_combined, X_round8_combined = (
    create_combined_manifold(
        X_round_raw,
        X_round8_raw,
        X_round_sem,
        X_round8_sem
    )
)


print(
    f"Combined manifold: "
    f"{X_round_combined.shape[1]} features"
)


# ============================================================
# 19. COMBINED LOGISTIC
# ============================================================

def fit_combined_logistic(
    X_train,
    y_train,
    X_test
):

    num_cols, cat_cols = get_feature_types(
        X_train
    )

    preprocessor = build_preprocessor(
        num_cols,
        cat_cols,
        quantile=False
    )

    pipe = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                combined_logistic()
            )
        ]
    )

    pipe.fit(
        X_train,
        y_train
    )

    return pipe.predict_proba(
        X_test
    )[:, 1]


# ============================================================
# 20. INTERACTION MANIFOLD
# ============================================================

def create_interaction_manifold(
    train_df,
    test_df
):

    train = train_df.copy()
    test = test_df.copy()

    important_pairs = [
        (
            "is_first_time",
            "employed_lag_num"
        ),
        (
            "employed_lag_num",
            "tenure_lag_log"
        ),
        (
            "employed_lag_num",
            "days_since_obs_log"
        ),
        (
            "tenure_lag_log",
            "days_since_obs_log"
        ),
        (
            "work_readiness_score",
            "school_quintile"
        ),
        (
            "age",
            "work_readiness_score"
        ),
        (
            "matric_average",
            "work_readiness_score"
        ),
        (
            "matric_average",
            "school_quintile"
        )
    ]

    for left, right in important_pairs:

        if (
            left in train.columns
            and right in train.columns
        ):

            train[
                f"interaction_{left}_{right}"
            ] = (
                pd.to_numeric(
                    train[left],
                    errors="coerce"
                )
                *
                pd.to_numeric(
                    train[right],
                    errors="coerce"
                )
            )

            test[
                f"interaction_{left}_{right}"
            ] = (
                pd.to_numeric(
                    test[left],
                    errors="coerce"
                )
                *
                pd.to_numeric(
                    test[right],
                    errors="coerce"
                )
            )

    return train, test


X_round_interaction, X_round8_interaction = (
    create_interaction_manifold(
        X_round_combined,
        X_round8_combined
    )
)


# ============================================================
# 21. INTERACTION LOGISTIC
# ============================================================

def fit_interaction_logistic(
    X_train,
    y_train,
    X_test
):

    num_cols, cat_cols = get_feature_types(
        X_train
    )

    preprocessor = build_preprocessor(
        num_cols,
        cat_cols,
        quantile=True
    )

    pipe = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                interaction_logistic()
            )
        ]
    )

    pipe.fit(
        X_train,
        y_train
    )

    return pipe.predict_proba(
        X_test
    )[:, 1]


# ============================================================
# 22. MLP
# ============================================================

def fit_mlp(
    X_train,
    y_train,
    X_test,
    seed=42
):

    num_cols, cat_cols = get_feature_types(
        X_train
    )

    preprocessor = build_preprocessor(
        num_cols,
        cat_cols,
        quantile=True
    )

    pipe = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                mlp_model(seed)
            )
        ]
    )

    pipe.fit(
        X_train,
        y_train
    )

    return pipe.predict_proba(
        X_test
    )[:, 1]


# ============================================================
# 23. ROUND 8 MODEL TOURNAMENT
# ============================================================

print()
print("=" * 60)
print("ROUND 8 MODEL TOURNAMENT")
print("=" * 60)

print()
print(
    f"HARD GATE: ROC-AUC >= {REQUIRED_ROUND_AUC:.4f}"
)
print()


round_predictions = {}
round_scores = {}


# ------------------------------------------------------------
# MODEL A — RAW LOGISTIC
# ------------------------------------------------------------

print("Testing Raw Logistic...")

pred = fit_raw_logistic(
    X_round_raw,
    y_round,
    X_round8_raw
)

score = roc_auc_score(
    y_round8,
    pred
)

round_predictions["Raw_Logistic"] = pred
round_scores["Raw_Logistic"] = score

print(
    f"Raw Logistic                  : {score:.6f}"
)


# ------------------------------------------------------------
# MODEL B — RAW ELASTICNET
# ------------------------------------------------------------

print("Testing Raw ElasticNet...")

pred = fit_raw_elastic(
    X_round_raw,
    y_round,
    X_round8_raw
)

score = roc_auc_score(
    y_round8,
    pred
)

round_predictions["Raw_ElasticNet"] = pred
round_scores["Raw_ElasticNet"] = score

print(
    f"Raw ElasticNet                : {score:.6f}"
)


# ------------------------------------------------------------
# MODEL C — SEM LATENT
# ------------------------------------------------------------

print("Testing SEM Latent Logistic...")

pred = fit_sem_logistic(
    X_round_sem,
    y_sem,
    X_round8_sem
)

score = roc_auc_score(
    y_round8,
    pred
)

round_predictions["SEM_Logistic"] = pred
round_scores["SEM_Logistic"] = score

print(
    f"SEM Latent Logistic           : {score:.6f}"
)


# ------------------------------------------------------------
# MODEL D — RAW + SEM
# ------------------------------------------------------------

print("Testing Raw + SEM Logistic...")

pred = fit_combined_logistic(
    X_round_combined,
    y_round,
    X_round8_combined
)

score = roc_auc_score(
    y_round8,
    pred
)

round_predictions["Raw_SEM_Logistic"] = pred
round_scores["Raw_SEM_Logistic"] = score

print(
    f"Raw + SEM Logistic             : {score:.6f}"
)


# ------------------------------------------------------------
# MODEL E — INTERACTION
# ------------------------------------------------------------

print("Testing Raw + SEM Interaction Logistic...")

pred = fit_interaction_logistic(
    X_round_interaction,
    y_round,
    X_round8_interaction
)

score = roc_auc_score(
    y_round8,
    pred
)

round_predictions["Interaction_Logistic"] = pred
round_scores["Interaction_Logistic"] = score

print(
    f"Raw + SEM Interaction         : {score:.6f}"
)


# ------------------------------------------------------------
# MODEL F — MLP
# ------------------------------------------------------------

print("Testing Combined MLP...")

pred = fit_mlp(
    X_round_combined,
    y_round,
    X_round8_combined,
    seed=42
)

score = roc_auc_score(
    y_round8,
    pred
)

round_predictions["Combined_MLP"] = pred
round_scores["Combined_MLP"] = score

print(
    f"Combined MLP                  : {score:.6f}"
)


# ============================================================
# 24. TOURNAMENT TABLE
# ============================================================

tournament = pd.DataFrame(
    [
        {
            "Model": name,
            "Round_8_AUC": score,
            "PASS": score >= REQUIRED_ROUND_AUC
        }
        for name, score
        in round_scores.items()
    ]
)

tournament = tournament.sort_values(
    "Round_8_AUC",
    ascending=False
).reset_index(drop=True)


print()
print("=" * 60)
print("ROUND 8 LEADERBOARD")
print("=" * 60)

print(
    tournament.to_string(
        index=False
    )
)


# ============================================================
# 25. HARD GATE
# ============================================================

passing_models = [
    name
    for name, score
    in round_scores.items()
    if score >= REQUIRED_ROUND_AUC
]


print()
print("=" * 60)
print("ROUND 8 HARD GATE")
print("=" * 60)

print(
    f"Required AUC : {REQUIRED_ROUND_AUC:.4f}"
)

print(
    f"Passing models: {len(passing_models)}"
)

for model_name in passing_models:

    print(
        f"  PASS -> {model_name}: "
        f"{round_scores[model_name]:.6f}"
    )


# ------------------------------------------------------------
# CRITICAL SAFETY STOP
# ------------------------------------------------------------

if len(passing_models) == 0:

    print()
    print("=" * 60)
    print("❌ EXPERIMENT STOPPED")
    print("=" * 60)

    print()
    print(
        "NO MODEL REACHED THE REQUIRED "
        f"ROUND 8 AUC OF {REQUIRED_ROUND_AUC:.4f}."
    )

    print()
    print(
        "The full dataset WILL NOT be used."
    )

    print()
    print(
        "Improve the model and rerun."
    )

    raise SystemExit(
        "ROUND 8 GATE FAILED."
    )


# ============================================================
# 26. OOF STACKING ENGINE
# ============================================================

print()
print("=" * 60)
print("BUILDING OOF STACKING FEATURES")
print("=" * 60)

print()
print(
    "Passing models:"
)

for model_name in passing_models:

    print(
        f"  - {model_name}"
    )

print()


# ------------------------------------------------------------
# We create OOF predictions for each passing architecture.
#
# The important part is that each training prediction is made
# by a model that DID NOT train on that observation.
#
# This gives the meta-classifier a realistic training signal.
# ------------------------------------------------------------

def generate_oof_predictions(
    model_name,
    X_raw,
    X_sem,
    X_combined,
    X_interaction,
    y,
    folds=5
):

    oof = np.zeros(
        len(y),
        dtype=float
    )

    skf = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    for fold_number, (
        train_idx,
        valid_idx
    ) in enumerate(
        skf.split(
            X_raw,
            y
        ),
        start=1
    ):

        print(
            f"    {model_name} "
            f"fold {fold_number}/{folds}"
        )

        y_train = y.iloc[
            train_idx
        ]

        # ----------------------------------------------------
        # RAW LOGISTIC
        # ----------------------------------------------------

        if model_name == "Raw_Logistic":

            Xtr = X_raw.iloc[
                train_idx
            ]

            Xva = X_raw.iloc[
                valid_idx
            ]

            oof[valid_idx] = fit_raw_logistic(
                Xtr,
                y_train,
                Xva
            )

        # ----------------------------------------------------
        # RAW ELASTIC
        # ----------------------------------------------------

        elif model_name == "Raw_ElasticNet":

            Xtr = X_raw.iloc[
                train_idx
            ]

            Xva = X_raw.iloc[
                valid_idx
            ]

            oof[valid_idx] = fit_raw_elastic(
                Xtr,
                y_train,
                Xva
            )

        # ----------------------------------------------------
        # SEM
        #
        # IMPORTANT:
        # Rebuild SEM within each fold so that PLS does not
        # see validation targets.
        # ----------------------------------------------------

        elif model_name == "SEM_Logistic":

            raw_train_fold = (
                round_train_clean
                .iloc[train_idx]
                .copy()
            )

            raw_valid_fold = (
                round_train_clean
                .iloc[valid_idx]
                .copy()
            )

            sem_tr, sem_va = (
                build_sem_features(
                    raw_train_fold,
                    raw_valid_fold
                )
            )

            sem_train_x, sem_valid_x = (
                align_features(
                    sem_tr,
                    sem_va
                )
            )

            oof[valid_idx] = fit_sem_logistic(
                sem_train_x,
                y_train,
                sem_valid_x
            )

        # ----------------------------------------------------
        # RAW + SEM
        # ----------------------------------------------------

        elif model_name == "Raw_SEM_Logistic":

            raw_train_fold = (
                round_train_clean
                .iloc[train_idx]
                .copy()
            )

            raw_valid_fold = (
                round_train_clean
                .iloc[valid_idx]
                .copy()
            )

            sem_tr, sem_va = (
                build_sem_features(
                    raw_train_fold,
                    raw_valid_fold
                )
            )

            raw_tr, raw_va = align_features(
                raw_train_fold,
                raw_valid_fold
            )

            sem_tr_x, sem_va_x = align_features(
                sem_tr,
                sem_va
            )

            comb_tr, comb_va = (
                create_combined_manifold(
                    raw_tr,
                    raw_va,
                    sem_tr_x,
                    sem_va_x
                )
            )

            oof[valid_idx] = fit_combined_logistic(
                comb_tr,
                y_train,
                comb_va
            )

        # ----------------------------------------------------
        # INTERACTION
        # ----------------------------------------------------

        elif model_name == "Interaction_Logistic":

            raw_train_fold = (
                round_train_clean
                .iloc[train_idx]
                .copy()
            )

            raw_valid_fold = (
                round_train_clean
                .iloc[valid_idx]
                .copy()
            )

            sem_tr, sem_va = (
                build_sem_features(
                    raw_train_fold,
                    raw_valid_fold
                )
            )

            raw_tr, raw_va = align_features(
                raw_train_fold,
                raw_valid_fold
            )

            sem_tr_x, sem_va_x = align_features(
                sem_tr,
                sem_va
            )

            comb_tr, comb_va = (
                create_combined_manifold(
                    raw_tr,
                    raw_va,
                    sem_tr_x,
                    sem_va_x
                )
            )

            int_tr, int_va = (
                create_interaction_manifold(
                    comb_tr,
                    comb_va
                )
            )

            oof[valid_idx] = (
                fit_interaction_logistic(
                    int_tr,
                    y_train,
                    int_va
                )
            )

        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        elif model_name == "Combined_MLP":

            Xtr = X_combined.iloc[
                train_idx
            ]

            Xva = X_combined.iloc[
                valid_idx
            ]

            oof[valid_idx] = fit_mlp(
                Xtr,
                y_train,
                Xva,
                seed=RANDOM_STATE
            )

        else:

            raise ValueError(
                f"Unknown model: {model_name}"
            )

    return oof


# ============================================================
# 27. CREATE OOF MATRIX
# ============================================================

oof_columns = []

for model_name in passing_models:

    print()
    print(
        f"Generating OOF predictions: "
        f"{model_name}"
    )

    oof_prediction = generate_oof_predictions(
        model_name=model_name,
        X_raw=X_round_raw,
        X_sem=X_round_sem,
        X_combined=X_round_combined,
        X_interaction=X_round_interaction,
        y=y_round,
        folds=OOF_FOLDS
    )

    oof_columns.append(
        oof_prediction
    )


oof_matrix = np.column_stack(
    oof_columns
)


print()
print(
    f"OOF matrix shape: "
    f"{oof_matrix.shape}"
)


# ============================================================
# 28. TRAIN META CLASSIFIER
# ============================================================

print()
print("=" * 60)
print("TRAINING META CLASSIFIER")
print("=" * 60)


# ------------------------------------------------------------
# Convert base probabilities to log-odds.
#
# This lets the meta-classifier work with the confidence
# structure of the individual models rather than treating
# probabilities as simple raw numbers.
# ------------------------------------------------------------

oof_clipped = np.clip(
    oof_matrix,
    1e-6,
    1 - 1e-6
)

oof_logits = np.log(
    oof_clipped
    /
    (1.0 - oof_clipped)
)


meta_model = LogisticRegression(
    C=0.20,
    penalty="l2",
    solver="lbfgs",
    max_iter=2000,
    random_state=RANDOM_STATE
)


meta_model.fit(
    oof_logits,
    y_round
)


print(
    "Meta classifier trained."
)


# ============================================================
# 29. BUILD ROUND 8 STACK FEATURES
# ============================================================

round8_base_predictions = []

for model_name in passing_models:

    print(
        f"Preparing Round 8 stack prediction: "
        f"{model_name}"
    )

    round8_base_predictions.append(
        round_predictions[model_name]
    )


round8_matrix = np.column_stack(
    round8_base_predictions
)


round8_clipped = np.clip(
    round8_matrix,
    1e-6,
    1 - 1e-6
)


round8_logits = np.log(
    round8_clipped
    /
    (1.0 - round8_clipped)
)


stack_round8_prediction = (
    meta_model
    .predict_proba(
        round8_logits
    )[:, 1]
)


stack_round8_auc = roc_auc_score(
    y_round8,
    stack_round8_prediction
)


print()
print("=" * 60)
print("ROUND 8 STACK RESULT")
print("=" * 60)

print(
    f"Stack ROC-AUC: "
    f"{stack_round8_auc:.6f}"
)

print(
    f"Required     : "
    f"{REQUIRED_ROUND_AUC:.6f}"
)


# ============================================================
# 30. SECOND HARD GATE — STACK
# ============================================================

if stack_round8_auc < REQUIRED_ROUND_AUC:

    print()
    print("=" * 60)
    print("❌ STACK GATE FAILED")
    print("=" * 60)

    print()
    print(
        "At least one individual model passed,"
    )

    print(
        "but the stacked classifier did NOT "
        "reach the required Round 8 AUC."
    )

    print()
    print(
        "The full dataset WILL NOT be used."
    )

    raise SystemExit(
        "ROUND 8 STACK GATE FAILED."
    )


print()
print("=" * 60)
print("✅ ROUND 8 STACK GATE PASSED")
print("=" * 60)

print()
print(
    f"Round 8 Stack AUC = "
    f"{stack_round8_auc:.6f}"
)

print(
    f"Required          = "
    f"{REQUIRED_ROUND_AUC:.6f}"
)

print()


# ============================================================
# 31. ONLY NOW LOAD COMPLETE DATASET
# ============================================================

print("=" * 60)
print("FULL DATASET GATE PASSED")
print("=" * 60)

print()
print(
    "The complete training dataset is now "
    "allowed to be used."
)

print()


train_file = DATA_DIR / "train.csv"
test_file = DATA_DIR / "test.csv"


if not train_file.exists():

    raise RuntimeError(
        f"Missing training file:\n{train_file}"
    )


if not test_file.exists():

    raise RuntimeError(
        f"Missing test file:\n{test_file}"
    )


full_train_raw = pd.read_csv(
    train_file
)

full_test_raw = pd.read_csv(
    test_file
)


full_train_clean = base_feature_engineering(
    full_train_raw
)

full_test_clean = base_feature_engineering(
    full_test_raw
)


full_test_ids = full_test_raw[
    ID_COLUMN
].copy()


y_full = (
    full_train_clean[TARGET]
    .reset_index(drop=True)
)


# ============================================================
# 32. BUILD FULL RAW MANIFOLD
# ============================================================

X_full_raw, X_full_test_raw = align_features(
    full_train_clean,
    full_test_clean
)


# ============================================================
# 33. BUILD FULL SEM MANIFOLD
# ============================================================

print()
print("=" * 60)
print("BUILDING FULL SEM MANIFOLD")
print("=" * 60)


X_full_sem, X_full_test_sem, y_full_sem = (
    prepare_sem_dataset(
        full_train_clean,
        full_test_clean
    )
)


# ============================================================
# 34. BUILD FULL COMBINED MANIFOLD
# ============================================================

X_full_combined, X_full_test_combined = (
    create_combined_manifold(
        X_full_raw,
        X_full_test_raw,
        X_full_sem,
        X_full_test_sem
    )
)


# ============================================================
# 35. BUILD FULL INTERACTION MANIFOLD
# ============================================================

X_full_interaction, X_full_test_interaction = (
    create_interaction_manifold(
        X_full_combined,
        X_full_test_combined
    )
)


print()
print(
    f"Full raw features        : "
    f"{X_full_raw.shape[1]}"
)

print(
    f"Full SEM features        : "
    f"{X_full_sem.shape[1]}"
)

print(
    f"Full combined features   : "
    f"{X_full_combined.shape[1]}"
)

print(
    f"Full interaction features: "
    f"{X_full_interaction.shape[1]}"
)


# ============================================================
# 36. TRAIN PASSING BASE MODELS ON FULL DATA
# ============================================================

print()
print("=" * 60)
print("TRAINING PASSING MODELS ON FULL DATASET")
print("=" * 60)


full_test_predictions = []


for model_name in passing_models:

    print()
    print(
        f"Training full model: "
        f"{model_name}"
    )

    if model_name == "Raw_Logistic":

        pred = fit_raw_logistic(
            X_full_raw,
            y_full,
            X_full_test_raw
        )

    elif model_name == "Raw_ElasticNet":

        pred = fit_raw_elastic(
            X_full_raw,
            y_full,
            X_full_test_raw
        )

    elif model_name == "SEM_Logistic":

        pred = fit_sem_logistic(
            X_full_sem,
            y_full_sem,
            X_full_test_sem
        )

    elif model_name == "Raw_SEM_Logistic":

        pred = fit_combined_logistic(
            X_full_combined,
            y_full,
            X_full_test_combined
        )

    elif model_name == "Interaction_Logistic":

        pred = fit_interaction_logistic(
            X_full_interaction,
            y_full,
            X_full_test_interaction
        )

    elif model_name == "Combined_MLP":

        # ----------------------------------------------------
        # Multi-seed MLP for stability.
        # ----------------------------------------------------

        seed_predictions = []

        for seed in SEEDS:

            print(
                f"    MLP seed {seed}"
            )

            seed_predictions.append(
                fit_mlp(
                    X_full_combined,
                    y_full,
                    X_full_test_combined,
                    seed=seed
                )
            )

        pred = np.mean(
            seed_predictions,
            axis=0
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    pred = np.asarray(
        pred,
        dtype=float
    )

    if np.isnan(pred).any():

        raise ValueError(
            f"{model_name} produced NaN predictions."
        )

    full_test_predictions.append(
        pred
    )


# ============================================================
# 37. FULL STACK PREDICTION
# ============================================================

print()
print("=" * 60)
print("APPLYING META CLASSIFIER")
print("=" * 60)


full_base_matrix = np.column_stack(
    full_test_predictions
)


full_base_clipped = np.clip(
    full_base_matrix,
    1e-6,
    1 - 1e-6
)


full_base_logits = np.log(
    full_base_clipped
    /
    (1.0 - full_base_clipped)
)


final_probabilities = (
    meta_model
    .predict_proba(
        full_base_logits
    )[:, 1]
)


# ============================================================
# 38. FINAL SAFETY CHECKS
# ============================================================

if len(final_probabilities) != len(
    full_test_raw
):

    raise ValueError(
        "Prediction count does not match "
        "test dataset."
    )


if np.isnan(
    final_probabilities
).any():

    raise ValueError(
        "Final predictions contain NaN."
    )


if np.isinf(
    final_probabilities
).any():

    raise ValueError(
        "Final predictions contain infinity."
    )


if (
    final_probabilities < 0
).any():

    raise ValueError(
        "Predictions below 0."
    )


if (
    final_probabilities > 1
).any():

    raise ValueError(
        "Predictions above 1."
    )


# ============================================================
# 39. SAVE SUBMISSION
# ============================================================

output_file = (
    "submission_exp62_sem_supervised_stack.csv"
)


submission = pd.DataFrame(
    {
        ID_COLUMN: full_test_ids,
        TARGET: final_probabilities
    }
)


submission.to_csv(
    output_file,
    index=False
)


# ============================================================
# 40. FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("EXPERIMENT 62 COMPLETE")
print("=" * 60)

print()
print(
    f"Round 8 Stack AUC : "
    f"{stack_round8_auc:.6f}"
)

print(
    f"Required AUC      : "
    f"{REQUIRED_ROUND_AUC:.6f}"
)

print()

print(
    "Passing models:"
)

for model_name in passing_models:

    print(
        f"  {model_name:<30} "
        f"{round_scores[model_name]:.6f}"
    )


print()
print(
    "Meta-model coefficients:"
)

for model_name, coefficient in zip(
    passing_models,
    meta_model.coef_[0]
):

    print(
        f"  {model_name:<30} "
        f"{coefficient:.6f}"
    )


print()
print(
    "Final prediction summary:"
)

print(
    pd.Series(
        final_probabilities
    ).describe()
)


print()
print(
    "First 10 predictions:"
)

print(
    submission.head(10)
)


print()
print("=" * 60)
print("BENCHMARKS")
print("=" * 60)

print(
    "Exp 59 Pure SEM Latent Titan    : 0.65832"
)

print(
    "Exp 60 Deep SEM Longitudinal    : 0.65735"
)

print(
    "Exp 61 Dual-Manifold Grand      : 0.65832 baseline"
)

print(
    f"Exp 62 Round 8 Stack            : "
    f"{stack_round8_auc:.6f}"
)

print()
print(
    f"Saved submission: {output_file}"
)

print()
print("=" * 60)
print("READY FOR KAGGLE SUBMISSION")
print("=" * 60)

