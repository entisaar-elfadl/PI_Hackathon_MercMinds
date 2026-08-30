
# ============================================================
# EXPERIMENT 48 — XGBOOST ENGINEERED TABULAR MODEL
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

from xgboost import XGBClassifier


print("============================================")
print("EXPERIMENT 48")
print("XGBOOST ENGINEERED TABULAR MODEL")
print("============================================")


# ============================================================
# 1. DIRECTORY
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (
    CURRENT_DIR / "../../assets/dataset"
).resolve()

if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR


print("\nDataset directory:")
print(DATA_DIR)


# ============================================================
# 2. LOAD DATA
# ============================================================

train_file = DATA_DIR / "train.csv"
test_file = DATA_DIR / "test.csv"


if not train_file.exists():
    raise FileNotFoundError(
        f"Training file not found:\n{train_file}"
    )


if not test_file.exists():
    raise FileNotFoundError(
        f"Testing file not found:\n{test_file}"
    )


train = pd.read_csv(train_file)
test = pd.read_csv(test_file)


print("\n============================================")
print("DATASET LOADED")
print("============================================")

print(
    "Training rows:",
    len(train)
)

print(
    "Training columns:",
    len(train.columns)
)

print(
    "Testing rows:",
    len(test)
)

print(
    "Testing columns:",
    len(test.columns)
)


# ============================================================
# 3. TARGET CLEANING
# ============================================================

TARGET = "employed_status"


train[TARGET] = pd.to_numeric(
    train[TARGET],
    errors="coerce"
)


missing_target = train[TARGET].isna().sum()


print(
    "\nMissing target values:",
    missing_target
)


train = train.dropna(
    subset=[TARGET]
).copy()


# FORCE TARGET TO 0 / 1

train[TARGET] = (
    train[TARGET]
    .astype(int)
)


# Safety check

unique_target = sorted(
    train[TARGET].unique()
)


print(
    "\nClean target values:",
    unique_target
)


if unique_target != [0, 1]:

    raise ValueError(
        f"Target is not exactly 0/1: {unique_target}"
    )


y = train[TARGET].copy()


# ============================================================
# 4. REMOVE TARGET / ID
# ============================================================

X = train.drop(
    columns=[TARGET]
).copy()

X_test = test.copy()


if "anonymised_id" in X.columns:

    X = X.drop(
        columns=["anonymised_id"]
    )


if "anonymised_id" in X_test.columns:

    X_test = X_test.drop(
        columns=["anonymised_id"]
    )


# ============================================================
# 5. DATE FEATURES
# ============================================================

print("\n============================================")
print("DATE ENGINEERING")
print("============================================")


if "survey_date" in X.columns:

    train_date = pd.to_datetime(
        X["survey_date"],
        errors="coerce"
    )

    test_date = pd.to_datetime(
        X_test["survey_date"],
        errors="coerce"
    )


    # Basic date components

    X["survey_year"] = (
        train_date.dt.year
    )

    X_test["survey_year"] = (
        test_date.dt.year
    )


    X["survey_month"] = (
        train_date.dt.month
    )

    X_test["survey_month"] = (
        test_date.dt.month
    )


    X["survey_quarter"] = (
        train_date.dt.quarter
    )

    X_test["survey_quarter"] = (
        test_date.dt.quarter
    )


    X["survey_dayofyear"] = (
        train_date.dt.dayofyear
    )

    X_test["survey_dayofyear"] = (
        test_date.dt.dayofyear
    )


    X["survey_week"] = (
        train_date.dt.isocalendar().week
        .astype(float)
    )

    X_test["survey_week"] = (
        test_date.dt.isocalendar().week
        .astype(float)
    )


    X = X.drop(
        columns=["survey_date"]
    )

    X_test = X_test.drop(
        columns=["survey_date"]
    )


    print(
        "Created date features."
    )


# ============================================================
# 6. ALIGN COLUMNS
# ============================================================

common_columns = [
    c
    for c in X.columns
    if c in X_test.columns
]


X = X[common_columns].copy()

X_test = X_test[common_columns].copy()


# ============================================================
# 7. IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_columns = (
    X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()
)


numerical_columns = (
    X.select_dtypes(
        include=[
            np.number
        ]
    ).columns.tolist()
)


print("\n============================================")
print("FEATURE INFORMATION")
print("============================================")

print(
    "Total features:",
    len(X.columns)
)

print(
    "Categorical:",
    len(categorical_columns)
)

print(
    "Numerical:",
    len(numerical_columns)
)


# ============================================================
# 8. FREQUENCY ENCODING
# ============================================================

print("\n============================================")
print("FREQUENCY ENCODING")
print("============================================")


for column in categorical_columns:

    # Convert to string while preserving missing values

    train_values = (
        X[column]
        .fillna("__MISSING__")
        .astype(str)
    )

    test_values = (
        X_test[column]
        .fillna("__MISSING__")
        .astype(str)
    )


    # Frequency calculated ONLY from training data

    frequencies = (
        train_values
        .value_counts(
            normalize=True
        )
    )


    # New frequency feature

    X[column + "_frequency"] = (
        train_values.map(
            frequencies
        ).fillna(0)
    )


    X_test[column + "_frequency"] = (
        test_values.map(
            frequencies
        ).fillna(0)
    )


    print(
        f" - {column}: "
        f"{train_values.nunique()} categories"
    )


# ============================================================
# 9. DROP ORIGINAL CATEGORICAL COLUMNS
# ============================================================

X = X.drop(
    columns=categorical_columns
)

X_test = X_test.drop(
    columns=categorical_columns
)


# ============================================================
# 10. NUMERIC CONVERSION
# ============================================================

for column in X.columns:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


for column in X_test.columns:

    X_test[column] = pd.to_numeric(
        X_test[column],
        errors="coerce"
    )


# ============================================================
# 11. ENGINEERED FEATURES
# ============================================================

print("\n============================================")
print("ENGINEERING INTERACTION FEATURES")
print("============================================")


def safe_ratio(
    df,
    numerator,
    denominator,
    output
):

    if (
        numerator in df.columns
        and
        denominator in df.columns
    ):

        denominator_values = (
            df[denominator]
            .replace(0, np.nan)
        )

        df[output] = (
            df[numerator]
            /
            denominator_values
        )


def safe_product(
    df,
    column_a,
    column_b,
    output
):

    if (
        column_a in df.columns
        and
        column_b in df.columns
    ):

        df[output] = (
            df[column_a]
            *
            df[column_b]
        )


def safe_difference(
    df,
    column_a,
    column_b,
    output
):

    if (
        column_a in df.columns
        and
        column_b in df.columns
    ):

        df[output] = (
            df[column_a]
            -
            df[column_b]
        )


# ------------------------------------------------------------
# ROUND FEATURES
# ------------------------------------------------------------

safe_difference(
    X,
    "current_round",
    "lag_round",
    "round_progress"
)

safe_difference(
    X_test,
    "current_round",
    "lag_round",
    "round_progress"
)


# ------------------------------------------------------------
# AGE / TENURE
# ------------------------------------------------------------

safe_ratio(
    X,
    "tenure_lag",
    "age",
    "tenure_age_ratio"
)

safe_ratio(
    X_test,
    "tenure_lag",
    "age",
    "tenure_age_ratio"
)


safe_product(
    X,
    "age",
    "work_readiness_score",
    "age_readiness_interaction"
)

safe_product(
    X_test,
    "age",
    "work_readiness_score",
    "age_readiness_interaction"
)


# ------------------------------------------------------------
# EMPLOYMENT / HISTORY
# ------------------------------------------------------------

safe_product(
    X,
    "employed_lag",
    "total_historical_rounds",
    "employment_history_strength"
)

safe_product(
    X_test,
    "employed_lag",
    "total_historical_rounds",
    "employment_history_strength"
)


safe_ratio(
    X,
    "employed_lag",
    "total_historical_rounds",
    "historical_employment_ratio"
)

safe_ratio(
    X_test,
    "employed_lag",
    "total_historical_rounds",
    "historical_employment_ratio"
)


# ------------------------------------------------------------
# WORK READINESS / EDUCATION
# ------------------------------------------------------------

safe_product(
    X,
    "work_readiness_score",
    "education_schooling_grade_twelve_equiv",
    "readiness_education_interaction"
)

safe_product(
    X_test,
    "work_readiness_score",
    "education_schooling_grade_twelve_equiv",
    "readiness_education_interaction"
)


# ------------------------------------------------------------
# WORK READINESS / AGE
# ------------------------------------------------------------

safe_ratio(
    X,
    "work_readiness_score",
    "age",
    "readiness_age_ratio"
)

safe_ratio(
    X_test,
    "work_readiness_score",
    "age",
    "readiness_age_ratio"
)


# ------------------------------------------------------------
# TENURE / HISTORICAL ROUNDS
# ------------------------------------------------------------

safe_ratio(
    X,
    "tenure_lag",
    "total_historical_rounds",
    "tenure_history_ratio"
)

safe_ratio(
    X_test,
    "tenure_lag",
    "total_historical_rounds",
    "tenure_history_ratio"
)


# ============================================================
# 12. HANDLE INFINITIES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 13. ALIGN FINAL FEATURES
# ============================================================

X_test = X_test.reindex(
    columns=X.columns
)


print(
    "\nFinal feature count:",
    len(X.columns)
)


# ============================================================
# 14. MISSING VALUES
# ============================================================

print(
    "\nTotal missing training values:",
    X.isna().sum().sum()
)

print(
    "Total missing testing values:",
    X_test.isna().sum().sum()
)


# XGBoost handles NaN values natively.
# No imputation is required.


# ============================================================
# 15. XGBOOST MODEL
# ============================================================

print("\n============================================")
print("TRAINING XGBOOST")
print("============================================")


model = XGBClassifier(

    objective="binary:logistic",

    eval_metric="auc",

    # --------------------------------------------------------
    # BOOSTING
    # --------------------------------------------------------

    n_estimators=1500,

    learning_rate=0.025,

    # --------------------------------------------------------
    # TREE STRUCTURE
    # --------------------------------------------------------

    max_depth=5,

    min_child_weight=5,

    gamma=0.05,

    # --------------------------------------------------------
    # SAMPLING
    # --------------------------------------------------------

    subsample=0.85,

    colsample_bytree=0.80,

    # --------------------------------------------------------
    # REGULARISATION
    # --------------------------------------------------------

    reg_alpha=0.10,

    reg_lambda=2.0,

    # --------------------------------------------------------
    # IMBALANCE
    # --------------------------------------------------------

    scale_pos_weight=1.0,

    # --------------------------------------------------------
    # HISTOGRAM TREE METHOD
    # --------------------------------------------------------

    tree_method="hist",

    # --------------------------------------------------------
    # RANDOMNESS
    # --------------------------------------------------------

    random_state=42,

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    n_jobs=-1,

    verbosity=0
)


print(
    "Training on",
    len(X),
    "rows..."
)


model.fit(
    X,
    y
)


print(
    "\nTraining complete."
)


# ============================================================
# 16. PREDICTIONS
# ============================================================

print("\n============================================")
print("GENERATING PREDICTIONS")
print("============================================")


probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 17. VALIDATION
# ============================================================

if len(probabilities) != len(test):

    raise ValueError(
        "Prediction count does not match test rows."
    )


if np.isnan(probabilities).any():

    raise ValueError(
        "Predictions contain NaN."
    )


if (
    (probabilities < 0).any()
    or
    (probabilities > 1).any()
):

    raise ValueError(
        "Predictions outside [0,1]."
    )


# ============================================================
# 18. SUBMISSION
# ============================================================

submission = pd.DataFrame({

    "anonymised_id":
        test["anonymised_id"],

    "employed_status":
        probabilities

})


output_file = (
    "submission_exp48_xgboost_engineered.csv"
)


submission.to_csv(
    output_file,
    index=False
)


# ============================================================
# 19. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature":
        X.columns,

    "importance":
        model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


# ============================================================
# 20. SUMMARY
# ============================================================

print("\n============================================")
print("EXPERIMENT 48 COMPLETE")
print("============================================")


print(
    "\nSaved:",
    output_file
)


print(
    "Rows:",
    len(submission)
)


print(
    "\nPrediction summary:"
)


print(
    submission[
        "employed_status"
    ].describe()
)


print(
    "\nFirst 10 predictions:"
)


print(
    submission.head(10)
)


# ============================================================
# 21. TOP FEATURES
# ============================================================

print("\n============================================")
print("TOP 20 FEATURES")
print("============================================")


print(
    importance.head(20).to_string(
        index=False
    )
)


# ============================================================
# 22. BENCHMARKS
# ============================================================

print("\n============================================")
print("BENCHMARKS")
print("============================================")

print(
    "Exp 41 Sequential Hybrid : 0.65325"
)

print(
    "Exp 43 LogReg Champion   : 0.65502"
)

print(
    "Exp 44 Top-3 Linear      : 0.65630"
)

print(
    "Exp 45 Dynamic Programming: 0.65613"
)

print(
    "Exp 46 Multilinear       : 0.65701"
)

print(
    "Exp 47 LightGBM          : 0.59252"
)

print(
    "Exp 48 XGBoost           : PENDING"
)


print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")
