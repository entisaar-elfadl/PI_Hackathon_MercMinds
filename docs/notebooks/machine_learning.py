# ============================================================
# EXPERIMENT 47 — LIGHTGBM FULL TABULAR MODEL
# ============================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, early_stopping, log_evaluation


warnings.filterwarnings("ignore")


print("============================================")
print("EXPERIMENT 47")
print("LIGHTGBM FULL TABULAR MODEL")
print("============================================")


# ============================================================
# 1. DIRECTORY
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (
    CURRENT_DIR / "../../assets/dataset"
).resolve()

if not DATA_DIR.exists():

    print("\nDataset directory not found at:")
    print(DATA_DIR)

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
        f"Could not find:\n{train_file}"
    )


if not test_file.exists():

    raise FileNotFoundError(
        f"Could not find:\n{test_file}"
    )


train_data = pd.read_csv(
    train_file
)

test_data = pd.read_csv(
    test_file
)


print("\n============================================")
print("DATASET LOADED")
print("============================================")

print(
    "Training rows:",
    len(train_data)
)

print(
    "Training columns:",
    len(train_data.columns)
)

print(
    "Testing rows:",
    len(test_data)
)

print(
    "Testing columns:",
    len(test_data.columns)
)


# ============================================================
# 3. TARGET CLEANING
# ============================================================

TARGET = "employed_status"

print("\n============================================")
print("TARGET CLEANING")
print("============================================")


train_data[TARGET] = pd.to_numeric(
    train_data[TARGET],
    errors="coerce"
)


print(
    "\nOriginal target distribution:"
)

print(
    train_data[TARGET].value_counts(
        dropna=False
    )
)


missing_target = train_data[TARGET].isna().sum()


print(
    "\nMissing target values:",
    missing_target
)


# Remove rows where target is missing

train_data = train_data.dropna(
    subset=[TARGET]
).copy()


# Force target to exactly 0 / 1

train_data[TARGET] = (
    train_data[TARGET]
    .astype(int)
)


# Safety check

unique_target = sorted(
    train_data[TARGET].unique()
)


print(
    "\nClean target distribution:"
)

print(
    train_data[TARGET].value_counts()
)


print(
    "\nUnique target values:",
    unique_target
)


if unique_target != [0, 1]:

    raise ValueError(
        f"Target is not binary 0/1: {unique_target}"
    )


# ============================================================
# 4. SEPARATE X / Y
# ============================================================

y = train_data[TARGET].copy()

X = train_data.drop(
    columns=[TARGET]
).copy()

X_test = test_data.copy()


# ============================================================
# 5. REMOVE ID
# ============================================================

if "anonymised_id" in X.columns:

    X = X.drop(
        columns=["anonymised_id"]
    )


if "anonymised_id" in X_test.columns:

    X_test = X_test.drop(
        columns=["anonymised_id"]
    )


# ============================================================
# 6. DATE FEATURES
# ============================================================

print("\n============================================")
print("DATE FEATURES")
print("============================================")


if "survey_date" in X.columns:

    print(
        "Processing survey_date..."
    )


    X["survey_date"] = pd.to_datetime(
        X["survey_date"],
        errors="coerce"
    )


    X_test["survey_date"] = pd.to_datetime(
        X_test["survey_date"],
        errors="coerce"
    )


    # Year

    X["survey_year"] = (
        X["survey_date"].dt.year
    )

    X_test["survey_year"] = (
        X_test["survey_date"].dt.year
    )


    # Month

    X["survey_month"] = (
        X["survey_date"].dt.month
    )

    X_test["survey_month"] = (
        X_test["survey_date"].dt.month
    )


    # Day of year

    X["survey_dayofyear"] = (
        X["survey_date"].dt.dayofyear
    )

    X_test["survey_dayofyear"] = (
        X_test["survey_date"].dt.dayofyear
    )


    # Quarter

    X["survey_quarter"] = (
        X["survey_date"].dt.quarter
    )

    X_test["survey_quarter"] = (
        X_test["survey_date"].dt.quarter
    )


    # Drop original date

    X = X.drop(
        columns=["survey_date"]
    )

    X_test = X_test.drop(
        columns=["survey_date"]
    )


# ============================================================
# 7. MAKE TRAIN / TEST COLUMNS IDENTICAL
# ============================================================

common_columns = [
    column
    for column in X.columns
    if column in X_test.columns
]


X = X[common_columns].copy()

X_test = X_test[common_columns].copy()


print(
    "\nFeatures:",
    len(common_columns)
)


# ============================================================
# 8. CONVERT OBJECT COLUMNS TO CATEGORICAL
# ============================================================

print("\n============================================")
print("CATEGORICAL FEATURES")
print("============================================")


categorical_features = []


for column in X.columns:

    if (
        X[column].dtype == "object"
        or
        str(X[column].dtype) == "category"
        or
        X[column].dtype == "bool"
    ):

        categorical_features.append(
            column
        )


print(
    "Categorical features:",
    len(categorical_features)
)


for column in categorical_features:

    # Combine train/test categories so
    # unknown test categories are handled safely.

    combined = pd.concat(
        [
            X[column],
            X_test[column]
        ],
        axis=0
    ).astype("string")


    categories = pd.Index(
        combined.dropna().unique()
    )


    X[column] = pd.Categorical(
        X[column].astype("string"),
        categories=categories
    )


    X_test[column] = pd.Categorical(
        X_test[column].astype("string"),
        categories=categories
    )


    print(
        f" - {column}: "
        f"{len(categories)} categories"
    )


# ============================================================
# 9. NUMERICAL FEATURES
# ============================================================

numerical_features = [
    column
    for column in X.columns
    if column not in categorical_features
]


print("\nNumerical features:")

for column in numerical_features:

    print(
        " -",
        column
    )


# ============================================================
# 10. VALIDATE DATA TYPES
# ============================================================

print("\n============================================")
print("FEATURE SUMMARY")
print("============================================")

print(
    "Total features:",
    len(X.columns)
)

print(
    "Categorical:",
    len(categorical_features)
)

print(
    "Numerical:",
    len(numerical_features)
)


# ============================================================
# 11. TRAIN LIGHTGBM
# ============================================================

print("\n============================================")
print("TRAINING LIGHTGBM")
print("============================================")

print(
    "Training on",
    len(X),
    "rows..."
)


model = LGBMClassifier(

    objective="binary",

    # --------------------------------------------------------
    # Core boosting
    # --------------------------------------------------------

    n_estimators=2000,

    learning_rate=0.025,

    # --------------------------------------------------------
    # Tree complexity
    # --------------------------------------------------------

    num_leaves=31,

    max_depth=-1,

    min_child_samples=30,

    min_split_gain=0.0,

    # --------------------------------------------------------
    # Regularisation
    # --------------------------------------------------------

    reg_alpha=0.10,

    reg_lambda=1.00,

    # --------------------------------------------------------
    # Feature / row sampling
    # --------------------------------------------------------

    subsample=0.85,

    subsample_freq=1,

    colsample_bytree=0.85,

    # --------------------------------------------------------
    # Categorical / histogram
    # --------------------------------------------------------

    max_bin=255,

    # --------------------------------------------------------
    # Class imbalance
    #
    # IMPORTANT:
    # We deliberately do NOT use class_weight here.
    # AUC is ranking based and we want probabilities that
    # preserve the natural target distribution.
    # --------------------------------------------------------

    class_weight=None,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random_state=42,

    n_jobs=-1,

    verbosity=-1
)


# ============================================================
# 12. FIT
# ============================================================

model.fit(

    X,
    y,

    categorical_feature=categorical_features,

    callbacks=[
        log_evaluation(100)
    ]
)


print(
    "\nTraining complete."
)


# ============================================================
# 13. PREDICT
# ============================================================

print("\n============================================")
print("GENERATING PREDICTIONS")
print("============================================")


probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 14. VALIDATION
# ============================================================

if len(probabilities) != len(test_data):

    raise ValueError(
        "Prediction count does not match test data."
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
        "Predictions outside [0, 1]."
    )


# ============================================================
# 15. CREATE SUBMISSION
# ============================================================

submission = pd.DataFrame({

    "anonymised_id":
        test_data["anonymised_id"],

    "employed_status":
        probabilities

})


# ============================================================
# 16. SAVE
# ============================================================

output_file = (
    "submission_exp47_lightgbm.csv"
)


submission.to_csv(
    output_file,
    index=False
)


# ============================================================
# 17. PREDICTION SUMMARY
# ============================================================

print("\n============================================")
print("EXPERIMENT 47 COMPLETE")
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
    "Columns:",
    len(submission.columns)
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
# 18. FEATURE IMPORTANCE
# ============================================================

print("\n============================================")
print("TOP FEATURE IMPORTANCE")
print("============================================")


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


print(
    importance.head(20).to_string(
        index=False
    )
)


# ============================================================
# 19. BENCHMARKS
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
    "Exp 47 LightGBM          : PENDING"
)


print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")

