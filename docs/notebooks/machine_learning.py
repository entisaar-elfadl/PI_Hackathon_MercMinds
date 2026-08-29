# ============================================================
# EXPERIMENT 32 — HISTOGRAM GRADIENT BOOSTING
# FIXED HIGH-CARDINALITY VERSION
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder


print("============================================")
print("EXPERIMENT 32")
print("HISTOGRAM GRADIENT BOOSTING")
print("============================================")


# ============================================================
# 1. DATASET PATH
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (
    CURRENT_DIR /
    "../../assets/dataset"
).resolve()


# ============================================================
# 2. LOAD DATA
# ============================================================

train_data = pd.read_csv(
    DATA_DIR / "train.csv"
)

test_data = pd.read_csv(
    DATA_DIR / "test.csv"
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
# 3. CLEAN TARGET
# ============================================================

target = "employed_status"


print(
    "\nMissing target values:",
    train_data[target].isna().sum()
)


train_clean = train_data.dropna(
    subset=[target]
).copy()


train_clean[target] = pd.to_numeric(
    train_clean[target],
    errors="coerce"
)


train_clean = train_clean.dropna(
    subset=[target]
).copy()


train_clean[target] = train_clean[
    target
].astype(int)


y = train_clean[target]


print(
    "Training rows after cleaning:",
    len(train_clean)
)


print("\nTarget distribution:")

print(
    y.value_counts()
)


# ============================================================
# 4. CREATE X
# ============================================================

X = train_clean.drop(
    columns=[target]
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

# survey_date is not treated as a categorical variable.
# Instead we extract useful numerical date information.

if "survey_date" in X.columns:

    X["survey_date"] = pd.to_datetime(
        X["survey_date"],
        errors="coerce"
    )

    X_test["survey_date"] = pd.to_datetime(
        X_test["survey_date"],
        errors="coerce"
    )


    X["survey_year"] = (
        X["survey_date"].dt.year
    )

    X_test["survey_year"] = (
        X_test["survey_date"].dt.year
    )


    X["survey_month"] = (
        X["survey_date"].dt.month
    )

    X_test["survey_month"] = (
        X_test["survey_date"].dt.month
    )


    X["survey_dayofyear"] = (
        X["survey_date"].dt.dayofyear
    )

    X_test["survey_dayofyear"] = (
        X_test["survey_date"].dt.dayofyear
    )


    X = X.drop(
        columns=["survey_date"]
    )

    X_test = X_test.drop(
        columns=["survey_date"]
    )


# ============================================================
# 7. IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()


numerical_features = X.select_dtypes(
    include=[
        np.number
    ]
).columns.tolist()


print("\n============================================")
print("INITIAL FEATURES")
print("============================================")

print(
    "Categorical:",
    len(categorical_features)
)

print(
    "Numerical:",
    len(numerical_features)
)


# ============================================================
# 8. REMOVE HIGH-CARDINALITY CATEGORICAL FEATURES
# ============================================================

# HistGradientBoosting supports at most 255 categories
# for an individual categorical feature.

high_cardinality = []

for column in categorical_features:

    unique_count = X[column].nunique(
        dropna=True
    )

    if unique_count > 255:

        high_cardinality.append(
            column
        )

        print(
            f"\nDropping high-cardinality feature:"
            f" {column}"
            f" ({unique_count} categories)"
        )


if high_cardinality:

    X = X.drop(
        columns=high_cardinality
    )

    X_test = X_test.drop(
        columns=high_cardinality
    )


# Update categorical list

categorical_features = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()


numerical_features = X.select_dtypes(
    include=[
        np.number
    ]
).columns.tolist()


print("\n============================================")
print("FINAL FEATURES")
print("============================================")

print(
    "Total:",
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


print("\nCategorical features:")

for column in categorical_features:

    print(
        f" - {column}: "
        f"{X[column].nunique(dropna=True)} categories"
    )


print("\nNumerical features:")

for column in numerical_features:

    print(
        " -",
        column
    )


# ============================================================
# 9. ENCODE CATEGORICAL FEATURES
# ============================================================

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=np.nan
)


if len(categorical_features) > 0:

    X[categorical_features] = (
        encoder.fit_transform(
            X[categorical_features]
        )
    )

    X_test[categorical_features] = (
        encoder.transform(
            X_test[categorical_features]
        )
    )


# ============================================================
# 10. CATEGORICAL MASK
# ============================================================

categorical_mask = [
    column in categorical_features
    for column in X.columns
]


print("\nCategorical mask:")

print(
    categorical_mask
)


# ============================================================
# 11. MODEL
# ============================================================

model = HistGradientBoostingClassifier(

    loss="log_loss",

    learning_rate=0.04,

    max_iter=600,

    max_leaf_nodes=31,

    min_samples_leaf=20,

    l2_regularization=1.0,

    max_features=0.8,

    categorical_features=categorical_mask,

    early_stopping=True,

    validation_fraction=0.15,

    n_iter_no_change=40,

    random_state=42

)


# ============================================================
# 12. TRAIN
# ============================================================

print("\n============================================")
print("TRAINING")
print("============================================")

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

print(
    "Iterations used:",
    model.n_iter_
)


# ============================================================
# 13. PREDICT
# ============================================================

print(
    "\nGenerating test predictions..."
)


probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 14. VALIDATE
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
    "submission_exp32_hist_gradient_boosting.csv"
)


submission.to_csv(
    output_file,
    index=False
)


# ============================================================
# 17. SUMMARY
# ============================================================

print("\n============================================")
print("EXPERIMENT 32 COMPLETE")
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
# 18. BENCHMARKS
# ============================================================

print("\n============================================")
print("BENCHMARKS")
print("============================================")

print(
    "Current best ensemble : 0.64516"
)

print(
    "Exp 30 Logistic       : 0.59229"
)

print(
    "Exp 31 ExtraTrees     : 0.61732"
)

print(
    "Exp 32 HistGB         : PENDING"
)


print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")