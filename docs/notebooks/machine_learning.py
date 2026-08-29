# ============================================================
# EXPERIMENT 31 — EXTRA TREES
# ALL FEATURES
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier


print("============================================")
print("EXPERIMENT 31")
print("EXTRA TREES — ALL FEATURES")
print("============================================")


# ============================================================
# 1. DATASET PATH
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (
    CURRENT_DIR /
    "../../assets/dataset"
).resolve()


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
        f"Test file not found:\n{test_file}"
    )


train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)


print("\n============================================")
print("DATASET LOADED")
print("============================================")

print(
    f"Training rows: {len(train_data)}"
)

print(
    f"Training columns: {len(train_data.columns)}"
)

print(
    f"Testing rows: {len(test_data)}"
)

print(
    f"Testing columns: {len(test_data.columns)}"
)


# ============================================================
# 3. TARGET
# ============================================================

target = "employed_status"

print(
    "\nMissing target values:",
    train_data[target].isna().sum()
)


# ============================================================
# 4. REMOVE MISSING TARGETS
# ============================================================

train_clean = train_data.dropna(
    subset=[target]
).copy()


# Make target explicitly 0/1

train_clean[target] = pd.to_numeric(
    train_clean[target],
    errors="coerce"
)

train_clean = train_clean.dropna(
    subset=[target]
)

train_clean[target] = train_clean[
    target
].astype(int)


y = train_clean[target]


print(
    "Training rows after cleaning:",
    len(train_clean)
)


print(
    "\nTarget distribution:"
)

print(
    y.value_counts()
)


# ============================================================
# 5. REMOVE TARGET FROM FEATURES
# ============================================================

X = train_clean.drop(
    columns=[target]
).copy()


X_test = test_data.copy()


# ============================================================
# 6. REMOVE ID FROM FEATURES
# ============================================================

# anonymised_id is an identifier, not a meaningful predictor.

id_column = "anonymised_id"

if id_column in X.columns:

    X = X.drop(
        columns=[id_column]
    )

if id_column in X_test.columns:

    X_test = X_test.drop(
        columns=[id_column]
    )


# ============================================================
# 7. IDENTIFY COLUMN TYPES
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
print("FEATURE INFORMATION")
print("============================================")

print(
    "\nTotal features:",
    len(X.columns)
)

print(
    "Categorical features:",
    len(categorical_features)
)

print(
    "Numerical features:",
    len(numerical_features)
)


print("\nCategorical:")

for column in categorical_features:
    print(" -", column)


print("\nNumerical:")

for column in numerical_features:
    print(" -", column)


# ============================================================
# 8. CATEGORICAL PIPELINE
# ============================================================

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
            handle_unknown="ignore"
        )
    )
])


# ============================================================
# 9. NUMERICAL PIPELINE
# ============================================================

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])


# ============================================================
# 10. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer([

    (
        "categorical",
        categorical_pipeline,
        categorical_features
    ),

    (
        "numerical",
        numerical_pipeline,
        numerical_features
    )

])


# ============================================================
# 11. EXTRA TREES
# ============================================================

model = ExtraTreesClassifier(

    n_estimators=800,

    max_depth=None,

    min_samples_split=4,

    min_samples_leaf=2,

    max_features="sqrt",

    class_weight="balanced",

    random_state=42,

    n_jobs=-1

)


# ============================================================
# 12. COMPLETE PIPELINE
# ============================================================

extra_trees = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        model
    )

])


# ============================================================
# 13. TRAIN
# ============================================================

print("\n============================================")
print("TRAINING EXTRA TREES")
print("============================================")

print(
    "Training on",
    len(X),
    "rows..."
)


extra_trees.fit(
    X,
    y
)


print(
    "Training complete."
)


# ============================================================
# 14. PREDICT
# ============================================================

print("\nGenerating test predictions...")


probabilities = extra_trees.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 15. VALIDATE
# ============================================================

if len(probabilities) != len(test_data):

    raise ValueError(
        "Prediction count does not match test rows."
    )


if np.isnan(probabilities).any():

    raise ValueError(
        "Predictions contain NaN values."
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
# 16. CREATE SUBMISSION
# ============================================================

submission = pd.DataFrame({

    "anonymised_id":
        test_data["anonymised_id"],

    "employed_status":
        probabilities

})


# ============================================================
# 17. VALIDATE SUBMISSION
# ============================================================

if len(submission) != len(test_data):

    raise ValueError(
        "Submission row count mismatch."
    )


if submission[
    "employed_status"
].isna().any():

    raise ValueError(
        "Submission contains NaN predictions."
    )


if not submission[
    "anonymised_id"
].equals(
    test_data["anonymised_id"]
):

    raise ValueError(
        "ID ordering does not match test data."
    )


# ============================================================
# 18. SAVE
# ============================================================

output_file = (
    "submission_exp31_extra_trees.csv"
)


submission.to_csv(
    output_file,
    index=False
)


# ============================================================
# 19. SUMMARY
# ============================================================

print("\n============================================")
print("EXPERIMENT 31 COMPLETE")
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


# ============================================================
# 20. BENCHMARKS
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
    "Exp 31 ExtraTrees     : PENDING"
)


print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")