from pathlib import Path
import pandas as pd

# ============================================================
# 1. FIND DATASET
# ============================================================

# Directory where this Python file/notebook is running
CURRENT_DIR = Path.cwd()

print("Current directory:")
print(CURRENT_DIR)

# Dataset location
DATA_DIR = CURRENT_DIR / "../../assets/dataset"

# Convert to absolute path
DATA_DIR = DATA_DIR.resolve()

print("\nDataset directory:")
print(DATA_DIR)


# ============================================================
# 2. LOAD DATA
# ============================================================

train_file = DATA_DIR / "train.csv"
test_file = DATA_DIR / "test.csv"

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n============================================")
print("DATASET LOADED")
print("============================================")

print(f"\nTraining rows: {len(train_data)}")
print(f"Training columns: {len(train_data.columns)}")

print(f"\nTesting rows: {len(test_data)}")
print(f"Testing columns: {len(test_data.columns)}")


# ============================================================
# 4. VERIFY TARGET
# ============================================================

target = "employed_status"

if target not in train_data.columns:
    raise ValueError(
        f"Target column '{target}' was not found in training data."
    )

if target in test_data.columns:
    raise ValueError(
        f"Target column '{target}' unexpectedly exists in test data."
    )

print(f"\nTarget column: {target}")

print("\nTarget distribution:")
print(train_data[target].value_counts())

print("\nTarget proportions:")
print(train_data[target].value_counts(normalize=True))

# ============================================================
# EXPERIMENT — UPGRADED LOGISTIC REGRESSION
# ============================================================

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


print("============================================")
print("UPGRADED LOGISTIC REGRESSION")
print("============================================")


# ============================================================
# 1. TARGET
# ============================================================

target = "employed_status"

y = train_data[target]


# ============================================================
# 2. FEATURES
# ============================================================

up_cat = [
    "gender",
    "education_level"
]

up_num = [
    "work_readiness_score"
]


features = up_cat + up_num

Xtr2 = train_data[features].copy()
Xte2 = test_data[features].copy()


print("\nFeatures:")
print(features)


# ============================================================
# 3. PREPROCESSING
# ============================================================

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
            handle_unknown="ignore"
        )
    )
])


num_pipe = Pipeline([
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


pre2 = ColumnTransformer([
    (
        "cat",
        cat_pipe,
        up_cat
    ),
    (
        "num",
        num_pipe,
        up_num
    )
])


# ============================================================
# 4. MODEL
# ============================================================

upgraded = Pipeline([
    (
        "pre",
        pre2
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=1000
        )
    )
])


# ============================================================
# 5. TRAIN
# ============================================================

print("\nTraining model...")

upgraded.fit(
    Xtr2,
    y
)

print("Training complete.")


# ============================================================
# 6. PREDICT
# ============================================================

up_prob = upgraded.predict_proba(
    Xte2
)[:, 1]


# ============================================================
# 7. CREATE SUBMISSION
# ============================================================

submission2 = pd.DataFrame({
    "anonymised_id":
        test_data["anonymised_id"],

    "employed_status":
        up_prob
})


# ============================================================
# 8. VALIDATE
# ============================================================

if len(submission2) != len(test_data):
    raise ValueError(
        "Submission row count does not match test data."
    )


if submission2["employed_status"].isna().any():
    raise ValueError(
        "Submission contains NA predictions."
    )


if not submission2["anonymised_id"].equals(
    test_data["anonymised_id"]
):
    raise ValueError(
        "anonymised_id ordering does not match test data."
    )


# ============================================================
# 9. SAVE
# ============================================================

output_file = "submission_exp30_upgraded_logistic.csv"

submission2.to_csv(
    output_file,
    index=False
)


# ============================================================
# 10. SUMMARY
# ============================================================

print("\n============================================")
print("EXPERIMENT COMPLETE")
print("============================================")

print(f"\nSaved: {output_file}")
print(f"Rows: {len(submission2)}")

print("\nPrediction summary:")
print(submission2["employed_status"].describe())

print("\nFirst 5 predictions:")
print(submission2.head())

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")