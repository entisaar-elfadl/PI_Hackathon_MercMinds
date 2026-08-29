# ============================================================
# EXPERIMENT 30 — UPGRADED LOGISTIC REGRESSION
# ============================================================

import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


print("============================================")
print("EXPERIMENT 30")
print("UPGRADED LOGISTIC REGRESSION")
print("============================================")


# ============================================================
# 1. DATASET PATH
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (
    CURRENT_DIR /
    "../../assets/dataset"
).resolve()


print("\nCurrent directory:")
print(CURRENT_DIR)

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


if target not in train_data.columns:
    raise ValueError(
        f"Target column '{target}' not found."
    )


if target in test_data.columns:
    raise ValueError(
        f"Target column '{target}' unexpectedly exists in test data."
    )


print("\nTarget:", target)


# ============================================================
# 4. CHECK ORIGINAL TARGET
# ============================================================

print("\nOriginal target values:")

print(
    train_data[target].value_counts(
        dropna=False
    )
)


print(
    "\nMissing target values:",
    train_data[target].isna().sum()
)


# ============================================================
# 5. CLEAN TARGET
# ============================================================

# Remove rows where the target is missing.
# We cannot train a supervised model without a label.

train_clean = train_data.dropna(
    subset=[target]
).copy()


# Convert target to numeric.

train_clean[target] = pd.to_numeric(
    train_clean[target],
    errors="coerce"
)


# Remove anything that became NaN after conversion.

train_clean = train_clean.dropna(
    subset=[target]
).copy()


# Convert target to integer 0/1.

y = train_clean[target].astype(int)


# ============================================================
# 6. VALIDATE TARGET
# ============================================================

unique_targets = sorted(
    y.unique()
)


print("\nClean target values:")
print(
    y.value_counts()
)


print(
    "\nUnique target values:",
    unique_targets
)


if not set(unique_targets).issubset({0, 1}):
    raise ValueError(
        "Target contains values other than 0 and 1."
    )


print(
    "\nTraining rows after target cleaning:",
    len(train_clean)
)


# ============================================================
# 7. FEATURES
# ============================================================

up_cat = [
    "gender",
    "education_level"
]


up_num = [
    "work_readiness_score"
]


features = up_cat + up_num


# Make sure all required columns exist.

for column in features:

    if column not in train_clean.columns:
        raise ValueError(
            f"Missing training feature: {column}"
        )

    if column not in test_data.columns:
        raise ValueError(
            f"Missing test feature: {column}"
        )


# ============================================================
# 8. CREATE X
# ============================================================

X_train = train_clean[
    features
].copy()


X_test = test_data[
    features
].copy()


print("\nFeatures:")
for feature in features:
    print(" -", feature)


# ============================================================
# 9. CATEGORICAL PREPROCESSING
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


# ============================================================
# 10. NUMERICAL PREPROCESSING
# ============================================================

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


# ============================================================
# 11. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([
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
# 12. LOGISTIC REGRESSION MODEL
# ============================================================

upgraded = Pipeline([
    (
        "pre",
        preprocessor
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=1000
        )
    )
])


# ============================================================
# 13. TRAIN
# ============================================================

print("\n============================================")
print("TRAINING")
print("============================================")

print(
    "Training on",
    len(X_train),
    "rows..."
)


upgraded.fit(
    X_train,
    y
)


print("Training complete.")


# ============================================================
# 14. PREDICT PROBABILITIES
# ============================================================

print("\nGenerating test predictions...")


up_prob = upgraded.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 15. VALIDATE PREDICTIONS
# ============================================================

if len(up_prob) != len(test_data):

    raise ValueError(
        "Prediction count does not match test rows."
    )


if pd.isna(up_prob).any():

    raise ValueError(
        "Predictions contain NaN values."
    )


if ((up_prob < 0) | (up_prob > 1)).any():

    raise ValueError(
        "Predictions are outside [0, 1]."
    )


# ============================================================
# 16. CREATE SUBMISSION
# ============================================================

submission2 = pd.DataFrame({

    "anonymised_id":
        test_data["anonymised_id"],

    "employed_status":
        up_prob

})


# ============================================================
# 17. VALIDATE SUBMISSION
# ============================================================

if len(submission2) != len(test_data):

    raise ValueError(
        "Submission row count does not match test data."
    )


if submission2[
    "employed_status"
].isna().any():

    raise ValueError(
        "Submission contains NaN predictions."
    )


if not submission2[
    "anonymised_id"
].equals(
    test_data["anonymised_id"]
):

    raise ValueError(
        "anonymised_id ordering does not match test data."
    )


# ============================================================
# 18. SAVE
# ============================================================

output_file = (
    "submission_exp30_upgraded_logistic.csv"
)


submission2.to_csv(
    output_file,
    index=False
)


# ============================================================
# 19. OUTPUT SUMMARY
# ============================================================

print("\n============================================")
print("EXPERIMENT 30 COMPLETE")
print("============================================")

print(
    "\nSaved:",
    output_file
)

print(
    "Rows:",
    len(submission2)
)

print(
    "Columns:",
    len(submission2.columns)
)


print("\nPrediction summary:")

print(
    submission2[
        "employed_status"
    ].describe()
)


print("\nFirst 10 predictions:")

print(
    submission2.head(10)
)


print("\nTarget distribution used for training:")

print(
    y.value_counts(
        normalize=True
    )
)


print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")