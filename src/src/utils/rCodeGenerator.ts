import { RoundId } from '../types';

export interface RScriptOptions {
  modelType: 'logistic' | 'randomForest' | 'xgboost' | 'glmnet';
  roundId: RoundId;
  outputFilename?: string;
  imputeMissing: boolean;
}

export function generateRScript(options: RScriptOptions): string {
  const { modelType, roundId, outputFilename = `submission_${options.modelType}_${roundId}.csv`, imputeMissing } = options;

  let trainFile = 'rounds_1_5.csv';
  let evalFile = 'round_6.csv';
  let prevRounds = 'Rounds 1–5';

  if (roundId === 'round_7') {
    trainFile = 'rounds_1_6.csv';
    evalFile = 'round_7.csv';
    prevRounds = 'Rounds 1–6';
  } else if (roundId === 'round_8') {
    trainFile = 'rounds_1_7.csv';
    evalFile = 'round_8.csv';
    prevRounds = 'Rounds 1–7';
  }

  let script = `# ==============================================================================
# Employment Prediction Competition - R Model Training & Validation Script
# Historical Evaluation Round: ${roundId.toUpperCase()} (${prevRounds} -> ${evalFile})
# Output: ${outputFilename}
# ==============================================================================

# 1. Load Required Packages
`;

  if (modelType === 'xgboost') {
    script += `library(xgboost)\nlibrary(dplyr)\nlibrary(readr)\n`;
  } else if (modelType === 'randomForest') {
    script += `library(randomForest)\nlibrary(dplyr)\nlibrary(readr)\n`;
  } else if (modelType === 'glmnet') {
    script += `library(glmnet)\nlibrary(dplyr)\nlibrary(readr)\n`;
  } else {
    script += `library(dplyr)\nlibrary(readr)\n`;
  }

  script += `
# 2. Load Datasets
# Note: Ensure files are in working directory or specify path
train_df <- read_csv("${trainFile}")
eval_df  <- read_csv("${evalFile}")

cat("Loaded Training Rows:", nrow(train_df), "\\n")
cat("Loaded Evaluation Rows:", nrow(eval_df), "\\n")

# CRITICAL DATA LEAKAGE PREVENTION:
# Preprocessing statistics MUST be derived strictly from training_data.
`;

  if (imputeMissing) {
    script += `
# Calculate median/mode from training set ONLY
train_readiness_median <- median(train_df$work_readiness_score, na.rm = TRUE)
train_age_median       <- median(train_df$age, na.rm = TRUE)

# Apply training medians to both training and evaluation data
train_df$work_readiness_score[is.na(train_df$work_readiness_score)] <- train_readiness_median
eval_df$work_readiness_score[is.na(eval_df$work_readiness_score)]   <- train_readiness_median

train_df$age[is.na(train_df$age)] <- train_age_median
eval_df$age[is.na(eval_df$age)]   <- train_age_median
`;
  }

  if (modelType === 'logistic') {
    script += `
# 3. Train Logistic Regression Model
cat("Training Logistic Regression Model on ${prevRounds}...\\n")

model <- glm(
  employed_status ~ age + education_level + work_readiness_score + prev_employment_months + region,
  data = train_df,
  family = binomial(link = "logit")
)

summary(model)

# 4. Predict Employment Probabilities on Evaluation Round (${evalFile})
eval_preds <- predict(model, newdata = eval_df, type = "response")
`;
  } else if (modelType === 'randomForest') {
    script += `
# 3. Train Random Forest Model
cat("Training Random Forest Model on ${prevRounds}...\\n")
train_df$employed_status <- as.factor(train_df$employed_status)

set.seed(42)
model <- randomForest(
  employed_status ~ age + education_level + work_readiness_score + prev_employment_months + region,
  data = train_df,
  ntree = 300,
  mtry = 3,
  importance = TRUE
)

print(model)

# 4. Predict Employment Probabilities on Evaluation Round (${evalFile})
eval_preds <- predict(model, newdata = eval_df, type = "prob")[, "1"]
`;
  } else if (modelType === 'xgboost') {
    script += `
# 3. Prepare Feature Matrices for XGBoost
features <- c("age", "education_level", "work_readiness_score", "prev_employment_months")

dtrain <- xgb.DMatrix(
  data = as.matrix(train_df[, features]),
  label = train_df$employed_status
)

deval <- as.matrix(eval_df[, features])

# Train XGBoost Model
set.seed(42)
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.05,
  max_depth = 5,
  subsample = 0.8
)

cat("Training XGBoost Model on ${prevRounds}...\\n")
model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 150,
  verbose = 1
)

# 4. Predict Employment Probabilities on Evaluation Round (${evalFile})
eval_preds <- predict(model, deval)
`;
  } else if (modelType === 'glmnet') {
    script += `
# 3. Train Regularized Logistic Regression (ElasticNet / Lasso)
features_matrix <- model.matrix(employed_status ~ age + education_level + work_readiness_score + prev_employment_months + region, data = train_df)[, -1]
eval_matrix     <- model.matrix(~ age + education_level + work_readiness_score + prev_employment_months + region, data = eval_df)[, -1]

set.seed(42)
cv_fit <- cv.glmnet(features_matrix, train_df$employed_status, family = "binomial", alpha = 0.5, type.measure = "auc")

cat("Best Lambda (AUC):", cv_fit$lambda.min, "\\n")

# 4. Predict Probabilities
eval_preds <- predict(cv_fit, newx = eval_matrix, s = "lambda.min", type = "response")[, 1]
`;
  }

  script += `
# 5. Create Submission File with Required Columns: anonymised_id & employed_status
submission <- data.frame(
  anonymised_id = eval_df$anonymised_id,
  employed_status = eval_preds
)

# Verify probability range [0, 1] and no NAs
cat("Summary of Predicted Probabilities:\\n")
print(summary(submission$employed_status))

stopifnot(all(submission$employed_status >= 0 & submission$employed_status <= 1))
stopifnot(sum(is.na(submission$employed_status)) == 0)

# Write output CSV
write_csv(submission, "${outputFilename}")
cat("\\nSuccessfully saved prediction file to: ${outputFilename}\\n")
cat("Upload this file to the Local Evaluation System to get your ROC AUC Score!\\n")
`;

  return script;
}
