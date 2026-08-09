# Employment Prediction — Local Kaggle-Style Testing System

A production-grade local evaluation and testing platform for the Employment Prediction Competition.

This system allows machine learning engineers and data scientists to evaluate their R/Python prediction models against historical survey rounds (Rounds 6, 7, and 8) where actual employment ground truth outcomes are known.

> **Important Distinction:**
> **Historical Round Evaluation ≠ Official Kaggle Score**
> Historical evaluation uses known ground-truth outcomes from survey Rounds 6, 7 & 8 to estimate model generalisation. The real competition's Round 9 test outcomes remain hidden.

---

## Competition Context

* **Target Variable**: `employed_status` (0 = unemployed, 1 = employed)
* **Model Output**: Predicted probability between `0.0` and `1.0` (not a hard 0/1 classification)
* **Evaluation Metric**: **ROC AUC** (Area Under Receiver Operating Characteristic Curve)

---

## Historical Validation Strategy

To simulate the hidden Round 9 competition evaluation without data leakage, the testing system supports three primary historical survey validation rounds:

| Evaluation Round | Training Dataset | Evaluation Dataset | Simulates |
|---|---|---|---|
| **Round 6** | Rounds 1–5 (`rounds_1_5.csv`) | Round 6 (`round_6.csv`) | Predicting 1-step future survey round |
| **Round 7** | Rounds 1–6 (`rounds_1_6.csv`) | Round 7 (`round_7.csv`) | Predicting 1-step future survey round |
| **Round 8** | Rounds 1–7 (`rounds_1_7.csv`) | Round 8 (`round_8.csv`) | Predicting 1-step future survey round |

Evaluating models across all three rounds answers the critical question:
> **"Does this model consistently generalise to a future survey round?"**

---

## Supported Prediction CSV Format

Prediction files generated from R or Python models must contain two columns:

```csv
anonymised_id,employed_status
ID_10,0.3147682
ID_100,0.5278193
ID_1000,0.8212341
```

* `anonymised_id`: Unique participant identifier string matching the evaluation dataset.
* `employed_status`: Predicted probability of employment between `0.0` and `1.0`.

---

## Key Features

1. **Strict CSV Validation & Probability Checks**: Detects missing columns, non-numeric values (`NA`, `NaN`, `Inf`), out-of-bounds values (`<0` or `>1`), duplicate IDs, and unmatched IDs.
2. **Wilcoxon Rank-Sum ROC AUC Calculator**: Computes exact ROC AUC scores handling probability ties cleanly. Displays score as both AUC decimal (e.g., `0.55366`) and percentage (e.g., `55.37%`).
3. **Interactive Visualizations**:
   - Area-shaded ROC Curve (TPR vs FPR) with diagonal random-guess baseline.
   - Predicted Probability Distribution Histogram (overlaid density for actual `0`s vs actual `1`s).
   - Dynamic Threshold Classifier (slide cutoff from `0.00` to `1.00` to inspect 2x2 Confusion Matrix, Precision, Recall, Accuracy, Specificity, F1 Score).
4. **Multi-Round Comparison Matrix**: Compares models side-by-side across Round 6, Round 7, and Round 8. Computes Average AUC, Min/Max, and Standard Deviation to flag high-variance overfitted models.
5. **R Code Generator**: Generates clean, leakage-free R scripts (`glm`, `randomForest`, `xgboost`, `glmnet`) that perform median imputation strictly on training data and export properly formatted submission CSVs.
6. **Local Leaderboard & Privacy**: Operates entirely in the browser using IndexedDB / LocalStorage. Zero prediction data is sent to external servers.

---

## Installation & Running

### Dependencies
* Node.js v18+
* npm

### Setup
```bash
# Clone repository
git clone <repo-url>
cd employment-prediction-evaluator

# Install dependencies
npm install

# Start development server
npm run dev
```

The application will launch on `http://localhost:3000`.

---

## Statistical Principles

### Generalisation Over Single-Round Peaks
A model with stable performance across rounds (e.g. `0.62`, `0.63`, `0.61` | Std Dev = `0.010`) is significantly more reliable for hidden Round 9 than an overfitted model with single-round spikes (e.g. `0.70`, `0.51`, `0.52` | Std Dev = `0.106`).

### Data Leakage Prevention
Preprocessing statistics (such as median imputation or scaling factors) must be calculated **strictly from the training dataset** (e.g. `rounds_1_5.csv`) and applied to the evaluation dataset (`round_6.csv`). Calculating preprocessing statistics from future evaluation data constitutes data leakage.
