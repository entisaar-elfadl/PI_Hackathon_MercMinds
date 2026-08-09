# 🚀 Employment Prediction — Local Kaggle-Style Testing System

A production-grade, local, browser-based evaluation system for the **Employment Prediction Competition**. This platform simulates the real Kaggle evaluation workflow by validating model predictions against historical survey rounds where true employment outcomes are already known. 

With this tool, you can easily answer: **"Does my R/Python model actually generalise to a future survey round?"**

---

## 📋 Table of Contents
- [Competition Context](#-competition-context)
- [How Historical Validation Works](#-how-historical-validation-works)
- [Supported Rounds](#-supported-rounds)
- [Prediction CSV Format](#-prediction-csv-format)
- [Area Under the Curve (AUC)](#-area-under-the-curve-auc)
- [How to Use the System](#-how-to-use-the-system)
- [Installation & Setup](#-installation--setup)

---

## 🎯 Competition Context
The objective of this competition is to predict the employment status of survey participants:
* `0` = unemployed
* `1` = employed

Rather than outputting a hard classification, models must output a **probability between 0 and 1** indicating the likelihood of being employed. 

### Key Concept
A probability of `0.73` should not be evaluated as a binary right/wrong prediction. Instead, the metric evaluates the model's ability to rank employed individuals higher than unemployed individuals.

---

## 🔄 How Historical Validation Works
To prevent **overfitting** to a single dataset, the platform implements a cross-round validation pipeline. This mimics the real competition's hidden Round 9 evaluation. 

The strategy ensures you do not use future data during model training (preventing **data leakage**):

```
Rounds 1–5 Training Data 
      ↓ (Train R/Python Model)
Predict probabilities for Round 6
      ↓ (Upload Predictions to Evaluator)
Calculate ROC AUC against true Round 6 outcomes
```

---

## 📊 Supported Rounds

### Round 6 Evaluation Pack
* **Training Data:** `rounds_1_5.csv` (Survey Rounds 1 through 5)
* **Evaluation Data:** `round_6.csv` (Round 6 outcomes, acting as your local answer key)

### Round 7 Evaluation Pack
* **Training Data:** `rounds_1_6.csv` (Survey Rounds 1 through 6)
* **Evaluation Data:** `round_7.csv` (Round 7 outcomes, acting as your local answer key)

### Round 8 Evaluation Pack
* **Training Data:** `rounds_1_7.csv` (Survey Rounds 1 through 7)
* **Evaluation Data:** `round_8.csv` (Round 8 outcomes, acting as your local answer key)

---

## 📄 Prediction CSV Format
The platform accepts standard CSV files with the following headers:

```csv
anonymised_id,employed_status
ID_10,0.31476
ID_100,0.52781
ID_1000,0.82123
```

* `anonymised_id`: Unique identifier used for ID matching. Rows do not need to be in any specific order.
* `employed_status`: Represents your model's **predicted probability** (must be numeric, between 0.0 and 1.0, and cannot contain `NA`, `NaN`, or infinite values).

---

## 📈 Area Under the Curve (AUC)
The primary evaluation metric is **ROC AUC (Area Under the Receiver Operating Characteristic Curve)**.
* **1.00** = Perfect separation of classes.
* **0.90** = Excellent generalisation.
* **0.70** = Moderate/Good performance.
* **0.50** = Equivalent to random guessing.

The score displayed in the platform is simply `AUC × 100` (e.g., `AUC: 0.75321` corresponds to a **Kaggle score of 75.32%**).

---

## 🖥️ How to Use the System

1. **Download Data**: Go to the **Survey Datasets** tab, and download your training (e.g., `rounds_1_5.csv`) and answer keys.
2. **Train your R Model**: Write your model script in R (e.g., Logistic Regression, Random Forest, XGBoost) using the training data.
3. **Generate Predictions**: Export a prediction CSV for the evaluation round (e.g., Round 6).
4. **Evaluate**: Go to the **Model Evaluator** tab, select **Round 6**, drag & drop your prediction CSV, and click **Run Historical Evaluation**.
5. **Analyze Diagnostics**: View your ROC Curve, prediction density separation chart, matched ID metrics, and save the run to your local **Experiment Ledger**.
6. **Compare**: View the **Model Comparison** tab to analyze model consistency and select your best candidate for Kaggle's final hidden Round 9.

---

## ⚙️ Installation & Setup

### Prerequisites
* [Node.js](https://nodejs.org/) (v18 or higher)
* npm

### Steps
1. Install dependencies:
   ```bash
   npm install
   ```
2. Start the development server:
   ```bash
   npm run dev
   ```
3. Open `http://localhost:3000` in your web browser.
4. Run the production build command:
   ```bash
   npm run build
   ```
