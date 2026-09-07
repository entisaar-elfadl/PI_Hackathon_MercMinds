# Usage Guide

## Run the Dashboard

From the repository root:

```powershell
cd src
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

## Evaluate a Prediction File

1. Open **Survey Datasets** and download a practice training set and its matching evaluation answers.
2. Train a model using the training data.
3. Export predictions as a CSV with these columns:

	```csv
	anonymised_id,employed_status
	ID_10,0.31476
	ID_100,0.52781
	```

4. Open **Model Evaluator** and select the matching practice round.
5. Upload the prediction CSV and run the evaluation.
6. Review the ROC AUC, matched IDs, missing IDs, extra IDs, ROC curve, and prediction distribution.
7. Save the result to the local **Experiment Ledger** when the evaluation is complete.

## Evaluate Custom Ground Truth

Choose **Real Evaluation**, then upload both files. The ground-truth CSV must contain `anonymised_id` and binary `employed_status` values (`0` or `1`). The prediction CSV must contain the same ID column and probability values between `0` and `1`.

## Python Modelling Pipeline

From the repository root, ensure the datasets are available in `assets/dataset/`, then run:

```powershell
python demo\final_submission.py
```

The generated submission file is `final_winning_submission.csv`.

## Notes

- Practice datasets are deterministic and are intended for local testing only.
- The dashboard stores experiment metadata in the browser's local storage.
- Do not use the practice answer files as a substitute for the competition's hidden Round 9 evaluation.