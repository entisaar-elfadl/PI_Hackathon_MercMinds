# MercMinds Project Overview

## Project Name

**MercMinds Employment Prediction**

## Summary

MercMinds develops machine-learning models that estimate whether a participant in a longitudinal South African youth survey will be employed in a future survey round. The project combines feature engineering, model experimentation, and a local browser-based evaluator for comparing prediction files with known historical outcomes.

## Key Features

- Longitudinal modelling using historical survey rounds.
- Feature engineering for employment history, education, geography, and socioeconomic indicators.
- Python modelling pipeline in [`demo/final_submission.py`](final_submission.py).
- Local React/Vite evaluator for CSV validation, ID matching, ROC AUC scoring, and experiment tracking.
- Deterministic practice datasets for Round 6, Round 7, and Round 8 evaluation workflows.
- Documentation and experiment outputs for comparing modelling approaches.

## Motivation

Youth unemployment is a major social and economic challenge in South Africa. Reliable estimates of employment transitions can support better research, programme evaluation, and policy decisions. This project focuses on honest out-of-round validation so that models are assessed on their ability to generalise beyond the data used for training.

## Future Improvements

- Add automated tests and continuous integration for the Python and TypeScript workflows.
- Move large browser-side CSV evaluations to Web Workers to keep the interface responsive.
- Add reproducible experiment configuration files and a single root-level command interface.
- Expand model explainability and fairness analysis across demographic and geographic groups.
