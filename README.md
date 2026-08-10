# MercMinds: Labour Market Status Prediction

This project addresses the Predictive Insights labour market challenge, where the goal is to predict whether a participant is employed in Round 9 of a longitudinal youth survey using data from earlier rounds.

## Overview
South Africa has one of the highest youth unemployment rates globally, and understanding who moves into and out of work is essential for labour market policy and economic planning. In this challenge, we use anonymised historical panel data to estimate the probability that each participant is employed in Round 9.

The dataset includes demographic, geographic, educational, and labour market history information. Our task is to build a model that learns from this longitudinal behaviour and predicts employment status as a binary target, evaluated using AUC.

## Problem Statement
Each participant is tracked across multiple survey rounds. The task is to predict whether they are employed in Round 9, based on their earlier-round characteristics and labour-market history.

This is a real-world machine learning problem involving:
- Longitudinal panel data
- Demographic and socioeconomic features
- Missing or noisy survey data
- Binary classification with class imbalance
- Evaluation by AUC rather than simple accuracy

## Objective
Build a predictive model that assigns a probability of employment for each participant in the Round 9 cohort. The target is to maximise ranking quality on the hidden test set, measured by AUC.

## Project Focus
This repository provides the base structure for:
- Exploratory data analysis
- Feature engineering across survey rounds
- Baseline modelling and benchmarking
- Model comparison and submission preparation
- Documentation and project reporting

## Repository Structure
```text
├── assets/
├── demo/
├── docs/
├── scripts/
├── src/
├── train.csv
├── test.csv
├── README.md
├── LICENSE
└── package.json
```

## Evaluation
Submissions are evaluated using the Area Under the ROC Curve (AUC), which measures how well the model ranks participants by likelihood of being employed. A score closer to 1.0 indicates stronger discrimination.

## Challenge Context
Hosted by Predictive Insights in collaboration with the Alphawave Group and Harambee, the challenge is designed to explore how data science and AI can help understand unemployment and opportunity in South Africa.

## Team
MercMinds

## Status
This repository is being used to develop the modelling workflow and supporting documentation for the hackathon challenge.

---
