# FINAL TECHNICAL REPORT: PREDICTIVE MODELING OF YOUTH LABOUR MARKET DYNAMICS IN SOUTH AFRICA

**Project / Competition:** PI Hackathon — MercMinds  
**Author / Candidate:** Moegamat Samsodien  
**Recipient / Reviewer:** `jean@predictiveinsights.net`  
**Evaluation Metric:** ROC-AUC / Binary Cross-Entropy (Log-Loss)  
**Final Peak Performance:** **`0.66054`** *(Achieved independently via Exp 68 & Exp 93)*  
**Baseline Progression:** `0.59229` $\rightarrow$ **`0.66054`** ($+0.06825$ AUC Lift)

---

## Executive Summary

This report provides a comprehensive summary of the machine learning methodology, domain feature engineering, mathematical modeling, and empirical findings developed over **100 iterative experiments** to predict youth employment status in South Africa (`employed_status`).

Starting from a baseline Logistic Regression score of **`0.59229`**, our pipeline systematically evolved through deep multi-layer perceptrons, domain-specific labor transition equations, **Structural Equation Modeling (SEM) Latent Factor representations**, and culminated in a **Simplex-Constrained Non-Negative (NNLS) Super-Learner** achieving an all-time peak public leaderboard score of **`0.66054`**.

```
[0.59229] Baseline Logistic Regression
    │
    ▼  (+0.0522)
[0.64453] Deep Multi-Layer Perceptrons (MLP)
    │
    ▼  (+0.0079)
[0.65247] Stratified 5-Fold Cross-Validation Blending
    │
    ▼  (+0.0045)
[0.65693] Titan Multi-Seed Linear-Neural Ensemble
    │
    ▼  (+0.0014)
[0.65832] Structural Equation Modeling (SEM Latent Manifold)
    │
    ▼  (+0.0022)
[0.66054] Simplex-Constrained Convex Super-Learner (All-Time Peak)
```

### Key Technical Findings:
1. **The Dataset Structure:** Every row in `train.csv` ($16,653$ cleaned observations) and `test.csv` ($3,230$ observations) represents a unique individual. Individual longitudinal histories were pre-summarized inline within the lag columns (`current_round`, `lag_round`, `employed_lag`, `tenure_lag`, `days_since_last_obs`).
2. **The Decision Tree Failure Mode:** Tree-based models (Random Forests, Extra Trees, HistGradientBoosting, and AutoGluon) consistently underperformed on the test set (**`0.613 – 0.645`**). The sparse one-hot encoded features and high-cardinality survey text caused decision trees to overfit the training waves ($1\dots8$) and collapse on the unseen future evaluation cohort (Round 9).
3. **The SEM Latent Factor Breakthrough:** Decomposing collinear questionnaire responses into **8 orthogonal latent factors** (*Academic Human Capital*, *Labour Market Momentum*, *Socio-Economic Readiness*, and *Supervised PLS Projections*) eliminated multicollinearity and unlocked direct linear pathways to the target.
4. **The Winning Convex Sweet Spot:** An ensemble combining **Quantile-Gaussian ElasticNet ($\sim 44\%$)**, **Deep Multi-Layer Perceptrons ($\sim 45\%$)**, and **Standard ElasticNet SAGA ($\sim 11\%$)** optimized on the unit simplex ($w_i \ge 0, \sum w_i = 1$) proved mathematically optimal across multiple seeds.

---

## 1. Problem Architecture & Domain Context

### 1.1 Objective & Target Definition
The objective is to predict the binary probability of formal/informal employment (`employed_status` $\in \{0, 1\}$) for a cohort of young South African jobseekers participating in **Round 9** of a longitudinal survey, trained exclusively on historical observations from **Rounds 1–8**.

* **Target Variable:** `employed_status` ($1 = \text{Employed}$, $0 = \text{Unemployed}$).
* **Target Distribution:** $68.31\%$ Class 0 (Unemployed), $31.69\%$ Class 1 (Employed).
* **Dataset Scale:** 
  * Training Set: $16,719$ rows ($16,653$ complete after removing target NaNs).
  * Evaluation Test Set: $3,230$ rows (strictly Round 9).
  * Competition Split: $50\%$ Public ($1,615$ rows) / $50\%$ Private ($1,615$ rows).

### 1.2 The 5 Structural Pillars of Youth Unemployment in South Africa
Through exploratory domain analysis, the dataset features were categorized into 5 structural pillars:
1. **Labour Market Transition & Momentum:** `employed_lag`, `status_lag`, `status_broad_lag`, `tenure_lag`, `days_since_last_obs`, `total_historical_rounds`.
2. **STEM & Cognitive Gateways:** `matric_mathpure`, `matric_mathlit`, `matric_physicalscience`, `matric_englishhome`, `matric_englishadd`.
3. **Institutional Education & Qualifications:** `education_level`, `education_schooling_grade_twelve_equiv`, `institution_type`, `school_quintile`, `education_field`, `seta`.
4. **Spatial & Economic Hub Agglomeration:** `province`, `district`, `municipality`.
5. **Demographics & Behavioural Employability:** `age`, `gender`, `race`, `sa_citizen`, `work_readiness_score`.

---

## 2. Forensic Exploratory Data Analysis & Empirical Discoveries

```
                               DATASET ARCHITECTURE
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │  Train Set (Rounds 1–8): 16,653 observations (Unique Respondents)           │
 │  Test Set  (Round 9):     3,230 observations (Unique Respondents)           │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │  • Participant Overlap between Train & Test: 0.0% (Pre-summarized inline)  │
 │  • Missingness in Lag Columns == Informative First-Time Entrant Signal      │
 │  • Matric Performance: Banded text strings ('50 - 59 %', '80 - 100 %')      │
 └─────────────────────────────────────────────────────────────────────────────┘
```

### Discovery 1: Zero ID Overlap Across Files (Pre-Summarized Inline Panels)
Cross-referencing `anonymised_id` between `train.csv` and `test.csv` revealed **$0$ overlapping IDs ($0.0\%$)**, and $16,653$ unique IDs in the training set. 
* **Implication:** The dataset is not a multi-row raw panel table; each row is a pre-aggregated cross-sectional snapshot summarizing the respondent's historical trajectory inside the `_lag` and `historical_rounds` columns.

### Discovery 2: Informative Missingness in Lag Features
Missing values in `employed_lag`, `days_since_last_obs`, and `tenure_lag` do not represent random omissions ($MCAR$). They denote **First-Time Survey Entrants** who have no prior round history. 
* Imputing missing lags with the mean/median collapsed new jobseekers with experienced unemployed participants.
* **Solution:** Creating an explicit tri-state variable (`employed_lag_num` $\in \{-1 = \text{First Time}, 0 = \text{Unemployed}, 1 = \text{Employed}\}$) and a binary flag `is_first_time` preserved this distinction.

### Discovery 3: Ordinal Structure in Matric Percentage Bands
Matric subjects were recorded as banded string intervals: `"80 - 100 %"`, `"70 - 79 %"`, `"50 - 59 %"`, `"< 30 %"`.
* One-hot encoding these strings destroyed their monotonic numerical ranking ($90\% > 74.5\% > 54.5\% > 20\%$).
* **Solution:** A dedicated parser extracted continuous interval midpoints, enabling the calculation of composite scores: `best_math_score = max(Pure Math, Math Lit)` and `stem_composite = (Pure Math + Physical Science) / 2`.

---

## 3. The Structural Equation Modeling (SEM) Framework

The primary mathematical breakthrough of the campaign (responsible for breaking through from `0.6569` to **`0.65832`** in Exp 59) was the introduction of **Structural Equation Modeling (SEM) Latent Measurement Blocks**.

### 3.1 Mathematical Theory of SEM in Survey Data
Individual questionnaire responses are noisy, highly collinear manifestations of unobserved latent socio-economic constructs. We structured the feature space into 4 orthogonal measurement blocks:

$$\mathbf{X}_{\text{raw}} \xrightarrow{\text{Factor Analysis}} \boldsymbol{\eta}_{\text{latent}} \xrightarrow{\text{Supervised PLS}} \mathbf{z}_{\text{path}} \xrightarrow{\text{Convex Ensemble}} P(\text{Employed} = 1)$$

```
                      SEM MEASUREMENT MODEL (LATENT SUBSPACE)
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                                                                            │
 │  [ Matric Math, English, Science, Grade 12 ] ──► ( η₁: Academic Capital )  │
 │                                                                            │
 │  [ Employed Lag, Tenure Log, Days Gap, First ] ──► ( η₂: Labour Momentum ) │
 │                                                                            │
 │  [ School Quintile, Work Readiness, Age ] ──► ( η₃: Socio-Economic Cap. )  │
 │                                                                            │
 │  [ Full Covariance Block x Target ] ──► ( z_pls: Supervised PLS Vector )   │
 │                                                                            │
 └────────────────────────────────────────────────────────────────────────────┘
```

1. **Block A (Academic Human Capital $\boldsymbol{\eta}_1$):** Factor analysis extracting shared variance across continuous matric marks (`matric_*_score`).
2. **Block B (Labour Market Momentum $\boldsymbol{\eta}_2$):** Factor analysis capturing employment persistence, tenure stability (`tenure_lag_log`), and survey contact gap (`days_since_obs_log`).
3. **Block C (Socio-Economic Readiness $\boldsymbol{\eta}_3$):** Factor analysis modeling school funding quintiles (`school_quintile_num`), behavioral work readiness (`work_readiness_num`), and maturity (`age_clean`).
4. **Block D (Supervised Partial Least Squares $\mathbf{z}_{\text{PLS}}$):** A 2-component PLS projection maximizing direct linear covariance with `employed_status`.

---

## 4. Comprehensive Experimental Journey & Milestones

The table below documents key progression milestones across the experimental campaign:

| Experiment | Paradigm / Methodology | Key Architectural Innovations | OOF CV (AUC) | Public Score | Status / Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 30** | Baseline Linear Model | Basic Logistic Regression on raw features | $0.5982$ | **`0.59229`** | Baseline anchor |
| **Exp 31** | Tree Ensemble | ExtraTrees Classifier with Gini criterion | $0.6215$ | **`0.61732`** | Tree sparsity penalty |
| **Exp 33** | Neural Network | Multi-Layer Perceptron (`MLP 64-32`, ReLU) | $0.6481$ | **`0.64453`** | Non-linear interaction lift |
| **Exp 35** | Hybrid Voting | Soft Voting: 75% MLP (`64-32`) + 25% LogReg | $0.6542$ | **`0.65089`** | Linear + Neural synergy |
| **Exp 37** | Stratified CV | 5-Fold Cross-Validation OOF Probability Blending | $0.6616$ | **`0.65247`** | Variance reduction |
| **Exp 41** | Sequential Validation | Expanding-window evaluation across Rounds 6–8 | $0.6588$ | **`0.65325`** | Temporal calibration |
| **Exp 44** | Regularized Linear Suite | Top-3 Linear Ensemble ($L_2$, ElasticNet SAGA, Quantile) | $0.6621$ | **`0.65630`** | Regularization dominance |
| **Exp 51** | Titan Multi-Seed | 80% Linear Champions + 20% Neural MLP (7 seeds) | $0.6634$ | **`0.65693`** | Previous best benchmark |
| **Exp 58** | Stacking Trees + MLP | Meta-Learner over Trees + Neural Networks | $0.6241$ | **`0.61899`** | Tree squashing failure |
| **Exp 59** | **Pure SEM Latent Titan** | **8 Orthogonal SEM Factors + 7-Seed Titan Suite** | **$0.6648$** | **`0.65832`** | **Major structural lift** |
| **Exp 68** | **5-Fold NNLS Super-Learner** | **Non-Negative Least Squares ($w \ge 0$) on SEM Manifold** | **$0.6668$** | **`0.66054`** | **All-Time Peak #1** |
| **Exp 71** | **Grand Master Stacker** | **Multi-Seed Consensus Fusion in Logit Space** | **$0.6669$** | **`0.66051`** | **Peak #2 (Safety Hedge)** |
| **Exp 78** | Caruana DP Stacker | Discrete State Memoization ($T=150$) on SEM | $0.6671$ | **`0.65995`** | Combinatorial optimization |
| **Exp 88** | Multi-Framework GBDT | HistGB + CatBoost + ElasticNet + MLP | $0.6602$ | **`0.65391`** | GBDT drag confirmation |
| **Exp 93** | **Simplex Convex SLSQP** | **Direct Simplex Cross-Entropy Minimization** | **$0.6669$** | **`0.66054`** | **All-Time Peak #2** |
| **Exp 96** | Prebuilt AutoGluon | AutoGluon `best_quality` Multi-Layer Stacker | $0.6684$ | **`0.63622`** | Text n-gram test collapse |
| **Exp 97** | Quantile Power Blend | Exact Prior Power Calibration ($R^{1.6576}$) | $0.6668$ | **`0.66027`** | Strict monotonic calibration |
| **Exp 100** | Centurion Grand Finale | 25-Fold Simplex SLSQP + Multi-Peak Fusion | $0.6672$ | **`0.66029`** | Master stability engine |

---

## 5. Forensic Failure Analysis (What Failed and Why)

Understanding which methods failed provided the guardrails required to achieve our peak:

```
                            POST-MORTEM COMPARISON
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ 1. DECISION TREES & AUTOGLUON (0.618 – 0.645)                               │
 │    • Sparse One-Hot splits created high-variance step functions.            │
 │    • AutoGluon memorized 111 text n-grams that failed on Round 9.           │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ 2. UNREGULARIZED STACKING (0.61899)                                         │
 │    • Meta-learner placed a -2.08 negative coefficient on trees.             │
 │    • Output probability std collapsed to 0.029 (max probability 0.44).      │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ 3. IN-SAMPLE TARGET ENCODING (0.61344)                                      │
 │    • Target encoding on 16k rows caused severe training memorization.       │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ 4. PSEUDO-LABELING / SELF-TRAINING (0.65154)                                │
 │    • At 0.66 AUC, confident predictions contain 20–25% false labels.       │
 │    • Feedback loop amplified confirmation bias on Round 9.                  │
 └─────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 The Decision Tree & AutoML Failure (Exp 31, 58, 73, 96: `0.618 – 0.645`)
While GBDTs dominate continuous dense tabular benchmarks, they underperformed on this dataset:
* **The Sparsity Penalty:** One-hot encoding categories produced dozens of sparse binary columns. Tree models partitioned these with rigid step boundaries, overfitting the training survey rounds.
* **AutoGluon's Overfitting Mode (Exp 96: `0.63622`):** AutoGluon generated $111$ text n-gram features from `education_tertiary_qualification_name`. While training CV reached `0.6684`, the trees memorized specific training qualification strings, collapsing when evaluated on the unseen Round 9 test wave.

### 5.2 The Unconstrained Stacking Collapse (Exp 58: `0.61899`)
In Exp 58, stacking base models using an unconstrained linear meta-learner resulted in coefficients of `[+1.59, +0.82, -0.18, -2.08, +0.73]`.
* **The Squeeze:** The meta-learner assigned a large negative weight (`-2.08`) to the noisy Gradient Tree model to subtract its overconfidence.
* This compressed the output standard deviation down to **`0.029`**, capping maximum predictions at $0.440$ and destroying classification margins.

### 5.3 In-Sample Target Encoding Leakage (Exp 38, 72: `0.61344`)
Target encoding high-cardinality geographic codes (`municipality`, `district`) caused in-sample target memorization on training folds. The neural networks over-relied on these synthetic target numbers, resulting in a breakdown on the test set.

### 5.4 Pseudo-Labeling Confirmation Bias (Exp 92: `0.65154`)
With an intrinsic error rate of $\sim 34\%$ (AUC $\approx 0.66$), even high-confidence pseudo-labeled test rows ($P > 0.78$) contain false positives. Feeding them back into training created an error-propagation feedback loop that dragged the score down from `0.66054` to `0.65154`.

---

## 6. The Winning Architecture: Non-Negative Simplex Super-Learner

The definitive winning solution (achieved in **Exp 68** and validated in **Exp 93**) is built on two core components: **The SEM Latent Manifold** and **Convex Optimization on the Unit Simplex**.

```
                         THE WINNING EXP 68 / 93 ENGINE
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                                                                             │
 │                   [ RAW SURVEY DATA (16,653 Clean Rows) ]                   │
 │                                      │                                      │
 │                                      ▼                                      │
 │                    [ SEM LATENT MEASUREMENT MODEL ]                         │
 │                    • 8 Continuous Orthogonal Latent Factors                 │
 │                    • Low-cardinality Categories One-Hot Encoded (<= 100)    │
 │                    • High-cardinality Text Excluded                         │
 │                                      │                                      │
 │                                      ▼                                      │
 │                 [ 5-FOLD CROSS-VALIDATION OOF EXTRACTION ]                  │
 │                                      │                                      │
 │             ┌────────────────────────┼────────────────────────┐             │
 │             ▼                        ▼                        ▼             │
 │   [ Quantile ElasticNet ]    [ Deep ReLU MLP ]      [ Medium ReLU MLP ]     │
 │    (C=0.10, L1=0.15 SAGA)      (128-64 Layers)         (64-32 Layers)       │
 │         Weight: 43.8%           Weight: 25.5%          Weight: 19.8%        │
 │             │                        │                        │             │
 │             └────────────────────────┼────────────────────────┘             │
 │                                      │                                      │
 │                                      ▼ [ + Standard ElasticNet SAGA (10.9%) │
 │                                                                             │
 │                [ CONVEX NON-NEGATIVE SIMPLEX OPTIMIZATION ]                 │
 │                • Solves: min Log-Loss(y, Sigmoid(Z * w))                    │
 │                • Subject to: w_i >= 0, sum(w) = 1.0                         │
 │                                      │                                      │
 │                                      ▼                                      │
 │                [ TEST PROBABILITIES: Score = 0.66054 ]                      │
 │                                                                             │
 └─────────────────────────────────────────────────────────────────────────────┘
```

### 6.1 Mathematical Formulation of Simplex Convex Optimization
To avoid the squared-error distortion of standard OLS, Exp 93 solves for optimal weights $w \in \mathbb{R}^M$ directly on the unit simplex $\Delta^M = \{w \mid w_i \ge 0, \sum w_i = 1\}$ using Sequential Least Squares Programming (SLSQP):

$$\min_{w \in \Delta^M} -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln\left(\sigma(\mathbf{z}_i^T w)\right) + (1 - y_i) \ln\left(1 - \sigma(\mathbf{z}_i^T w)\right) \right]$$

$$\text{subject to } w_i \ge 0 \quad \forall i, \quad \sum_{i=1}^M w_i = 1.0$$

### 6.2 The Optimal 4-Pillar Convex Equilibrium
The optimizer converged to the following weight distribution:

| Base Estimator | Mathematical Role | Optimal Weight |
| :--- | :--- | :--- |
| **Quantile ElasticNet SAGA** | Gaussian-projected $L_1/L_2$ shrinkage; outlier-free linear baseline | **`43.8%`** |
| **Deep Neural MLP (128-64)** | High-capacity non-linear interactions across SEM latent variables | **`25.5%`** |
| **Medium Neural MLP (64-32)** | Regularized smooth non-linear boundaries; variance anchor | **`19.8%`** |
| **Standard ElasticNet SAGA** | Raw category log-odds regularizer | **`10.9%`** |
| **Pure $L_2$ Models** | Pruned automatically by optimizer (redundant vs ElasticNet) | **`0.0%`** |

* **Total Regularized Linear Weight:** **`54.7%`**
* **Total Deep Neural Network Weight:** **`45.3%`**

---

## 7. Competition Strategy & Final Submission Selections

The competition features a **50% Public / 50% Private Leaderboard Split** ($1,615$ public rows vs $1,615$ private rows):
* The difference between our peak (**`0.66054`**) and 1st place (**`0.66372`**) is **`+0.00318`**, which represents **just 5 out of 1,615 individuals** classified with slightly different confidence margins.
* Chasing those 5 public rows with complex, unregularized models risks severe private test set shake-ups.

### The Final 2 Selected Submissions for Grading:

```
                          FINAL SUBMISSION PORTFOLIO
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                                                                             │
 │  SELECTION 1: PRIMARY STANDALONE PEAK CHAMPION                              │
 │  • File: submission_exp93_simplex_convex_optimization.csv                   │
 │  • Score: 0.66054                                                           │
 │  • Architecture: 8 SEM Latent Factors + Simplex SLSQP Log-Loss Optimization  │
 │  • Role: Maximum Public Validation Alignment & Exact Loss Minimization      │
 │                                                                             │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │                                                                             │
 │  SELECTION 2: MULTI-SEED CONSENSUS SAFETY HEDGE                             │
 │  • File: submission_exp71_grand_master_calibrated_superlearner.csv          │
 │  • Score: 0.66051                                                           │
 │  • Architecture: Multi-Seed Fusion (Exp 68 NNLS + Exp 70 Repeated CV + SEM)│
 │  • Role: Defense Against Private Leaderboard Distribution Shift             │
 │                                                                             │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Conclusion & Methodological Recommendations

The 100-experiment campaign confirms that **domain-aligned representation learning and constrained meta-optimization are superior to raw algorithmic complexity on survey data**:

1. **Latent Decomposition Over Raw Item Fitting:** Reducing survey dimensionality through Structural Equation Modeling (Factor Analysis + PLS) allowed linear and neural models to generalize across survey rounds without memorizing noise.
2. **Convex Non-Negative Meta-Learning:** Constraining ensemble weights to positive values on the unit simplex ($w_i \ge 0, \sum w_i = 1$) solved the negative-weight variance collapse that impacted unconstrained stacking models.
3. **Linear-Neural Complementarity:** The optimal architecture on South African youth labor dynamics is a balanced cooperation between **Quantile-Gaussian ElasticNet ($55\%$)** and **Multi-Scale Multi-Layer Perceptrons ($45\%$)**.

The submitted solution is reproducible, mathematically grounded, and optimized for generalization on the final private leaderboard evaluation.