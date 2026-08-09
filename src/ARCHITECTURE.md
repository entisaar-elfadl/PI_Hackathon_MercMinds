# System Architecture & Technical Specification

## Overview

The **Employment Prediction Evaluator** is a local, client-side evaluation system for testing statistical and machine learning models on the Employment Prediction Competition datasets.

---

## Architecture & Data Flow

```text
[ Prediction CSV File / R Output ]
               │
               ▼
   ┌──────────────────────┐
   │    PapaParse CSV     │
   │    Streaming Parser  │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Validation Engine    │
   │  - Col Check         │
   │  - NA / NaN / Inf    │
   │  - Range [0, 1]      │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Map-Based ID Matcher │
   │  (anonymised_id)     │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Mann-Whitney U Engine│
   │  - Rank ascending    │
   │  - Average ties      │
   │  - ROC Curve Points  │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Local Storage State  │
   │ & Experiment Matrix  │
   └──────────────────────┘
```

---

## Core Technical Components

### 1. ROC AUC Evaluation Engine (`src/utils/aucCalculator.ts`)
Calculates ROC AUC using the Wilcoxon Mann-Whitney U statistic:

$$U = R_1 - \frac{n_1(n_1 + 1)}{2}$$

$$\text{AUC} = \frac{U}{n_1 \times n_0}$$

* Handles tied predictions by assigning average fractional ranks.
* Time Complexity: $O(N \log N)$ due to probability sorting.
* Space Complexity: $O(N)$ for pair storage.

### 2. ID Matcher (`src/utils/idMatcher.ts`)
* Uses a `Map<string, number>` lookup for ground truth IDs.
* Never assumes prediction rows match ground truth row order.
* Detects duplicate prediction IDs, missing prediction IDs, and unexpected extra IDs.

### 3. Prediction Validator (`src/utils/validation.ts`)
* Ensures file contains required headers `anonymised_id` and `employed_status`.
* Validates that all prediction values are finite numbers between 0 and 1.
* Generates clear, non-cryptic error and warning logs with affected ID samples.

### 4. Experiment & History Storage (`src/services/experimentStorage.ts`)
* Maintains local persistence using browser `localStorage` / `IndexedDB`.
* Groups individual evaluation runs into model experiments linked by `modelName` and `modelVersion`.
* Calculates summary statistics across rounds:
  * Average AUC: $\mu = \frac{1}{K} \sum \text{AUC}_k$
  * Standard Deviation: $\sigma = \sqrt{\frac{1}{K-1} \sum (\text{AUC}_k - \mu)^2}$
  * Min / Max AUC across historical rounds.

---

## Directory Structure

```text
src/
├── types/
│   └── index.ts                 # Explicit TypeScript interfaces
├── utils/
│   ├── aucCalculator.ts         # Mann-Whitney U AUC & ROC points
│   ├── validation.ts            # CSV structure & probability checks
│   ├── idMatcher.ts             # Map-based ID alignment
│   ├── stats.ts                 # Mean, std dev, formatting
│   └── rCodeGenerator.ts        # R template generator
├── data/
│   ├── historicalDatasets.ts    # Round 6-8 ground truth datasets
│   └── sampleModels.ts          # Benchmark models for testing
├── services/
│   ├── csvParser.ts             # PapaParse wrapper
│   └── experimentStorage.ts     # LocalStorage state manager
├── components/
│   ├── layout/                  # Navbar, DisclaimerBanner
│   ├── common/                  # MetricCard, StatusBadge
│   ├── charts/                  # RocCurveChart, ProbabilityDistChart, ModelComparisonChart
│   ├── evaluation/              # CsvUploader, ValidationReportView, ThresholdAnalyzer, IdAuditTable
│   ├── rgenerator/              # RCodeModal
│   └── leaderboard/             # LeaderboardTable
├── pages/
│   ├── DashboardPage.tsx
│   ├── EvaluationPage.tsx
│   ├── ResultsPage.tsx
│   ├── ComparisonPage.tsx
│   ├── LeaderboardPage.tsx
│   └── DatasetExplorerPage.tsx
├── tests/
│   └── evaluationEngine.test.ts # Unit tests for math & logic
└── App.tsx                      # Root application & router
```

---

## Performance & Optimization

* **In-Memory Calculations**: All array transformations and sorting operate in client memory for instant execution (< 50ms for 10,000 rows).
* **Roc Point Downsampling**: ROC curve coordinates are downsampled to a maximum of 200 points for smooth Recharts rendering without canvas lag.
* **Responsive Layout**: Designed with Tailwind CSS supporting mobile, tablet, and desktop views with responsive tables and flex containers.
