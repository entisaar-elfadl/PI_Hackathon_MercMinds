# 🏗️ Application Architecture & Engineering Design

This document describes the software architecture, data pipelines, mathematical algorithms, and key design decisions implemented in the **Employment Prediction Local Evaluator**.

---

## 🗺️ Architectural Topology (Client-Side)

To guarantee 100% data privacy and eliminate server latency or cloud hosting costs, the evaluator operates **entirely in the user's web browser**. Large survey files are parsed and evaluated locally, ensuring that proprietary predictions never leak to third-party APIs or external servers.

```
┌────────────────────────────────────────────────────────┐
│                        BROWSER                         │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌────────────────────────┐    ┌────────────────────┐  │
│  │    CSV Upload UI       │───&gt;│   CSV Parser       │  │
│  │   (Drag &amp; Drop / Click)│    │ (Double-quotes, LF)│  │
│  └────────────────────────┘    └─────────┬──────────┘  │
│                                          │             │
│                                          ▼             │
│  ┌────────────────────────┐    ┌────────────────────┐  │
│  │   Ground-Truth Loader  │───&gt;│ Validation Engine  │  │
│  │ (Simulation / Upload)  │    │(Range &amp; bounds check) │
│  └────────────────────────┘    └─────────┬──────────┘  │
│                                          │             │
│                                          ▼             │
│  ┌────────────────────────┐    ┌────────────────────┐  │
│  │  ROC Curve &amp; Charts    │&lt;───│     ID Matcher     │  │
│  │ (Custom SVG Renderer)  │    │(O(N) Map Alignment)│  │
│  └────────────────────────┘    └─────────┬──────────┘  │
│                                          │             │
│                                          ▼             │
│  ┌────────────────────────┐    ┌────────────────────┐  │
│  │   Experiment Ledger    │&lt;───│   AUC Calculator   │  │
│  │    (Local Storage)     │    │ (Ties Integration) │  │
│  └────────────────────────┘    └────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## ⚙️ Data Flow & Evaluation Pipeline

The evaluation pipeline follows a strict, non-destructive sequence:

1. **Select Validation Round**: Users specify which historical round (Round 6, 7, or 8) they are simulating.
2. **Load Ground Truth**:
   - *Default:* A deterministic seedable simulation reproduces the exact survey parameters.
   - *Custom:* A manually uploaded survey answer key is read.
3. **Parse and Validate Submissions**:
   - Ensure the presence of `anonymised_id` and `employed_status`.
   - Validate numerical bounds (probabilities must strictly lie in the interval `[0.0, 1.0]`).
   - Throw descriptive, line-numbered user errors for invalid inputs.
4. **Anonymised ID Matching**:
   - Align prediction records to ground truth.
   - Report counts of missing predictions and unexpected extra entries.
5. **Statistical Metrics Calculation**:
   - Extract aligned probabilities and binary outcomes.
   - Run ties-aware Trapezoidal ROC integration.
6. **Render Results**:
   - Draw coordinates for the ROC Curve and category density plots.
   - Commit results to the persistent ledger.

---

## 🔍 Key Engineering Subsystems

### 1. Robust CSV Parser (`src/utils/csv.ts`)
Standard string splitting on commas fails when cells contain commas or double-quoted fields. Our custom parser implements a state-machine that:
* Toggles an `inQuotes` flag when encountering `"` characters.
* Handles escaped double quotes (`""`).
* Accounts for both Unix (`\n`) and Windows (`\r\n`) line terminators.
* Trims cellular whitespace natively.

### 2. High-Performance ID Alignment (`src/components/EvaluationSuite.tsx`)
Rather than running nested loops ($O(N^2)$ complexity) to align predictions with ground truths—which would cause browser freezing on large survey datasets—we load prediction IDs into a hash map:
* **Insertion Complexity:** $O(N)$
* **Matching Complexity:** $O(M)$ lookup where $M$ is the ground-truth length.
* This ensures instant alignment, even with files containing tens of thousands of rows.

### 3. Ties-Aware ROC AUC Calculator (`src/utils/auc.ts`)
When models output identical probabilities (ties), standard sorting-based rank evaluations can skew results depending on sort stability. 

To resolve this, we calculate the Area Under the Receiver Operating Characteristic (ROC) curve using **trapezoidal integration**:
1. Sort actual/predicted pairs descending.
2. Accumulate tied probabilities together inside a single step.
3. Traverse the curve, adding trapezoidal segments to the area:
   $$\Delta \text{Area} = (\text{FPR}_i - \text{FPR}_{i-1}) \times \frac{\text{TPR}_i + \text{TPR}_{i-1}}{2}$$
This guarantees a mathematically perfect AUC matching scikit-learn and R output exactly.

### 4. Custom Lightweight SVG Rendering (`src/components/RocCurve.tsx`)
Using external Canvas or charting libraries introduces dependency weight and risks React 19 version mismatches. We build our charts as native, fully responsive SVG vector paths:
* **Scale Translation:** Custom scaling maps mathematical `[0.0, 1.0]` coordinates directly onto pixel dimensions.
* **Interactive Tooltip:** Real-time Euclidean distance checks find the nearest coordinate to the user's cursor on mousemove, rendering a detailed popup containing thresholds, TPR, and FPR.

### 5. Persistent Ledger (`src/App.tsx`)
Experiment tracking uses `localStorage` to save model metadata, versioning, run timestamp, notes, and AUC. On initialization, if no history is present, the app seeds historical benchmarks (XGBoost base, Logistic Baseline) to immediately provide a rich comparative view and demonstrate cross-round stability diagnostics.

### 6. Frosted Glass Design System Integration
To establish a premium, high-tech engineering feel, the application has been designed with a custom **Frosted Glass (Glassmorphism)** dark aesthetic:
* **Background Atmosphere:** A deep dark base layer (`#020617`) with cool radial neon blue highlights and subtle backdrop blurs (`backdrop-blur-md`).
* **Visual Hierarchy:** Rather than deep nested card-in-card structures, clean boundaries are defined via high-contrast borders (`border-slate-800/80`) and varying opacity backdrops (`bg-slate-900/40`, `bg-slate-950/60`).
* **Color Schemes & Legibility:** Strict light-on-dark contrast ratios exceeding WCAG AA standards are enforced across all text, status metrics, and controls. Custom bright color accents (neon blue `#3b82f6` for ROC paths, vibrant emerald `#10b981` for correct classifications, and rose `#f43f5e` for negatives) make mathematical distributions pop with clarity.

