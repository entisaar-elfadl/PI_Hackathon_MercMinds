# ============================================================
# EXPERIMENT 87 — THE FINAL GRAND MASTER CONSENSUS
# (PRECISION LOG-ODDS FUSION OF ALL >= 0.6598 TOP CHAMPIONS)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


print("============================================")
print("EXPERIMENT 87")
print("THE FINAL GRAND MASTER CONSENSUS")
print("============================================")


# ============================================================
# 1. DIRECTORY PATHS & DATA LOADING
# ============================================================

CURRENT_DIR = Path.cwd()

DATA_DIR = (CURRENT_DIR / "../../assets/dataset").resolve()
if not DATA_DIR.exists():
    DATA_DIR = CURRENT_DIR

test_raw = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_raw["anonymised_id"].copy()
target = "employed_status"


# ============================================================
# 2. DISCOVER & FUSE TOP 5 CHAMPION SUBMISSIONS
# ============================================================

print("\n============================================")
print("FUSING TOP >= 0.6598 CHAMPION PREDICTIONS IN LOGIT SPACE")
print("============================================")

champion_registry = {
    "submission_exp68_nnls_sem_superlearner.csv": 0.35,            # 0.66054 (Peak #1)
    "submission_exp71_grand_master_calibrated_superlearner.csv": 0.25, # 0.66051 (Peak #2)
    "submission_exp76_master_calibration_fusion.csv": 0.15,         # 0.66040
    "submission_exp70_repeated_5fold_nnls_superlearner.csv": 0.15,  # 0.65998
    "submission_exp86_inline_longitudinal_sem_titan.csv": 0.10     # 0.65985
}

discovered_logits = []
discovered_weights = []

for filename, weight in champion_registry.items():
    f_path = CURRENT_DIR / filename
    if f_path.exists():
        sub_df = pd.read_csv(f_path)
        if target in sub_df.columns and len(sub_df) == len(test_raw):
            p = np.clip(sub_df[target].to_numpy(), 1e-6, 1.0 - 1e-6)
            z = np.log(p / (1.0 - p))
            discovered_logits.append(z)
            discovered_weights.append(weight)
            print(f" -> [FUSED] {filename:<60} (Weight: {weight*100:.0f}%)")
    else:
        print(f" -> [MISSING] {filename}")

if len(discovered_logits) == 0:
    raise FileNotFoundError("No past champion submission files found in the current directory.")

# Normalize weights
total_w = sum(discovered_weights)
norm_weights = [w / total_w for w in discovered_weights]

# Master Logit Fusion
final_master_logits = sum(w * z for w, z in zip(norm_weights, discovered_logits))
final_probabilities = 1.0 / (1.0 + np.exp(-final_master_logits))

print(f"\nSuccessfully integrated {len(discovered_logits)} champion models.")


# ============================================================
# 3. VALIDATE & SAVE SUBMISSION FILE
# ============================================================

if len(final_probabilities) != len(test_raw):
    raise ValueError("Prediction count does not match test data.")
if np.isnan(final_probabilities).any():
    raise ValueError("Predictions contain NaN.")
if (final_probabilities < 0).any() or (final_probabilities > 1).any():
    raise ValueError("Predictions fall outside [0, 1].")

output_file = "submission_exp87_final_grand_consensus.csv"

submission = pd.DataFrame({
    "anonymised_id": test_ids,
    "employed_status": final_probabilities
})

submission.to_csv(output_file, index=False)


# ============================================================
# 4. SUMMARY & INSPECTION
# ============================================================

print("\n============================================")
print("EXPERIMENT 87 COMPLETE")
print("============================================")
print(f"Saved: {output_file}")
print(f"Rows: {len(submission)}")

print("\nPrediction summary:")
print(submission["employed_status"].describe())

print("\nFirst 10 predictions:")
print(submission.head(10))

print("\n============================================")
print("BENCHMARKS INCLUDED IN CONSENSUS")
print("============================================")
print("Exp 86 Inline-Longitudinal Titan : 0.65985")
print("Exp 70 Repeated 5-Fold NNLS      : 0.65998")
print("Exp 76 Master Calibration Fusion : 0.66040")
print("Exp 71 Grand Master Superlearner : 0.66051")
print("Exp 68 5-Fold NNLS Super Learner : 0.66054 (Personal Best)")
print("Exp 87 Final Grand Consensus     : READY")

print("\n============================================")
print("READY FOR KAGGLE SUBMISSION")
print("============================================")