/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

interface RocPoint {
  fpr: number;
  tpr: number;
  threshold: number;
}

/**
 * Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC)
 * and returns coordinates for plotting the ROC curve.
 * 
 * It handles tied prediction probabilities correctly and yields identical results
 * to Python scikit-learn (sklearn.metrics.roc_auc_score) and R (pROC).
 * 
 * @param actual Binary true labels (0 or 1)
 * @param predicted Predicted probabilities (0.0 to 1.0)
 */
export function calculateAUC(
  actual: number[],
  predicted: number[]
): {
  auc: number;
  fpr: number[];
  tpr: number[];
  thresholds: number[];
} {
  const n = actual.length;
  if (n === 0) {
    return { auc: 0.5, fpr: [0, 1], tpr: [0, 1], thresholds: [1, 0] };
  }

  // Create pairs
  const pairs = actual.map((label, idx) => ({
    label,
    prob: predicted[idx],
  }));

  // Sort descending by predicted probability
  pairs.sort((a, b) => b.prob - a.prob);

  // Count positives (1) and negatives (0)
  let posCount = 0;
  let negCount = 0;
  for (let i = 0; i < n; i++) {
    if (pairs[i].label === 1) posCount++;
    else negCount++;
  }

  // Handle boundary cases (only one class present in actual labels)
  if (posCount === 0 || negCount === 0) {
    return { auc: 0.5, fpr: [0, 1], tpr: [0, 1], thresholds: [1, 0] };
  }

  const fpr: number[] = [0];
  const tpr: number[] = [0];
  const thresholds: number[] = [1];

  let currentFp = 0;
  let currentTp = 0;
  let auc = 0;
  let prevFp = 0;
  let prevTp = 0;

  let i = 0;
  while (i < n) {
    const currentProb = pairs[i].prob;

    // Accumulate all elements with identical predicted probability (ties)
    while (i < n && pairs[i].prob === currentProb) {
      if (pairs[i].label === 1) {
        currentTp++;
      } else {
        currentFp++;
      }
      i++;
    }

    const currentFpr = currentFp / negCount;
    const currentTpr = currentTp / posCount;

    fpr.push(currentFpr);
    tpr.push(currentTpr);
    thresholds.push(currentProb);

    // Calculate trapezoidal area under ROC step
    const deltaFpr = currentFpr - (prevFp / negCount);
    const avgTpr = (currentTpr + (prevTp / posCount)) / 2;
    auc += deltaFpr * avgTpr;

    prevFp = currentFp;
    prevTp = currentTp;
  }

  // Ensure end-point coordinates
  if (fpr[fpr.length - 1] !== 1 || tpr[tpr.length - 1] !== 1) {
    fpr.push(1);
    tpr.push(1);
    thresholds.push(0);
  }

  return {
    auc: parseFloat(auc.toFixed(5)),
    fpr,
    tpr,
    thresholds,
  };
}

/**
 * Standard deviation of a list of numbers (for cross-round evaluations).
 */
export function calculateStdDev(values: number[]): number {
  const len = values.length;
  if (len <= 1) return 0;
  const mean = values.reduce((sum, v) => sum + v, 0) / len;
  const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (len - 1);
  return parseFloat(Math.sqrt(variance).toFixed(5));
}
