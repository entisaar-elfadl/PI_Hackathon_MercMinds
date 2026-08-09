import { MatchedPair, RocPoint, ThresholdMetrics } from '../types';

/**
 * Calculates the exact ROC AUC score using the Mann-Whitney U statistic (Wilcoxon rank-sum method).
 * This method handles ties in predicted probabilities cleanly and accurately.
 *
 * @param pairs Array of matched ground truth (0 or 1) and predicted probability (0.0 - 1.0)
 * @returns Object with AUC (0.0 to 1.0) and percentage score (0.00 to 100.00)
 */
export function calculateRocAuc(pairs: MatchedPair[]): {
  auc: number;
  scorePercentage: number;
  positivesCount: number;
  negativesCount: number;
} {
  if (!pairs || pairs.length === 0) {
    return { auc: 0.5, scorePercentage: 50.0, positivesCount: 0, negativesCount: 0 };
  }

  // Filter valid binary targets (actual === 0 or 1) and finite predicted numbers
  const validPairs = pairs.filter(
    (p) => (p.actual === 0 || p.actual === 1) && typeof p.predicted === 'number' && !isNaN(p.predicted) && isFinite(p.predicted)
  );

  let n1 = 0; // Number of positive class (employed = 1)
  let n0 = 0; // Number of negative class (unemployed = 0)

  for (let i = 0; i < validPairs.length; i++) {
    if (validPairs[i].actual === 1) n1++;
    else if (validPairs[i].actual === 0) n0++;
  }

  // If there are no positive or negative examples, AUC cannot be determined
  if (n1 === 0 || n0 === 0) {
    return { auc: 0.5, scorePercentage: 50.0, positivesCount: n1, negativesCount: n0 };
  }

  // Sort pairs in ascending order by predicted probability
  // Use index tracking to handle ties
  const sorted = [...validPairs].sort((a, b) => a.predicted - b.predicted);

  // Assign ranks with average ranking for ties
  let sumPositiveRanks = 0;
  let i = 0;

  while (i < sorted.length) {
    let j = i;
    // Find all items with equal predicted probability
    while (j < sorted.length && sorted[j].predicted === sorted[i].predicted) {
      j++;
    }

    // Average rank for tied group (1-indexed ranks)
    // Ranks from (i + 1) to j
    const averageRank = (i + 1 + j) / 2;

    // Add rank for each positive in the tied group
    for (let k = i; k < j; k++) {
      if (sorted[k].actual === 1) {
        sumPositiveRanks += averageRank;
      }
    }

    i = j;
  }

  // U statistic calculation: U = R1 - (n1 * (n1 + 1)) / 2
  const uStat = sumPositiveRanks - (n1 * (n1 + 1)) / 2;
  const rawAuc = uStat / (n1 * n0);

  // Clamp AUC between 0 and 1 (rounding precision to 5 decimal places)
  const clampedAuc = Math.max(0, Math.min(1, rawAuc));
  const roundedAuc = Math.round(clampedAuc * 100000) / 100000;
  const scorePercentage = Math.round(clampedAuc * 10000) / 100;

  return {
    auc: roundedAuc,
    scorePercentage,
    positivesCount: n1,
    negativesCount: n0,
  };
}

/**
 * Generates ROC curve coordinate points (FPR vs TPR) for charting.
 * Downsamples points if dataset is extremely large to maintain smooth rendering performance.
 */
export function generateRocPoints(pairs: MatchedPair[], maxPoints: number = 200): RocPoint[] {
  if (!pairs || pairs.length === 0) {
    return [
      { fpr: 0, tpr: 0, threshold: 1 },
      { fpr: 1, tpr: 1, threshold: 0 },
    ];
  }

  const validPairs = pairs.filter(
    (p) => (p.actual === 0 || p.actual === 1) && typeof p.predicted === 'number' && !isNaN(p.predicted)
  );

  const totalPositives = validPairs.filter((p) => p.actual === 1).length;
  const totalNegatives = validPairs.filter((p) => p.actual === 0).length;

  if (totalPositives === 0 || totalNegatives === 0) {
    return [
      { fpr: 0, tpr: 0, threshold: 1 },
      { fpr: 1, tpr: 1, threshold: 0 },
    ];
  }

  // Sort descending by predicted probability
  const sorted = [...validPairs].sort((a, b) => b.predicted - a.predicted);

  const points: RocPoint[] = [{ fpr: 0, tpr: 0, threshold: 1.01 }];

  let tp = 0;
  let fp = 0;

  for (let i = 0; i < sorted.length; i++) {
    const item = sorted[i];
    if (item.actual === 1) {
      tp++;
    } else {
      fp++;
    }

    // Record point whenever probability changes or at the last element
    const nextItem = sorted[i + 1];
    if (!nextItem || nextItem.predicted !== item.predicted) {
      const tpr = tp / totalPositives;
      const fpr = fp / totalNegatives;
      points.push({
        fpr: Math.round(fpr * 10000) / 10000,
        tpr: Math.round(tpr * 10000) / 10000,
        threshold: Math.round(item.predicted * 10000) / 10000,
      });
    }
  }

  // Ensure start and end points exist
  if (points[points.length - 1].fpr < 1 || points[points.length - 1].tpr < 1) {
    points.push({ fpr: 1, tpr: 1, threshold: 0 });
  }

  // Downsample if points exceed maxPoints
  if (points.length <= maxPoints) {
    return points;
  }

  const sampledPoints: RocPoint[] = [points[0]];
  const step = (points.length - 1) / (maxPoints - 1);

  for (let i = 1; i < maxPoints - 1; i++) {
    const idx = Math.floor(i * step);
    sampledPoints.push(points[idx]);
  }

  sampledPoints.push(points[points.length - 1]);
  return sampledPoints;
}

/**
 * Calculates classification threshold metrics (Accuracy, Precision, Recall, F1, Specificity)
 * for a given cutoff probability threshold (default 0.50).
 */
export function calculateThresholdMetrics(pairs: MatchedPair[], threshold: number = 0.5): ThresholdMetrics {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;

  for (const pair of pairs) {
    const isPredictedPositive = pair.predicted >= threshold;
    const isActualPositive = pair.actual === 1;

    if (isPredictedPositive && isActualPositive) tp++;
    else if (isPredictedPositive && !isActualPositive) fp++;
    else if (!isPredictedPositive && !isActualPositive) tn++;
    else if (!isPredictedPositive && isActualPositive) fn++;
  }

  const total = tp + fp + tn + fn;
  const accuracy = total > 0 ? (tp + tn) / total : 0;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;

  return {
    threshold,
    tp,
    fp,
    tn,
    fn,
    accuracy: Math.round(accuracy * 10000) / 10000,
    precision: Math.round(precision * 10000) / 10000,
    recall: Math.round(recall * 10000) / 10000,
    f1Score: Math.round(f1Score * 10000) / 10000,
    specificity: Math.round(specificity * 10000) / 10000,
  };
}
