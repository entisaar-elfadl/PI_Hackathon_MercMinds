import { GroundTruthRow, MatchedPair, PredictionRow } from '../types';

export interface IdMatchResult {
  matchedPairs: MatchedPair[];
  matchedCount: number;
  missingInPredictionsCount: number; // Ground truth IDs not present in prediction
  extraInPredictionsCount: number; // Prediction IDs not present in ground truth
  duplicatePredictionIds: string[];
  unmatchedPredictionIds: string[];
  missingGroundTruthIds: string[];
}

/**
 * Matches prediction rows with ground truth rows using anonymised_id.
 * Never assumes rows are in the same order.
 */
export function matchPredictionsWithGroundTruth(
  predictions: PredictionRow[],
  groundTruth: GroundTruthRow[]
): IdMatchResult {
  // Map ground truth by anonymised_id
  const groundTruthMap = new Map<string, number>();
  for (const gt of groundTruth) {
    if (gt.anonymised_id && typeof gt.employed_status === 'number') {
      groundTruthMap.set(gt.anonymised_id.toString().trim(), gt.employed_status);
    }
  }

  const matchedPairs: MatchedPair[] = [];
  const seenPredictionIds = new Set<string>();
  const duplicatePredictionIdsSet = new Set<string>();
  const unmatchedPredictionIds: string[] = [];
  const matchedGtIdsSet = new Set<string>();

  for (const pred of predictions) {
    const rawId = pred.anonymised_id?.toString().trim();
    if (!rawId) continue;

    // Check for duplicates in predictions
    if (seenPredictionIds.has(rawId)) {
      duplicatePredictionIdsSet.add(rawId);
      continue; // Skip processing duplicate rows or log warning
    }
    seenPredictionIds.add(rawId);

    if (groundTruthMap.has(rawId)) {
      const actual = groundTruthMap.get(rawId)!;
      matchedGtIdsSet.add(rawId);
      matchedPairs.push({
        anonymised_id: rawId,
        actual,
        predicted: pred.employed_status,
        diff: Math.abs(actual - pred.employed_status),
      });
    } else {
      unmatchedPredictionIds.push(rawId);
    }
  }

  // Find ground truth IDs that were not matched in predictions
  const missingGroundTruthIds: string[] = [];
  groundTruthMap.forEach((_, id) => {
    if (!matchedGtIdsSet.has(id)) {
      missingGroundTruthIds.push(id);
    }
  });

  return {
    matchedPairs,
    matchedCount: matchedPairs.length,
    missingInPredictionsCount: missingGroundTruthIds.length,
    extraInPredictionsCount: unmatchedPredictionIds.length,
    duplicatePredictionIds: Array.from(duplicatePredictionIdsSet),
    unmatchedPredictionIds,
    missingGroundTruthIds,
  };
}
