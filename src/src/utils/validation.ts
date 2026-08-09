import { GroundTruthRow, PredictionRow, ValidationError, ValidationSummary } from '../types';
import { matchPredictionsWithGroundTruth } from './idMatcher';

/**
 * Performs rigorous CSV structure, probability range, and ID integrity validation.
 */
export function validatePredictions(
  rawParsedData: any[],
  groundTruth: GroundTruthRow[]
): {
  validPredictions: PredictionRow[];
  validationSummary: ValidationSummary;
} {
  const errors: ValidationError[] = [];
  const warnings: ValidationError[] = [];

  if (!rawParsedData || rawParsedData.length === 0) {
    errors.push({
      type: 'parse_error',
      severity: 'error',
      message: 'Prediction file is empty or could not be parsed.',
    });

    return {
      validPredictions: [],
      validationSummary: {
        isValid: false,
        totalRowsParsed: 0,
        validPredictionCount: 0,
        invalidPredictionCount: 0,
        errors,
        warnings,
        hasRequiredColumns: false,
        idMatchStats: {
          totalGroundTruth: groundTruth.length,
          matchedCount: 0,
          missingInPredictions: groundTruth.length,
          extraInPredictions: 0,
          duplicateInPredictions: 0,
        },
      },
    };
  }

  // Column name normalization check
  const sampleRow = rawParsedData[0] || {};
  const columns = Object.keys(sampleRow).map((col) => col.trim());

  const hasIdCol = columns.some((col) => col.toLowerCase() === 'anonymised_id');
  const hasStatusCol = columns.some((col) => col.toLowerCase() === 'employed_status');

  if (!hasIdCol || !hasStatusCol) {
    const missingCols: string[] = [];
    if (!hasIdCol) missingCols.push('anonymised_id');
    if (!hasStatusCol) missingCols.push('employed_status');

    errors.push({
      type: 'missing_column',
      severity: 'error',
      message: `Prediction file is missing required column(s): ${missingCols.join(', ')}.`,
      details: `Found columns: ${columns.length > 0 ? columns.join(', ') : 'none'}`,
    });

    return {
      validPredictions: [],
      validationSummary: {
        isValid: false,
        totalRowsParsed: rawParsedData.length,
        validPredictionCount: 0,
        invalidPredictionCount: rawParsedData.length,
        errors,
        warnings,
        hasRequiredColumns: false,
        idMatchStats: {
          totalGroundTruth: groundTruth.length,
          matchedCount: 0,
          missingInPredictions: groundTruth.length,
          extraInPredictions: 0,
          duplicateInPredictions: 0,
        },
      },
    };
  }

  // Find exact key names from raw data (e.g. case insensitive match)
  const idKey = Object.keys(sampleRow).find((k) => k.trim().toLowerCase() === 'anonymised_id') || 'anonymised_id';
  const statusKey = Object.keys(sampleRow).find((k) => k.trim().toLowerCase() === 'employed_status') || 'employed_status';

  const validPredictions: PredictionRow[] = [];
  let invalidCount = 0;
  const sampleInvalidIds: string[] = [];
  const sampleOutOfBoundsIds: string[] = [];

  let naNanInfCount = 0;
  let outOfBoundsCount = 0;

  for (let idx = 0; idx < rawParsedData.length; idx++) {
    const row = rawParsedData[idx];
    const rawId = row[idKey] !== undefined && row[idKey] !== null ? String(row[idKey]).trim() : '';
    const rawVal = row[statusKey];

    if (!rawId) {
      invalidCount++;
      continue;
    }

    // Parse probability
    const valString = String(rawVal).trim().toLowerCase();
    const parsedNum = Number(valString);

    if (
      rawVal === undefined ||
      rawVal === null ||
      valString === '' ||
      valString === 'na' ||
      valString === 'nan' ||
      valString === 'null' ||
      valString === 'inf' ||
      valString === '-inf' ||
      isNaN(parsedNum) ||
      !isFinite(parsedNum)
    ) {
      naNanInfCount++;
      invalidCount++;
      if (sampleInvalidIds.length < 5) sampleInvalidIds.push(rawId);
      continue;
    }

    if (parsedNum < 0 || parsedNum > 1) {
      outOfBoundsCount++;
      invalidCount++;
      if (sampleOutOfBoundsIds.length < 5) sampleOutOfBoundsIds.push(`${rawId} (${parsedNum})`);
      continue;
    }

    validPredictions.push({
      anonymised_id: rawId,
      employed_status: parsedNum,
      rawRowNumber: idx + 1,
    });
  }

  if (naNanInfCount > 0) {
    errors.push({
      type: 'invalid_value',
      severity: 'error',
      message: `Found ${naNanInfCount} prediction value(s) that are NA, NaN, infinite, or non-numeric.`,
      count: naNanInfCount,
      sampleIds: sampleInvalidIds,
      details: 'Predictions must be finite numbers representing probabilities.',
    });
  }

  if (outOfBoundsCount > 0) {
    errors.push({
      type: 'out_of_bounds',
      severity: 'error',
      message: `Found ${outOfBoundsCount} prediction value(s) outside the valid range [0, 1].`,
      count: outOfBoundsCount,
      sampleIds: sampleOutOfBoundsIds,
      details: 'Competition metric (ROC AUC) requires probabilities between 0 and 1.',
    });
  }

  // ID matching & duplicate check against ground truth
  const idMatch = matchPredictionsWithGroundTruth(validPredictions, groundTruth);

  if (idMatch.duplicatePredictionIds.length > 0) {
    warnings.push({
      type: 'duplicate_id',
      severity: 'warning',
      message: `Found ${idMatch.duplicatePredictionIds.length} duplicate anonymised_id(s) in prediction file.`,
      count: idMatch.duplicatePredictionIds.length,
      sampleIds: idMatch.duplicatePredictionIds.slice(0, 5),
      details: 'First occurrence was retained; subsequent duplicates were ignored.',
    });
  }

  if (idMatch.missingInPredictionsCount > 0) {
    warnings.push({
      type: 'unmatched_id',
      severity: 'warning',
      message: `${idMatch.missingInPredictionsCount} observation(s) in evaluation dataset were missing in predictions.`,
      count: idMatch.missingInPredictionsCount,
      sampleIds: idMatch.missingGroundTruthIds.slice(0, 5),
    });
  }

  if (idMatch.extraInPredictionsCount > 0) {
    warnings.push({
      type: 'unmatched_id',
      severity: 'warning',
      message: `${idMatch.extraInPredictionsCount} ID(s) in prediction file were not found in the evaluation dataset.`,
      count: idMatch.extraInPredictionsCount,
      sampleIds: idMatch.unmatchedPredictionIds.slice(0, 5),
    });
  }

  const isValid = errors.length === 0 && idMatch.matchedCount > 0;

  return {
    validPredictions,
    validationSummary: {
      isValid,
      totalRowsParsed: rawParsedData.length,
      validPredictionCount: validPredictions.length,
      invalidPredictionCount: invalidCount,
      errors,
      warnings,
      hasRequiredColumns: true,
      idMatchStats: {
        totalGroundTruth: groundTruth.length,
        matchedCount: idMatch.matchedCount,
        missingInPredictions: idMatch.missingInPredictionsCount,
        extraInPredictions: idMatch.extraInPredictionsCount,
        duplicateInPredictions: idMatch.duplicatePredictionIds.length,
      },
    },
  };
}
