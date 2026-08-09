import { calculateRocAuc, calculateThresholdMetrics } from '../utils/aucCalculator';
import { matchPredictionsWithGroundTruth } from '../utils/idMatcher';
import { validatePredictions } from '../utils/validation';
import { MatchedPair, GroundTruthRow, PredictionRow } from '../types';

export function runEvaluationEngineTests(): { passed: number; total: number; logs: string[] } {
  const logs: string[] = [];
  let passed = 0;
  let total = 0;

  function assert(condition: boolean, testName: string) {
    total++;
    if (condition) {
      passed++;
      logs.push(`[PASS] ${testName}`);
    } else {
      logs.push(`[FAIL] ${testName}`);
    }
  }

  // TEST 1: Exact AUC calculation from competition prompt example
  // Actual: 1, 0, 1, 0
  // Predicted: 0.90, 0.70, 0.60, 0.20
  // Expected AUC: 0.75
  const promptPairs: MatchedPair[] = [
    { anonymised_id: 'ID_1', actual: 1, predicted: 0.9 },
    { anonymised_id: 'ID_2', actual: 0, predicted: 0.7 },
    { anonymised_id: 'ID_3', actual: 1, predicted: 0.6 },
    { anonymised_id: 'ID_4', actual: 0, predicted: 0.2 },
  ];

  const res1 = calculateRocAuc(promptPairs);
  assert(res1.auc === 0.75, `Prompt Example AUC = 0.75 (Got: ${res1.auc})`);
  assert(res1.scorePercentage === 75.0, `Prompt Example Percentage = 75.00% (Got: ${res1.scorePercentage}%)`);

  // TEST 2: Perfect AUC (1.0)
  const perfectPairs: MatchedPair[] = [
    { anonymised_id: 'ID_1', actual: 1, predicted: 0.95 },
    { anonymised_id: 'ID_2', actual: 1, predicted: 0.85 },
    { anonymised_id: 'ID_3', actual: 0, predicted: 0.30 },
    { anonymised_id: 'ID_4', actual: 0, predicted: 0.10 },
  ];
  const res2 = calculateRocAuc(perfectPairs);
  assert(res2.auc === 1.0, `Perfect Separation AUC = 1.0 (Got: ${res2.auc})`);

  // TEST 3: Ties handling
  const tiedPairs: MatchedPair[] = [
    { anonymised_id: 'ID_1', actual: 1, predicted: 0.5 },
    { anonymised_id: 'ID_2', actual: 0, predicted: 0.5 },
    { anonymised_id: 'ID_3', actual: 1, predicted: 0.8 },
    { anonymised_id: 'ID_4', actual: 0, predicted: 0.2 },
  ];
  const res3 = calculateRocAuc(tiedPairs);
  assert(typeof res3.auc === 'number' && !isNaN(res3.auc), 'Tied predictions AUC calculated cleanly');

  // TEST 4: ID Matcher with duplicate & missing IDs
  const groundTruth: GroundTruthRow[] = [
    { anonymised_id: 'ID_10', employed_status: 0 },
    { anonymised_id: 'ID_100', employed_status: 1 },
    { anonymised_id: 'ID_1000', employed_status: 1 },
  ];

  const predictions: PredictionRow[] = [
    { anonymised_id: 'ID_1000', employed_status: 0.82 }, // Reordered ID
    { anonymised_id: 'ID_10', employed_status: 0.31 },
    { anonymised_id: 'ID_10', employed_status: 0.99 }, // Duplicate
    { anonymised_id: 'ID_UNKNOWN', employed_status: 0.50 }, // Extra ID
  ];

  const matchRes = matchPredictionsWithGroundTruth(predictions, groundTruth);
  assert(matchRes.matchedCount === 2, `ID Matcher matched 2 valid IDs regardless of order (Got: ${matchRes.matchedCount})`);
  assert(matchRes.duplicatePredictionIds.includes('ID_10'), 'Detected duplicate ID_10');
  assert(matchRes.extraInPredictionsCount === 1, 'Detected 1 extra unknown ID');
  assert(matchRes.missingInPredictionsCount === 1, 'Detected 1 missing ground truth ID (ID_100)');

  // TEST 5: Validator detects NA and Out-Of-Bounds values
  const rawDataWithErrors = [
    { anonymised_id: 'ID_1', employed_status: '0.72' },
    { anonymised_id: 'ID_2', employed_status: 'NA' },
    { anonymised_id: 'ID_3', employed_status: '1.45' }, // Out of bounds > 1
    { anonymised_id: 'ID_4', employed_status: '-0.20' }, // Out of bounds < 0
  ];

  const valRes = validatePredictions(rawDataWithErrors, groundTruth);
  assert(valRes.validationSummary.invalidPredictionCount === 3, `Validator flagged 3 invalid rows (Got: ${valRes.validationSummary.invalidPredictionCount})`);
  assert(valRes.validationSummary.validPredictionCount === 1, `Validator extracted 1 valid probability row (Got: ${valRes.validationSummary.validPredictionCount})`);

  return { passed, total, logs };
}
