import { GroundTruthRow, PredictionRow, RoundId, SampleModel } from '../types';
import { HISTORICAL_GROUND_TRUTH } from './historicalDatasets';

function generateModelPredictions(
  groundTruth: GroundTruthRow[],
  modelNoiseLevel: number,
  bias: number,
  inconsistencyFactor: number = 0
): PredictionRow[] {
  return groundTruth.map((gt, idx) => {
    // Generate realistic probability prediction based on actual status
    const actual = gt.employed_status;
    const baseProb = actual === 1 ? 0.65 : 0.35;

    // Pseudo-random noise
    const pseudoRand = ((idx * 17 + 13) % 100) / 100 - 0.5;
    let pred = baseProb + (actual === 1 ? 1 : -1) * (0.25 - modelNoiseLevel) + pseudoRand * modelNoiseLevel + bias;

    if (inconsistencyFactor !== 0) {
      pred += (idx % 2 === 0 ? 1 : -1) * inconsistencyFactor;
    }

    // Clamp between 0.01 and 0.99
    const clamped = Math.max(0.01, Math.min(0.99, pred));
    return {
      anonymised_id: gt.anonymised_id,
      employed_status: Math.round(clamped * 10000) / 10000,
    };
  });
}

export const SAMPLE_MODELS: SampleModel[] = [
  {
    id: 'logistic_3_3',
    name: 'Logistic Regression 3.3',
    version: '3.3.0',
    description: 'Refined logit model with non-linear interaction terms and median imputation.',
    expectedAucByRound: { round_6: 0.5919, round_7: 0.6231, round_8: 0.6382 },
    predictionsByRound: {
      round_6: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_6, 0.22, 0.02),
      round_7: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_7, 0.20, 0.01),
      round_8: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_8, 0.19, 0.01),
    },
  },
  {
    id: 'logistic_3_2',
    name: 'Logistic Regression 3.2',
    version: '3.2.0',
    description: 'Standard logit model with age and education level polynomial features.',
    expectedAucByRound: { round_6: 0.5712, round_7: 0.6014, round_8: 0.6158 },
    predictionsByRound: {
      round_6: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_6, 0.26, 0.01),
      round_7: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_7, 0.24, 0.01),
      round_8: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_8, 0.22, 0.01),
    },
  },
  {
    id: 'logistic_3_1',
    name: 'Logistic Regression 3.1',
    version: '3.1.0',
    description: 'Basic linear logistic regression baseline.',
    expectedAucByRound: { round_6: 0.5537, round_7: 0.5821, round_8: 0.6013 },
    predictionsByRound: {
      round_6: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_6, 0.30, 0.0),
      round_7: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_7, 0.28, 0.0),
      round_8: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_8, 0.26, 0.0),
    },
  },
  {
    id: 'xgboost_2_1',
    name: 'XGBoost Tuned 2.1',
    version: '2.1.0',
    description: 'Gradient boosted trees with max_depth=4 and eta=0.03 trained across rounds.',
    expectedAucByRound: { round_6: 0.6210, round_7: 0.6480, round_8: 0.6610 },
    predictionsByRound: {
      round_6: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_6, 0.17, 0.02),
      round_7: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_7, 0.15, 0.02),
      round_8: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_8, 0.14, 0.02),
    },
  },
  {
    id: 'overfitted_tree',
    name: 'Overfitted Decision Tree (High Variance)',
    version: '1.0.0-overfit',
    description: 'Deep unpruned tree that fits noise in earlier rounds but suffers in generalisation.',
    expectedAucByRound: { round_6: 0.7020, round_7: 0.5140, round_8: 0.5210 },
    predictionsByRound: {
      round_6: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_6, 0.10, 0.05),
      round_7: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_7, 0.38, -0.05, 0.15),
      round_8: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_8, 0.37, -0.05, 0.14),
    },
  },
  {
    id: 'random_baseline',
    name: 'Random Guessing Baseline',
    version: '0.0.1',
    description: 'Uninformative uniform random prediction benchmark.',
    expectedAucByRound: { round_6: 0.5010, round_7: 0.4980, round_8: 0.5020 },
    predictionsByRound: {
      round_6: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_6, 0.50, 0.0),
      round_7: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_7, 0.50, 0.0),
      round_8: generateModelPredictions(HISTORICAL_GROUND_TRUTH.round_8, 0.50, 0.0),
    },
  },
];
