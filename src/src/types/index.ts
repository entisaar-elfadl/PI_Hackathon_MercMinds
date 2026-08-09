/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type RoundId = 'round_6' | 'round_7' | 'round_8' | 'custom';

export interface RoundConfig {
  id: RoundId;
  name: string;
  trainingDesc: string;
  evaluationDesc: string;
  trainingDatasetName: string;
  evaluationDatasetName: string;
  sampleSize: number;
}

export interface Prediction {
  anonymised_id: string;
  employed_status: number; // This represents the predicted probability (0 to 1)
}

export interface GroundTruth {
  anonymised_id: string;
  employed_status: number; // This represents the actual binary class (0 or 1)
  [key: string]: any;      // Other features can be present
}

export interface ValidationError {
  row?: number;
  column?: string;
  value?: string;
  message: string;
  severity: 'error' | 'warning';
}

export interface EvaluationResult {
  roundId: RoundId;
  modelName: string;
  modelVersion: string;
  auc: number;
  score: number; // AUC * 100
  totalObservations: number;
  validObservations: number;
  removedObservations: number;
  missingGroundTruth: number;
  matchedCount: number;
  missingPredictionCount: number; // IDs in ground truth but missing in prediction
  extraPredictionCount: number;   // IDs in prediction but missing in ground truth
  duplicatePredictionCount: number;
  predictionFile: string;
  timestamp: string;
  notes: string;
  rocCurve: {
    fpr: number[];
    tpr: number[];
    thresholds: number[];
  };
}

export interface Experiment {
  id: string;
  modelName: string;
  modelVersion: string;
  roundId: RoundId;
  auc: number;
  score: number;
  totalObservations: number;
  validObservations: number;
  predictionFile: string;
  timestamp: string;
  notes: string;
}

export interface ModelComparison {
  modelName: string;
  modelVersion: string;
  roundScores: Record<RoundId, number | null>; // Score (AUC) per round
  averageAuc: number;
  minAuc: number;
  maxAuc: number;
  stdDev: number;
}
