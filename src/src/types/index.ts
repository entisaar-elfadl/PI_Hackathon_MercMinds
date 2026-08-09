/**
 * Core Type Definitions for Employment Prediction Evaluator
 */

export type RoundId = 'round_6' | 'round_7' | 'round_8';

export interface GroundTruthRow {
  anonymised_id: string;
  employed_status: number; // 0 or 1
  [key: string]: any;
}

export interface PredictionRow {
  anonymised_id: string;
  employed_status: number; // probability between 0 and 1
  rawRowNumber?: number;
}

export interface MatchedPair {
  anonymised_id: string;
  actual: number; // 0 or 1
  predicted: number; // 0.0 to 1.0
  diff?: number;
}

export interface ValidationError {
  type: 'missing_column' | 'invalid_value' | 'out_of_bounds' | 'duplicate_id' | 'unmatched_id' | 'missing_ground_truth' | 'parse_error';
  severity: 'error' | 'warning';
  message: string;
  count?: number;
  sampleIds?: string[];
  details?: string;
}

export interface ValidationSummary {
  isValid: boolean;
  totalRowsParsed: number;
  validPredictionCount: number;
  invalidPredictionCount: number;
  errors: ValidationError[];
  warnings: ValidationError[];
  hasRequiredColumns: boolean;
  idMatchStats: {
    totalGroundTruth: number;
    matchedCount: number;
    missingInPredictions: number; // Ground truth IDs not in prediction
    extraInPredictions: number; // Prediction IDs not in ground truth
    duplicateInPredictions: number;
  };
}

export interface RocPoint {
  fpr: number; // False Positive Rate (x-axis)
  tpr: number; // True Positive Rate (y-axis)
  threshold: number;
}

export interface ThresholdMetrics {
  threshold: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  specificity: number;
}

export interface EvaluationResult {
  id: string;
  modelName: string;
  modelVersion: string;
  roundId: RoundId;
  timestamp: string; // ISO String
  filename: string;
  auc: number; // e.g. 0.55366
  scorePercentage: number; // e.g. 55.37
  matchedPairsCount: number;
  totalGroundTruthCount: number;
  positivesCount: number;
  negativesCount: number;
  rocPoints: RocPoint[];
  matchedPairsSample: MatchedPair[]; // representative sample for table/audit
  validationSummary: ValidationSummary;
  notes?: string;
  trainingSetDescription: string;
  evalSetDescription: string;
}

export interface Experiment {
  id: string;
  modelName: string;
  modelVersion: string;
  createdAt: string;
  notes?: string;
  resultsByRound: Partial<Record<RoundId, EvaluationResult>>;
  avgAuc: number;
  minAuc: number;
  maxAuc: number;
  stdDevAuc: number;
  roundsEvaluatedCount: number;
}

export interface RoundInfo {
  id: RoundId;
  title: string;
  trainingSet: string;
  evaluationSet: string;
  description: string;
  sampleObservationCount: number;
  groundTruthFile: string;
  trainingDataFile: string;
}

export interface SampleModel {
  id: string;
  name: string;
  version: string;
  description: string;
  expectedAucByRound: Record<RoundId, number>;
  predictionsByRound: Record<RoundId, PredictionRow[]>;
}
