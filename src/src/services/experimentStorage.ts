import { EvaluationResult, Experiment, RoundId } from '../types';
import { calculateMean, calculateStdDev } from '../utils/stats';

const STORAGE_KEY_EXPERIMENTS = 'employment_eval_experiments_v1';
const STORAGE_KEY_EVALUATIONS = 'employment_eval_results_v1';

export class ExperimentStorageService {
  /**
   * Loads all saved experiments from LocalStorage
   */
  static getExperiments(): Experiment[] {
    try {
      const data = localStorage.getItem(STORAGE_KEY_EXPERIMENTS);
      if (!data) return [];
      return JSON.parse(data) as Experiment[];
    } catch (e) {
      console.error('Failed to parse experiments from storage:', e);
      return [];
    }
  }

  /**
   * Loads all saved evaluation results
   */
  static getEvaluations(): EvaluationResult[] {
    try {
      const data = localStorage.getItem(STORAGE_KEY_EVALUATIONS);
      if (!data) return [];
      return JSON.parse(data) as EvaluationResult[];
    } catch (e) {
      console.error('Failed to parse evaluations from storage:', e);
      return [];
    }
  }

  /**
   * Saves a new evaluation result and updates/links its experiment
   */
  static saveEvaluationResult(result: EvaluationResult): Experiment {
    const existingEvals = this.getEvaluations();
    const existingExps = this.getExperiments();

    // Check if an experiment with same model name and version exists
    let experiment = existingExps.find(
      (e) => e.modelName.toLowerCase().trim() === result.modelName.toLowerCase().trim() &&
             e.modelVersion.toLowerCase().trim() === result.modelVersion.toLowerCase().trim()
    );

    if (!experiment) {
      experiment = {
        id: `exp_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
        modelName: result.modelName,
        modelVersion: result.modelVersion,
        createdAt: new Date().toISOString(),
        notes: result.notes || '',
        resultsByRound: {},
        avgAuc: 0,
        minAuc: 0,
        maxAuc: 0,
        stdDevAuc: 0,
        roundsEvaluatedCount: 0,
      };
      existingExps.push(experiment);
    }

    // Attach result to round
    experiment.resultsByRound[result.roundId] = result;

    // Recalculate summary metrics across evaluated rounds
    const roundResults = Object.values(experiment.resultsByRound).filter(Boolean) as EvaluationResult[];
    const aucValues = roundResults.map((r) => r.auc);

    experiment.roundsEvaluatedCount = roundResults.length;
    experiment.avgAuc = Math.round(calculateMean(aucValues) * 100000) / 100000;
    experiment.minAuc = Math.min(...aucValues);
    experiment.maxAuc = Math.max(...aucValues);
    experiment.stdDevAuc = Math.round(calculateStdDev(aucValues) * 100000) / 100000;

    // Save back to storage
    existingEvals.unshift(result);

    try {
      localStorage.setItem(STORAGE_KEY_EVALUATIONS, JSON.stringify(existingEvals));
      localStorage.setItem(STORAGE_KEY_EXPERIMENTS, JSON.stringify(existingExps));
    } catch (e) {
      console.error('LocalStorage write error:', e);
    }

    return experiment;
  }

  /**
   * Delete an experiment
   */
  static deleteExperiment(experimentId: string): void {
    const exps = this.getExperiments().filter((e) => e.id !== experimentId);
    try {
      localStorage.setItem(STORAGE_KEY_EXPERIMENTS, JSON.stringify(exps));
    } catch (e) {
      console.error('Error deleting experiment:', e);
    }
  }

  /**
   * Delete single evaluation
   */
  static deleteEvaluation(evalId: string): void {
    const evals = this.getEvaluations().filter((e) => e.id !== evalId);
    try {
      localStorage.setItem(STORAGE_KEY_EVALUATIONS, JSON.stringify(evals));
    } catch (e) {
      console.error('Error deleting evaluation:', e);
    }
  }

  /**
   * Clear all stored history
   */
  static clearAll(): void {
    localStorage.removeItem(STORAGE_KEY_EXPERIMENTS);
    localStorage.removeItem(STORAGE_KEY_EVALUATIONS);
  }
}
