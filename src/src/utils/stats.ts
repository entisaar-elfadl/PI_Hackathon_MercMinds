/**
 * Statistical Helper Functions for Multi-Round Model Analysis
 */

export function calculateMean(values: number[]): number {
  if (!values || values.length === 0) return 0;
  const sum = values.reduce((acc, val) => acc + val, 0);
  return sum / values.length;
}

export function calculateStdDev(values: number[]): number {
  if (!values || values.length <= 1) return 0;
  const mean = calculateMean(values);
  const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / (values.length - 1);
  return Math.sqrt(variance);
}

export function formatAuc(auc: number): string {
  if (typeof auc !== 'number' || isNaN(auc)) return '0.50000';
  return auc.toFixed(5);
}

export function formatScorePct(scorePct: number): string {
  if (typeof scorePct !== 'number' || isNaN(scorePct)) return '50.00%';
  return `${scorePct.toFixed(2)}%`;
}
