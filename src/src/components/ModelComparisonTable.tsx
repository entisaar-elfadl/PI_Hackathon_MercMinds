/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { TrendingUp, Award, Activity, AlertCircle } from 'lucide-react';
import { ModelComparison } from '../types';

interface ModelComparisonTableProps {
  comparisons: ModelComparison[];
}

export function ModelComparisonTable({ comparisons }: ModelComparisonTableProps) {
  // Sort by average AUC descending
  const sortedComparisons = [...comparisons].sort((a, b) => b.averageAuc - a.averageAuc);

  return (
    <div id="comparison_suite_card" className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
      <div className="flex items-center gap-2 mb-2">
        <Activity className="w-5 h-5 text-indigo-500" />
        <h2 className="text-base font-bold text-slate-800 font-display">Cross-Round Generalisation Analysis</h2>
      </div>
      <p className="text-xs text-slate-500 mb-6">
        Compare model robustness side-by-side. Robust models exhibit <strong>high average AUC</strong> and 
        <strong>low standard deviation</strong> across all validation rounds.
      </p>

      {sortedComparisons.length === 0 ? (
        <div className="text-center py-10 border border-dashed border-slate-150 rounded-xl bg-slate-50/50">
          <AlertCircle className="w-8 h-8 text-slate-400 mx-auto mb-2" />
          <p className="text-xs font-medium text-slate-600">No evaluation data available to compare.</p>
          <p className="text-[11px] text-slate-400 mt-1 max-w-xs mx-auto">
            Evaluate a model against Round 6, Round 7, and Round 8 in the evaluator to see performance comparative stats.
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Main Table */}
          <div className="overflow-x-auto border border-slate-100 rounded-xl">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50/50 text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                  <th className="py-3 px-4">Model & Version</th>
                  <th className="py-3 px-4 text-center">Round 6 AUC</th>
                  <th className="py-3 px-4 text-center">Round 7 AUC</th>
                  <th className="py-3 px-4 text-center">Round 8 AUC</th>
                  {comparisons.some(m => m.roundScores.custom !== null) && (
                    <th className="py-3 px-4 text-center text-blue-600 bg-blue-50/50 font-bold">Real Round AUC</th>
                  )}
                  <th className="py-3 px-4 text-right">Avg AUC</th>
                  <th className="py-3 px-4 text-center">Min / Max AUC</th>
                  <th className="py-3 px-4 text-right">Std Dev (SD)</th>
                  <th className="py-3 px-4 text-center">Consistency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50 text-xs">
                {sortedComparisons.map((m, idx) => {
                  const isTop = idx === 0;
                  const sd = m.stdDev;
                  
                  // Label consistency level
                  let consistencyLabel = 'Unknown';
                  let consistencyClass = 'bg-slate-100 text-slate-500';
                  
                  if (sd > 0) {
                    if (sd < 0.015) {
                      consistencyLabel = 'High Consistency';
                      consistencyClass = 'bg-emerald-50 text-emerald-700 border border-emerald-100';
                    } else if (sd < 0.04) {
                      consistencyLabel = 'Moderate Consistency';
                      consistencyClass = 'bg-blue-50 text-blue-700 border border-blue-100';
                    } else {
                      consistencyLabel = 'Volatile Performance';
                      consistencyClass = 'bg-amber-50 text-amber-700 border border-amber-100';
                    }
                  } else if (m.roundScores.round_6 !== null && m.roundScores.round_7 === null && m.roundScores.round_8 === null) {
                    consistencyLabel = 'Single Round Evaluated';
                    consistencyClass = 'bg-slate-100 text-slate-500 border border-slate-200';
                  }

                  return (
                    <tr key={`${m.modelName}-${m.modelVersion}`} className="hover:bg-slate-50/40">
                      <td className="py-3.5 px-4 font-semibold text-slate-800">
                        <div className="flex items-center gap-2">
                          {isTop && <Award className="w-4 h-4 text-amber-500 shrink-0" />}
                          <div>
                            <div>{m.modelName}</div>
                            <span className="text-[9px] text-slate-400">Ver {m.modelVersion}</span>
                          </div>
                        </div>
                      </td>

                      {/* R6 score */}
                      <td className="py-3.5 px-4 text-center font-mono font-medium text-slate-600">
                        {m.roundScores.round_6 !== null ? m.roundScores.round_6.toFixed(4) : <span className="text-slate-300">—</span>}
                      </td>

                      {/* R7 score */}
                      <td className="py-3.5 px-4 text-center font-mono font-medium text-slate-600">
                        {m.roundScores.round_7 !== null ? m.roundScores.round_7.toFixed(4) : <span className="text-slate-300">—</span>}
                      </td>

                      {/* R8 score */}
                      <td className="py-3.5 px-4 text-center font-mono font-medium text-slate-600">
                        {m.roundScores.round_8 !== null ? m.roundScores.round_8.toFixed(4) : <span className="text-slate-300">—</span>}
                      </td>

                      {/* Custom score */}
                      {comparisons.some(x => x.roundScores.custom !== null) && (
                        <td className="py-3.5 px-4 text-center font-mono font-bold text-blue-600 bg-blue-500/5 border-x border-blue-500/10">
                          {m.roundScores.custom !== null ? m.roundScores.custom.toFixed(4) : <span className="text-slate-300">—</span>}
                        </td>
                      )}

                      {/* Avg AUC */}
                      <td className="py-3.5 px-4 text-right font-mono font-bold text-slate-800 text-sm">
                        {m.averageAuc.toFixed(5)}
                      </td>

                      {/* Min / Max */}
                      <td className="py-3.5 px-4 text-center font-mono text-[11px] text-slate-500">
                        {m.minAuc > 0 ? `${m.minAuc.toFixed(3)} - ${m.maxAuc.toFixed(3)}` : '—'}
                      </td>

                      {/* Std Dev */}
                      <td className="py-3.5 px-4 text-right font-mono text-slate-500 font-medium">
                        {sd > 0 ? sd.toFixed(5) : '—'}
                      </td>

                      {/* Consistency Badge */}
                      <td className="py-3.5 px-4 text-center">
                        <span className={`inline-block px-2.5 py-0.5 rounded-full text-[10px] font-semibold ${consistencyClass}`}>
                          {consistencyLabel}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Analysis / Insight Card */}
          <div className="p-4 bg-indigo-50/50 border border-indigo-100 rounded-xl">
            <h3 className="text-xs font-bold text-indigo-900 flex items-center gap-1.5 mb-2">
              <TrendingUp className="w-4 h-4 text-indigo-600" />
              Statistical Insight for Your Submission Choice
            </h3>
            <div className="text-xs text-indigo-950 leading-relaxed space-y-2">
              <p>
                A high score on a single validation round (e.g. 0.75 on Round 6) is promising, but if that same model 
                scores 0.51 on Round 7, it is highly likely to contain <strong>data leakage</strong> or be heavily 
                <strong>overfitted</strong> to the specific covariates of that round's pool.
              </p>
              <p>
                When selecting your final model for the unseen <strong>Kaggle Round 9</strong> evaluation, choose the model 
                with the highest <strong>Average AUC</strong> and <strong>High Consistency</strong>. Models that generalise 
                consistently to Round 6, 7, and 8 have a mathematically superior probability of excelling on Round 9.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
