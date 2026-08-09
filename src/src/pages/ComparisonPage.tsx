import React from 'react';
import { BarChart3, ShieldCheck, AlertTriangle, Trophy, Sparkles, HelpCircle } from 'lucide-react';
import { Experiment, RoundId } from '../types';
import { ModelComparisonChart } from '../components/charts/ModelComparisonChart';
import { PlayCircle } from 'lucide-react';

interface ComparisonPageProps {
  experiments: Experiment[];
  onNavigateToEvaluate: (roundId: RoundId) => void;
}

export const ComparisonPage: React.FC<ComparisonPageProps> = ({ experiments, onNavigateToEvaluate }) => {
  // Sort experiments by Average AUC
  const sortedExperiments = [...experiments].sort((a, b) => b.avgAuc - a.avgAuc);

  return (
    <div className="space-y-6 max-w-6xl mx-auto font-sans">
      {/* Header */}
      <div className="border-b border-slate-200 pb-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-slate-900 uppercase tracking-tight flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-blue-600" />
            <span>Multi-Round Historical Validation Matrix</span>
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Compare model generalisation across Round 6, Round 7 &amp; Round 8 to evaluate performance consistency before Round 9 submission.
          </p>
        </div>

        <button
          onClick={() => onNavigateToEvaluate('round_6')}
          className="flex items-center space-x-2 px-4 py-2 rounded bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs shadow-xs transition-all self-start sm:self-auto cursor-pointer uppercase tracking-wider"
        >
          <PlayCircle className="w-4 h-4" />
          <span>Evaluate Model</span>
        </button>
      </div>

      {/* Generalisation Principle Banner */}
      <div className="bg-white border border-slate-200 rounded p-4 text-xs text-slate-700 space-y-1.5 shadow-2xs">
        <div className="flex items-center space-x-2 text-blue-700 font-bold uppercase tracking-wider text-[11px]">
          <ShieldCheck className="w-4 h-4 text-blue-600" />
          <span>Generalisation Principle: Stability Over Single-Round Peaks</span>
        </div>
        <p className="leading-relaxed">
          A model that scores <strong className="text-emerald-700 font-mono">0.62 in R6, 0.63 in R7, and 0.61 in R8</strong> (Low Std Dev: 0.010) is significantly more trustworthy for hidden Round 9 than a model scoring <strong className="text-amber-700 font-mono">0.70 in R6, 0.51 in R7, and 0.52 in R8</strong> (High Std Dev: 0.106).
        </p>
      </div>

      {/* Multi-Round Matrix Chart */}
      {experiments.length > 0 && <ModelComparisonChart experiments={experiments} />}

      {/* Comparison Table */}
      <div className="bg-white border border-slate-200 rounded p-5 space-y-4 shadow-2xs">
        <div className="flex items-center justify-between pb-3 border-b border-slate-200">
          <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
            Model Generalisation Matrix ({experiments.length} Models Tracked)
          </h2>
          <span className="text-xs text-slate-500 font-mono">Metric: ROC AUC</span>
        </div>

        {experiments.length === 0 ? (
          <div className="text-center py-12 text-slate-500 text-sm">
            No models available for comparison. Run evaluations or select benchmark models to populate the matrix.
          </div>
        ) : (
          <div className="overflow-x-auto rounded border border-slate-200">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-50 text-slate-500 uppercase text-[10px] tracking-wider font-bold border-b border-slate-200">
                <tr>
                  <th className="px-4 py-2.5">Rank</th>
                  <th className="px-4 py-2.5">Model Name</th>
                  <th className="px-4 py-2.5">Version</th>
                  <th className="px-4 py-2.5">Round 6 AUC</th>
                  <th className="px-4 py-2.5">Round 7 AUC</th>
                  <th className="px-4 py-2.5">Round 8 AUC</th>
                  <th className="px-4 py-2.5 text-right font-bold text-blue-700">Average AUC</th>
                  <th className="px-4 py-2.5 text-right">Min / Max</th>
                  <th className="px-4 py-2.5 text-right">Std Dev (&sigma;)</th>
                  <th className="px-4 py-2.5 text-right">Consistency Rating</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {sortedExperiments.map((exp, idx) => {
                  const r6 = exp.resultsByRound.round_6?.auc;
                  const r7 = exp.resultsByRound.round_7?.auc;
                  const r8 = exp.resultsByRound.round_8?.auc;

                  const isHighVariance = exp.stdDevAuc > 0.05;
                  const isConsistent = exp.stdDevAuc <= 0.02 && exp.roundsEvaluatedCount >= 2;

                  return (
                    <tr key={exp.id} className="hover:bg-blue-50/50 transition-colors">
                      <td className="px-4 py-2.5 font-bold text-slate-700">#{idx + 1}</td>
                      <td className="px-4 py-2.5 font-bold text-slate-900">
                        <div>{exp.modelName}</div>
                        {exp.notes && <div className="text-[10px] text-slate-500 font-normal line-clamp-1">{exp.notes}</div>}
                      </td>
                      <td className="px-4 py-2.5 text-slate-500">{exp.modelVersion}</td>

                      <td className="px-4 py-2.5">
                        {r6 !== undefined ? (
                          <span className="font-bold text-blue-700">{r6.toFixed(5)}</span>
                        ) : (
                          <button
                            onClick={() => onNavigateToEvaluate('round_6')}
                            className="text-[10px] text-slate-400 hover:text-blue-600 underline cursor-pointer"
                          >
                            + Eval R6
                          </button>
                        )}
                      </td>

                      <td className="px-4 py-2.5">
                        {r7 !== undefined ? (
                          <span className="font-bold text-indigo-700">{r7.toFixed(5)}</span>
                        ) : (
                          <button
                            onClick={() => onNavigateToEvaluate('round_7')}
                            className="text-[10px] text-slate-400 hover:text-indigo-600 underline cursor-pointer"
                          >
                            + Eval R7
                          </button>
                        )}
                      </td>

                      <td className="px-4 py-2.5">
                        {r8 !== undefined ? (
                          <span className="font-bold text-amber-700">{r8.toFixed(5)}</span>
                        ) : (
                          <button
                            onClick={() => onNavigateToEvaluate('round_8')}
                            className="text-[10px] text-slate-400 hover:text-amber-600 underline cursor-pointer"
                          >
                            + Eval R8
                          </button>
                        )}
                      </td>

                      <td className="px-4 py-2.5 text-right font-extrabold text-blue-700 text-sm">
                        {exp.avgAuc.toFixed(5)}
                      </td>

                      <td className="px-4 py-2.5 text-right text-slate-500 text-[11px]">
                        {exp.minAuc.toFixed(4)} - {exp.maxAuc.toFixed(4)}
                      </td>

                      <td className="px-4 py-2.5 text-right font-semibold">
                        <span className={isHighVariance ? 'text-amber-700' : 'text-slate-700'}>
                          {exp.stdDevAuc.toFixed(5)}
                        </span>
                      </td>

                      <td className="px-4 py-2.5 text-right">
                        {isConsistent ? (
                          <span className="px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 text-[10px] font-bold">
                            High Consistency
                          </span>
                        ) : isHighVariance ? (
                          <span className="px-2 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200 text-[10px] font-bold">
                            High Variance / Risk
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded bg-slate-100 border border-slate-200 text-slate-600 text-[10px]">
                            Moderate
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
