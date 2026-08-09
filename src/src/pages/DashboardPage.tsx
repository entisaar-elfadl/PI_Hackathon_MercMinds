import React from 'react';
import { PlayCircle, Trophy, BarChart3, ShieldCheck, ArrowRight, Sparkles, CheckCircle2, FileSpreadsheet, Layers, Info } from 'lucide-react';
import { Experiment, EvaluationResult, RoundId } from '../types';
import { HISTORICAL_ROUNDS } from '../data/historicalDatasets';
import { MetricCard } from '../components/common/MetricCard';

interface DashboardPageProps {
  experiments: Experiment[];
  evaluations: EvaluationResult[];
  onNavigate: (tab: 'evaluate' | 'comparison' | 'leaderboard' | 'datasets') => void;
  onSelectRoundForEval: (roundId: RoundId) => void;
  onOpenRGenerator: () => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  experiments,
  evaluations,
  onNavigate,
  onSelectRoundForEval,
  onOpenRGenerator,
}) => {
  // Find top performing experiment by average AUC
  const topModel = [...experiments].sort((a, b) => b.avgAuc - a.avgAuc)[0];

  return (
    <div className="space-y-6">
      {/* Hero Welcome Banner */}
      <div className="bg-white border border-slate-200 rounded p-6 shadow-2xs relative overflow-hidden">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
          <div className="space-y-2 max-w-2xl">
            <div className="inline-flex items-center space-x-2 px-2.5 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200 text-[11px] font-mono font-bold uppercase tracking-wider">
              <Sparkles className="w-3.5 h-3.5 text-blue-600" />
              <span>Local Validation Pipeline</span>
            </div>

            <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight uppercase">
              Employment <span className="text-blue-600">Prediction Engine</span>
            </h1>

            <p className="text-xs sm:text-sm text-slate-600 leading-relaxed font-sans">
              Validate your R or Python prediction models locally against historical survey rounds (Rounds 6, 7 &amp; 8) where actual employment outcomes are known. Evaluate model generalisation before final submission to hidden Round 9.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <button
              onClick={() => onNavigate('evaluate')}
              className="flex items-center space-x-2 px-4 py-2.5 rounded bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs shadow-xs transition-all cursor-pointer"
            >
              <PlayCircle className="w-4 h-4" />
              <span>Evaluate Prediction CSV</span>
            </button>

            <button
              onClick={() => onNavigate('comparison')}
              className="flex items-center space-x-2 px-4 py-2.5 rounded bg-white hover:bg-slate-50 text-slate-700 font-bold text-xs border border-slate-300 shadow-2xs transition-all cursor-pointer"
            >
              <BarChart3 className="w-4 h-4 text-blue-600" />
              <span>Multi-Round Matrix</span>
            </button>

            <button
              onClick={onOpenRGenerator}
              className="flex items-center space-x-2 px-3.5 py-2.5 rounded bg-blue-50 hover:bg-blue-100 text-blue-700 font-bold text-xs border border-blue-200 transition-all cursor-pointer"
            >
              <span>Get R Script</span>
            </button>
          </div>
        </div>
      </div>

      {/* Summary Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Model Experiments"
          value={experiments.length}
          subtitle="Tracked across validation rounds"
          icon={<Layers className="w-4 h-4 text-slate-600" />}
        />

        <MetricCard
          title="Top Average AUC"
          value={topModel ? topModel.avgAuc.toFixed(5) : '0.00000'}
          subtitle={topModel ? `${topModel.modelName} (v${topModel.modelVersion})` : 'No models evaluated'}
          icon={<Trophy className="w-4 h-4 text-amber-500" />}
          badgeText={topModel ? `${(topModel.avgAuc * 100).toFixed(2)}%` : undefined}
          badgeType="success"
          highlight
        />

        <MetricCard
          title="Total Evaluations Run"
          value={evaluations.length}
          subtitle="Completed AUC scoring runs"
          icon={<CheckCircle2 className="w-4 h-4 text-emerald-600" />}
        />

        <MetricCard
          title="Validation Metric"
          value="ROC AUC"
          subtitle="Area Under Receiver Operating Characteristic"
          icon={<ShieldCheck className="w-4 h-4 text-blue-600" />}
          badgeText="Standard"
          badgeType="info"
        />
      </div>

      {/* Historical Validation Strategy Rounds Cards */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest flex items-center space-x-2">
              <FileSpreadsheet className="w-4 h-4 text-blue-600" />
              <span>Historical Evaluation Rounds</span>
            </h2>
          </div>
          <button
            onClick={() => onNavigate('datasets')}
            className="text-xs font-bold text-blue-600 hover:text-blue-800 flex items-center space-x-1"
          >
            <span>Explore Datasets</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.values(HISTORICAL_ROUNDS).map((round) => {
            const roundEvals = evaluations.filter((e) => e.roundId === round.id);
            const bestRoundEval = [...roundEvals].sort((a, b) => b.auc - a.auc)[0];

            return (
              <div
                key={round.id}
                className="bg-white border border-slate-200 hover:border-blue-300 rounded p-5 flex flex-col justify-between space-y-4 shadow-2xs transition-all"
              >
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="px-2 py-0.5 text-[10px] font-bold uppercase rounded bg-slate-100 text-slate-700 font-mono border border-slate-200">
                      {round.id.replace('_', ' ')}
                    </span>
                    <span className="text-[11px] text-slate-400 font-mono">
                      ~{round.sampleObservationCount} Obs
                    </span>
                  </div>

                  <h3 className="text-sm font-bold text-slate-900">{round.title}</h3>
                  <p className="text-xs text-slate-600 leading-relaxed">{round.description}</p>

                  <div className="pt-2 text-xs font-mono space-y-1">
                    <div className="text-slate-500">
                      Train: <span className="text-slate-800 font-semibold">{round.trainingSet}</span>
                    </div>
                    <div className="text-slate-500">
                      Eval: <span className="text-blue-600 font-semibold">{round.evaluationSet}</span>
                    </div>
                  </div>
                </div>

                <div className="pt-3 border-t border-slate-100 space-y-3">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-slate-500 uppercase text-[10px] font-bold">Best AUC:</span>
                    <span className="font-bold text-emerald-600">
                      {bestRoundEval ? `${bestRoundEval.auc.toFixed(5)} (${(bestRoundEval.auc * 100).toFixed(2)}%)` : '—'}
                    </span>
                  </div>

                  <button
                    onClick={() => {
                      onSelectRoundForEval(round.id);
                      onNavigate('evaluate');
                    }}
                    className="w-full py-1.5 px-3 rounded bg-slate-100 hover:bg-blue-600 hover:text-white text-slate-700 text-xs font-bold transition-all flex items-center justify-center space-x-1.5 cursor-pointer border border-slate-200 hover:border-blue-600"
                  >
                    <PlayCircle className="w-3.5 h-3.5" />
                    <span>Evaluate {round.id.replace('_', ' ').toUpperCase()}</span>
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Leaderboard Snapshot Table */}
      <div className="bg-white border border-slate-200 rounded p-5 space-y-4 shadow-2xs">
        <div className="flex items-center justify-between pb-3 border-b border-slate-200">
          <div>
            <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2">
              <Trophy className="w-4 h-4 text-amber-500" />
              <span>Model Performance Leaderboard (Top 5)</span>
            </h2>
            <p className="text-[11px] text-slate-500 mt-0.5 font-mono">Rankings based on Average ROC AUC across historical rounds</p>
          </div>

          <button
            onClick={() => onNavigate('leaderboard')}
            className="text-xs font-bold text-blue-600 hover:underline flex items-center space-x-1"
          >
            <span>Full Leaderboard</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {experiments.length === 0 ? (
          <div className="text-center py-8 text-slate-500 text-xs font-mono">
            No models evaluated yet. Upload a prediction CSV or select a benchmark model to evaluate.
          </div>
        ) : (
          <div className="overflow-x-auto border border-slate-200 rounded">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-50 text-slate-500 uppercase text-[10px] tracking-wider italic font-bold border-b border-slate-200">
                <tr>
                  <th className="px-4 py-2.5">Rank</th>
                  <th className="px-4 py-2.5">Model Name</th>
                  <th className="px-4 py-2.5">Version</th>
                  <th className="px-4 py-2.5">R6 AUC</th>
                  <th className="px-4 py-2.5">R7 AUC</th>
                  <th className="px-4 py-2.5">R8 AUC</th>
                  <th className="px-4 py-2.5 text-right font-bold text-blue-600">Average AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {[...experiments]
                  .sort((a, b) => b.avgAuc - a.avgAuc)
                  .slice(0, 5)
                  .map((exp, idx) => (
                    <tr key={exp.id} className="hover:bg-blue-50/50 transition-colors">
                      <td className="px-4 py-2.5 font-bold text-slate-400">0{idx + 1}</td>
                      <td className="px-4 py-2.5 font-bold text-slate-900">{exp.modelName}</td>
                      <td className="px-4 py-2.5 text-slate-500">{exp.modelVersion}</td>
                      <td className="px-4 py-2.5 text-slate-700">
                        {exp.resultsByRound.round_6 ? exp.resultsByRound.round_6.auc.toFixed(5) : '-'}
                      </td>
                      <td className="px-4 py-2.5 text-slate-700">
                        {exp.resultsByRound.round_7 ? exp.resultsByRound.round_7.auc.toFixed(5) : '-'}
                      </td>
                      <td className="px-4 py-2.5 text-slate-700">
                        {exp.resultsByRound.round_8 ? exp.resultsByRound.round_8.auc.toFixed(5) : '-'}
                      </td>
                      <td className="px-4 py-2.5 text-right font-extrabold text-blue-600">
                        {exp.avgAuc.toFixed(5)} ({(exp.avgAuc * 100).toFixed(2)}%)
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
