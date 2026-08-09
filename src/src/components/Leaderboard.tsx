/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Trophy, TrendingUp, HelpCircle, ShieldAlert } from 'lucide-react';
import { ModelComparison } from '../types';

interface LeaderboardProps {
  models: ModelComparison[];
}

export function Leaderboard({ models }: LeaderboardProps) {
  // Sort models by average AUC descending
  const sortedModels = [...models].sort((a, b) => b.averageAuc - a.averageAuc);

  return (
    <div id="leaderboard_card" className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-md relative overflow-hidden shadow-xl">
      <div className="absolute top-0 right-0 w-32 h-32 bg-blue-600/5 blur-[50px] rounded-full pointer-events-none" />
      
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6 relative z-10">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Trophy className="w-5 h-5 text-amber-400" />
            <h2 className="text-base font-extrabold text-white">Local Historical Leaderboard</h2>
          </div>
          <p className="text-xs text-slate-400">
            Rankings of model generalisation across simulated survey rounds (Rounds 6, 7, and 8).
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-amber-500/10 border border-amber-500/20 rounded-xl text-amber-400 max-w-fit text-xs font-bold">
          <ShieldAlert className="w-4 h-4 shrink-0 text-amber-400" />
          <span>Local Historical Evaluation ≠ Official Kaggle Score</span>
        </div>
      </div>

      <div className="overflow-x-auto relative z-10">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-800/80 text-[10px] font-black text-slate-500 uppercase tracking-widest bg-slate-950/30">
              <th className="py-3.5 px-4">Rank</th>
              <th className="py-3.5 px-4">Model & Version</th>
              <th className="py-3.5 px-4 text-center">Round 6</th>
              <th className="py-3.5 px-4 text-center">Round 7</th>
              <th className="py-3.5 px-4 text-center">Round 8</th>
              <th className="py-3.5 px-4 text-right bg-slate-950/40 text-slate-400 rounded-t-xl">Avg AUC</th>
              <th className="py-3.5 px-4 text-right">Avg Score</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/40 text-xs text-slate-300">
            {sortedModels.length === 0 ? (
              <tr>
                <td colSpan={7} className="py-12 text-center text-slate-500 font-semibold">
                  No models have been evaluated yet. Run evaluations on different rounds to populate the leaderboard.
                </td>
              </tr>
            ) : (
              sortedModels.map((m, index) => {
                const isTop = index === 0;
                return (
                  <tr
                    key={`${m.modelName}-${m.modelVersion}`}
                    className={`hover:bg-slate-800/25 transition-all duration-150 ${
                      isTop ? 'bg-amber-500/5 font-medium text-slate-200' : ''
                    }`}
                  >
                    {/* Rank */}
                    <td className="py-4 px-4 font-bold">
                      <div className="flex items-center justify-center w-6 h-6 rounded-lg text-center">
                        {index === 0 ? (
                          <span className="text-base text-amber-400">🥇</span>
                        ) : index === 1 ? (
                          <span className="text-base text-slate-400">🥈</span>
                        ) : index === 2 ? (
                          <span className="text-base text-amber-600">🥉</span>
                        ) : (
                          <span className="text-slate-500 font-mono font-bold">{index + 1}</span>
                        )}
                      </div>
                    </td>

                    {/* Model Name */}
                    <td className="py-4 px-4">
                      <div>
                        <div className="font-extrabold text-white text-xs">{m.modelName}</div>
                        <div className="text-[10px] text-slate-500 mt-0.5">Version {m.modelVersion}</div>
                      </div>
                    </td>

                    {/* Round 6 AUC */}
                    <td className="py-4 px-4 text-center font-mono text-[11px]">
                      {m.roundScores.round_6 !== null ? (
                        <span className="px-2 py-0.5 bg-slate-950/40 border border-slate-800/80 rounded text-slate-300 font-semibold">
                          {m.roundScores.round_6.toFixed(4)}
                        </span>
                      ) : (
                        <span className="text-slate-700">—</span>
                      )}
                    </td>

                    {/* Round 7 AUC */}
                    <td className="py-4 px-4 text-center font-mono text-[11px]">
                      {m.roundScores.round_7 !== null ? (
                        <span className="px-2 py-0.5 bg-slate-950/40 border border-slate-800/80 rounded text-slate-300 font-semibold">
                          {m.roundScores.round_7.toFixed(4)}
                        </span>
                      ) : (
                        <span className="text-slate-700">—</span>
                      )}
                    </td>

                    {/* Round 8 AUC */}
                    <td className="py-4 px-4 text-center font-mono text-[11px]">
                      {m.roundScores.round_8 !== null ? (
                        <span className="px-2 py-0.5 bg-slate-950/40 border border-slate-800/80 rounded text-slate-300 font-semibold">
                          {m.roundScores.round_8.toFixed(4)}
                        </span>
                      ) : (
                        <span className="text-slate-700">—</span>
                      )}
                    </td>

                    {/* Average AUC */}
                    <td className="py-4 px-4 text-right font-mono bg-slate-950/40 font-black text-emerald-400 text-[11px]">
                      {m.averageAuc.toFixed(5)}
                    </td>

                    {/* Average Score */}
                    <td className="py-4 px-4 text-right font-bold">
                      <span
                        className={`inline-block px-2 py-0.5 rounded-lg text-[11px] font-black border ${
                          m.averageAuc >= 0.8
                            ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                            : m.averageAuc >= 0.7
                            ? 'bg-teal-500/10 text-teal-400 border-teal-500/20'
                            : m.averageAuc >= 0.6
                            ? 'bg-blue-500/10 text-blue-400 border-blue-500/20'
                            : 'bg-slate-800 text-slate-400 border-slate-700'
                        }`}
                      >
                        {(m.averageAuc * 100).toFixed(2)}%
                      </span>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      <div className="mt-5 p-4 bg-slate-950/40 border border-slate-800/80 rounded-xl flex items-start gap-3 relative z-10">
        <HelpCircle className="w-5 h-5 text-slate-500 shrink-0 mt-0.5" />
        <div className="text-xs text-slate-400 leading-relaxed">
          <p className="font-extrabold text-slate-300 mb-1">How is the Average calculated?</p>
          Each model is ranked by its average Area Under the Curve (AUC) across all three rounds. Evaluating across 
          multiple rounds ensures that your R model has <strong>robust generalisation</strong> and is not simply 
          overfitting to a single validation set.
        </div>
      </div>
    </div>
  );
}
