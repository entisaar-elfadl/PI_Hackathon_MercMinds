import React, { useState } from 'react';
import { Trophy, Download, Trash2, Award, Search, RefreshCw, FileText } from 'lucide-react';
import { Experiment, RoundId } from '../types';
import { ExperimentStorageService } from '../services/experimentStorage';

interface LeaderboardPageProps {
  experiments: Experiment[];
  onRefresh: () => void;
}

export const LeaderboardPage: React.FC<LeaderboardPageProps> = ({ experiments, onRefresh }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRoundFilter, setSelectedRoundFilter] = useState<'all' | RoundId>('all');

  const filtered = experiments.filter((exp) => {
    const matchesSearch =
      exp.modelName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      exp.modelVersion.toLowerCase().includes(searchTerm.toLowerCase());

    if (!matchesSearch) return false;

    if (selectedRoundFilter !== 'all') {
      return exp.resultsByRound[selectedRoundFilter] !== undefined;
    }
    return true;
  });

  const sorted = [...filtered].sort((a, b) => {
    if (selectedRoundFilter === 'all') {
      return b.avgAuc - a.avgAuc;
    }
    const aucA = a.resultsByRound[selectedRoundFilter]?.auc || 0;
    const aucB = b.resultsByRound[selectedRoundFilter]?.auc || 0;
    return aucB - aucA;
  });

  const handleExportJson = () => {
    const jsonStr = JSON.stringify(experiments, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `local_historical_leaderboard_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear all local experiment history?')) {
      ExperimentStorageService.clearAll();
      onRefresh();
    }
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto font-sans">
      {/* Page Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-200 pb-3 gap-3">
        <div>
          <div className="flex items-center space-x-2">
            <Trophy className="w-5 h-5 text-amber-600" />
            <h1 className="text-xl font-bold text-slate-900 uppercase tracking-tight">Local Historical Leaderboard</h1>
            <span className="px-2 py-0.5 text-[10px] font-mono font-bold uppercase rounded bg-amber-50 text-amber-700 border border-amber-200">
              Historical Validation
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Rankings calculated from known historical outcomes (Rounds 6–8). Does NOT represent official Kaggle score.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={handleExportJson}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-white hover:bg-slate-50 text-slate-700 text-xs font-bold border border-slate-300 transition-all cursor-pointer shadow-2xs"
          >
            <Download className="w-3.5 h-3.5 text-blue-600" />
            <span>Export History JSON</span>
          </button>

          <button
            onClick={handleClearHistory}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-rose-50 hover:bg-rose-100 text-rose-700 text-xs font-bold border border-rose-200 transition-all cursor-pointer"
          >
            <Trash2 className="w-3.5 h-3.5 text-rose-600" />
            <span>Clear History</span>
          </button>
        </div>
      </div>

      {/* Filters Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-white border border-slate-200 p-4 rounded shadow-2xs">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
          <input
            type="text"
            placeholder="Search model name or version..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-slate-50 text-xs text-slate-900 pl-9 pr-3 py-2 rounded border border-slate-300 focus:outline-none focus:border-blue-600 font-medium"
          />
        </div>

        <div className="flex items-center space-x-2 text-xs">
          <span className="text-slate-600 font-bold">Sort By Round:</span>
          <select
            value={selectedRoundFilter}
            onChange={(e) => setSelectedRoundFilter(e.target.value as any)}
            className="bg-slate-50 text-slate-800 px-3 py-2 rounded border border-slate-300 focus:outline-none focus:border-blue-600 font-semibold"
          >
            <option value="all">Average AUC (Overall Rank)</option>
            <option value="round_6">Round 6 AUC</option>
            <option value="round_7">Round 7 AUC</option>
            <option value="round_8">Round 8 AUC</option>
          </select>
        </div>
      </div>

      {/* Leaderboard Table */}
      <div className="bg-white border border-slate-200 rounded p-5 space-y-4 shadow-2xs">
        {sorted.length === 0 ? (
          <div className="text-center py-12 text-slate-500 text-sm">
            No models found matching criteria.
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
                  <th className="px-4 py-2.5 text-right">Avg Percentage Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {sorted.map((exp, idx) => {
                  const r6 = exp.resultsByRound.round_6?.auc;
                  const r7 = exp.resultsByRound.round_7?.auc;
                  const r8 = exp.resultsByRound.round_8?.auc;

                  return (
                    <tr key={exp.id} className="hover:bg-blue-50/50 transition-colors">
                      <td className="px-4 py-2.5 font-bold">
                        {idx === 0 ? (
                          <span className="text-amber-700 flex items-center space-x-1 font-extrabold text-sm">
                            <Award className="w-4 h-4 text-amber-600" />
                            <span>#1</span>
                          </span>
                        ) : idx === 1 ? (
                          <span className="text-slate-700 font-bold text-sm">#2</span>
                        ) : idx === 2 ? (
                          <span className="text-amber-800 font-bold text-sm">#3</span>
                        ) : (
                          <span className="text-slate-500">#{idx + 1}</span>
                        )}
                      </td>

                      <td className="px-4 py-2.5 font-bold text-slate-900">
                        {exp.modelName}
                        {exp.notes && <div className="text-[10px] text-slate-500 font-normal line-clamp-1">{exp.notes}</div>}
                      </td>

                      <td className="px-4 py-2.5 text-slate-500">{exp.modelVersion}</td>

                      <td className="px-4 py-2.5 text-blue-700 font-bold">
                        {r6 !== undefined ? r6.toFixed(5) : '-'}
                      </td>

                      <td className="px-4 py-2.5 text-indigo-700 font-bold">
                        {r7 !== undefined ? r7.toFixed(5) : '-'}
                      </td>

                      <td className="px-4 py-2.5 text-amber-700 font-bold">
                        {r8 !== undefined ? r8.toFixed(5) : '-'}
                      </td>

                      <td className="px-4 py-2.5 text-right font-extrabold text-blue-700 text-sm">
                        {exp.avgAuc.toFixed(5)}
                      </td>

                      <td className="px-4 py-2.5 text-right font-extrabold text-slate-900 text-sm">
                        {(exp.avgAuc * 100).toFixed(2)}%
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
