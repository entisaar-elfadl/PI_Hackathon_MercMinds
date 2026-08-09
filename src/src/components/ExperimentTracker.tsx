/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { Search, Calendar, FileCode, Trash2, ClipboardList, Eye } from 'lucide-react';
import { Experiment, RoundId } from '../types';

interface ExperimentTrackerProps {
  experiments: Experiment[];
  onDeleteExperiment: (id: string) => void;
  onClearAll: () => void;
  onSelectExperiment: (id: string) => void;
}

export function ExperimentTracker({
  experiments,
  onDeleteExperiment,
  onClearAll,
  onSelectExperiment,
}: ExperimentTrackerProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [roundFilter, setRoundFilter] = useState<string>('all');

  const filteredExperiments = experiments.filter(e => {
    const matchesSearch =
      e.modelName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      e.predictionFile.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (e.notes && e.notes.toLowerCase().includes(searchTerm.toLowerCase()));

    const matchesRound = roundFilter === 'all' || e.roundId === roundFilter;

    return matchesSearch && matchesRound;
  });

  const formatDate = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  const getRoundBadgeClass = (roundId: RoundId) => {
    switch (roundId) {
      case 'round_6':
        return 'bg-blue-50 text-blue-700 border border-blue-100';
      case 'round_7':
        return 'bg-purple-50 text-purple-700 border border-purple-100';
      case 'round_8':
        return 'bg-pink-50 text-pink-700 border border-pink-100';
      default:
        return 'bg-slate-100 text-slate-700';
    }
  };

  const getRoundLabel = (roundId: RoundId) => {
    return roundId.replace('_', ' ').toUpperCase();
  };

  return (
    <div id="experiment_tracker_card" className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <ClipboardList className="w-5 h-5 text-indigo-500" />
            <h2 className="text-lg font-bold text-slate-800 font-display">Experiment Ledger</h2>
          </div>
          <p className="text-xs text-slate-500">
            Persistent log of local validation runs. Keep track of features and R hyperparameter tuning.
          </p>
        </div>
        {experiments.length > 0 && (
          <button
            onClick={() => {
              if (confirm('Are you sure you want to clear your local evaluation history? This cannot be undone.')) {
                onClearAll();
              }
            }}
            className="px-3 py-1.5 border border-rose-200 text-rose-600 hover:bg-rose-50 font-semibold text-xs rounded-xl active:scale-95 transition-all flex items-center gap-1.5 shrink-0 self-start sm:self-center"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear Ledger
          </button>
        )}
      </div>

      {experiments.length === 0 ? (
        <div className="text-center py-12 border border-dashed border-slate-150 rounded-xl bg-slate-50/50">
          <ClipboardList className="w-10 h-10 text-slate-300 mx-auto mb-2" />
          <h3 className="text-sm font-semibold text-slate-700">No Experiments Tracked</h3>
          <p className="text-xs text-slate-400 mt-1 max-w-xs mx-auto">
            Uploaded model predictions that pass evaluation are automatically logged here to track development progress.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Filters Bar */}
          <div className="flex flex-col sm:flex-row gap-3">
            {/* Search Input */}
            <div className="relative flex-1">
              <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                placeholder="Search models, files, notes..."
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 text-slate-800 rounded-xl text-xs focus:bg-white focus:outline-hidden focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 transition-all placeholder:text-slate-400"
              />
            </div>
            {/* Round Filter */}
            <select
              value={roundFilter}
              onChange={e => setRoundFilter(e.target.value)}
              className="px-3 py-2 bg-slate-50 border border-slate-200 text-slate-700 rounded-xl text-xs font-medium focus:bg-white focus:outline-hidden focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 transition-all cursor-pointer shrink-0"
            >
              <option value="all">All Rounds</option>
              <option value="round_6">Round 6</option>
              <option value="round_7">Round 7</option>
              <option value="round_8">Round 8</option>
            </select>
          </div>

          {/* Ledger Items */}
          <div className="space-y-3 max-h-[480px] overflow-y-auto pr-1">
            {filteredExperiments.length === 0 ? (
              <p className="text-center text-xs text-slate-400 py-6">No matching records found.</p>
            ) : (
              filteredExperiments.map(exp => (
                <div
                  key={exp.id}
                  className="p-4 border border-slate-100 rounded-xl hover:border-slate-200 hover:shadow-xs transition-all bg-white flex flex-col md:flex-row md:items-center justify-between gap-4"
                >
                  {/* Left block: Model Meta */}
                  <div className="flex-1 space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="font-bold text-slate-800 text-sm leading-none">{exp.modelName}</h3>
                      <span className="text-[10px] bg-slate-100 text-slate-500 font-bold px-1.5 py-0.5 rounded-sm">
                        Ver {exp.modelVersion}
                      </span>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${getRoundBadgeClass(exp.roundId)}`}>
                        {getRoundLabel(exp.roundId)}
                      </span>
                    </div>

                    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-400">
                      <div className="flex items-center gap-1">
                        <FileCode className="w-3.5 h-3.5 shrink-0" />
                        <span className="max-w-[140px] sm:max-w-[200px] truncate" title={exp.predictionFile}>
                          {exp.predictionFile}
                        </span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Calendar className="w-3.5 h-3.5 shrink-0" />
                        <span>{formatDate(exp.timestamp)}</span>
                      </div>
                    </div>

                    {exp.notes && (
                      <p className="text-[11px] text-slate-500 bg-slate-50 px-2.5 py-1 rounded-lg border-l-2 border-slate-300 italic max-w-2xl">
                        {exp.notes}
                      </p>
                    )}
                  </div>

                  {/* Right block: Stats + Controls */}
                  <div className="flex items-center justify-between md:justify-end gap-6 border-t md:border-t-0 border-slate-50 pt-2 md:pt-0">
                    <div className="text-right">
                      <div className="text-slate-400 text-[10px] font-semibold uppercase tracking-wider">Evaluation Score</div>
                      <div className="flex items-baseline gap-1 justify-end mt-0.5">
                        <span className="text-lg font-bold text-slate-800">{(exp.auc * 100).toFixed(2)}%</span>
                        <span className="text-xs font-semibold font-mono text-slate-400">({exp.auc.toFixed(5)})</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => onSelectExperiment(exp.id)}
                        className="p-1.5 hover:bg-slate-50 text-slate-500 hover:text-indigo-600 rounded-lg active:scale-95 transition-all"
                        title="View Full Diagnostics"
                      >
                        <Eye className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => {
                          if (confirm('Delete this experiment from the local ledger?')) {
                            onDeleteExperiment(exp.id);
                          }
                        }}
                        className="p-1.5 hover:bg-rose-50 text-slate-400 hover:text-rose-600 rounded-lg active:scale-95 transition-all"
                        title="Delete record"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
