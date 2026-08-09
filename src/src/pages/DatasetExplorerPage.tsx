import React, { useState } from 'react';
import { Database, Download, FileSpreadsheet, Eye, Info, CheckCircle2 } from 'lucide-react';
import { RoundId } from '../types';
import { HISTORICAL_GROUND_TRUTH, HISTORICAL_ROUNDS } from '../data/historicalDatasets';
import { convertToCsv } from '../services/csvParser';

export const DatasetExplorerPage: React.FC = () => {
  const [selectedRound, setSelectedRound] = useState<RoundId>('round_6');
  const [activeTab, setActiveTab] = useState<'ground_truth' | 'features'>('ground_truth');

  const roundInfo = HISTORICAL_ROUNDS[selectedRound];
  const groundTruthRows = HISTORICAL_GROUND_TRUTH[selectedRound] || [];

  const handleDownloadGroundTruthCsv = () => {
    // Export standard submission format
    const formatted = groundTruthRows.map((r) => ({
      anonymised_id: r.anonymised_id,
      employed_status: r.employed_status,
    }));
    const csvStr = convertToCsv(formatted);
    const blob = new Blob([csvStr], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedRound}_known_answers.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadSampleSubmissionCsv = () => {
    // Sample submission with 0.50 default predictions
    const formatted = groundTruthRows.map((r) => ({
      anonymised_id: r.anonymised_id,
      employed_status: 0.5,
    }));
    const csvStr = convertToCsv(formatted);
    const blob = new Blob([csvStr], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedRound}_sample_submission.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto font-sans">
      {/* Title */}
      <div className="border-b border-slate-200 pb-3">
        <h1 className="text-xl font-bold text-slate-900 uppercase tracking-tight flex items-center space-x-2">
          <Database className="w-5 h-5 text-blue-600" />
          <span>Historical Rounds &amp; Dataset Explorer</span>
        </h1>
        <p className="text-xs text-slate-500 mt-1">
          Inspect ground truth answer keys and feature distributions for survey validation rounds 6, 7 &amp; 8.
        </p>
      </div>

      {/* Round Selector Bar */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {Object.values(HISTORICAL_ROUNDS).map((round) => (
          <button
            key={round.id}
            onClick={() => setSelectedRound(round.id)}
            className={`p-4 rounded border text-left transition-all cursor-pointer ${
              selectedRound === round.id
                ? 'bg-blue-50/60 border-blue-600 shadow-2xs'
                : 'bg-white border-slate-200 hover:border-slate-300'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="font-bold text-sm text-slate-900">{round.title}</span>
              {selectedRound === round.id && <CheckCircle2 className="w-4 h-4 text-blue-600" />}
            </div>
            <p className="text-xs text-slate-500 mt-1 line-clamp-2">{round.description}</p>
            <div className="mt-3 text-[11px] font-mono font-bold text-blue-700 pt-2 border-t border-slate-200">
              Training: {round.trainingSet}
            </div>
          </button>
        ))}
      </div>

      {/* Selected Round Inspector Panel */}
      <div className="bg-white border border-slate-200 rounded p-5 space-y-4 shadow-2xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-200 gap-3">
          <div>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2">
              <FileSpreadsheet className="w-4 h-4 text-emerald-600" />
              <span>
                {roundInfo.title} Inspector ({groundTruthRows.length.toLocaleString()} Observations)
              </span>
            </h2>
            <p className="text-xs text-slate-500">Ground truth answers used to calculate local ROC AUC</p>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={handleDownloadSampleSubmissionCsv}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-white hover:bg-slate-50 text-slate-700 text-xs font-bold border border-slate-300 transition-all cursor-pointer shadow-2xs"
            >
              <Download className="w-3.5 h-3.5 text-blue-600" />
              <span>Sample Submission Template CSV</span>
            </button>

            <button
              onClick={handleDownloadGroundTruthCsv}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold transition-all cursor-pointer shadow-2xs uppercase tracking-wider"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download Ground Truth CSV</span>
            </button>
          </div>
        </div>

        {/* Data Sample Table */}
        <div className="overflow-x-auto rounded border border-slate-200">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-50 text-slate-500 uppercase text-[10px] tracking-wider font-bold border-b border-slate-200">
              <tr>
                <th className="px-4 py-2.5">anonymised_id</th>
                <th className="px-4 py-2.5">employed_status (Target)</th>
                <th className="px-4 py-2.5">Age</th>
                <th className="px-4 py-2.5">Education Level</th>
                <th className="px-4 py-2.5">Work Readiness Score</th>
                <th className="px-4 py-2.5">Previous Employment Months</th>
                <th className="px-4 py-2.5">Region</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {groundTruthRows.slice(0, 15).map((row) => (
                <tr key={row.anonymised_id} className="hover:bg-blue-50/50 transition-colors">
                  <td className="px-4 py-2.5 font-bold text-slate-800">{row.anonymised_id}</td>
                  <td className="px-4 py-2.5">
                    {row.employed_status === 1 ? (
                      <span className="px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 font-bold border border-emerald-200">
                        1 (Employed)
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded bg-rose-50 text-rose-700 font-bold border border-rose-200">
                        0 (Unemployed)
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2.5 text-slate-700">{row.age}</td>
                  <td className="px-4 py-2.5 text-slate-700">Level {row.education_level}</td>
                  <td className="px-4 py-2.5 text-slate-700">{row.work_readiness_score}</td>
                  <td className="px-4 py-2.5 text-slate-700">{row.prev_employment_months} mos</td>
                  <td className="px-4 py-2.5 text-slate-500">{row.region}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="text-right text-[11px] text-slate-500 font-mono italic">
          Showing 15 sample observations out of {groundTruthRows.length.toLocaleString()} total rows in {selectedRound.toUpperCase()} answer key.
        </div>
      </div>
    </div>
  );
};
