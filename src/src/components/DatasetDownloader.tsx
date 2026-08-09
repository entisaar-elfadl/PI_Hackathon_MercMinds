/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { Download, FileSpreadsheet, Loader2, Info } from 'lucide-react';
import { generateRoundDatasets, generateSamplePredictions } from '../utils/generator';
import { convertToCSV } from '../utils/csv';
import { RoundId } from '../types';

export function DatasetDownloader() {
  const [downloading, setDownloading] = useState<string | null>(null);

  const triggerDownload = (filename: string, csvContent: string) => {
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDownloadDataset = async (roundId: RoundId, type: 'training' | 'evaluation') => {
    const taskKey = `${roundId}-${type}`;
    setDownloading(taskKey);

    // Run in short timeout to allow spinner to paint
    setTimeout(() => {
      try {
        const { training, evaluation } = generateRoundDatasets(roundId);
        const data = type === 'training' ? training : evaluation;
        const filename = 
          type === 'training' 
            ? roundId === 'round_6' ? 'rounds_1_5.csv' : roundId === 'round_7' ? 'rounds_1_6.csv' : 'rounds_1_7.csv'
            : `${roundId}.csv`;

        const csvContent = convertToCSV(data);
        triggerDownload(filename, csvContent);
      } catch (err) {
        console.error('Download failed', err);
      } finally {
        setDownloading(null);
      }
    }, 100);
  };

  const handleDownloadSample = (modelType: 'perfect' | 'random' | 'weak_logistic' | 'strong_boosting', roundId: RoundId) => {
    const taskKey = `sample-${modelType}-${roundId}`;
    setDownloading(taskKey);

    setTimeout(() => {
      try {
        const { evaluation } = generateRoundDatasets(roundId);
        const predictions = generateSamplePredictions(evaluation, modelType);
        const filename = `prediction_${modelType}_${roundId}.csv`;
        const csvContent = convertToCSV(predictions);
        triggerDownload(filename, csvContent);
      } catch (err) {
        console.error('Download failed', err);
      } finally {
        setDownloading(null);
      }
    }, 100);
  };

  return (
    <div id="dataset_downloader_card" className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Datasets Download Section */}
      <div className="lg:col-span-2 bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
        <div className="flex items-center gap-2 mb-4">
          <FileSpreadsheet className="w-5 h-5 text-indigo-500" />
          <h2 className="text-base font-bold text-slate-800 font-display">Competition Datasets Generator</h2>
        </div>
        <p className="text-xs text-slate-500 mb-6 leading-relaxed">
          Generate and download the actual historical survey datasets to train your R/Python machine learning models. 
          To prevent <strong>data leakage</strong>, ensure your models are trained <i>only</i> on the training datasets 
          prior to being evaluated on corresponding survey rounds.
        </p>

        <div className="space-y-4">
          {/* Round 6 Card */}
          <div className="p-4 bg-slate-50 border border-slate-100 rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <span className="inline-block px-2 py-0.5 bg-indigo-100 text-indigo-700 text-[10px] font-bold rounded-full mb-1">
                SIMULATION ROUND 6
              </span>
              <h3 className="text-sm font-bold text-slate-800">Round 6 Evaluation Pack</h3>
              <p className="text-xs text-slate-500 mt-1">
                Train on survey rounds 1–5 (3,800 rows). Evaluate against round 6 outcomes (1,200 rows).
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => handleDownloadDataset('round_6', 'training')}
                disabled={downloading !== null}
                className="px-3 py-1.5 bg-white border border-slate-200 text-xs font-semibold text-slate-700 rounded-lg hover:bg-slate-100 active:scale-95 transition-all flex items-center gap-1.5 disabled:opacity-50"
              >
                {downloading === 'round_6-training' ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-slate-400" />
                ) : (
                  <Download className="w-3.5 h-3.5" />
                )}
                rounds_1_5.csv (Train)
              </button>
              <button
                onClick={() => handleDownloadDataset('round_6', 'evaluation')}
                disabled={downloading !== null}
                className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-xs font-semibold text-white rounded-lg active:scale-95 transition-all flex items-center gap-1.5 disabled:opacity-50"
              >
                {downloading === 'round_6-evaluation' ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Download className="w-3.5 h-3.5" />
                )}
                round_6.csv (Answers)
              </button>
            </div>
          </div>

          {/* Round 7 Card */}
          <div className="p-4 bg-slate-50 border border-slate-100 rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <span className="inline-block px-2 py-0.5 bg-indigo-100 text-indigo-700 text-[10px] font-bold rounded-full mb-1">
                SIMULATION ROUND 7
              </span>
              <h3 className="text-sm font-bold text-slate-800">Round 7 Evaluation Pack</h3>
              <p className="text-xs text-slate-500 mt-1">
                Train on survey rounds 1–6 (4,600 rows). Evaluate against round 7 outcomes (1,300 rows).
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => handleDownloadDataset('round_7', 'training')}
                disabled={downloading !== null}
                className="px-3 py-1.5 bg-white border border-slate-200 text-xs font-semibold text-slate-700 rounded-lg hover:bg-slate-100 active:scale-95 transition-all flex items-center gap-1.5 disabled:opacity-50"
              >
                {downloading === 'round_7-training' ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-slate-400" />
                ) : (
                  <Download className="w-3.5 h-3.5" />
                )}
                rounds_1_6.csv (Train)
              </button>
              <button
                onClick={() => handleDownloadDataset('round_7', 'evaluation')}
                disabled={downloading !== null}
                className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-xs font-semibold text-white rounded-lg active:scale-95 transition-all flex items-center gap-1.5 disabled:opacity-50"
              >
                {downloading === 'round_7-evaluation' ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Download className="w-3.5 h-3.5" />
                )}
                round_7.csv (Answers)
              </button>
            </div>
          </div>

          {/* Round 8 Card */}
          <div className="p-4 bg-slate-50 border border-slate-100 rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <span className="inline-block px-2 py-0.5 bg-indigo-100 text-indigo-700 text-[10px] font-bold rounded-full mb-1">
                SIMULATION ROUND 8
              </span>
              <h3 className="text-sm font-bold text-slate-800">Round 8 Evaluation Pack</h3>
              <p className="text-xs text-slate-500 mt-1">
                Train on survey rounds 1–7 (5,400 rows). Evaluate against round 8 outcomes (1,400 rows).
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => handleDownloadDataset('round_8', 'training')}
                disabled={downloading !== null}
                className="px-3 py-1.5 bg-white border border-slate-200 text-xs font-semibold text-slate-700 rounded-lg hover:bg-slate-100 active:scale-95 transition-all flex items-center gap-1.5 disabled:opacity-50"
              >
                {downloading === 'round_8-training' ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-slate-400" />
                ) : (
                  <Download className="w-3.5 h-3.5" />
                )}
                rounds_1_7.csv (Train)
              </button>
              <button
                onClick={() => handleDownloadDataset('round_8', 'evaluation')}
                disabled={downloading !== null}
                className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-xs font-semibold text-white rounded-lg active:scale-95 transition-all flex items-center gap-1.5 disabled:opacity-50"
              >
                {downloading === 'round_8-evaluation' ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Download className="w-3.5 h-3.5" />
                )}
                round_8.csv (Answers)
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Templates / Sandbox Section */}
      <div className="bg-slate-900 text-white rounded-2xl p-6 flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2 mb-3">
            <Info className="w-5 h-5 text-indigo-400" />
            <h2 className="text-base font-bold font-display">Instant Sandbox Templates</h2>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed mb-6">
            Don't have an R model running yet? Generate and download pre-fitted submission files representing 
            different mathematical tiers to test the platform's diagnostic charts, ID-matching, and leaderboard:
          </p>

          <div className="space-y-2">
            <div>
              <label className="text-[10px] font-bold text-slate-500 uppercase">Pre-made Submissions for Round 6</label>
              <div className="grid grid-cols-2 gap-2 mt-1.5">
                <button
                  onClick={() => handleDownloadSample('perfect', 'round_6')}
                  className="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-[11px] font-semibold text-emerald-400 transition-colors text-left truncate flex items-center gap-1"
                >
                  🥇 Perfect (AUC 1.0)
                </button>
                <button
                  onClick={() => handleDownloadSample('strong_boosting', 'round_6')}
                  className="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-[11px] font-semibold text-teal-400 transition-colors text-left truncate flex items-center gap-1"
                >
                  🚀 Strong (AUC ~0.76)
                </button>
                <button
                  onClick={() => handleDownloadSample('weak_logistic', 'round_6')}
                  className="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-[11px] font-semibold text-amber-400 transition-colors text-left truncate flex items-center gap-1"
                >
                  📉 Weak (AUC ~0.61)
                </button>
                <button
                  onClick={() => handleDownloadSample('random', 'round_6')}
                  className="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-[11px] font-semibold text-rose-400 transition-colors text-left truncate flex items-center gap-1"
                >
                  🎲 Random (AUC 0.5)
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-slate-800 text-[10px] text-slate-400 leading-relaxed">
          <strong>Tip:</strong> Download the <code className="text-slate-300 font-mono">prediction_strong_boosting_round_6.csv</code> and 
          upload it in the <strong>Model Evaluator</strong> tab to see a fully populated diagnostics report.
        </div>
      </div>
    </div>
  );
}
