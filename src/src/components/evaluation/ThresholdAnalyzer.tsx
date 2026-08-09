import React, { useState } from 'react';
import { Sliders, Target, CheckSquare, XSquare } from 'lucide-react';
import { MatchedPair } from '../../types';
import { calculateThresholdMetrics } from '../../utils/aucCalculator';

interface ThresholdAnalyzerProps {
  matchedPairs: MatchedPair[];
}

export const ThresholdAnalyzer: React.FC<ThresholdAnalyzerProps> = ({ matchedPairs }) => {
  const [cutoff, setCutoff] = useState<number>(0.5);

  const metrics = calculateThresholdMetrics(matchedPairs, cutoff);

  return (
    <div className="bg-white border border-slate-200 rounded p-5 space-y-4 shadow-2xs font-sans">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-200 gap-2">
        <div className="flex items-center space-x-2">
          <Sliders className="w-5 h-5 text-blue-600" />
          <div>
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Classification Threshold &amp; Confusion Matrix</h3>
            <p className="text-xs text-slate-500">Slide decision threshold to inspect hard classification outcomes at probability cutoff</p>
          </div>
        </div>

        <div className="flex items-center space-x-3 bg-slate-50 px-3 py-1.5 rounded border border-slate-200">
          <span className="text-xs text-slate-600 font-semibold">Probability Cutoff:</span>
          <span className="text-sm font-bold text-blue-600 font-mono">{cutoff.toFixed(2)}</span>
        </div>
      </div>

      {/* Slider Control */}
      <div className="space-y-2 pt-1">
        <input
          type="range"
          min="0.01"
          max="0.99"
          step="0.01"
          value={cutoff}
          onChange={(e) => setCutoff(parseFloat(e.target.value))}
          className="w-full h-2 bg-slate-200 rounded appearance-none cursor-pointer accent-blue-600"
        />
        <div className="flex justify-between text-[11px] text-slate-500 font-mono">
          <span>0.00 (Predict all Employed = 1)</span>
          <span>0.50 (Default)</span>
          <span>1.00 (Predict all Unemployed = 0)</span>
        </div>
      </div>

      {/* Confusion Matrix & Classification Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
        {/* 2x2 Confusion Matrix */}
        <div className="bg-slate-50 p-4 rounded border border-slate-200 space-y-3">
          <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center justify-between">
            <span>2x2 Confusion Matrix</span>
            <span className="text-[10px] text-slate-500 font-mono">Threshold: {cutoff.toFixed(2)}</span>
          </h4>

          <div className="grid grid-cols-2 gap-2 text-xs font-mono text-center">
            {/* TP */}
            <div className="bg-emerald-50 border border-emerald-200 p-3 rounded">
              <span className="text-[10px] text-emerald-800 font-bold block uppercase">True Positive (TP)</span>
              <span className="text-xl font-bold text-emerald-700 mt-1 block">{metrics.tp}</span>
              <span className="text-[10px] text-slate-500 mt-1 block">Actual 1, Pred &ge; {cutoff.toFixed(2)}</span>
            </div>

            {/* FP */}
            <div className="bg-rose-50 border border-rose-200 p-3 rounded">
              <span className="text-[10px] text-rose-800 font-bold block uppercase">False Positive (FP)</span>
              <span className="text-xl font-bold text-rose-700 mt-1 block">{metrics.fp}</span>
              <span className="text-[10px] text-slate-500 mt-1 block">Actual 0, Pred &ge; {cutoff.toFixed(2)}</span>
            </div>

            {/* FN */}
            <div className="bg-amber-50 border border-amber-200 p-3 rounded">
              <span className="text-[10px] text-amber-800 font-bold block uppercase">False Negative (FN)</span>
              <span className="text-xl font-bold text-amber-700 mt-1 block">{metrics.fn}</span>
              <span className="text-[10px] text-slate-500 mt-1 block">Actual 1, Pred &lt; {cutoff.toFixed(2)}</span>
            </div>

            {/* TN */}
            <div className="bg-blue-50 border border-blue-200 p-3 rounded">
              <span className="text-[10px] text-blue-800 font-bold block uppercase">True Negative (TN)</span>
              <span className="text-xl font-bold text-blue-700 mt-1 block">{metrics.tn}</span>
              <span className="text-[10px] text-slate-500 mt-1 block">Actual 0, Pred &lt; {cutoff.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* Calculated Metrics Breakdown */}
        <div className="bg-slate-50 p-4 rounded border border-slate-200 flex flex-col justify-between space-y-3">
          <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
            Derived Classification Metrics
          </h4>

          <div className="grid grid-cols-2 gap-3 font-mono">
            <div className="bg-white p-2.5 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 font-bold uppercase block">Accuracy</span>
              <span className="text-base font-bold text-slate-900">{(metrics.accuracy * 100).toFixed(2)}%</span>
            </div>

            <div className="bg-white p-2.5 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 font-bold uppercase block">Precision</span>
              <span className="text-base font-bold text-emerald-700">{(metrics.precision * 100).toFixed(2)}%</span>
            </div>

            <div className="bg-white p-2.5 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 font-bold uppercase block">Recall (Sensitivity)</span>
              <span className="text-base font-bold text-blue-600">{(metrics.recall * 100).toFixed(2)}%</span>
            </div>

            <div className="bg-white p-2.5 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 font-bold uppercase block">F1 Score</span>
              <span className="text-base font-bold text-amber-700">{metrics.f1Score.toFixed(4)}</span>
            </div>
          </div>

          <p className="text-[11px] text-slate-500 italic pt-1 border-t border-slate-200">
            Note: ROC AUC evaluates ranking quality across ALL thresholds simultaneously and is threshold-invariant.
          </p>
        </div>
      </div>
    </div>
  );
};
