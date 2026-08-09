import React from 'react';
import { ArrowLeft, CheckCircle2, Award, Calendar, FileText, Layers, Download, Share2 } from 'lucide-react';
import { EvaluationResult } from '../types';
import { MetricCard } from '../components/common/MetricCard';
import { RocCurveChart } from '../components/charts/RocCurveChart';
import { ProbabilityDistChart } from '../components/charts/ProbabilityDistChart';
import { ThresholdAnalyzer } from '../components/evaluation/ThresholdAnalyzer';
import { IdAuditTable } from '../components/evaluation/IdAuditTable';

interface ResultsPageProps {
  result: EvaluationResult;
  onBack: () => void;
  onNavigateToMatrix: () => void;
}

export const ResultsPage: React.FC<ResultsPageProps> = ({ result, onBack, onNavigateToMatrix }) => {
  return (
    <div className="space-y-6 max-w-6xl mx-auto font-sans">
      {/* Navigation Header */}
      <div className="flex items-center justify-between border-b border-slate-200 pb-3">
        <button
          onClick={onBack}
          className="flex items-center space-x-2 text-xs font-bold text-slate-700 hover:text-slate-900 bg-white border border-slate-300 hover:bg-slate-50 px-3 py-1.5 rounded transition-all shadow-2xs cursor-pointer"
        >
          <ArrowLeft className="w-4 h-4 text-blue-600" />
          <span>Back to Evaluator</span>
        </button>

        <div className="flex items-center space-x-2">
          <button
            onClick={onNavigateToMatrix}
            className="px-3 py-1.5 rounded bg-blue-50 text-blue-700 border border-blue-200 text-xs font-bold hover:bg-blue-100 transition-all cursor-pointer"
          >
            Compare in Multi-Round Matrix
          </button>
        </div>
      </div>

      {/* Main Score Callout Box */}
      <div className="bg-white border border-slate-200 rounded p-6 shadow-2xs relative overflow-hidden">
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center space-x-2">
              <span className="px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200 text-xs font-bold font-mono uppercase">
                {result.roundId.replace('_', ' ').toUpperCase()}
              </span>
              <span className="text-xs text-slate-500 font-mono">File: {result.filename}</span>
            </div>

            <span className="text-xs text-slate-500 font-mono">
              Evaluated on {new Date(result.timestamp).toLocaleTimeString()}
            </span>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-4 border-y border-slate-200 py-4">
            <div>
              <h1 className="text-2xl font-bold text-slate-900 uppercase tracking-tight">{result.modelName}</h1>
              <p className="text-xs text-slate-600 mt-1 font-mono">
                Version Tag: <span className="text-slate-900 font-bold">{result.modelVersion}</span>
                {result.notes && <span className="ml-2 font-sans">({result.notes})</span>}
              </p>
            </div>

            <div className="text-left sm:text-right bg-blue-600 text-white p-4 rounded shadow-xs">
              <span className="text-[10px] uppercase tracking-wider font-bold text-blue-100 block">Primary Evaluation Score</span>
              <div className="flex items-baseline space-x-3 mt-1">
                <span className="text-3xl font-light font-mono tracking-tight">
                  AUC {result.auc.toFixed(5)}
                </span>
                <span className="text-lg font-bold font-mono text-blue-100">
                  {result.scorePercentage.toFixed(2)}%
                </span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono pt-1">
            <div className="text-slate-600">
              Train: <span className="text-slate-900 font-semibold">{result.trainingSetDescription}</span>
            </div>
            <div className="text-slate-600">
              Eval Target: <span className="text-blue-600 font-semibold">{result.evalSetDescription}</span>
            </div>
            <div className="text-slate-600">
              Matched Observations: <span className="text-emerald-600 font-semibold">{result.matchedPairsCount.toLocaleString()} / {result.totalGroundTruthCount.toLocaleString()}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="ROC AUC Metric"
          value={result.auc.toFixed(5)}
          subtitle="Area Under ROC Curve"
          badgeText="Primary"
          badgeType="success"
          highlight
        />

        <MetricCard
          title="Percentage Score"
          value={`${result.scorePercentage.toFixed(2)}%`}
          subtitle="AUC x 100"
          badgeText="Score"
          badgeType="info"
        />

        <MetricCard
          title="Matched Ground Truth"
          value={result.matchedPairsCount.toLocaleString()}
          subtitle={`Out of ${result.totalGroundTruthCount.toLocaleString()} total`}
          badgeText="100% ID Match"
          badgeType="success"
        />

        <MetricCard
          title="Class Balance"
          value={`${result.positivesCount} / ${result.negativesCount}`}
          subtitle="Employed (1) vs Unemployed (0)"
          badgeText={`Ratio ${(result.positivesCount / (result.negativesCount || 1)).toFixed(2)}`}
          badgeType="neutral"
        />
      </div>

      {/* Visual Charts: ROC Curve & Probability Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RocCurveChart rocPoints={result.rocPoints} auc={result.auc} />
        <ProbabilityDistChart matchedPairs={result.matchedPairsSample} />
      </div>

      {/* Interactive Threshold Slider & Confusion Matrix */}
      <ThresholdAnalyzer matchedPairs={result.matchedPairsSample} />

      {/* ID Audit Table */}
      <IdAuditTable matchedPairs={result.matchedPairsSample} />
    </div>
  );
};
