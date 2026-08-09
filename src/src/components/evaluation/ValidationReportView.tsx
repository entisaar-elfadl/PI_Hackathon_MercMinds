import React from 'react';
import { AlertCircle, AlertTriangle, CheckCircle2, FileCheck, Layers, HelpCircle, Users } from 'lucide-react';
import { ValidationSummary } from '../../types';

interface ValidationReportViewProps {
  summary: ValidationSummary;
}

export const ValidationReportView: React.FC<ValidationReportViewProps> = ({ summary }) => {
  return (
    <div className="bg-white border border-slate-200 rounded p-5 space-y-4 shadow-2xs font-sans">
      <div className="flex items-center justify-between pb-3 border-b border-slate-200">
        <div className="flex items-center space-x-2">
          {summary.isValid ? (
            <CheckCircle2 className="w-5 h-5 text-emerald-600" />
          ) : (
            <AlertCircle className="w-5 h-5 text-rose-600" />
          )}
          <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
            Prediction CSV Validation Status
          </h3>
        </div>
        <span
          className={`px-2 py-0.5 text-[11px] font-bold rounded uppercase font-mono ${
            summary.isValid
              ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
              : 'bg-rose-50 text-rose-700 border border-rose-200'
          }`}
        >
          {summary.isValid ? 'Valid & Ready' : 'Validation Errors'}
        </span>
      </div>

      {/* Grid Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-slate-50 p-3 rounded border border-slate-200">
          <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Parsed Rows</span>
          <p className="text-lg font-bold text-slate-900 font-mono mt-0.5">{summary.totalRowsParsed.toLocaleString()}</p>
        </div>

        <div className="bg-slate-50 p-3 rounded border border-slate-200">
          <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Valid Probabilities</span>
          <p className="text-lg font-bold text-emerald-600 font-mono mt-0.5">
            {summary.validPredictionCount.toLocaleString()}
          </p>
        </div>

        <div className="bg-slate-50 p-3 rounded border border-slate-200">
          <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Invalid / NA / Out of Range</span>
          <p
            className={`text-lg font-bold font-mono mt-0.5 ${
              summary.invalidPredictionCount > 0 ? 'text-rose-600' : 'text-slate-400'
            }`}
          >
            {summary.invalidPredictionCount.toLocaleString()}
          </p>
        </div>

        <div className="bg-slate-50 p-3 rounded border border-slate-200">
          <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Matched Ground Truth IDs</span>
          <p className="text-lg font-bold text-blue-600 font-mono mt-0.5">
            {summary.idMatchStats.matchedCount.toLocaleString()} / {summary.idMatchStats.totalGroundTruth.toLocaleString()}
          </p>
        </div>
      </div>

      {/* ID Match Details */}
      <div className="bg-slate-50 p-3.5 rounded border border-slate-200 text-xs space-y-2">
        <div className="flex items-center justify-between text-slate-800 font-semibold border-b border-slate-200 pb-2">
          <span className="flex items-center space-x-1.5">
            <Users className="w-4 h-4 text-blue-600" />
            <span className="font-bold uppercase text-[11px] text-slate-700">ID Matching &amp; Ground Truth Audit</span>
          </span>
          <span className="font-mono text-blue-600 font-bold">
            {((summary.idMatchStats.matchedCount / (summary.idMatchStats.totalGroundTruth || 1)) * 100).toFixed(1)}% Coverage
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11px] pt-1">
          <div className="text-slate-600">
            Ground Truth Missing Predictions:{' '}
            <strong className={summary.idMatchStats.missingInPredictions > 0 ? 'text-amber-700 font-mono' : 'text-slate-500 font-mono'}>
              {summary.idMatchStats.missingInPredictions}
            </strong>
          </div>
          <div className="text-slate-600">
            Extra IDs Not In Evaluation Set:{' '}
            <strong className={summary.idMatchStats.extraInPredictions > 0 ? 'text-amber-700 font-mono' : 'text-slate-500 font-mono'}>
              {summary.idMatchStats.extraInPredictions}
            </strong>
          </div>
          <div className="text-slate-600">
            Duplicate Prediction IDs:{' '}
            <strong className={summary.idMatchStats.duplicateInPredictions > 0 ? 'text-amber-700 font-mono' : 'text-slate-500 font-mono'}>
              {summary.idMatchStats.duplicateInPredictions}
            </strong>
          </div>
        </div>
      </div>

      {/* Errors Section */}
      {summary.errors.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-bold text-rose-700 uppercase tracking-wider flex items-center space-x-1">
            <AlertCircle className="w-4 h-4 text-rose-600" />
            <span>Blocking Errors ({summary.errors.length})</span>
          </p>
          {summary.errors.map((err, idx) => (
            <div key={idx} className="bg-rose-50 border border-rose-200 rounded p-3 text-xs text-rose-900 space-y-1">
              <p className="font-bold flex items-center justify-between">
                <span>❌ {err.message}</span>
                {err.count && <span className="font-mono text-rose-700 font-bold">{err.count} issues</span>}
              </p>
              {err.details && <p className="text-rose-800">{err.details}</p>}
              {err.sampleIds && err.sampleIds.length > 0 && (
                <div className="pt-1">
                  <span className="text-[10px] uppercase text-rose-700 font-mono">Sample Affected IDs:</span>
                  <p className="font-mono text-[11px] bg-rose-100 border border-rose-200 px-2 py-1 rounded text-rose-900 mt-0.5">
                    {err.sampleIds.join(', ')}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Warnings Section */}
      {summary.warnings.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-bold text-amber-700 uppercase tracking-wider flex items-center space-x-1">
            <AlertTriangle className="w-4 h-4 text-amber-600" />
            <span>Validation Warnings ({summary.warnings.length})</span>
          </p>
          {summary.warnings.map((warn, idx) => (
            <div key={idx} className="bg-amber-50 border border-amber-200 rounded p-3 text-xs text-amber-900 space-y-1">
              <p className="font-semibold">{warn.message}</p>
              {warn.sampleIds && warn.sampleIds.length > 0 && (
                <div className="pt-1">
                  <span className="text-[10px] uppercase text-amber-700 font-mono">Sample IDs:</span>
                  <p className="font-mono text-[11px] bg-amber-100 border border-amber-200 px-2 py-1 rounded text-amber-900 mt-0.5">
                    {warn.sampleIds.join(', ')}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
