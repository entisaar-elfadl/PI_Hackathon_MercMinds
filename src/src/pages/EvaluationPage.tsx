import React, { useState } from 'react';
import { PlayCircle, Upload, CheckCircle2, AlertTriangle, Layers, FileText, Info } from 'lucide-react';
import { EvaluationResult, Experiment, GroundTruthRow, PredictionRow, RoundId } from '../types';
import { HISTORICAL_GROUND_TRUTH, HISTORICAL_ROUNDS } from '../data/historicalDatasets';
import { validatePredictions } from '../utils/validation';
import { matchPredictionsWithGroundTruth } from '../utils/idMatcher';
import { calculateRocAuc, generateRocPoints } from '../utils/aucCalculator';
import { ExperimentStorageService } from '../services/experimentStorage';
import { CsvUploader } from '../components/evaluation/CsvUploader';
import { ValidationReportView } from '../components/evaluation/ValidationReportView';

interface EvaluationPageProps {
  selectedRound: RoundId;
  setSelectedRound: (round: RoundId) => void;
  onEvaluationComplete: (result: EvaluationResult) => void;
}

export const EvaluationPage: React.FC<EvaluationPageProps> = ({
  selectedRound,
  setSelectedRound,
  onEvaluationComplete,
}) => {
  const [modelName, setModelName] = useState('Logistic Regression 3.3');
  const [modelVersion, setModelVersion] = useState('3.3.0');
  const [notes, setNotes] = useState('');

  const [rawParsedRows, setRawParsedRows] = useState<any[] | null>(null);
  const [filename, setFilename] = useState<string>('');

  const groundTruth: GroundTruthRow[] = HISTORICAL_GROUND_TRUTH[selectedRound] || [];

  // Perform validation when parsed data or round changes
  const validationResult = rawParsedRows
    ? validatePredictions(rawParsedRows, groundTruth)
    : null;

  const handleDataParsed = (rows: any[], fileName: string, sampleModelName?: string) => {
    setRawParsedRows(rows);
    setFilename(fileName);
    if (sampleModelName) {
      setModelName(sampleModelName);
    }
  };

  const handleRunEvaluation = () => {
    if (!validationResult || !validationResult.validationSummary.isValid) return;

    const matched = matchPredictionsWithGroundTruth(validationResult.validPredictions, groundTruth);
    const aucResult = calculateRocAuc(matched.matchedPairs);
    const rocPoints = generateRocPoints(matched.matchedPairs);

    const roundInfo = HISTORICAL_ROUNDS[selectedRound];

    const result: EvaluationResult = {
      id: `eval_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
      modelName,
      modelVersion,
      roundId: selectedRound,
      timestamp: new Date().toISOString(),
      filename: filename || 'predictions.csv',
      auc: aucResult.auc,
      scorePercentage: aucResult.scorePercentage,
      matchedPairsCount: matched.matchedCount,
      totalGroundTruthCount: groundTruth.length,
      positivesCount: aucResult.positivesCount,
      negativesCount: aucResult.negativesCount,
      rocPoints,
      matchedPairsSample: matched.matchedPairs,
      validationSummary: validationResult.validationSummary,
      notes,
      trainingSetDescription: roundInfo.trainingSet,
      evalSetDescription: roundInfo.evaluationSet,
    };

    // Save to local experiment storage
    ExperimentStorageService.saveEvaluationResult(result);

    // Notify parent to switch view to detailed result page
    onEvaluationComplete(result);
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto font-sans">
      {/* Page Heading */}
      <div className="border-b border-slate-200 pb-3">
        <h1 className="text-xl font-bold text-slate-900 uppercase tracking-tight flex items-center space-x-2">
          <PlayCircle className="w-5 h-5 text-blue-600" />
          <span>Historical Model Evaluation Pipeline</span>
        </h1>
        <p className="text-xs text-slate-500 mt-1">
          Upload a prediction CSV output from R or Python, select target historical survey round, and calculate ROC AUC.
        </p>
      </div>

      {/* Step 1: Round & Model Information */}
      <div className="bg-white border border-slate-200 rounded p-5 space-y-4 shadow-2xs">
        <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2">
          <span className="w-5 h-5 rounded bg-blue-50 border border-blue-200 text-blue-700 text-xs flex items-center justify-center font-bold font-mono">
            1
          </span>
          <span>Target Round &amp; Model Metadata</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div>
            <label className="block text-slate-700 font-bold mb-1">Historical Evaluation Round</label>
            <select
              value={selectedRound}
              onChange={(e) => setSelectedRound(e.target.value as RoundId)}
              className="w-full bg-slate-50 text-slate-900 p-2 rounded border border-slate-300 focus:outline-none focus:border-blue-600 font-mono font-semibold"
            >
              <option value="round_6">Round 6 (Trained on Rounds 1–5)</option>
              <option value="round_7">Round 7 (Trained on Rounds 1–6)</option>
              <option value="round_8">Round 8 (Trained on Rounds 1–7)</option>
            </select>
          </div>

          <div>
            <label className="block text-slate-700 font-bold mb-1">Model Name</label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="e.g. Logistic Regression 3.3"
              className="w-full bg-slate-50 text-slate-900 p-2 rounded border border-slate-300 focus:outline-none focus:border-blue-600 font-medium"
            />
          </div>

          <div>
            <label className="block text-slate-700 font-bold mb-1">Version Tag</label>
            <input
              type="text"
              value={modelVersion}
              onChange={(e) => setModelVersion(e.target.value)}
              placeholder="e.g. 3.3.0"
              className="w-full bg-slate-50 text-slate-900 p-2 rounded border border-slate-300 focus:outline-none focus:border-blue-600 font-mono"
            />
          </div>
        </div>

        <div>
          <label className="block text-slate-700 font-bold mb-1 text-xs">Experiment Notes (Optional)</label>
          <input
            type="text"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="e.g. Added interaction feature between work_readiness and age, median imputed."
            className="w-full bg-slate-50 text-slate-900 p-2 rounded border border-slate-300 text-xs focus:outline-none focus:border-blue-600"
          />
        </div>
      </div>

      {/* Step 2: Prediction CSV Upload / Benchmark Selection */}
      <div className="bg-white border border-slate-200 rounded p-5 space-y-4 shadow-2xs">
        <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2">
          <span className="w-5 h-5 rounded bg-blue-50 border border-blue-200 text-blue-700 text-xs flex items-center justify-center font-bold font-mono">
            2
          </span>
          <span>Upload Prediction CSV or Select Benchmark Model</span>
        </h2>

        <CsvUploader onDataParsed={handleDataParsed} selectedRound={selectedRound} />
      </div>

      {/* Step 3: Automated Validation Summary */}
      {validationResult && (
        <div className="space-y-4">
          <ValidationReportView summary={validationResult.validationSummary} />

          {/* Action Button */}
          <div className="pt-2 flex justify-end">
            <button
              disabled={!validationResult.validationSummary.isValid}
              onClick={handleRunEvaluation}
              className="px-6 py-2.5 rounded bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white font-bold text-xs shadow-xs transition-all flex items-center space-x-2 cursor-pointer uppercase tracking-wider"
            >
              <PlayCircle className="w-4 h-4" />
              <span>Calculate ROC AUC &amp; Save Experiment</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
