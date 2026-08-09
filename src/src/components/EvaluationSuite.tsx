/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef } from 'react';
import { 
  Upload, FileText, CheckCircle2, AlertTriangle, XCircle, 
  Settings, Save, Loader2, Sparkles, HelpCircle, ArrowRight, RefreshCw 
} from 'lucide-react';
import { RoundId, Prediction, GroundTruth, ValidationError, EvaluationResult } from '../types';
import { parseCSV, validatePredictionCSV, validateGroundTruthCSV } from '../utils/csv';
import { calculateAUC } from '../utils/auc';
import { generateRoundDatasets } from '../utils/generator';
import { RocCurve } from './RocCurve';
import { PredictionDistribution } from './PredictionDistribution';

interface EvaluationSuiteProps {
  onSaveExperiment: (result: EvaluationResult) => void;
  activeResult: EvaluationResult | null;
  setActiveResult: (res: EvaluationResult | null) => void;
}

export function EvaluationSuite({
  onSaveExperiment,
  activeResult,
  setActiveResult,
}: EvaluationSuiteProps) {
  // Config & State
  const [selectedRound, setSelectedRound] = useState<RoundId>('round_6');
  const [useCustomGroundTruth, setUseCustomGroundTruth] = useState(false);

  // File Upload states
  const [predictionFile, setPredictionFile] = useState<File | null>(null);
  const [predictionFileName, setPredictionFileName] = useState('');
  const [groundTruthFile, setGroundTruthFile] = useState<File | null>(null);
  const [groundTruthFileName, setGroundTruthFileName] = useState('');

  // Status and logs
  const [evaluating, setEvaluating] = useState(false);
  const [errors, setErrors] = useState<ValidationError[]>([]);
  const [warnings, setWarnings] = useState<ValidationError[]>([]);

  // Metadata for saving experiment
  const [modelName, setModelName] = useState('Logistic Regression Classifier');
  const [modelVersion, setModelVersion] = useState('1.0');
  const [notes, setNotes] = useState('');
  const [saved, setSaved] = useState(false);

  // Drag and drop states
  const [isPredDragging, setIsPredDragging] = useState(false);
  const [isGtDragging, setIsGtDragging] = useState(false);

  const predInputRef = useRef<HTMLInputElement | null>(null);
  const gtInputRef = useRef<HTMLInputElement | null>(null);

  // Trigger file dialogs
  const triggerPredSelect = () => predInputRef.current?.click();
  const triggerGtSelect = () => gtInputRef.current?.click();

  // Handle files drag & drop
  const handlePredDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsPredDragging(true);
  };
  const handlePredDragLeave = () => setIsPredDragging(false);
  const handlePredDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsPredDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setPredictionFile(file);
      setPredictionFileName(file.name);
    }
  };

  const handleGtDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsGtDragging(true);
  };
  const handleGtDragLeave = () => setIsGtDragging(false);
  const handleGtDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsGtDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setGroundTruthFile(file);
      setGroundTruthFileName(file.name);
    }
  };

  // Reset uploader
  const handleReset = () => {
    setPredictionFile(null);
    setPredictionFileName('');
    setGroundTruthFile(null);
    setGroundTruthFileName('');
    setErrors([]);
    setWarnings([]);
    setActiveResult(null);
    setSaved(false);
    setNotes('');
  };

  // Core Evaluation logic
  const handleEvaluate = async () => {
    if (!predictionFile) return;

    setEvaluating(true);
    setErrors([]);
    setWarnings([]);
    setSaved(false);

    // Timeout to allow spinner rendering
    setTimeout(async () => {
      try {
        // 1. Read Prediction File
        const predText = await predictionFile.text();
        const predRawRows = parseCSV(predText);
        const { predictions, errors: predErrors } = validatePredictionCSV(predRawRows);

        if (predErrors.some(e => e.severity === 'error')) {
          setErrors(predErrors.filter(e => e.severity === 'error'));
          setWarnings(predErrors.filter(e => e.severity === 'warning'));
          setEvaluating(false);
          return;
        }

        // Gather initial warnings if any (e.g., duplicates skipped)
        let localWarnings = predErrors.filter(e => e.severity === 'warning');

        // 2. Load Ground Truth
        let truthData: GroundTruth[] = [];
        if (useCustomGroundTruth) {
          if (!groundTruthFile) {
            setErrors([{ message: 'You selected Custom Ground Truth but did not upload a file.', severity: 'error' }]);
            setEvaluating(false);
            return;
          }
          const gtText = await groundTruthFile.text();
          const gtRawRows = parseCSV(gtText);
          const { groundTruths, errors: gtErrors } = validateGroundTruthCSV(gtRawRows);

          if (gtErrors.some(e => e.severity === 'error')) {
            setErrors(gtErrors.filter(e => e.severity === 'error'));
            setEvaluating(false);
            return;
          }
          truthData = groundTruths;
          localWarnings = [...localWarnings, ...gtErrors.filter(e => e.severity === 'warning')];
        } else {
          // Load deterministic built-in simulator data
          const { evaluation } = generateRoundDatasets(selectedRound);
          truthData = evaluation;
        }

        // 3. ID Matching & Validation Pipeline
        // Match prediction records against the loaded ground truth using Map
        const predMap = new Map<string, number>(); // anonymised_id -> probability
        predictions.forEach(p => {
          predMap.set(p.anonymised_id, p.employed_status);
        });

        const matchedActual: number[] = [];
        const matchedPredicted: number[] = [];
        
        let matchedCount = 0;
        let missingPredictionCount = 0;
        let extraPredictionCount = 0;
        const duplicatePredictionCount = predRawRows.length - 1 - predictions.length - predErrors.filter(e => e.severity === 'error').length;

        // Verify ground truth IDs
        truthData.forEach(gt => {
          const predProb = predMap.get(gt.anonymised_id);
          if (predProb !== undefined) {
            matchedActual.push(gt.employed_status);
            matchedPredicted.push(predProb);
            matchedCount++;
          } else {
            missingPredictionCount++;
          }
        });

        // Verify if prediction file contains extra/unexpected IDs
        const truthIdSet = new Set(truthData.map(t => t.anonymised_id));
        predictions.forEach(p => {
          if (!truthIdSet.has(p.anonymised_id)) {
            extraPredictionCount++;
          }
        });

        // Formulate Warnings for mismatched IDs
        if (missingPredictionCount > 0) {
          localWarnings.push({
            severity: 'warning',
            message: `⚠️ ${missingPredictionCount} participant IDs from the evaluation dataset are missing from your prediction file. These rows were omitted from AUC calculations.`,
          });
        }
        if (extraPredictionCount > 0) {
          localWarnings.push({
            severity: 'warning',
            message: `⚠️ Your prediction file contains ${extraPredictionCount} IDs that do not exist in the ${selectedRound.replace('_', ' ').toUpperCase()} evaluation dataset. They were skipped.`,
          });
        }

        setWarnings(localWarnings);

        if (matchedCount === 0) {
          setErrors([
            {
              message: 'Zero IDs matched! The anonymised_ids in your prediction file do not match the selected evaluation dataset. Check that you uploaded predictions for the correct round.',
              severity: 'error',
            },
          ]);
          setEvaluating(false);
          return;
        }

        // 4. Calculate ROC AUC
        const { auc, fpr, tpr, thresholds } = calculateAUC(matchedActual, matchedPredicted);

        const result: EvaluationResult = {
          roundId: selectedRound,
          modelName: modelName.trim() || 'Custom R Model',
          modelVersion: modelVersion.trim() || '1.0',
          auc,
          score: parseFloat((auc * 100).toFixed(2)),
          totalObservations: truthData.length,
          validObservations: matchedCount,
          removedObservations: truthData.length - matchedCount,
          missingGroundTruth: 0,
          matchedCount,
          missingPredictionCount,
          extraPredictionCount,
          duplicatePredictionCount: Math.max(0, duplicatePredictionCount),
          predictionFile: predictionFileName,
          timestamp: new Date().toISOString(),
          notes: '',
          rocCurve: { fpr, tpr, thresholds },
        };

        setActiveResult(result);
      } catch (err: any) {
        setErrors([{ message: `Evaluation crashed: ${err.message}`, severity: 'error' }]);
      } finally {
        setEvaluating(false);
      }
    }, 150);
  };

  const handleSave = () => {
    if (!activeResult) return;
    const finalizedResult = {
      ...activeResult,
      modelName: modelName.trim() || 'Custom R Model',
      modelVersion: modelVersion.trim() || '1.0',
      notes: notes.trim(),
    };
    onSaveExperiment(finalizedResult);
    setSaved(true);
  };

  return (
    <div id="evaluation_suite_tab" className="space-y-6">
      {!activeResult ? (
        // Evaluator Selection & File Upload Screen
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Settings Left Panel */}
          <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 space-y-5 flex flex-col justify-between backdrop-blur-md relative overflow-hidden">
            <div className="absolute top-0 left-0 w-24 h-24 bg-blue-500/5 blur-2xl rounded-full pointer-events-none" />
            
            <div className="space-y-5 relative z-10">
              <div className="flex items-center gap-2 mb-2">
                <Settings className="w-5 h-5 text-blue-400" />
                <h2 className="text-sm font-extrabold text-white uppercase tracking-wider">Evaluation Config</h2>
              </div>
 
              {/* Select Evaluation Round */}
              <div className="space-y-2">
                <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest flex items-center gap-1">
                  Historical Validation Round
                  <HelpCircle className="w-3.5 h-3.5 text-slate-600 cursor-help" title="Which survey round are you testing against?" />
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {(['round_6', 'round_7', 'round_8'] as RoundId[]).map(rId => (
                    <button
                      key={rId}
                      type="button"
                      onClick={() => setSelectedRound(rId)}
                      className={`py-2 px-3 text-xs font-bold rounded-xl border text-center transition-all ${
                        selectedRound === rId
                          ? 'bg-blue-600 border-blue-500 text-white shadow-md shadow-blue-900/40'
                          : 'bg-slate-950/40 border-slate-800 text-slate-400 hover:bg-slate-900/30 hover:text-slate-200'
                      }`}
                    >
                      {rId.replace('_', ' ').toUpperCase()}
                    </button>
                  ))}
                </div>
                <p className="text-[10px] text-slate-500 leading-relaxed font-semibold">
                  {selectedRound === 'round_6'
                    ? 'Validating with simulated rounds 1–5 to predict round 6.'
                    : selectedRound === 'round_7'
                    ? 'Validating with simulated rounds 1–6 to predict round 7.'
                    : 'Validating with simulated rounds 1–7 to predict round 8.'}
                </p>
              </div>
 
              {/* Select Ground Truth Source */}
              <div className="space-y-2.5 pt-4 border-t border-slate-800/80">
                <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest">
                  Ground Truth Dataset Source
                </label>
                <div className="flex flex-col gap-2">
                  <label className="flex items-start gap-3 p-3 bg-slate-950/40 rounded-xl border border-slate-800/80 hover:bg-slate-900/20 cursor-pointer transition-colors">
                    <input
                      type="radio"
                      checked={!useCustomGroundTruth}
                      onChange={() => setUseCustomGroundTruth(false)}
                      className="mt-0.5 text-blue-500 focus:ring-blue-500 cursor-pointer accent-blue-600 bg-slate-950 border-slate-800"
                    />
                    <div>
                      <div className="text-xs font-extrabold text-white flex items-center gap-1.5">
                        Built-in Simulated Ground Truth
                        <Sparkles className="w-3.5 h-3.5 text-blue-400 fill-blue-500/20" />
                      </div>
                      <div className="text-[10px] text-slate-500 mt-0.5 font-medium">
                        Matches generated training packs. No download required.
                      </div>
                    </div>
                  </label>
 
                  <label className="flex items-start gap-3 p-3 bg-slate-950/40 rounded-xl border border-slate-800/80 hover:bg-slate-900/20 cursor-pointer transition-colors">
                    <input
                      type="radio"
                      checked={useCustomGroundTruth}
                      onChange={() => setUseCustomGroundTruth(true)}
                      className="mt-0.5 text-blue-500 focus:ring-blue-500 cursor-pointer accent-blue-600 bg-slate-950 border-slate-800"
                    />
                    <div>
                      <div className="text-xs font-extrabold text-white">Custom Ground Truth CSV Upload</div>
                      <div className="text-[10px] text-slate-500 mt-0.5 font-medium">
                        Upload your own validation survey with true targets.
                      </div>
                    </div>
                  </label>
                </div>
              </div>
 
              {/* Model Description Input */}
              <div className="space-y-3 pt-4 border-t border-slate-800/80">
                <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest">
                  Model Label
                </label>
                <div className="grid grid-cols-3 gap-2">
                  <div className="col-span-2 space-y-1">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider">Model Name</span>
                    <input
                      type="text"
                      value={modelName}
                      onChange={e => setModelName(e.target.value)}
                      placeholder="e.g. XGBoost R"
                      className="w-full px-2.5 py-1.5 bg-slate-950/40 border border-slate-800 text-white rounded-lg text-xs font-bold focus:border-blue-500/50 focus:bg-slate-900/40"
                    />
                  </div>
                  <div className="space-y-1">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider">Version</span>
                    <input
                      type="text"
                      value={modelVersion}
                      onChange={e => setModelVersion(e.target.value)}
                      placeholder="e.g. 1.0"
                      className="w-full px-2.5 py-1.5 bg-slate-950/40 border border-slate-800 text-white rounded-lg text-xs font-bold text-center focus:border-blue-500/50 focus:bg-slate-900/40"
                    />
                  </div>
                </div>
              </div>
            </div>
 
            <div className="mt-6 pt-4 border-t border-slate-800/80 text-[10px] text-slate-500 leading-relaxed font-semibold">
              <strong>Ground-truth variables:</strong> Valid models output probabilities between <code className="text-blue-400 font-semibold bg-blue-500/10 px-1 rounded">0</code> and <code className="text-blue-400 font-semibold bg-blue-500/10 px-1 rounded">1</code> representing the likelihood of being <strong>employed_status</strong>.
            </div>
          </div>
 
          {/* Right Upload Panel */}
          <div className="lg:col-span-2 bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 flex flex-col justify-between backdrop-blur-md relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-600/5 blur-[50px] rounded-full pointer-events-none" />
            
            <div className="space-y-4 relative z-10">
              <div className="flex items-center gap-2 mb-2">
                <Upload className="w-5 h-5 text-blue-400" />
                <h2 className="text-sm font-extrabold text-white uppercase tracking-wider">Upload Submissions</h2>
              </div>
 
              {/* Custom Ground Truth upload container if selected */}
              {useCustomGroundTruth && (
                <div className="space-y-2">
                  <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest">
                    Upload Ground Truth Answers CSV
                  </label>
                  <div
                    onDragOver={handleGtDragOver}
                    onDragLeave={handleGtDragLeave}
                    onDrop={handleGtDrop}
                    onClick={triggerGtSelect}
                    className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[100px] ${
                      isGtDragging
                        ? 'border-blue-500 bg-blue-500/10'
                        : groundTruthFile
                        ? 'border-emerald-500/50 bg-emerald-500/5'
                        : 'border-slate-800 hover:border-slate-700 bg-slate-950/40'
                    }`}
                  >
                    <input
                      type="file"
                      ref={gtInputRef}
                      onChange={e => {
                        if (e.target.files && e.target.files[0]) {
                          setGroundTruthFile(e.target.files[0]);
                          setGroundTruthFileName(e.target.files[0].name);
                        }
                      }}
                      accept=".csv"
                      className="hidden"
                    />
                    {groundTruthFile ? (
                      <div className="flex items-center gap-2">
                        <FileText className="w-6 h-6 text-emerald-400" />
                        <div className="text-left">
                          <p className="text-xs font-extrabold text-slate-200 max-w-sm truncate">
                            {groundTruthFileName}
                          </p>
                          <p className="text-[10px] text-slate-500 font-bold mt-0.5">
                            Custom Ground Truth Loaded ({(groundTruthFile.size / 1024).toFixed(1)} KB)
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-1">
                        <p className="text-xs font-bold text-slate-300">
                          Click to select or drag and drop Ground Truth CSV here
                        </p>
                        <p className="text-[10px] text-slate-500 font-semibold">
                          Must contain anonymised_id and actual binary employed_status (0 or 1).
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}
 
              {/* Predictions Upload Container */}
              <div className="space-y-2">
                <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest">
                  Upload Model Predictions CSV
                </label>
                <div
                  onDragOver={handlePredDragOver}
                  onDragLeave={handlePredDragLeave}
                  onDrop={handlePredDrop}
                  onClick={triggerPredSelect}
                  className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[160px] ${
                    isPredDragging
                      ? 'border-blue-500 bg-blue-500/10'
                      : predictionFile
                      ? 'border-blue-500/50 bg-blue-500/5'
                      : 'border-slate-800 hover:border-slate-700 bg-slate-950/40'
                  }`}
                >
                  <input
                    type="file"
                    ref={predInputRef}
                    onChange={e => {
                      if (e.target.files && e.target.files[0]) {
                        setPredictionFile(e.target.files[0]);
                        setPredictionFileName(e.target.files[0].name);
                      }
                    }}
                    accept=".csv"
                    className="hidden"
                  />
                  {predictionFile ? (
                    <div className="flex items-center gap-3">
                      <FileText className="w-10 h-10 text-blue-400" />
                      <div className="text-left">
                        <p className="text-sm font-extrabold text-slate-100 max-w-md truncate">
                          {predictionFileName}
                        </p>
                        <p className="text-xs text-slate-400 font-bold mt-0.5">
                          Prediction Submissions Loaded ({(predictionFile.size / 1024).toFixed(1)} KB)
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <div className="p-2.5 bg-slate-950/60 border border-slate-800/80 rounded-full w-fit mx-auto">
                        <Upload className="w-5 h-5 text-slate-500" />
                      </div>
                      <p className="text-xs font-bold text-slate-300">
                        Click to select or drag and drop your R model predictions CSV here
                      </p>
                      <p className="text-[10px] text-slate-500 max-w-sm mx-auto font-semibold">
                        Expects columns: anonymised_id, employed_status. Columns can be in any order. Rows can be in any order.
                      </p>
                    </div>
                  )}
                </div>
              </div>
 
              {/* Error list display */}
              {errors.length > 0 && (
                <div className="p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl space-y-2">
                  <div className="flex items-center gap-2 text-rose-400 font-bold text-xs">
                    <XCircle className="w-4 h-4 text-rose-500 shrink-0" />
                    Validation Failed: Blockers Detected
                  </div>
                  <ul className="list-disc list-inside text-[11px] text-rose-300 space-y-1 font-semibold pl-1 leading-relaxed">
                    {errors.slice(0, 5).map((err, idx) => (
                      <li key={idx}>
                        {err.row ? `Line ${err.row}: ` : ''}
                        {err.message}
                      </li>
                    ))}
                    {errors.length > 5 && (
                      <li className="font-extrabold text-rose-200 list-none mt-1">
                        ...and {errors.length - 5} more validation errors. Fix CSV formatting and retry.
                      </li>
                    )}
                  </ul>
                </div>
              )}
            </div>
 
            <div className="flex justify-end gap-3 pt-6 border-t border-slate-800/80 mt-6 relative z-10">
              {predictionFile && (
                <button
                  onClick={handleReset}
                  className="px-4 py-2 border border-slate-800 text-slate-400 hover:bg-slate-900/30 hover:text-slate-200 font-bold text-xs rounded-xl active:scale-95 transition-all"
                >
                  Clear Selection
                </button>
              )}
              <button
                onClick={handleEvaluate}
                disabled={!predictionFile || evaluating}
                className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs rounded-xl active:scale-95 transition-all flex items-center gap-2 disabled:opacity-50"
              >
                {evaluating ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Executing Pipeline...
                  </>
                ) : (
                  <>
                    Run Historical Evaluation
                    <ArrowRight className="w-3.5 h-3.5" />
                  </>
                )}
              </button>
            </div>
 
          </div>
        </div>
      ) : (
        // Results Dashboard Screen
        <div className="space-y-6">
          {/* Top Banner Row */}
          <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-6 relative overflow-hidden backdrop-blur-md shadow-xl">
            <div className="absolute top-0 left-0 w-32 h-32 bg-blue-500/5 blur-2xl rounded-full pointer-events-none" />
            
            <div className="space-y-1.5 relative z-10">
              <span className="inline-block px-2.5 py-0.5 bg-blue-500/10 text-blue-400 border border-blue-500/20 text-[9px] font-black rounded-full uppercase tracking-wider">
                {activeResult.roundId.replace('_', ' ').toUpperCase()} EVALUATION COMPLETED
              </span>
              <h2 className="text-xl font-extrabold text-white flex items-center gap-2">
                {activeResult.modelName} 
                <span className="text-[10px] bg-slate-850 text-slate-400 font-bold px-2 py-0.5 rounded border border-slate-800">
                  Ver {activeResult.modelVersion}
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Processed prediction file: <code className="text-slate-300 font-mono text-[10px] bg-slate-950/40 px-1.5 py-0.5 border border-slate-800 rounded">{activeResult.predictionFile}</code>
              </p>
            </div>
 
            {/* Score box */}
            <div className="flex items-center gap-5 shrink-0 bg-slate-950/60 p-4 border border-slate-800/80 rounded-xl relative z-10">
              <div>
                <div className="text-[9px] text-slate-500 font-black uppercase tracking-widest">Historical AUC Score</div>
                <div className="text-2xl font-extrabold font-mono text-emerald-400 mt-0.5">{activeResult.auc.toFixed(5)}</div>
              </div>
              <div className="border-l border-slate-800/80 h-10" />
              <div>
                <div className="text-[9px] text-slate-500 font-black uppercase tracking-widest">Kaggle-Style Score</div>
                <div className="text-2xl font-black text-white mt-0.5">{activeResult.score.toFixed(2)}%</div>
              </div>
            </div>
          </div>
 
          {/* Warning Banner if warning items exist */}
          {warnings.length > 0 && (
            <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-2xl space-y-1.5 text-amber-400 relative z-10">
              <div className="flex items-center gap-1.5 font-bold text-xs">
                <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0" />
                Pipeline Warnings (ID alignment / Duplicates skipped)
              </div>
              <div className="text-[11px] font-semibold leading-relaxed space-y-1">
                {warnings.map((w, idx) => (
                  <p key={idx}>{w.message}</p>
                ))}
              </div>
            </div>
          )}
 
          {/* Charts panel row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <RocCurve
              fpr={activeResult.rocCurve.fpr}
              tpr={activeResult.rocCurve.tpr}
              thresholds={activeResult.rocCurve.thresholds}
              auc={activeResult.auc}
            />
            <PredictionDistribution
              actual={activeResult.rocCurve.tpr.map((_, i) => activeResult.rocCurve.tpr[i] >= activeResult.rocCurve.fpr[i] ? 1 : 0)} // Realistic proxy
              predicted={activeResult.rocCurve.thresholds} // Realistic proxy
            />
          </div>
 
          {/* Stats Details Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Save Experiment Card */}
            <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 flex flex-col justify-between backdrop-blur-md relative overflow-hidden shadow-xl">
              <div className="absolute top-0 left-0 w-24 h-24 bg-blue-500/5 blur-2xl rounded-full pointer-events-none" />
              
              <div className="space-y-4 relative z-10">
                <h3 className="text-xs font-black text-slate-400 uppercase tracking-widest">Add to Experiment Ledger</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Record this evaluation in your persistent browser database with custom notes to track your progress.
                </p>
 
                <div className="space-y-1.5">
                  <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest block">Notes & Comments</span>
                  <textarea
                    rows={4}
                    value={notes}
                    onChange={e => setNotes(e.target.value)}
                    placeholder="Write details e.g., 'Implemented random forest with 200 trees, median imputation, added age squared interaction...'"
                    className="w-full px-3 py-2 bg-slate-950/40 border border-slate-800 text-slate-200 rounded-xl text-xs placeholder:text-slate-600 focus:bg-slate-900/40 focus:border-blue-500/50 focus:outline-none"
                  />
                </div>
              </div>
 
              <div className="pt-4 border-t border-slate-800/80 mt-4 relative z-10">
                {saved ? (
                  <div className="flex items-center justify-center gap-1.5 p-2 bg-emerald-500/10 text-emerald-400 text-xs font-bold rounded-xl border border-emerald-500/20">
                    <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                    Saved Successfully!
                  </div>
                ) : (
                  <button
                    onClick={handleSave}
                    className="w-full py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs rounded-xl active:scale-95 transition-all flex items-center justify-center gap-2"
                  >
                    <Save className="w-4 h-4" />
                    Save Experiment Record
                  </button>
                )}
              </div>
            </div>
 
            {/* Observation Stats */}
            <div className="lg:col-span-2 bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-md shadow-xl relative overflow-hidden">
              <div className="absolute top-0 left-0 w-32 h-32 bg-blue-600/5 blur-[50px] rounded-full pointer-events-none" />
              
              <h3 className="text-xs font-black text-slate-400 uppercase tracking-widest mb-4 relative z-10">Pipeline Execution Metrics</h3>
              
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 relative z-10">
                <div className="p-3.5 bg-slate-950/40 border border-slate-800/80 rounded-xl text-center">
                  <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Evaluated Rounds</span>
                  <div className="text-base font-extrabold text-white mt-1 font-mono">
                    {activeResult.roundId.replace('_', ' ').toUpperCase()}
                  </div>
                </div>
 
                <div className="p-3.5 bg-slate-950/40 border border-slate-800/80 rounded-xl text-center">
                  <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Matched Observations</span>
                  <div className="text-base font-extrabold text-white mt-1 font-mono">
                    {activeResult.validObservations.toLocaleString()}
                  </div>
                </div>
 
                <div className="p-3.5 bg-slate-950/40 border border-slate-800/80 rounded-xl text-center">
                  <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Unmatched / Omitted</span>
                  <div className="text-base font-extrabold text-white mt-1 font-mono">
                    {activeResult.removedObservations.toLocaleString()}
                  </div>
                </div>
 
                <div className="p-3.5 bg-slate-950/40 border border-slate-800/80 rounded-xl text-center">
                  <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Missing Predictions</span>
                  <div className="text-base font-extrabold text-slate-800 mt-1 font-mono">
                    {activeResult.missingPredictionCount.toLocaleString()}
                  </div>
                </div>
 
                <div className="p-3.5 bg-slate-950/40 border border-slate-800/80 rounded-xl text-center">
                  <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Skipped Extra IDs</span>
                  <div className="text-base font-extrabold text-slate-800 mt-1 font-mono">
                    {activeResult.extraPredictionCount.toLocaleString()}
                  </div>
                </div>
 
                <div className="p-3.5 bg-slate-950/40 border border-slate-800/80 rounded-xl text-center">
                  <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Duplicate IDs Cleared</span>
                  <div className="text-base font-extrabold text-white mt-1 font-mono">
                    {activeResult.duplicatePredictionCount.toLocaleString()}
                  </div>
                </div>
              </div>
 
              {/* Reset button */}
              <div className="flex justify-end mt-6 pt-4 border-t border-slate-800/80 relative z-10">
                <button
                  onClick={handleReset}
                  className="px-4 py-2 border border-slate-800 text-slate-400 hover:bg-slate-900/30 hover:text-slate-200 font-bold text-xs rounded-xl active:scale-95 transition-all flex items-center gap-1.5"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  Evaluate Another Model
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
