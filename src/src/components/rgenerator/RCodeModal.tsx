import React, { useState } from 'react';
import { X, Code2, Copy, Check, Download, Info } from 'lucide-react';
import { RoundId } from '../../types';
import { generateRScript } from '../../utils/rCodeGenerator';

interface RCodeModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const RCodeModal: React.FC<RCodeModalProps> = ({ isOpen, onClose }) => {
  const [modelType, setModelType] = useState<'logistic' | 'randomForest' | 'xgboost' | 'glmnet'>('logistic');
  const [roundId, setRoundId] = useState<RoundId>('round_6');
  const [imputeMissing, setImputeMissing] = useState(true);
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const scriptText = generateRScript({
    modelType,
    roundId,
    imputeMissing,
    outputFilename: `submission_${modelType}_${roundId}.csv`,
  });

  const handleCopy = () => {
    navigator.clipboard.writeText(scriptText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([scriptText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `train_and_predict_${modelType}_${roundId}.R`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-3xl w-full p-6 shadow-2xl space-y-5 my-8">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
              <Code2 className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">R Script Generator</h2>
              <p className="text-xs text-slate-400">
                Generate clean, leakage-free R code to train models and export submission CSVs
              </p>
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Configurations */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 bg-slate-950 p-4 rounded-xl border border-slate-800 text-xs">
          <div>
            <label className="block text-slate-400 font-medium mb-1.5">Model Architecture</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value as any)}
              className="w-full bg-slate-900 text-slate-200 p-2 rounded-lg border border-slate-800 focus:outline-none focus:border-cyan-500 font-medium"
            >
              <option value="logistic">Logistic Regression (glm)</option>
              <option value="xgboost">XGBoost (xgb.train)</option>
              <option value="randomForest">Random Forest (randomForest)</option>
              <option value="glmnet">Regularized Glmnet (Lasso/ElasticNet)</option>
            </select>
          </div>

          <div>
            <label className="block text-slate-400 font-medium mb-1.5">Historical Target Round</label>
            <select
              value={roundId}
              onChange={(e) => setRoundId(e.target.value as any)}
              className="w-full bg-slate-900 text-slate-200 p-2 rounded-lg border border-slate-800 focus:outline-none focus:border-cyan-500 font-medium"
            >
              <option value="round_6">Round 6 (Train on Rounds 1–5)</option>
              <option value="round_7">Round 7 (Train on Rounds 1–6)</option>
              <option value="round_8">Round 8 (Train on Rounds 1–7)</option>
            </select>
          </div>

          <div className="flex items-center space-x-2 pt-5">
            <input
              type="checkbox"
              id="impute"
              checked={imputeMissing}
              onChange={(e) => setImputeMissing(e.target.checked)}
              className="rounded bg-slate-900 border-slate-800 text-cyan-500 focus:ring-cyan-500"
            />
            <label htmlFor="impute" className="text-slate-300 font-medium cursor-pointer">
              Train-Set Median Imputation
            </label>
          </div>
        </div>

        {/* Warning callout */}
        <div className="flex items-start space-x-2 bg-emerald-950/30 border border-emerald-500/20 p-3 rounded-xl text-xs text-emerald-200">
          <Info className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
          <span>
            <strong>Data Leakage Protection:</strong> Imputation statistics (e.g. median) are calculated strictly on training data (`train_df`) and subsequently applied to evaluation data (`eval_df`).
          </span>
        </div>

        {/* Code View */}
        <div className="relative">
          <pre className="bg-slate-950 border border-slate-800 rounded-xl p-4 text-xs font-mono text-emerald-400/90 overflow-x-auto max-h-72 leading-relaxed">
            <code>{scriptText}</code>
          </pre>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between pt-2">
          <button
            onClick={handleCopy}
            className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-white font-semibold text-xs border border-slate-700 transition-all"
          >
            {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
            <span>{copied ? 'Copied to Clipboard!' : 'Copy Code'}</span>
          </button>

          <button
            onClick={handleDownload}
            className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs shadow-lg shadow-cyan-500/20 transition-all"
          >
            <Download className="w-4 h-4" />
            <span>Download .R Script</span>
          </button>
        </div>
      </div>
    </div>
  );
};
