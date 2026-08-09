/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect } from 'react';
import { 
  BarChart3, BrainCircuit, Activity, ClipboardList, FileSpreadsheet, 
  ShieldCheck, HelpCircle, GraduationCap, Award, CheckCircle, Flame, Sparkles,
  ArrowRight
} from 'lucide-react';
import { Experiment, EvaluationResult, ModelComparison, RoundId } from './types';
import { calculateStdDev } from './utils/auc';

// Modular components
import { EvaluationSuite } from './components/EvaluationSuite';
import { Leaderboard } from './components/Leaderboard';
import { DatasetDownloader } from './components/DatasetDownloader';
import { ModelComparisonTable } from './components/ModelComparisonTable';
import { ExperimentTracker } from './components/ExperimentTracker';

// Seed default benchmark experiments for a professional first-time load
const BENCHMARK_EXPERIMENTS: Experiment[] = [
  {
    id: 'bench-1-r6',
    modelName: 'XGBoost Robust Classifier',
    modelVersion: '2.1.0',
    roundId: 'round_6',
    auc: 0.76124,
    score: 76.12,
    totalObservations: 1200,
    validObservations: 1200,
    predictionFile: 'submission_xgb_2.1_r6.csv',
    timestamp: '2026-08-08T14:30:00Z',
    notes: 'Tuned with hyperparameter search in R. Included interaction terms between work readiness and education level.',
  },
  {
    id: 'bench-1-r7',
    modelName: 'XGBoost Robust Classifier',
    modelVersion: '2.1.0',
    roundId: 'round_7',
    auc: 0.75892,
    score: 75.89,
    totalObservations: 1300,
    validObservations: 1300,
    predictionFile: 'submission_xgb_2.1_r7.csv',
    timestamp: '2026-08-08T14:45:00Z',
    notes: 'Retrained on Rounds 1-6. Maintains high consistency across rounds.',
  },
  {
    id: 'bench-1-r8',
    modelName: 'XGBoost Robust Classifier',
    modelVersion: '2.1.0',
    roundId: 'round_8',
    auc: 0.76412,
    score: 76.41,
    totalObservations: 1400,
    validObservations: 1400,
    predictionFile: 'submission_xgb_2.1_r8.csv',
    timestamp: '2026-08-08T15:00:00Z',
    notes: 'Retrained on Rounds 1-7. Excellent generalisation score.',
  },
  {
    id: 'bench-2-r6',
    modelName: 'Logistic Baseline (R GLM)',
    modelVersion: '1.0.0',
    roundId: 'round_6',
    auc: 0.61234,
    score: 61.23,
    totalObservations: 1200,
    validObservations: 1200,
    predictionFile: 'glm_logistic_baseline_r6.csv',
    timestamp: '2026-08-07T09:12:00Z',
    notes: 'Simple R glm(employed_status ~ age + education_level, family=binomial) baseline. Lacks interaction terms.',
  },
  {
    id: 'bench-2-r7',
    modelName: 'Logistic Baseline (R GLM)',
    modelVersion: '1.0.0',
    roundId: 'round_7',
    auc: 0.60871,
    score: 60.87,
    totalObservations: 1300,
    validObservations: 1300,
    predictionFile: 'glm_logistic_baseline_r7.csv',
    timestamp: '2026-08-07T09:30:00Z',
    notes: 'Glm trained on rounds 1-6. Score remains consistent but moderately weak.',
  },
  {
    id: 'bench-2-r8',
    modelName: 'Logistic Baseline (R GLM)',
    modelVersion: '1.0.0',
    roundId: 'round_8',
    auc: 0.61543,
    score: 61.54,
    totalObservations: 1400,
    validObservations: 1400,
    predictionFile: 'glm_logistic_baseline_r8.csv',
    timestamp: '2026-08-07T09:45:00Z',
    notes: 'Glm baseline on rounds 1-7.',
  },
  {
    id: 'bench-3-r6',
    modelName: 'Overfitted Decision Tree',
    modelVersion: '3.4.1',
    roundId: 'round_6',
    auc: 0.72451,
    score: 72.45,
    totalObservations: 1200,
    validObservations: 1200,
    predictionFile: 'decision_tree_deep_r6.csv',
    timestamp: '2026-08-06T11:00:00Z',
    notes: 'Unpruned decision tree using rpart. Performs well on Round 6 but we expect it to crash on subsequent rounds due to lack of tree pruning.',
  },
  {
    id: 'bench-3-r7',
    modelName: 'Overfitted Decision Tree',
    modelVersion: '3.4.1',
    roundId: 'round_7',
    auc: 0.54213,
    score: 54.21,
    totalObservations: 1300,
    validObservations: 1300,
    predictionFile: 'decision_tree_deep_r7.csv',
    timestamp: '2026-08-06T11:15:00Z',
    notes: 'Performances dropped severely as expected. Demonstrates high variance (volatile performance).',
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'evaluator' | 'comparison' | 'ledger' | 'datasets'>('dashboard');
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [activeResult, setActiveResult] = useState<EvaluationResult | null>(null);

  // Load from local storage
  useEffect(() => {
    const stored = localStorage.getItem('kaggle_local_experiments');
    if (stored) {
      setExperiments(JSON.parse(stored));
    } else {
      // Seed default benchmarks
      localStorage.setItem('kaggle_local_experiments', JSON.stringify(BENCHMARK_EXPERIMENTS));
      setExperiments(BENCHMARK_EXPERIMENTS);
    }
  }, []);

  // Save experiment to ledger
  const handleSaveExperiment = (res: EvaluationResult) => {
    const newExp: Experiment = {
      id: `exp-${Date.now()}`,
      modelName: res.modelName,
      modelVersion: res.modelVersion,
      roundId: res.roundId,
      auc: res.auc,
      score: res.score,
      totalObservations: res.totalObservations,
      validObservations: res.validObservations,
      predictionFile: res.predictionFile,
      timestamp: res.timestamp,
      notes: res.notes,
    };

    const updated = [newExp, ...experiments];
    setExperiments(updated);
    localStorage.setItem('kaggle_local_experiments', JSON.stringify(updated));
  };

  // Delete experiment
  const handleDeleteExperiment = (id: string) => {
    const updated = experiments.filter(e => e.id !== id);
    setExperiments(updated);
    localStorage.setItem('kaggle_local_experiments', JSON.stringify(updated));
  };

  // Clear all
  const handleClearAll = () => {
    setExperiments([]);
    localStorage.setItem('kaggle_local_experiments', JSON.stringify([]));
  };

  // View full diagnostics for a historic experiment
  const handleSelectExperiment = (id: string) => {
    const found = experiments.find(e => e.id === id);
    if (!found) return;

    // Build dummy curve coords to display since we don't save entire ROC curve in localstorage
    const steps = 20;
    const fpr: number[] = [0];
    const tpr: number[] = [0];
    const thresholds: number[] = [1];
    
    for (let i = 1; i <= steps; i++) {
      const f = i / steps;
      // Generates a nice convex curve matching the AUC score
      const t = Math.min(1.0, Math.pow(f, 1 - found.auc) * 0.95 + f * 0.05);
      fpr.push(parseFloat(f.toFixed(2)));
      tpr.push(parseFloat(t.toFixed(4)));
      thresholds.push(parseFloat((1 - f).toFixed(2)));
    }

    const dummyResult: EvaluationResult = {
      roundId: found.roundId,
      modelName: found.modelName,
      modelVersion: found.modelVersion,
      auc: found.auc,
      score: found.score,
      totalObservations: found.totalObservations,
      validObservations: found.validObservations,
      removedObservations: found.totalObservations - found.validObservations,
      missingGroundTruth: 0,
      matchedCount: found.validObservations,
      missingPredictionCount: 0,
      extraPredictionCount: 0,
      duplicatePredictionCount: 0,
      predictionFile: found.predictionFile,
      timestamp: found.timestamp,
      notes: found.notes,
      rocCurve: { fpr, tpr, thresholds },
    };

    setActiveResult(dummyResult);
    setActiveTab('evaluator');
  };

  // Process and group experiments into side-by-side comparative Models
  const getModelComparisons = (): ModelComparison[] => {
    const modelMap = new Map<string, Record<RoundId, number | null>>();
    const modelMeta = new Map<string, { name: string; version: string }>();

    experiments.forEach(e => {
      const key = `${e.modelName.toLowerCase().trim()}-v${e.modelVersion.trim()}`;
      if (!modelMap.has(key)) {
        modelMap.set(key, { round_6: null, round_7: null, round_8: null });
        modelMeta.set(key, { name: e.modelName, version: e.modelVersion });
      }
      const roundScores = modelMap.get(key)!;
      // In case of multiple runs for same round, keep the highest (or most recent)
      if (roundScores[e.roundId] === null || roundScores[e.roundId]! < e.auc) {
        roundScores[e.roundId] = e.auc;
      }
    });

    const comparisons: ModelComparison[] = [];

    modelMap.forEach((scores, key) => {
      const meta = modelMeta.get(key)!;
      const validScores: number[] = [];
      
      // Collect valid numeric scores
      if (scores.round_6 !== null) validScores.push(scores.round_6);
      if (scores.round_7 !== null) validScores.push(scores.round_7);
      if (scores.round_8 !== null) validScores.push(scores.round_8);

      const count = validScores.length;
      if (count === 0) return;

      const sum = validScores.reduce((acc, v) => acc + v, 0);
      const averageAuc = sum / count;
      const minAuc = Math.min(...validScores);
      const maxAuc = Math.max(...validScores);
      const stdDev = calculateStdDev(validScores);

      comparisons.push({
        modelName: meta.name,
        modelVersion: meta.version,
        roundScores: scores,
        averageAuc,
        minAuc,
        maxAuc,
        stdDev,
      });
    });

    return comparisons;
  };

  const comparisons = getModelComparisons();

  // Highlight stats for Overview dashboard cards
  const totalExperiments = experiments.length;
  const bestModel = comparisons.length > 0 
    ? [...comparisons].sort((a, b) => b.averageAuc - a.averageAuc)[0] 
    : null;

  return (
    <div id="app_root" className="min-h-screen bg-[#020617] flex flex-col lg:flex-row font-sans text-slate-200 selection:bg-blue-600/30 selection:text-blue-200">
      
      {/* 1. DESKTOP SIDEBAR NAVIGATION (Large screens) */}
      <aside id="desktop_sidebar" className="hidden lg:flex flex-col w-64 border-r border-slate-800/80 bg-slate-950/50 backdrop-blur-xl shrink-0 h-screen sticky top-0 z-40">
        
        {/* Sidebar Header */}
        <div className="p-6 border-b border-slate-800/80">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-blue-600 rounded-xl text-white shadow-[0_0_15px_rgba(37,99,235,0.4)] shrink-0">
              <BrainCircuit className="w-5 h-5" />
            </div>
            <div>
              <div className="flex flex-col">
                <span className="font-extrabold text-sm tracking-tight text-white leading-tight">Employment</span>
                <span className="font-bold text-xs tracking-tight text-blue-400 leading-tight">Predictor</span>
              </div>
              <p className="text-[9px] text-slate-500 mt-1 font-bold tracking-wide uppercase">by MercMinds</p>
            </div>
          </div>
        </div>

        {/* Sidebar Navigation */}
        <nav className="flex-1 p-4 space-y-6 overflow-y-auto text-xs font-medium">
          
          {/* Section: Main Menu */}
          <div className="space-y-1">
            <div className="text-[10px] uppercase text-slate-500 font-bold tracking-widest px-3 mb-2">Main Menu</div>
            {[
              { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
              { id: 'evaluator', label: 'Model Evaluator', icon: BrainCircuit },
              { id: 'comparison', label: 'Model Comparison', icon: Activity },
              { id: 'ledger', label: 'Experiment Ledger', icon: ClipboardList },
            ].map(tab => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => {
                    setActiveTab(tab.id as any);
                    if (tab.id !== 'evaluator') setActiveResult(null);
                  }}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg border transition-all text-left font-semibold cursor-pointer ${
                    isActive
                       ? 'bg-slate-800/50 border-slate-700 text-blue-400 shadow-md shadow-slate-950/40'
                      : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/30'
                  }`}
                >
                  <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-blue-400' : 'text-slate-500'}`} />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* Section: Utilities */}
          <div className="space-y-1">
            <div className="text-[10px] uppercase text-slate-500 font-bold tracking-widest px-3 mb-2">Utilities</div>
            {[
              { id: 'datasets', label: 'Survey Datasets', icon: FileSpreadsheet },
            ].map(tab => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => {
                    setActiveTab(tab.id as any);
                    if (tab.id !== 'evaluator') setActiveResult(null);
                  }}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg border transition-all text-left font-semibold cursor-pointer ${
                    isActive
                       ? 'bg-slate-800/50 border-slate-700 text-blue-400 shadow-md shadow-slate-950/40'
                      : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/30'
                  }`}
                >
                  <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-blue-400' : 'text-slate-500'}`} />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* Section: Historical Rounds Info */}
          <div className="space-y-2 pt-2 border-t border-slate-900">
            <div className="text-[10px] uppercase text-slate-500 font-bold tracking-widest px-3 mb-2">Historical Rounds</div>
            <div className="space-y-1.5 px-1">
              <div className="flex items-center justify-between px-2 py-1.5 text-[10px] bg-emerald-500/10 text-emerald-400 rounded-md border border-emerald-500/20 font-semibold">
                <span>Round 6</span>
                <span className="opacity-75 font-mono text-[9px] uppercase tracking-wider">Trained R1-5</span>
              </div>
              <div className="flex items-center justify-between px-2 py-1.5 text-[10px] bg-blue-500/10 text-blue-400 rounded-md border border-blue-500/20 font-semibold">
                <span>Round 7</span>
                <span className="opacity-75 font-mono text-[9px] uppercase tracking-wider">Trained R1-6</span>
              </div>
              <div className="flex items-center justify-between px-2 py-1.5 text-[10px] bg-amber-500/10 text-amber-400 rounded-md border border-amber-500/20 font-semibold">
                <span>Round 8</span>
                <span className="opacity-75 font-mono text-[9px] uppercase tracking-wider">Trained R1-7</span>
              </div>
            </div>
          </div>

        </nav>

      </aside>

      {/* 2. MOBILE HEADER & NAVIGATION (Small screens) */}
      <header id="mobile_header" className="flex lg:hidden flex-col sticky top-0 z-40 bg-slate-950/85 backdrop-blur-xl border-b border-slate-800/80">
        <div className="px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-blue-600 rounded-lg text-white">
              <BrainCircuit className="w-4 h-4" />
            </div>
            <div>
              <span className="font-extrabold text-xs text-white tracking-tight">Employment Predictor</span>
              <span className="text-[8px] bg-blue-500/15 text-blue-400 px-1 py-0.5 ml-1.5 rounded font-black">by MercMinds</span>
            </div>
          </div>
          
          <div className="text-right">
            <span className="text-[8px] text-slate-500 uppercase tracking-widest font-bold block">Best AUC</span>
            <span className="text-xs font-mono font-black text-emerald-400">
              {bestModel ? bestModel.averageAuc.toFixed(4) : '0.0000'}
            </span>
          </div>
        </div>

        {/* Scrollable Mobile Tabs Menu */}
        <nav className="flex items-center gap-1.5 overflow-x-auto px-4 pb-2.5 scrollbar-none">
          {[
            { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
            { id: 'evaluator', label: 'Evaluator', icon: BrainCircuit },
            { id: 'comparison', label: 'Comparison', icon: Activity },
            { id: 'ledger', label: 'Ledger', icon: ClipboardList },
            { id: 'datasets', label: 'Datasets', icon: FileSpreadsheet },
          ].map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id as any);
                  if (tab.id !== 'evaluator') setActiveResult(null);
                }}
                className={`px-3 py-1.5 rounded-lg text-[11px] font-bold flex items-center gap-1.5 whitespace-nowrap transition-all border shrink-0 cursor-pointer ${
                  isActive
                    ? 'bg-slate-800 border-slate-700 text-blue-400'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                <Icon className="w-3.5 h-3.5 shrink-0" />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </header>

      {/* 3. MAIN CONTAINER AREA */}
      <div className="flex-1 flex flex-col min-h-screen bg-[#020617] text-slate-200 overflow-x-hidden">
        
        {/* Main Content Header Area (Layout Header from Design) */}
        <header id="main_content_header" className="px-6 lg:px-8 pt-8 pb-4 flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-900/40 mb-2">
          <div>
            <h1 className="text-2xl lg:text-3xl font-black tracking-tight text-white">
              Employment Predictor
            </h1>
            <p className="text-slate-400 text-xs mt-1">Designed and developed by <span className="text-blue-400 font-semibold">MercMinds</span></p>
          </div>
        </header>

        {/* Dynamic content rendering with custom margins */}
        <main id="main_content_area" className="flex-1 px-6 lg:px-8 pb-12 space-y-6">
          
          {/* TAB 1: OVERVIEW DASHBOARD */}
          {activeTab === 'dashboard' && (
            <div className="space-y-6">
              
              {/* Quick stats metrics widgets */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                {/* Stat 1: Best Model */}
                <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-5 backdrop-blur-md relative overflow-hidden flex items-center gap-4 group hover:border-slate-700 transition-all">
                  <div className="absolute top-0 right-0 w-20 h-20 bg-amber-500/5 blur-[30px] rounded-full pointer-events-none" />
                  <div className="p-3 bg-amber-500/10 rounded-xl text-amber-400 shrink-0 border border-amber-500/20">
                    <Award className="w-5 h-5" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider block">Best Generalising Model</span>
                    <span className="text-sm font-extrabold text-white mt-1 block truncate">
                      {bestModel ? bestModel.modelName : 'None'}
                    </span>
                    <span className="text-[11px] font-bold text-slate-400 font-mono mt-0.5 block">
                      {bestModel ? `Avg AUC: ${bestModel.averageAuc.toFixed(5)} (${(bestModel.averageAuc * 100).toFixed(1)}%)` : 'No runs recorded'}
                    </span>
                  </div>
                </div>

                {/* Stat 2: Active Ledger Runs */}
                <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-5 backdrop-blur-md relative overflow-hidden flex items-center gap-4 group hover:border-slate-700 transition-all">
                  <div className="absolute top-0 right-0 w-20 h-20 bg-blue-500/5 blur-[30px] rounded-full pointer-events-none" />
                  <div className="p-3 bg-blue-500/10 rounded-xl text-blue-400 shrink-0 border border-blue-500/20">
                    <ClipboardList className="w-5 h-5" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider block">Total Tracked Runs</span>
                    <span className="text-xl font-black text-white mt-1 block">{totalExperiments}</span>
                    <span className="text-[10px] text-slate-400 block mt-0.5">Logged evaluations in browser</span>
                  </div>
                </div>

                {/* Stat 3: Rounds Coverage */}
                <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-5 backdrop-blur-md relative overflow-hidden flex items-center gap-4 group hover:border-slate-700 transition-all">
                  <div className="absolute top-0 right-0 w-20 h-20 bg-emerald-500/5 blur-[30px] rounded-full pointer-events-none" />
                  <div className="p-3 bg-emerald-50/10 rounded-xl text-emerald-400 shrink-0 border border-emerald-500/20">
                    <GraduationCap className="w-5 h-5" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider block">Rounds Evaluated</span>
                    <span className="text-xl font-black text-white mt-1 block">
                      {comparisons.reduce((acc, m) => {
                        let c = 0;
                        if (m.roundScores.round_6 !== null) c++;
                        if (m.roundScores.round_7 !== null) c++;
                        if (m.roundScores.round_8 !== null) c++;
                        return Math.max(acc, c);
                      }, 0)} / 3
                    </span>
                    <span className="text-[10px] text-slate-400 block mt-0.5">Simulated survey rounds</span>
                  </div>
                </div>

              </div>

              {/* Explanatory Banner Section with Frosted Gradient accent */}
              <div className="p-6 bg-slate-900/40 border border-slate-800 rounded-3xl text-slate-200 shadow-xl flex flex-col md:flex-row items-center justify-between gap-6 relative overflow-hidden backdrop-blur-md">
                <div className="absolute top-0 right-0 w-[400px] h-[400px] bg-blue-600/10 blur-[80px] rounded-full -mr-40 -mt-40 pointer-events-none" />
                
                <div className="space-y-2 relative z-10">
                  <div className="flex items-center gap-2">
                    <Flame className="w-4 h-4 text-amber-400 fill-amber-400 shrink-0" />
                    <span className="text-[10px] font-bold text-blue-400 uppercase tracking-widest">Historical Generalisation Strategy</span>
                  </div>
                  <h2 className="text-base lg:text-lg font-black leading-tight max-w-xl text-white">
                    Does your model generalize to a future survey round?
                  </h2>
                  <p className="text-xs text-slate-400 leading-relaxed max-w-2xl">
                    By validating your R predictions against Round 6, 7, and 8 outcomes, you mimic the real Kaggle 
                    unseen Round 9. If a model scores high on one round but fails on another, it has overfitted the survey pool covariates.
                  </p>
                </div>
                
                <button
                  onClick={() => setActiveTab('evaluator')}
                  className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs rounded-xl shadow-lg shadow-blue-900/20 hover:shadow-blue-900/40 hover:-translate-y-0.5 transition-all shrink-0 active:scale-95 flex items-center gap-2 relative z-10"
                >
                  Evaluate Model Now
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Leaderboard Table section */}
              <Leaderboard models={comparisons} />

            </div>
          )}

          {/* TAB 2: MODEL EVALUATOR */}
          {activeTab === 'evaluator' && (
            <EvaluationSuite
              onSaveExperiment={handleSaveExperiment}
              activeResult={activeResult}
              setActiveResult={setActiveResult}
            />
          )}

          {/* TAB 3: MODEL COMPARISON */}
          {activeTab === 'comparison' && (
            <ModelComparisonTable comparisons={comparisons} />
          )}

          {/* TAB 4: EXPERIMENT LEDGER */}
          {activeTab === 'ledger' && (
            <ExperimentTracker
              experiments={experiments}
              onDeleteExperiment={handleDeleteExperiment}
              onClearAll={handleClearAll}
              onSelectExperiment={handleSelectExperiment}
            />
          )}

          {/* TAB 5: SURVEY DATASETS DOWNLOADER */}
          {activeTab === 'datasets' && (
            <DatasetDownloader />
          )}

        </main>

        {/* Aesthetic Footer */}
        <footer id="main_footer" className="border-t border-slate-900 py-6 mt-auto text-center text-[11px] text-slate-500 bg-slate-950/40">
          <div className="max-w-7xl mx-auto px-4 space-y-1.5">
            <p className="font-bold text-slate-400 uppercase tracking-wider">Employment Predictor Local Evaluation System</p>
            <p>
              Designed and developed by <span className="text-blue-400 font-semibold">MercMinds</span> to ensure statistical rigor and mathematical validation.
            </p>
          </div>
        </footer>

      </div>

    </div>
  );
}
