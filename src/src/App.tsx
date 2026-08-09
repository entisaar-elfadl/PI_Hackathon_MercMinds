import React, { useState, useEffect } from 'react';
import { RoundId, EvaluationResult, Experiment } from './types';
import { ExperimentStorageService } from './services/experimentStorage';
import { SAMPLE_MODELS } from './data/sampleModels';
import { HISTORICAL_GROUND_TRUTH, HISTORICAL_ROUNDS } from './data/historicalDatasets';
import { matchPredictionsWithGroundTruth } from './utils/idMatcher';
import { calculateRocAuc, generateRocPoints } from './utils/aucCalculator';
import { Navbar } from './components/layout/Navbar';
import { DisclaimerBanner } from './components/layout/DisclaimerBanner';
import { RCodeModal } from './components/rgenerator/RCodeModal';
import { DashboardPage } from './pages/DashboardPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { ResultsPage } from './pages/ResultsPage';
import { ComparisonPage } from './pages/ComparisonPage';
import { LeaderboardPage } from './pages/LeaderboardPage';
import { DatasetExplorerPage } from './pages/DatasetExplorerPage';
import { runEvaluationEngineTests } from './tests/evaluationEngine.test';

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'evaluate' | 'results' | 'comparison' | 'leaderboard' | 'datasets'>('dashboard');
  const [selectedRound, setSelectedRound] = useState<RoundId>('round_6');
  const [currentResult, setCurrentResult] = useState<EvaluationResult | null>(null);
  const [isRModalOpen, setIsRModalOpen] = useState(false);

  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationResult[]>([]);

  // Load experiments on mount or initialize with benchmark sample models if empty
  const loadData = () => {
    let loadedExps = ExperimentStorageService.getExperiments();
    let loadedEvals = ExperimentStorageService.getEvaluations();

    // If empty, pre-populate benchmark models across historical rounds for instant test drive
    if (loadedExps.length === 0) {
      for (const sampleModel of SAMPLE_MODELS) {
        for (const roundId of ['round_6', 'round_7', 'round_8'] as RoundId[]) {
          const gt = HISTORICAL_GROUND_TRUTH[roundId];
          const preds = sampleModel.predictionsByRound[roundId] || [];
          const matched = matchPredictionsWithGroundTruth(preds, gt);
          const aucRes = calculateRocAuc(matched.matchedPairs);
          const rocPts = generateRocPoints(matched.matchedPairs);
          const roundInfo = HISTORICAL_ROUNDS[roundId];

          const evalResult: EvaluationResult = {
            id: `eval_${sampleModel.id}_${roundId}`,
            modelName: sampleModel.name,
            modelVersion: sampleModel.version,
            roundId,
            timestamp: new Date().toISOString(),
            filename: `sample_${sampleModel.id}_${roundId}.csv`,
            auc: aucRes.auc,
            scorePercentage: aucRes.scorePercentage,
            matchedPairsCount: matched.matchedCount,
            totalGroundTruthCount: gt.length,
            positivesCount: aucRes.positivesCount,
            negativesCount: aucRes.negativesCount,
            rocPoints: rocPts,
            matchedPairsSample: matched.matchedPairs,
            validationSummary: {
              isValid: true,
              totalRowsParsed: preds.length,
              validPredictionCount: preds.length,
              invalidPredictionCount: 0,
              errors: [],
              warnings: [],
              hasRequiredColumns: true,
              idMatchStats: {
                totalGroundTruth: gt.length,
                matchedCount: matched.matchedCount,
                missingInPredictions: 0,
                extraInPredictions: 0,
                duplicateInPredictions: 0,
              },
            },
            notes: sampleModel.description,
            trainingSetDescription: roundInfo.trainingSet,
            evalSetDescription: roundInfo.evaluationSet,
          };

          ExperimentStorageService.saveEvaluationResult(evalResult);
        }
      }

      loadedExps = ExperimentStorageService.getExperiments();
      loadedEvals = ExperimentStorageService.getEvaluations();
    }

    setExperiments(loadedExps);
    setEvaluations(loadedEvals);
  };

  useEffect(() => {
    loadData();

    // Run statistical engine tests in console / verification
    const testResult = runEvaluationEngineTests();
    console.log(`[EmpEval Test Suite] ${testResult.passed}/${testResult.total} tests passed.`);
    testResult.logs.forEach((log) => console.log(log));
  }, []);

  const handleEvaluationComplete = (result: EvaluationResult) => {
    setCurrentResult(result);
    loadData();
    setActiveTab('results');
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans selection:bg-blue-600 selection:text-white flex flex-col">
      {/* Disclaimer Top Banner */}
      <DisclaimerBanner />

      {/* Main Header / Navigation */}
      <Navbar
        activeTab={activeTab === 'results' ? 'evaluate' : activeTab}
        setActiveTab={(tab) => {
          setActiveTab(tab);
        }}
        onOpenRGenerator={() => setIsRModalOpen(true)}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'dashboard' && (
          <DashboardPage
            experiments={experiments}
            evaluations={evaluations}
            onNavigate={(tab) => setActiveTab(tab)}
            onSelectRoundForEval={(r) => {
              setSelectedRound(r);
              setActiveTab('evaluate');
            }}
            onOpenRGenerator={() => setIsRModalOpen(true)}
          />
        )}

        {activeTab === 'evaluate' && (
          <EvaluationPage
            selectedRound={selectedRound}
            setSelectedRound={setSelectedRound}
            onEvaluationComplete={handleEvaluationComplete}
          />
        )}

        {activeTab === 'results' && currentResult && (
          <ResultsPage
            result={currentResult}
            onBack={() => setActiveTab('evaluate')}
            onNavigateToMatrix={() => setActiveTab('comparison')}
          />
        )}

        {activeTab === 'comparison' && (
          <ComparisonPage
            experiments={experiments}
            onNavigateToEvaluate={(r) => {
              setSelectedRound(r);
              setActiveTab('evaluate');
            }}
          />
        )}

        {activeTab === 'leaderboard' && (
          <LeaderboardPage experiments={experiments} onRefresh={loadData} />
        )}

        {activeTab === 'datasets' && <DatasetExplorerPage />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-4 text-slate-500 text-xs text-center">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
            <span className="font-semibold text-slate-800">EmpEval</span>
            <span>&bull; Local Kaggle-Style Employment Prediction Evaluator</span>
          </div>
          <div className="font-mono text-[11px] text-slate-400">
            Simulated Rounds 6–8 Validation vs Hidden Competition Round 9
          </div>
        </div>
      </footer>

      {/* R Code Modal */}
      <RCodeModal isOpen={isRModalOpen} onClose={() => setIsRModalOpen(false)} />
    </div>
  );
}
