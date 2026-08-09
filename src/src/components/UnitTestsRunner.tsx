/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { runSystemTests, TestResult } from '../utils/tests';
import { CheckCircle2, XCircle, Play, ShieldCheck, Terminal } from 'lucide-react';

export function UnitTestsRunner() {
  const [tests, setTests] = useState<TestResult[]>([]);
  const [hasRun, setHasRun] = useState(false);

  const runTests = () => {
    const results = runSystemTests();
    setTests(results);
    setHasRun(true);
  };

  const passedCount = tests.filter(t => t.status === 'passed').length;
  const failedCount = tests.filter(t => t.status === 'failed').length;

  return (
    <div id="unit_tests_card" className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <ShieldCheck className="w-5 h-5 text-emerald-500" />
            <h2 className="text-lg font-bold text-slate-800 font-display">Statistical Test Runner</h2>
          </div>
          <p className="text-xs text-slate-500">
            Verify the mathematical integrity, ID matching correctness, and bounds validation of the evaluation engine on-the-fly.
          </p>
        </div>
        <button
          onClick={runTests}
          className="px-4 py-2 bg-slate-900 text-white font-semibold text-xs rounded-xl hover:bg-slate-800 active:scale-95 transition-all flex items-center gap-2 shrink-0 self-start sm:self-center"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          Run Verification Tests
        </button>
      </div>

      {!hasRun ? (
        <div className="flex flex-col items-center justify-center p-12 bg-slate-50 border border-dashed border-slate-200 rounded-xl">
          <Terminal className="w-10 h-10 text-slate-400 mb-3" />
          <h3 className="text-sm font-semibold text-slate-700">Test Engine Idle</h3>
          <p className="text-xs text-slate-400 mt-1 mb-4 text-center max-w-sm">
            Click the run button to execute the full integration testing suite and verify calculations in real-time.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Summary Banner */}
          <div
            className={`p-4 rounded-xl flex items-center justify-between border ${
              failedCount === 0
                ? 'bg-emerald-50 border-emerald-100 text-emerald-800'
                : 'bg-rose-50 border-rose-100 text-rose-800'
            }`}
          >
            <div className="flex items-center gap-2.5">
              {failedCount === 0 ? (
                <CheckCircle2 className="w-5 h-5 text-emerald-500" />
              ) : (
                <XCircle className="w-5 h-5 text-rose-500" />
              )}
              <div>
                <h4 className="text-sm font-bold">
                  {failedCount === 0 ? 'All Systems Verified' : 'Test Failures Detected'}
                </h4>
                <p className="text-xs opacity-90 mt-0.5">
                  Executed {tests.length} unit tests. {passedCount} passed, {failedCount} failed.
                </p>
              </div>
            </div>
            <span className="font-mono text-xs font-bold bg-white/60 px-2 py-1 rounded-md">
              {passedCount} / {tests.length} PASSED
            </span>
          </div>

          {/* Test cases list */}
          <div className="divide-y divide-slate-100 border border-slate-100 rounded-xl overflow-hidden">
            {tests.map((test, index) => {
              const isPassed = test.status === 'passed';
              return (
                <div
                  key={`${test.name}-${index}`}
                  className="p-4 flex flex-col md:flex-row md:items-center justify-between gap-3 bg-white hover:bg-slate-50/40 transition-colors"
                >
                  <div className="flex items-start gap-3">
                    <span className="mt-0.5 shrink-0">
                      {isPassed ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                      ) : (
                        <XCircle className="w-4 h-4 text-rose-500" />
                      )}
                    </span>
                    <div>
                      <div className="flex items-center gap-2">
                        <h4 className="text-xs font-bold text-slate-800">{test.name}</h4>
                        <span className="px-1.5 py-0.5 bg-slate-100 text-slate-500 font-medium text-[9px] uppercase tracking-wider rounded-md">
                          {test.category}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-500 mt-1 leading-relaxed">
                        {test.message}
                      </p>
                    </div>
                  </div>

                  {test.expected !== undefined && test.actual !== undefined && (
                    <div className="flex items-center gap-3 self-end md:self-center font-mono text-[10px] bg-slate-50 border border-slate-100 px-3 py-1 rounded-lg shrink-0">
                      <div>
                        <span className="text-slate-400">Expected:</span>{' '}
                        <span className="text-slate-700 font-semibold">
                          {JSON.stringify(test.expected)}
                        </span>
                      </div>
                      <div className="border-l border-slate-200 h-3.5" />
                      <div>
                        <span className="text-slate-400">Actual:</span>{' '}
                        <span
                          className={`font-semibold ${
                            isPassed ? 'text-emerald-600' : 'text-rose-600'
                          }`}
                        >
                          {JSON.stringify(test.actual)}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
