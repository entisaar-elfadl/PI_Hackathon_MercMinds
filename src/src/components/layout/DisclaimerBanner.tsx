import React from 'react';
import { ShieldAlert, Info } from 'lucide-react';

export const DisclaimerBanner: React.FC = () => {
  return (
    <div className="bg-amber-50 border-b border-amber-200 px-4 py-2 text-amber-900 text-xs font-sans">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
        <div className="flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4 text-amber-700 shrink-0" />
          <span>
            <strong className="font-bold text-amber-950 uppercase tracking-tight text-[11px]">
              Historical Evaluation ≠ Official Kaggle Score:
            </strong>{' '}
            Evaluations use known ground-truth outcomes from survey Rounds 6, 7 &amp; 8 to estimate generalisation. Official competition Round 9 test outcomes remain strictly hidden.
          </span>
        </div>
        <div className="flex items-center space-x-1 text-amber-800 text-[11px] shrink-0 font-mono">
          <Info className="w-3.5 h-3.5 text-amber-700" />
          <span>Metric: ROC AUC [0.0 - 1.0]</span>
        </div>
      </div>
    </div>
  );
};
