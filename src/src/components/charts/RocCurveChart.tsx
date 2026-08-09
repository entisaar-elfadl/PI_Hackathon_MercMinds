import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { RocPoint } from '../../types';

interface RocCurveChartProps {
  rocPoints: RocPoint[];
  auc: number;
}

export const RocCurveChart: React.FC<RocCurveChartProps> = ({ rocPoints, auc }) => {
  // Generate diagonal baseline data (0,0 to 1,1)
  const chartData = rocPoints.map((p) => ({
    fpr: p.fpr,
    tpr: p.tpr,
    threshold: p.threshold,
    diagonal: p.fpr, // reference line value
  }));

  return (
    <div className="bg-white border border-slate-200 rounded p-5 shadow-2xs font-sans">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-200 gap-2">
        <div>
          <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Receiver Operating Characteristic (ROC Curve)</h3>
          <p className="text-xs text-slate-500">Trade-off between True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity)</p>
        </div>
        <div className="bg-blue-50 border border-blue-200 px-3 py-1 rounded text-xs font-mono font-bold text-blue-700 self-start sm:self-auto">
          AUC = {auc.toFixed(5)} ({ (auc * 100).toFixed(2) }%)
        </div>
      </div>

      <div className="h-72 w-full mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 20 }}>
            <defs>
              <linearGradient id="rocGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#2563eb" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#2563eb" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="fpr"
              type="number"
              domain={[0, 1]}
              tickCount={6}
              stroke="#64748b"
              tick={{ fontSize: 11 }}
              label={{ value: 'False Positive Rate (FPR)', position: 'insideBottom', offset: -12, fill: '#475569', fontSize: 11 }}
            />
            <YAxis
              dataKey="tpr"
              type="number"
              domain={[0, 1]}
              tickCount={6}
              stroke="#64748b"
              tick={{ fontSize: 11 }}
              label={{ value: 'True Positive Rate (TPR)', angle: -90, position: 'insideLeft', offset: 15, fill: '#475569', fontSize: 11 }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload as { fpr: number; tpr: number; threshold: number };
                  return (
                    <div className="bg-white border border-slate-300 p-3 rounded shadow-md text-xs font-mono text-slate-800">
                      <p className="font-bold text-blue-600 mb-1">ROC Point Details</p>
                      <p>Probability Threshold: <span className="text-slate-900 font-bold">{data.threshold.toFixed(4)}</span></p>
                      <p>True Positive Rate (TPR): <span className="text-emerald-700 font-bold">{data.tpr.toFixed(4)}</span></p>
                      <p>False Positive Rate (FPR): <span className="text-rose-700 font-bold">{data.fpr.toFixed(4)}</span></p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: 'Random Guess (AUC=0.5)', fill: '#64748b', fontSize: 10, position: 'center' }} />
            <Area
              type="monotone"
              dataKey="tpr"
              stroke="#2563eb"
              strokeWidth={2.5}
              fillOpacity={1}
              fill="url(#rocGradient)"
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center justify-between text-xs text-slate-600 pt-3 border-t border-slate-200">
        <div className="flex items-center space-x-2">
          <span className="w-3 h-3 rounded bg-blue-600 inline-block"></span>
          <span className="font-medium">Model Curve (AUC = {auc})</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="w-3 h-[2px] bg-slate-400 inline-block border-t border-dashed border-slate-500"></span>
          <span>Diagonal Baseline (0.5 AUC)</span>
        </div>
      </div>
    </div>
  );
};
