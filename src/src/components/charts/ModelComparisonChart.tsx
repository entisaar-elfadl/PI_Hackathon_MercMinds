import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';
import { Experiment } from '../../types';

interface ModelComparisonChartProps {
  experiments: Experiment[];
}

export const ModelComparisonChart: React.FC<ModelComparisonChartProps> = ({ experiments }) => {
  const chartData = experiments.map((exp) => ({
    modelName: `${exp.modelName} (v${exp.modelVersion})`,
    r6: exp.resultsByRound.round_6?.auc || 0,
    r7: exp.resultsByRound.round_7?.auc || 0,
    r8: exp.resultsByRound.round_8?.auc || 0,
    avgAuc: exp.avgAuc || 0,
  }));

  return (
    <div className="bg-white border border-slate-200 rounded p-5 shadow-2xs font-sans">
      <div className="pb-3 border-b border-slate-200 flex items-center justify-between">
        <div>
          <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Multi-Round Historical AUC Matrix</h3>
          <p className="text-xs text-slate-500">Comparison of model ROC AUC scores across historical survey validation rounds</p>
        </div>
      </div>

      <div className="h-80 w-full mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="modelName"
              stroke="#64748b"
              tick={{ fontSize: 11 }}
              interval={0}
              angle={-15}
              textAnchor="end"
            />
            <YAxis
              domain={[0.45, 0.75]}
              stroke="#64748b"
              tick={{ fontSize: 11 }}
              tickFormatter={(v) => v.toFixed(2)}
              label={{ value: 'ROC AUC', angle: -90, position: 'insideLeft', offset: 15, fill: '#475569', fontSize: 11 }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white border border-slate-300 p-3 rounded shadow-md text-xs font-mono text-slate-800">
                      <p className="font-bold text-blue-600 mb-1">{data.modelName}</p>
                      <p className="text-blue-700">Round 6 AUC: <span className="font-bold">{data.r6 ? data.r6.toFixed(5) : 'N/A'}</span></p>
                      <p className="text-indigo-700">Round 7 AUC: <span className="font-bold">{data.r7 ? data.r7.toFixed(5) : 'N/A'}</span></p>
                      <p className="text-amber-700">Round 8 AUC: <span className="font-bold">{data.r8 ? data.r8.toFixed(5) : 'N/A'}</span></p>
                      <p className="text-emerald-700 border-t border-slate-200 pt-1 mt-1 font-bold">
                        Average AUC: {(data.avgAuc).toFixed(5)} ({ (data.avgAuc * 100).toFixed(2) }%)
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px', color: '#475569' }} />
            <ReferenceLine y={0.5} stroke="#94a3b8" strokeDasharray="3 3" label={{ value: 'Random 0.5', fill: '#64748b', fontSize: 10 }} />
            <Bar dataKey="r6" name="Round 6 (Train 1–5)" fill="#2563eb" radius={[2, 2, 0, 0]} />
            <Bar dataKey="r7" name="Round 7 (Train 1–6)" fill="#4f46e5" radius={[2, 2, 0, 0]} />
            <Bar dataKey="r8" name="Round 8 (Train 1–7)" fill="#d97706" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
