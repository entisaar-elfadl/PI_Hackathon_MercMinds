import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { MatchedPair } from '../../types';

interface ProbabilityDistChartProps {
  matchedPairs: MatchedPair[];
}

export const ProbabilityDistChart: React.FC<ProbabilityDistChartProps> = ({ matchedPairs }) => {
  // Create histogram bins [0-0.1, 0.1-0.2, ..., 0.9-1.0]
  const bins = Array.from({ length: 10 }, (_, i) => {
    const min = i / 10;
    const max = (i + 1) / 10;
    return {
      binLabel: `${min.toFixed(1)}-${max.toFixed(1)}`,
      unemployedCount: 0,
      employedCount: 0,
    };
  });

  for (const pair of matchedPairs) {
    if (typeof pair.predicted !== 'number' || isNaN(pair.predicted)) continue;
    let idx = Math.floor(pair.predicted * 10);
    if (idx >= 10) idx = 9;
    if (idx < 0) idx = 0;

    if (pair.actual === 1) {
      bins[idx].employedCount++;
    } else {
      bins[idx].unemployedCount++;
    }
  }

  return (
    <div className="bg-white border border-slate-200 rounded p-5 shadow-2xs font-sans">
      <div className="pb-3 border-b border-slate-200">
        <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Prediction Probability Distribution</h3>
        <p className="text-xs text-slate-500">Distribution of predicted probabilities grouped by actual outcome (Separation test)</p>
      </div>

      <div className="h-72 w-full mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins} margin={{ top: 10, right: 20, left: -10, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="binLabel"
              stroke="#64748b"
              tick={{ fontSize: 11 }}
              label={{ value: 'Predicted Probability Range', position: 'insideBottom', offset: -12, fill: '#475569', fontSize: 11 }}
            />
            <YAxis
              stroke="#64748b"
              tick={{ fontSize: 11 }}
              label={{ value: 'Count of Participants', angle: -90, position: 'insideLeft', offset: 15, fill: '#475569', fontSize: 11 }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white border border-slate-300 p-3 rounded shadow-md text-xs font-mono text-slate-800">
                      <p className="font-bold text-blue-600 mb-1">Probability Bin: {data.binLabel}</p>
                      <p className="text-rose-700">Actual Unemployed (0): <span className="font-bold">{data.unemployedCount}</span></p>
                      <p className="text-emerald-700">Actual Employed (1): <span className="font-bold">{data.employedCount}</span></p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px', color: '#475569' }} />
            <Bar dataKey="unemployedCount" name="Actual Unemployed (0)" fill="#e11d48" opacity={0.85} radius={[2, 2, 0, 0]} />
            <Bar dataKey="employedCount" name="Actual Employed (1)" fill="#059669" opacity={0.85} radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
