/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

interface PredictionDistributionProps {
  actual: number[];
  predicted: number[];
}

export function PredictionDistribution({ actual, predicted }: PredictionDistributionProps) {
  // We'll divide probabilities into 10 bins (0.0 to 1.0)
  const numBins = 10;
  const binsUnemployed = Array(numBins).fill(0);
  const binsEmployed = Array(numBins).fill(0);

  // Distribute counts into bins
  for (let i = 0; i < actual.length; i++) {
    const prob = predicted[i];
    const label = actual[i];

    // Find bin index [0, 9]
    let binIdx = Math.floor(prob * numBins);
    if (binIdx >= numBins) binIdx = numBins - 1;
    if (binIdx < 0) binIdx = 0;

    if (label === 1) {
      binsEmployed[binIdx]++;
    } else {
      binsUnemployed[binIdx]++;
    }
  }

  // Find max count to scale bars
  const maxUnemployed = Math.max(...binsUnemployed, 1);
  const maxEmployed = Math.max(...binsEmployed, 1);
  const globalMax = Math.max(maxUnemployed, maxEmployed);

  // Chart size parameters
  const height = 180;
  const width = 400;
  const paddingLeft = 40;
  const paddingRight = 10;
  const paddingTop = 20;
  const paddingBottom = 30;

  const chartHeight = height - paddingTop - paddingBottom;
  const chartWidth = width - paddingLeft - paddingRight;
  const barWidth = chartWidth / numBins;

  return (
    <div id="pred_distribution_container" className="relative p-6 bg-slate-900/40 border border-slate-800/80 rounded-2xl backdrop-blur-md shadow-xl transition-all duration-300 hover:shadow-2xl overflow-hidden">
      <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500/5 blur-2xl rounded-full pointer-events-none" />
      
      <div className="relative z-10">
        <h3 className="text-sm font-extrabold text-white uppercase tracking-wider">Prediction Density & Separation</h3>
        <p className="text-xs text-slate-400 mt-0.5">Distribution of predicted probabilities grouped by actual outcome</p>
      </div>
 
      <div className="flex justify-center relative z-10 mt-2">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full max-w-[380px] h-auto overflow-visible select-none">
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
            const y = paddingTop + (1 - ratio) * chartHeight;
            const label = Math.round(ratio * globalMax);
            return (
              <g key={`grid-y-${ratio}`} className="opacity-30">
                <line
                  x1={paddingLeft}
                  y1={y}
                  x2={width - paddingRight}
                  y2={y}
                  stroke="#334155"
                  strokeWidth="1"
                />
                <text
                  x={paddingLeft - 8}
                  y={y + 3}
                  className="text-[9px] font-bold fill-slate-500 text-right"
                  textAnchor="end"
                >
                  {label}
                </text>
              </g>
            );
          })}
 
          {/* Render grouped bars for each bin */}
          {Array(numBins)
            .fill(null)
            .map((_, idx) => {
              const uCount = binsUnemployed[idx];
              const eCount = binsEmployed[idx];
 
              const uHeight = (uCount / globalMax) * chartHeight;
              const eHeight = (eCount / globalMax) * chartHeight;
 
              const xBase = paddingLeft + idx * barWidth;
              // Divide bar space into two narrow bars
              const spacing = 1.5;
              const subBarWidth = (barWidth - spacing * 3) / 2;
 
              const uX = xBase + spacing;
              const uY = height - paddingBottom - uHeight;
 
              const eX = xBase + subBarWidth + spacing * 2;
              const eY = height - paddingBottom - eHeight;
 
              return (
                <g key={`bin-${idx}`}>
                  {/* Unemployed (0) Bar */}
                  {uCount > 0 && (
                    <rect
                      x={uX}
                      y={uY}
                      width={subBarWidth}
                      height={uHeight}
                      fill="rgba(244, 63, 94, 0.2)" // rose opacity
                      stroke="#f43f5e" // rose-500
                      strokeWidth="1"
                      className="transition-all duration-300 hover:brightness-125 cursor-pointer"
                      title={`Unemployed in [${(idx/10).toFixed(1)}-${((idx+1)/10).toFixed(1)}]: ${uCount}`}
                    />
                  )}
 
                  {/* Employed (1) Bar */}
                  {eCount > 0 && (
                    <rect
                      x={eX}
                      y={eY}
                      width={subBarWidth}
                      height={eHeight}
                      fill="rgba(16, 185, 129, 0.2)" // emerald opacity
                      stroke="#10b981" // emerald-500
                      strokeWidth="1"
                      className="transition-all duration-300 hover:brightness-125 cursor-pointer"
                      title={`Employed in [${(idx/10).toFixed(1)}-${((idx+1)/10).toFixed(1)}]: ${eCount}`}
                    />
                  )}
 
                  {/* X axis bin label */}
                  <text
                    x={xBase + barWidth / 2}
                    y={height - paddingBottom + 14}
                    className="text-[8px] font-bold fill-slate-500"
                    textAnchor="middle"
                  >
                    {`${idx / 10}`}
                  </text>
                </g>
              );
            })}
 
          {/* Bottom horizontal axis line */}
          <line
            x1={paddingLeft}
            y1={height - paddingBottom}
            x2={width - paddingRight}
            y2={height - paddingBottom}
            stroke="#475569"
            strokeWidth="1.5"
          />
 
          {/* Axis Labels */}
          <text
            x={paddingLeft + chartWidth / 2}
            y={height - 4}
            className="text-[9px] font-bold fill-slate-400"
            textAnchor="middle"
          >
            Predicted Probability Bin
          </text>
        </svg>
      </div>
 
      {/* Legend */}
      <div className="flex justify-center gap-6 mt-3 border-t border-slate-800/80 pt-3 relative z-10">
        <div className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 bg-rose-500/20 border border-rose-500/60 rounded-xs" />
          <span className="text-xs font-bold text-slate-400">Unemployed (0)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 bg-emerald-500/20 border border-emerald-500/60 rounded-xs" />
          <span className="text-xs font-bold text-slate-400">Employed (1)</span>
        </div>
      </div>
    </div>
  );
}
