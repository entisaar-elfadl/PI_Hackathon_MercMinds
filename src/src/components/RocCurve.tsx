/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';

interface RocCurveProps {
  fpr: number[];
  tpr: number[];
  thresholds: number[];
  auc: number;
}

export function RocCurve({ fpr, tpr, thresholds, auc }: RocCurveProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Layout parameters for the chart inside the 500x500 box
  const padding = 55;
  const size = 450;
  const plotSize = size - 2 * padding;

  // Convert (fpr, tpr) coordinates to SVG pixels
  const getSvgCoords = (f: number, t: number) => {
    const x = padding + f * plotSize;
    const y = size - padding - t * plotSize; // Invert Y
    return { x, y };
  };

  // Build the path d-attribute
  let pathD = '';
  const points: { x: number; y: number; fpr: number; tpr: number; threshold: number }[] = [];

  for (let i = 0; i < fpr.length; i++) {
    const { x, y } = getSvgCoords(fpr[i], tpr[i]);
    points.push({ x, y, fpr: fpr[i], tpr: tpr[i], threshold: thresholds[i] });
    if (i === 0) {
      pathD += `M ${x} ${y}`;
    } else {
      pathD += ` L ${x} ${y}`;
    }
  }

  // Handle mouse moves on SVG to find closest point
  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement, MouseEvent>) => {
    if (!svgRef.current || points.length === 0) return;

    const rect = svgRef.current.getBoundingClientRect();
    const mouseX = ((e.clientX - rect.left) / rect.width) * size;
    const mouseY = ((e.clientY - rect.top) / rect.height) * size;

    // Find the closest point in Euclidean distance
    let minDistance = Infinity;
    let closestIdx = 0;

    points.forEach((pt, index) => {
      const dist = Math.sqrt(Math.pow(pt.x - mouseX, 2) + Math.pow(pt.y - mouseY, 2));
      if (dist < minDistance) {
        minDistance = dist;
        closestIdx = index;
      }
    });

    // If mouse is within a reasonable distance, show tooltip
    if (minDistance < 60) {
      setHoverIndex(closestIdx);
      // Calculate tooltip position relative to client page
      setTooltipPos({
        x: e.clientX - rect.left + 15,
        y: e.clientY - rect.top - 15,
      });
    } else {
      setHoverIndex(null);
      setTooltipPos(null);
    }
  };

  const handleMouseLeave = () => {
    setHoverIndex(null);
    setTooltipPos(null);
  };

  return (
    <div id="roc_curve_container" className="relative p-6 bg-slate-900/40 border border-slate-800/80 rounded-2xl backdrop-blur-md shadow-xl transition-all duration-300 hover:shadow-2xl">
      <div className="absolute top-0 left-0 w-24 h-24 bg-blue-500/5 blur-2xl rounded-full pointer-events-none" />
      
      <div className="flex justify-between items-center mb-4 relative z-10">
        <div>
          <h3 className="text-sm font-extrabold text-white uppercase tracking-wider">Receiver Operating Characteristic (ROC)</h3>
          <p className="text-xs text-slate-400 mt-0.5">True Positive vs. False Positive Trade-off</p>
        </div>
        <div className="px-3 py-1 bg-blue-500/10 border border-blue-500/20 rounded-full text-center">
          <span className="text-xs text-blue-400 font-bold mr-1">AUC:</span>
          <span className="text-sm font-extrabold text-emerald-400">{auc.toFixed(5)}</span>
        </div>
      </div>
 
      <div className="relative flex justify-center z-10">
        <svg
          id="roc_svg"
          ref={svgRef}
          viewBox={`0 0 ${size} ${size}`}
          className="w-full max-w-[340px] md:max-w-[380px] h-auto overflow-visible select-none cursor-crosshair"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          {/* Grid lines */}
          {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((val) => {
            const hLine = getSvgCoords(0, val);
            const vLine = getSvgCoords(val, 0);
            return (
              <g key={`grid-${val}`} className="opacity-40">
                {/* Horizontal Grid */}
                <line
                  x1={padding}
                  y1={hLine.y}
                  x2={size - padding}
                  y2={hLine.y}
                  stroke="#334155"
                  strokeWidth="1"
                  strokeDasharray="4 4"
                />
                {/* Vertical Grid */}
                <line
                  x1={vLine.x}
                  y1={padding}
                  x2={vLine.x}
                  y2={size - padding}
                  stroke="#334155"
                  strokeWidth="1"
                  strokeDasharray="4 4"
                />
                {/* Horizontal values labels */}
                <text
                  x={vLine.x}
                  y={size - padding + 20}
                  className="text-[10px] font-bold fill-slate-500 text-center"
                  textAnchor="middle"
                >
                  {val.toFixed(1)}
                </text>
                {/* Vertical values labels */}
                <text
                  x={padding - 10}
                  y={hLine.y + 3}
                  className="text-[10px] font-bold fill-slate-500 text-right"
                  textAnchor="end"
                >
                  {val.toFixed(1)}
                </text>
              </g>
            );
          })}
 
          {/* Core Axes */}
          <line
            x1={padding}
            y1={size - padding}
            x2={size - padding}
            y2={size - padding}
            stroke="#475569"
            strokeWidth="1.5"
          />
          <line
            x1={padding}
            y1={padding}
            x2={padding}
            y2={size - padding}
            stroke="#475569"
            strokeWidth="1.5"
          />
 
          {/* Axis Titles */}
          <text
            x={size / 2}
            y={size - 10}
            className="text-xs font-bold fill-slate-400"
            textAnchor="middle"
          >
            False Positive Rate (FPR)
          </text>
          <text
            x={15}
            y={size / 2}
            className="text-xs font-bold fill-slate-400"
            textAnchor="middle"
            transform={`rotate(-90 15 ${size / 2})`}
          >
            True Positive Rate (TPR)
          </text>
 
          {/* 50% Random Guess Line */}
          <line
            x1={padding}
            y1={size - padding}
            x2={size - padding}
            y2={padding}
            stroke="#334155"
            strokeWidth="2"
            strokeDasharray="5 5"
          />
          
          {/* ROC Curve Path */}
          <path
            d={pathD}
            fill="none"
            stroke="#3b82f6"
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
 
          {/* Active Hover Point highlight */}
          {hoverIndex !== null && points[hoverIndex] && (
            <g>
              {/* Vertical line to axis */}
              <line
                x1={points[hoverIndex].x}
                y1={points[hoverIndex].y}
                x2={points[hoverIndex].x}
                y2={size - padding}
                stroke="#3b82f6"
                strokeWidth="1"
                strokeDasharray="2 2"
              />
              {/* Horizontal line to axis */}
              <line
                x1={padding}
                y1={points[hoverIndex].y}
                x2={points[hoverIndex].x}
                y2={points[hoverIndex].y}
                stroke="#3b82f6"
                strokeWidth="1"
                strokeDasharray="2 2"
              />
              {/* Ring indicator */}
              <circle
                cx={points[hoverIndex].x}
                cy={points[hoverIndex].y}
                r="7"
                fill="#3b82f6"
                className="opacity-30"
              />
              <circle
                cx={points[hoverIndex].x}
                cy={points[hoverIndex].y}
                r="5"
                fill="#3b82f6"
                stroke="#020617"
                strokeWidth="2"
              />
            </g>
          )}
        </svg>
 
        {/* Floating HTML Tooltip */}
        {hoverIndex !== null && points[hoverIndex] && tooltipPos && (
          <div
            id="roc_tooltip"
            className="absolute z-20 pointer-events-none p-3 bg-slate-950/90 text-white text-xs rounded-xl shadow-2xl flex flex-col gap-1 min-w-[140px] border border-slate-800 backdrop-blur-md transition-all duration-75"
            style={{
              left: `${tooltipPos.x}px`,
              top: `${tooltipPos.y}px`,
            }}
          >
            <div className="font-extrabold border-b border-slate-800 pb-1 mb-1 text-blue-400">
              Cutoff Threshold: {points[hoverIndex].threshold.toFixed(4)}
            </div>
            <div className="flex justify-between gap-4 font-semibold">
              <span className="text-slate-400">True Positive (TPR):</span>
              <span className="font-extrabold text-emerald-400">{(points[hoverIndex].tpr * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between gap-4 font-semibold">
              <span className="text-slate-400">False Positive (FPR):</span>
              <span className="font-extrabold text-rose-400">{(points[hoverIndex].fpr * 100).toFixed(1)}%</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
