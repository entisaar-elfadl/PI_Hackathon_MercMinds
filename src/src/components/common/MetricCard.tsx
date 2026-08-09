import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  badgeText?: string;
  badgeType?: 'success' | 'warning' | 'info' | 'neutral';
  highlight?: boolean;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  badgeText,
  badgeType = 'neutral',
  highlight = false,
}) => {
  const badgeStyles = {
    success: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    warning: 'bg-amber-50 text-amber-700 border-amber-200',
    info: 'bg-blue-50 text-blue-700 border-blue-200',
    neutral: 'bg-slate-100 text-slate-600 border-slate-200',
  };

  if (highlight) {
    return (
      <div className="bg-blue-600 text-white border border-blue-700 rounded p-4 flex flex-col justify-between shadow-xs">
        <div>
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-bold text-blue-100 uppercase tracking-wider">{title}</span>
            {badgeText && (
              <span className="px-2 py-0.5 text-[10px] font-mono font-bold rounded bg-blue-500/40 text-white border border-blue-400/40">
                {badgeText}
              </span>
            )}
          </div>
          <div className="text-3xl font-light font-mono mt-2 tracking-tight">{value}</div>
        </div>
        {subtitle && <p className="mt-2 text-xs text-blue-100 leading-tight">{subtitle}</p>}
      </div>
    );
  }

  return (
    <div className="bg-white border border-slate-200 rounded p-4 flex flex-col justify-between shadow-2xs hover:border-slate-300 transition-colors">
      <div>
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">{title}</span>
          {icon && <div className="p-1.5 rounded bg-slate-50 text-slate-600 border border-slate-200">{icon}</div>}
        </div>

        <div className="mt-2 flex items-baseline justify-between gap-2">
          <span className="text-2xl sm:text-3xl font-light font-mono text-slate-900 tracking-tight">{value}</span>
          {badgeText && (
            <span className={`px-2 py-0.5 text-[10px] font-mono font-bold rounded border ${badgeStyles[badgeType]}`}>
              {badgeText}
            </span>
          )}
        </div>
      </div>

      {subtitle && <p className="mt-2 text-xs text-slate-500">{subtitle}</p>}
    </div>
  );
};
