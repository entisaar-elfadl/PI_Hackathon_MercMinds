import React, { useState } from 'react';
import { Search, Filter, CheckCircle2, XCircle, FileSpreadsheet } from 'lucide-react';
import { MatchedPair } from '../../types';

interface IdAuditTableProps {
  matchedPairs: MatchedPair[];
}

export const IdAuditTable: React.FC<IdAuditTableProps> = ({ matchedPairs }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'employed' | 'unemployed' | 'discrepancies'>('all');
  const [page, setPage] = useState(1);
  const pageSize = 15;

  const filtered = matchedPairs.filter((pair) => {
    const matchesSearch = pair.anonymised_id.toLowerCase().includes(searchTerm.toLowerCase());

    if (!matchesSearch) return false;

    if (filterType === 'employed') return pair.actual === 1;
    if (filterType === 'unemployed') return pair.actual === 0;
    if (filterType === 'discrepancies') {
      // Large difference between actual status and prediction
      return (pair.actual === 1 && pair.predicted < 0.4) || (pair.actual === 0 && pair.predicted > 0.6);
    }
    return true;
  });

  const totalPages = Math.ceil(filtered.length / pageSize) || 1;
  const paginated = filtered.slice((page - 1) * pageSize, page * pageSize);

  return (
    <div className="bg-white border border-slate-200 rounded p-5 space-y-4 shadow-2xs font-sans">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-200 gap-3">
        <div>
          <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2">
            <FileSpreadsheet className="w-4 h-4 text-blue-600" />
            <span>ID Matching &amp; Outcome Inspection Table</span>
          </h3>
          <p className="text-xs text-slate-500">
            Audit individual participant predictions (`anonymised_id`) matched against actual ground truth (`employed_status`)
          </p>
        </div>

        {/* Search & Filter Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-400" />
            <input
              type="text"
              placeholder="Search ID..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setPage(1);
              }}
              className="bg-slate-50 text-xs text-slate-900 pl-8 pr-3 py-1.5 rounded border border-slate-300 focus:outline-none focus:border-blue-600 font-mono"
            />
          </div>

          <select
            value={filterType}
            onChange={(e) => {
              setFilterType(e.target.value as any);
              setPage(1);
            }}
            className="bg-slate-50 text-xs text-slate-800 px-3 py-1.5 rounded border border-slate-300 focus:outline-none focus:border-blue-600 font-medium"
          >
            <option value="all">All Pairs ({matchedPairs.length})</option>
            <option value="employed">Actual Employed (1)</option>
            <option value="unemployed">Actual Unemployed (0)</option>
            <option value="discrepancies">High Discrepancies</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded border border-slate-200">
        <table className="w-full text-left text-xs font-mono">
          <thead className="bg-slate-50 text-slate-500 uppercase text-[10px] tracking-wider italic font-bold border-b border-slate-200">
            <tr>
              <th className="px-4 py-2.5">anonymised_id</th>
              <th className="px-4 py-2.5">Actual Outcome (`employed_status`)</th>
              <th className="px-4 py-2.5">Predicted Probability</th>
              <th className="px-4 py-2.5">Absolute Error (|y - p|)</th>
              <th className="px-4 py-2.5 text-right">Alignment Quality</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white">
            {paginated.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-slate-500 italic">
                  No observations matched search/filter criteria.
                </td>
              </tr>
            ) : (
              paginated.map((row) => {
                const diff = row.diff ?? Math.abs(row.actual - row.predicted);
                const isGoodRank = (row.actual === 1 && row.predicted >= 0.5) || (row.actual === 0 && row.predicted < 0.5);

                return (
                  <tr key={row.anonymised_id} className="hover:bg-blue-50/50 transition-colors">
                    <td className="px-4 py-2 text-slate-900 font-bold">{row.anonymised_id}</td>
                    <td className="px-4 py-2">
                      {row.actual === 1 ? (
                        <span className="px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 font-bold text-[10px]">
                          1 (Employed)
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded bg-rose-50 text-rose-700 border border-rose-200 font-bold text-[10px]">
                          0 (Unemployed)
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-2 text-slate-900 font-bold">{row.predicted.toFixed(4)}</td>
                    <td className="px-4 py-2 text-slate-500">{diff.toFixed(4)}</td>
                    <td className="px-4 py-2 text-right">
                      {isGoodRank ? (
                        <span className="inline-flex items-center space-x-1 text-emerald-700 text-[11px] font-bold">
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                          <span>Well-Ranked</span>
                        </span>
                      ) : (
                        <span className="inline-flex items-center space-x-1 text-amber-700 text-[11px] font-bold">
                          <XCircle className="w-3.5 h-3.5 text-amber-600" />
                          <span>Misaligned</span>
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div className="flex items-center justify-between pt-2 text-xs text-slate-500">
        <span>
          Showing {filtered.length === 0 ? 0 : (page - 1) * pageSize + 1} to{' '}
          {Math.min(page * pageSize, filtered.length)} of {filtered.length} matched observations
        </span>

        <div className="flex items-center space-x-2 font-mono">
          <button
            disabled={page <= 1}
            onClick={() => setPage((p) => p - 1)}
            className="px-3 py-1 rounded bg-white border border-slate-300 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-50 text-slate-700 font-bold"
          >
            Prev
          </button>
          <span className="font-semibold text-slate-700">
            Page {page} of {totalPages}
          </span>
          <button
            disabled={page >= totalPages}
            onClick={() => setPage((p) => p + 1)}
            className="px-3 py-1 rounded bg-white border border-slate-300 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-50 text-slate-700 font-bold"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
};
