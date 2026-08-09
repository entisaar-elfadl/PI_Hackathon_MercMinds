import React from 'react';
import { LayoutDashboard, PlayCircle, BarChart3, Trophy, Database, Code2, Award, Sparkles } from 'lucide-react';

interface NavbarProps {
  activeTab: 'dashboard' | 'evaluate' | 'comparison' | 'leaderboard' | 'datasets';
  setActiveTab: (tab: 'dashboard' | 'evaluate' | 'comparison' | 'leaderboard' | 'datasets') => void;
  onOpenRGenerator: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, onOpenRGenerator }) => {
  return (
    <header className="sticky top-0 z-40 bg-white/95 backdrop-blur-md border-b border-slate-200 text-slate-900 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          {/* Brand Logo & Title */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('dashboard')}>
            <div className="bg-blue-600 text-white p-2 rounded shadow-xs flex items-center justify-center font-bold">
              <Award className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-base tracking-tight uppercase text-slate-900">
                  Employment <span className="text-blue-600">Predictor</span>
                </span>
                <span className="px-2 py-0.5 bg-slate-100 text-slate-600 border border-slate-200 text-[10px] font-mono font-bold rounded">
                  v2.4.0-LOCAL
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-mono">Local Kaggle Validation Engine</p>
            </div>
          </div>

          {/* Desktop Nav Items */}
          <nav className="hidden md:flex items-center space-x-1">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded text-xs font-semibold transition-all ${
                activeTab === 'dashboard'
                  ? 'bg-blue-50 text-blue-700 border border-blue-200 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              <LayoutDashboard className="w-3.5 h-3.5" />
              <span>Dashboard</span>
            </button>

            <button
              onClick={() => setActiveTab('evaluate')}
              className={`flex items-center space-x-2 px-3.5 py-1.5 rounded text-xs font-bold transition-all ${
                activeTab === 'evaluate'
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border border-emerald-200'
              }`}
            >
              <PlayCircle className="w-3.5 h-3.5" />
              <span>Evaluate Model</span>
            </button>

            <button
              onClick={() => setActiveTab('comparison')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded text-xs font-semibold transition-all ${
                activeTab === 'comparison'
                  ? 'bg-blue-50 text-blue-700 border border-blue-200 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              <span>Multi-Round Matrix</span>
            </button>

            <button
              onClick={() => setActiveTab('leaderboard')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded text-xs font-semibold transition-all ${
                activeTab === 'leaderboard'
                  ? 'bg-blue-50 text-blue-700 border border-blue-200 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              <Trophy className="w-3.5 h-3.5 text-amber-600" />
              <span>Leaderboard</span>
            </button>

            <button
              onClick={() => setActiveTab('datasets')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded text-xs font-semibold transition-all ${
                activeTab === 'datasets'
                  ? 'bg-blue-50 text-blue-700 border border-blue-200 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              <span>Rounds &amp; Data</span>
            </button>
          </nav>

          {/* Action Button & Status Indicator */}
          <div className="flex items-center space-x-3">
            <div className="hidden lg:flex items-center space-x-2 px-2.5 py-1 bg-emerald-50 border border-emerald-200 rounded text-emerald-700 text-[11px] font-mono">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="font-semibold uppercase tracking-wider">Engine: Ready</span>
            </div>

            <button
              onClick={onOpenRGenerator}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded text-xs font-bold bg-slate-900 hover:bg-slate-800 text-white transition-all cursor-pointer shadow-xs"
              title="Generate R Script for Model Training and Prediction CSV Export"
            >
              <Code2 className="w-3.5 h-3.5 text-cyan-400" />
              <span>R Script Generator</span>
            </button>
          </div>
        </div>

        {/* Mobile Tab Navigation */}
        <div className="md:hidden flex items-center justify-between py-2 border-t border-slate-200 overflow-x-auto space-x-2 text-xs">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`px-3 py-1.5 rounded flex items-center space-x-1 shrink-0 ${
              activeTab === 'dashboard' ? 'bg-blue-50 text-blue-700 font-semibold' : 'text-slate-600'
            }`}
          >
            <LayoutDashboard className="w-3.5 h-3.5" />
            <span>Dashboard</span>
          </button>
          <button
            onClick={() => setActiveTab('evaluate')}
            className={`px-3 py-1.5 rounded flex items-center space-x-1 shrink-0 ${
              activeTab === 'evaluate' ? 'bg-blue-600 text-white font-bold' : 'text-blue-700 bg-blue-50'
            }`}
          >
            <PlayCircle className="w-3.5 h-3.5" />
            <span>Evaluate</span>
          </button>
          <button
            onClick={() => setActiveTab('comparison')}
            className={`px-3 py-1.5 rounded flex items-center space-x-1 shrink-0 ${
              activeTab === 'comparison' ? 'bg-blue-50 text-blue-700 font-semibold' : 'text-slate-600'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Matrix</span>
          </button>
          <button
            onClick={() => setActiveTab('leaderboard')}
            className={`px-3 py-1.5 rounded flex items-center space-x-1 shrink-0 ${
              activeTab === 'leaderboard' ? 'bg-blue-50 text-blue-700 font-semibold' : 'text-slate-600'
            }`}
          >
            <Trophy className="w-3.5 h-3.5 text-amber-600" />
            <span>Leaderboard</span>
          </button>
          <button
            onClick={() => setActiveTab('datasets')}
            className={`px-3 py-1.5 rounded flex items-center space-x-1 shrink-0 ${
              activeTab === 'datasets' ? 'bg-blue-50 text-blue-700 font-semibold' : 'text-slate-600'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>Rounds</span>
          </button>
        </div>
      </div>
    </header>
  );
};
