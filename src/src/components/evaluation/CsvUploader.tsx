import React, { useRef, useState } from 'react';
import { Upload, FileText, CheckCircle2, AlertTriangle, Sparkles, Download, Layers } from 'lucide-react';
import { SAMPLE_MODELS } from '../../data/sampleModels';
import { PredictionRow, RoundId, SampleModel } from '../../types';
import { parseCsvContent } from '../../services/csvParser';

interface CsvUploaderProps {
  onDataParsed: (parsedRows: any[], fileName: string, sampleModelName?: string) => void;
  selectedRound: RoundId;
}

export const CsvUploader: React.FC<CsvUploaderProps> = ({ onDataParsed, selectedRound }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (file: File) => {
    if (!file) return;
    setLoading(true);
    setFileName(file.name);

    try {
      const result = await parseCsvContent(file);
      onDataParsed(result.data, file.name);
    } catch (err) {
      console.error('Failed to parse CSV:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleSelectSampleModel = (model: SampleModel) => {
    const predictions = model.predictionsByRound[selectedRound] || [];
    setFileName(`sample_${model.id}_${selectedRound}.csv`);
    onDataParsed(predictions, `sample_${model.id}_${selectedRound}.csv`, model.name);
  };

  return (
    <div className="space-y-4 font-sans">
      {/* Drag & Drop Box */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded p-8 text-center cursor-pointer transition-all ${
          isDragging
            ? 'border-blue-600 bg-blue-50/50'
            : fileName
            ? 'border-emerald-500/80 bg-emerald-50/30'
            : 'border-slate-300 hover:border-blue-500 bg-slate-50/50 hover:bg-white'
        }`}
      >
        <input
          type="file"
          ref={fileInputRef}
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.[0]) handleFileChange(e.target.files[0]);
          }}
        />

        <div className="flex flex-col items-center justify-center space-y-3">
          <div className="w-12 h-12 rounded bg-white border border-slate-200 flex items-center justify-center text-blue-600 shadow-2xs">
            {loading ? (
              <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
            ) : fileName ? (
              <CheckCircle2 className="w-6 h-6 text-emerald-600" />
            ) : (
              <Upload className="w-6 h-6 text-blue-600" />
            )}
          </div>

          <div>
            {fileName ? (
              <div>
                <p className="text-sm font-bold text-slate-900 flex items-center justify-center space-x-2">
                  <FileText className="w-4 h-4 text-blue-600" />
                  <span>{fileName}</span>
                </p>
                <p className="text-xs text-emerald-600 font-semibold mt-1">File loaded successfully. Click to replace file.</p>
              </div>
            ) : (
              <div>
                <p className="text-sm font-semibold text-slate-800">
                  Drop model prediction CSV here, or <span className="text-blue-600 underline font-bold">browse file</span>
                </p>
                <p className="text-xs text-slate-500 mt-1 font-mono">
                  Required columns: <code className="bg-slate-200 px-1 py-0.5 rounded text-slate-800">anonymised_id</code>,{' '}
                  <code className="bg-slate-200 px-1 py-0.5 rounded text-slate-800">employed_status</code> (probability 0 to 1)
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Benchmark Sample Model Selector */}
      <div className="bg-white border border-slate-200 rounded p-4 shadow-2xs">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-blue-600" />
            <span className="text-xs font-bold text-slate-900 uppercase tracking-wider">
              Or Select Benchmark Sample Model Predictions ({selectedRound.replace('_', ' ').toUpperCase()})
            </span>
          </div>
          <span className="text-[11px] text-slate-500 font-mono">Quick Test Engine</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
          {SAMPLE_MODELS.map((model) => (
            <button
              key={model.id}
              onClick={() => handleSelectSampleModel(model)}
              className="flex flex-col items-start p-3 rounded bg-slate-50 border border-slate-200 hover:border-blue-500 hover:bg-blue-50/40 transition-all text-left group cursor-pointer"
            >
              <div className="flex items-center justify-between w-full">
                <span className="text-xs font-bold text-slate-800 group-hover:text-blue-600 transition-colors">
                  {model.name}
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white border border-slate-200 text-slate-600">
                  v{model.version}
                </span>
              </div>
              <p className="text-[11px] text-slate-500 mt-1 line-clamp-1">{model.description}</p>
              <div className="mt-2 text-[10px] text-blue-700 font-mono font-bold">
                R6: {model.expectedAucByRound.round_6} | R7: {model.expectedAucByRound.round_7} | R8: {model.expectedAucByRound.round_8}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
