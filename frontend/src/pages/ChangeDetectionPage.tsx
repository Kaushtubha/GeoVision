import React, { useState, useRef, useEffect } from 'react';
import {
  GitCompare,
  Upload,
  Sliders,
  Play,
  RotateCcw,
  AlertCircle,
  Clock,
  TrendingUp,
  Loader2,
  Calendar,
} from 'lucide-react';
import { geoVisionApi } from '@/services/api';
import type { ChangeResponse } from '@/types/api';

export const ChangeDetectionPage: React.FC = () => {
  const [fileT1, setFileT1] = useState<File | null>(null);
  const [fileT2, setFileT2] = useState<File | null>(null);

  const [previewT1, setPreviewT1] = useState<string | null>(null);
  const [previewT2, setPreviewT2] = useState<string | null>(null);

  const [resolutionM, setResolutionM] = useState<number>(1.0);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ChangeResponse | null>(null);

  const inputT1Ref = useRef<HTMLInputElement>(null);
  const inputT2Ref = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (previewT1) URL.revokeObjectURL(previewT1);
      if (previewT2) URL.revokeObjectURL(previewT2);
    };
  }, [previewT1, previewT2]);

  const handleT1Change = (file: File) => {
    if (previewT1) URL.revokeObjectURL(previewT1);
    setFileT1(file);
    setPreviewT1(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleT2Change = (file: File) => {
    if (previewT2) URL.revokeObjectURL(previewT2);
    setFileT2(file);
    setPreviewT2(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const runChangeDetection = async () => {
    if (!fileT1 || !fileT2) {
      setError('Please upload both Baseline (T1) and Target (T2) satellite images.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await geoVisionApi.detectChange(fileT1, fileT2, resolutionM);
      setResult(response);
    } catch (err: any) {
      setError(err.message || 'Change detection failed. Ensure the GeoVision backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    if (previewT1) URL.revokeObjectURL(previewT1);
    if (previewT2) URL.revokeObjectURL(previewT2);
    setFileT1(null);
    setFileT2(null);
    setPreviewT1(null);
    setPreviewT2(null);
    setResult(null);
    setError(null);
    if (inputT1Ref.current) inputT1Ref.current.value = '';
    if (inputT2Ref.current) inputT2Ref.current.value = '';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="p-5 rounded-lg bg-space-900 border border-space-700/80 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-cyan-400 font-mono text-xs uppercase tracking-wider mb-1">
            <GitCompare className="w-4 h-4" />
            <span>Bi-Temporal Core</span>
          </div>
          <h1 className="text-xl font-bold text-white font-sans">
            Bi-Temporal Satellite Change Detection
          </h1>
          <p className="text-xs text-slate-400 mt-1 font-sans">
            Compare multi-temporal satellite observations across two distinct epochs to identify surface transformations and structural changes.
          </p>
        </div>
      </div>

      {/* Control Parameters Bar */}
      <div className="p-4 rounded-lg bg-space-900/80 border border-space-700/70 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3 text-xs font-mono">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <span className="text-slate-300">Ground Resolution (m/px):</span>
          <input
            type="number"
            min="0.1"
            max="30.0"
            step="0.1"
            value={resolutionM}
            onChange={(e) => setResolutionM(parseFloat(e.target.value) || 1.0)}
            className="w-20 px-2 py-1 bg-space-950 border border-space-700 rounded text-cyan-300 text-center font-bold focus:outline-none focus:border-cyan-500"
          />
          <span className="text-slate-400 text-[11px]">meters/pixel</span>
        </div>

        <div className="flex items-center space-x-3">
          {(fileT1 || fileT2) && (
            <button
              onClick={clearAll}
              className="px-3 py-1.5 rounded bg-space-800 hover:bg-space-750 text-slate-400 hover:text-slate-200 text-xs font-mono flex items-center space-x-1.5 border border-space-700"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset</span>
            </button>
          )}

          <button
            onClick={runChangeDetection}
            disabled={loading || !fileT1 || !fileT2}
            className={`px-4 py-2 rounded-md text-xs font-mono font-bold flex items-center space-x-2 transition-all shadow-md ${
              loading || !fileT1 || !fileT2
                ? 'bg-space-800 text-slate-500 border border-space-700 cursor-not-allowed'
                : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-cyan-900/30'
            }`}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-cyan-200" />
                <span>Comparing Temporal Scenes...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Execute Change Analysis</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-800/80 text-rose-300 text-xs flex items-center space-x-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
          <div>
            <p className="font-semibold font-mono">Bi-Temporal Analysis Error</p>
            <p className="text-rose-400 font-mono mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Upload & Scene Comparison Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* T1 Scene (Baseline) */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="font-semibold text-cyan-400 flex items-center space-x-1.5">
              <Calendar className="w-3.5 h-3.5" />
              <span>T1: BASELINE CAPTURE (PRE-EVENT)</span>
            </span>
            {fileT1 && <span className="text-slate-400 truncate max-w-xs">{fileT1.name}</span>}
          </div>

          <div
            onClick={() => !previewT1 && inputT1Ref.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f) handleT1Change(f);
            }}
            className={`relative rounded-lg border-2 border-dashed transition-all overflow-hidden flex flex-col items-center justify-center min-h-[300px] bg-space-900/90 ${
              previewT1
                ? 'border-space-700 p-2'
                : 'border-space-700/80 hover:border-cyan-500/50 p-6 cursor-pointer'
            }`}
          >
            <input
              ref={inputT1Ref}
              type="file"
              accept="image/*,.tif,.tiff"
              onChange={(e) => e.target.files?.[0] && handleT1Change(e.target.files[0])}
              className="hidden"
            />

            {previewT1 ? (
              <img
                src={previewT1}
                alt="Baseline T1"
                className="max-h-[360px] w-auto object-contain rounded select-none"
              />
            ) : (
              <div className="flex flex-col items-center justify-center text-center space-y-2">
                <div className="w-12 h-12 rounded-full bg-space-850 border border-space-700 flex items-center justify-center text-cyan-400">
                  <Upload className="w-5 h-5" />
                </div>
                <p className="text-xs font-semibold text-white">Upload Baseline Scene (T1)</p>
                <p className="text-[11px] text-slate-400 font-mono">GeoTIFF, PNG, JPEG</p>
              </div>
            )}
          </div>
        </div>

        {/* T2 Scene (Target) */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="font-semibold text-emerald-400 flex items-center space-x-1.5">
              <Calendar className="w-3.5 h-3.5" />
              <span>T2: TARGET CAPTURE (POST-EVENT)</span>
            </span>
            {fileT2 && <span className="text-slate-400 truncate max-w-xs">{fileT2.name}</span>}
          </div>

          <div
            onClick={() => !previewT2 && inputT2Ref.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f) handleT2Change(f);
            }}
            className={`relative rounded-lg border-2 border-dashed transition-all overflow-hidden flex flex-col items-center justify-center min-h-[300px] bg-space-900/90 ${
              previewT2
                ? 'border-space-700 p-2'
                : 'border-space-700/80 hover:border-emerald-500/50 p-6 cursor-pointer'
            }`}
          >
            <input
              ref={inputT2Ref}
              type="file"
              accept="image/*,.tif,.tiff"
              onChange={(e) => e.target.files?.[0] && handleT2Change(e.target.files[0])}
              className="hidden"
            />

            {previewT2 ? (
              <img
                src={previewT2}
                alt="Target T2"
                className="max-h-[360px] w-auto object-contain rounded select-none"
              />
            ) : (
              <div className="flex flex-col items-center justify-center text-center space-y-2">
                <div className="w-12 h-12 rounded-full bg-space-850 border border-space-700 flex items-center justify-center text-emerald-400">
                  <Upload className="w-5 h-5" />
                </div>
                <p className="text-xs font-semibold text-white">Upload Target Scene (T2)</p>
                <p className="text-[11px] text-slate-400 font-mono">GeoTIFF, PNG, JPEG</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Results Telemetry & Change Metrics */}
      {result && (
        <div className="p-6 rounded-lg bg-space-900 border border-space-700 space-y-5">
          <div className="flex items-center justify-between border-b border-space-750 pb-3">
            <div className="flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-cyan-400" />
              <h3 className="text-xs font-mono uppercase tracking-wider font-bold text-white">
                Bi-Temporal Change Telemetry
              </h3>
            </div>
            <div className="flex items-center space-x-1.5 text-xs font-mono text-slate-400">
              <Clock className="w-3.5 h-3.5 text-cyan-400" />
              <span>Inference: {result.latency_ms.toFixed(1)} ms</span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 rounded bg-space-850 border border-space-750">
              <p className="text-[10px] font-mono text-slate-400 uppercase">Change Ratio</p>
              <p className="text-2xl font-mono font-bold text-cyan-300 mt-1">
                {(result.change_ratio * 100).toFixed(2)}%
              </p>
              <p className="text-[11px] text-slate-400 font-mono mt-1">
                Altered Scene Fraction
              </p>
            </div>

            <div className="p-4 rounded bg-space-850 border border-space-750">
              <p className="text-[10px] font-mono text-slate-400 uppercase">Changed Pixels</p>
              <p className="text-2xl font-mono font-bold text-white mt-1">
                {result.changed_pixels.toLocaleString()}
              </p>
              <p className="text-[11px] text-slate-400 font-mono mt-1">
                of {result.total_pixels.toLocaleString()} total
              </p>
            </div>

            <div className="p-4 rounded bg-space-850 border border-space-750">
              <p className="text-[10px] font-mono text-slate-400 uppercase">Altered Surface Area</p>
              <p className="text-2xl font-mono font-bold text-emerald-300 mt-1">
                {result.changed_area_ha.toFixed(2)} <span className="text-xs text-slate-400">ha</span>
              </p>
              <p className="text-[11px] text-slate-400 font-mono mt-1">
                {(result.changed_area_ha / 100).toFixed(4)} km²
              </p>
            </div>

            <div className="p-4 rounded bg-space-850 border border-space-750">
              <p className="text-[10px] font-mono text-slate-400 uppercase">Analysis Resolution</p>
              <p className="text-2xl font-mono font-bold text-slate-200 mt-1">
                {result.resolution_m} <span className="text-xs text-slate-400">m/px</span>
              </p>
              <p className="text-[11px] text-slate-400 font-mono mt-1">
                Ground Sample GSD
              </p>
            </div>
          </div>

          {/* Change Severity Status Banner */}
          <div className="p-3.5 rounded bg-space-850 border border-space-750 flex items-center justify-between text-xs font-mono">
            <div className="flex items-center space-x-2">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  result.change_ratio > 0.2
                    ? 'bg-rose-400'
                    : result.change_ratio > 0.05
                    ? 'bg-amber-400'
                    : 'bg-emerald-400'
                }`}
              />
              <span className="font-semibold text-slate-200">
                {result.change_ratio > 0.2
                  ? 'Significant Land/Structural Transformation Detected'
                  : result.change_ratio > 0.05
                  ? 'Moderate Land-Cover Alteration Detected'
                  : 'Minimal / Minor Surface Variation'}
              </span>
            </div>
            <span className="text-slate-400 text-[11px]">
              Grounded Siamese Difference
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
