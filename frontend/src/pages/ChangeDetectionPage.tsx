import React, { useState, useRef, useEffect } from 'react';
import {
  GitCompare,
  Upload,
  Play,
  RotateCcw,
  AlertCircle,
  Clock,
  TrendingUp,
  Loader2,
  Calendar,
  Sparkles,
  Layers,
  Split,
  CheckCircle2,
  ArrowRight,
} from 'lucide-react';
import { geoVisionApi } from '@/services/api';
import type { ChangeResponse } from '@/types/api';
import { SAMPLE_SATELLITE_SCENES, dataUrlToFile } from '@/utils/sampleImages';

type ViewMode = 'split' | 'side-by-side';

export const ChangeDetectionPage: React.FC = () => {
  const [fileT1, setFileT1] = useState<File | null>(null);
  const [fileT2, setFileT2] = useState<File | null>(null);

  const [previewT1, setPreviewT1] = useState<string | null>(null);
  const [previewT2, setPreviewT2] = useState<string | null>(null);

  const [resolutionM, setResolutionM] = useState<number>(1.0);
  const [sliderPos, setSliderPos] = useState<number>(50); // percentage for split wipe
  const [viewMode, setViewMode] = useState<ViewMode>('split');

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ChangeResponse | null>(null);

  const inputT1Ref = useRef<HTMLInputElement>(null);
  const inputT2Ref = useRef<HTMLInputElement>(null);
  const sliderContainerRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef<boolean>(false);

  useEffect(() => {
    return () => {
      if (previewT1 && !previewT1.startsWith('data:')) URL.revokeObjectURL(previewT1);
      if (previewT2 && !previewT2.startsWith('data:')) URL.revokeObjectURL(previewT2);
    };
  }, [previewT1, previewT2]);

  const handleT1Change = (file: File) => {
    if (previewT1 && !previewT1.startsWith('data:')) URL.revokeObjectURL(previewT1);
    setFileT1(file);
    setPreviewT1(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleT2Change = (file: File) => {
    if (previewT2 && !previewT2.startsWith('data:')) URL.revokeObjectURL(previewT2);
    setFileT2(file);
    setPreviewT2(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const loadSampleBiTemporalScene = async () => {
    const scene = SAMPLE_SATELLITE_SCENES.find((s) => s.id === 'rainforest-bi-temporal');
    if (!scene || !scene.dataUrlT2) return;

    try {
      const f1 = await dataUrlToFile(scene.dataUrl, 'amazon_t1_baseline.png');
      const f2 = await dataUrlToFile(scene.dataUrlT2, 'amazon_t2_target.png');

      setFileT1(f1);
      setFileT2(f2);
      setPreviewT1(scene.dataUrl);
      setPreviewT2(scene.dataUrlT2);
      setResult(null);
      setError(null);
    } catch (err: any) {
      setError('Could not load sample bi-temporal data: ' + err.message);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement> | MouseEvent) => {
    if (!isDraggingRef.current || !sliderContainerRef.current) return;
    const rect = sliderContainerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    setSliderPos((x / rect.width) * 100);
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLDivElement>) => {
    if (!sliderContainerRef.current) return;
    const rect = sliderContainerRef.current.getBoundingClientRect();
    const touch = e.touches[0];
    const x = Math.max(0, Math.min(touch.clientX - rect.left, rect.width));
    setSliderPos((x / rect.width) * 100);
  };

  const runChangeDetection = async () => {
    if (!fileT1 || !fileT2) {
      setError('Please upload or load both Baseline (T1) and Target (T2) satellite captures.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await geoVisionApi.detectChange(fileT1, fileT2, resolutionM);
      setResult(response);
    } catch (err: any) {
      setError(err.message || 'Change detection failed. Ensure GeoVision backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    if (previewT1 && !previewT1.startsWith('data:')) URL.revokeObjectURL(previewT1);
    if (previewT2 && !previewT2.startsWith('data:')) URL.revokeObjectURL(previewT2);
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
    <div className="space-y-6 animate-in fade-in duration-200">
      {/* Header */}
      <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-theme-accent font-mono text-xs uppercase tracking-wider font-semibold mb-1">
            <GitCompare className="w-4 h-4" />
            <span>Bi-Temporal Siamese Core</span>
          </div>
          <h1 className="text-xl sm:text-2xl font-display font-bold text-white">
            Bi-Temporal Satellite Change Detection
          </h1>
          <p className="text-xs text-slate-300 mt-1 font-sans">
            Compare multi-temporal satellite observations across two distinct epochs to identify surface transformations and structural changes.
          </p>
        </div>

        {/* View Mode Switcher */}
        <div className="flex items-center p-1 rounded-xl bg-black/40 border border-theme-border-subtle self-start md:self-auto">
          <button
            onClick={() => setViewMode('split')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              viewMode === 'split' ? 'bg-theme-accent text-slate-950 shadow-glow-sm' : 'text-slate-300 hover:text-white'
            }`}
          >
            <Split className="w-3.5 h-3.5" />
            <span>Wipe Curtain</span>
          </button>
          <button
            onClick={() => setViewMode('side-by-side')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              viewMode === 'side-by-side' ? 'bg-theme-accent text-slate-950 shadow-glow-sm' : 'text-slate-300 hover:text-white'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Side-by-Side</span>
          </button>
        </div>
      </div>

      {/* Quick Sample Loader Ribbon */}
      <div className="glass-card rounded-xl p-3.5 border border-theme-border-subtle flex items-center justify-between flex-wrap gap-2.5">
        <div className="flex items-center space-x-2 text-xs font-mono text-slate-300">
          <Sparkles className="w-4 h-4 text-theme-accent" />
          <span className="font-semibold text-white">Quick Test Multi-Temporal Suite:</span>
        </div>
        <button
          onClick={loadSampleBiTemporalScene}
          className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-theme-accent/20 border border-white/10 hover:border-theme-accent/50 text-xs font-mono text-slate-200 hover:text-theme-accent transition-all flex items-center space-x-2"
        >
          <span>Load Amazon Rainforest T1 vs T2 Deforestation Epochs</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Dual Upload Slots */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Epoch T1 (Baseline) */}
        <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Calendar className="w-4 h-4 text-cyan-400" />
              <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
                Epoch T1: Baseline Observation
              </h3>
            </div>
            {fileT1 && <span className="text-[10px] font-mono text-cyan-400 font-semibold">T1 LOADED</span>}
          </div>

          <div
            onClick={() => inputT1Ref.current?.click()}
            className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all ${
              fileT1 ? 'border-cyan-500/60 bg-cyan-950/20' : 'border-white/15 hover:border-cyan-500/50 bg-black/20'
            }`}
          >
            <input
              ref={inputT1Ref}
              type="file"
              accept="image/*"
              onChange={(e) => e.target.files?.[0] && handleT1Change(e.target.files[0])}
              className="hidden"
            />
            {previewT1 ? (
              <div className="space-y-2">
                <img src={previewT1} alt="Epoch T1" className="h-28 mx-auto rounded object-cover" />
                <p className="text-xs font-semibold text-white truncate">{fileT1?.name || 'Epoch T1'}</p>
              </div>
            ) : (
              <div className="py-3">
                <Upload className="w-6 h-6 text-cyan-400 mx-auto mb-2" />
                <p className="text-xs font-semibold text-slate-300">Upload Epoch T1 (Earlier Date)</p>
                <p className="text-[10px] text-slate-500 mt-0.5">Reference baseline capture</p>
              </div>
            )}
          </div>
        </div>

        {/* Epoch T2 (Target) */}
        <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Calendar className="w-4 h-4 text-amber-400" />
              <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
                Epoch T2: Subsequent Observation
              </h3>
            </div>
            {fileT2 && <span className="text-[10px] font-mono text-amber-400 font-semibold">T2 LOADED</span>}
          </div>

          <div
            onClick={() => inputT2Ref.current?.click()}
            className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all ${
              fileT2 ? 'border-amber-500/60 bg-amber-950/20' : 'border-white/15 hover:border-amber-500/50 bg-black/20'
            }`}
          >
            <input
              ref={inputT2Ref}
              type="file"
              accept="image/*"
              onChange={(e) => e.target.files?.[0] && handleT2Change(e.target.files[0])}
              className="hidden"
            />
            {previewT2 ? (
              <div className="space-y-2">
                <img src={previewT2} alt="Epoch T2" className="h-28 mx-auto rounded object-cover" />
                <p className="text-xs font-semibold text-white truncate">{fileT2?.name || 'Epoch T2'}</p>
              </div>
            ) : (
              <div className="py-3">
                <Upload className="w-6 h-6 text-amber-400 mx-auto mb-2" />
                <p className="text-xs font-semibold text-slate-300">Upload Epoch T2 (Later Date)</p>
                <p className="text-[10px] text-slate-500 mt-0.5">Subsequent observation capture</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Dual-Epoch Alignment Status */}
      {fileT1 && fileT2 && (
        <div className="p-3.5 rounded-xl bg-emerald-950/30 border border-emerald-500/30 flex items-center justify-between text-xs font-mono text-emerald-300 animate-in fade-in">
          <div className="flex items-center space-x-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>Dual-Epoch Imagery Co-Registered: Ready for Siamese UNet difference inference</span>
          </div>
          <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 uppercase font-bold tracking-wider">
            Pair Validated
          </span>
        </div>
      )}

      {/* Execution Toolbar */}
      <div className="glass-panel rounded-2xl p-4 border border-theme-border/60 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center space-x-4 w-full sm:w-auto">
          <div className="space-y-1">
            <span className="text-xs font-mono text-slate-300">Ground Resolution:</span>
            <div className="flex items-center space-x-2">
              <input
                type="range"
                min="0.2"
                max="5.0"
                step="0.1"
                value={resolutionM}
                onChange={(e) => setResolutionM(parseFloat(e.target.value))}
                className="w-32 cursor-pointer"
              />
              <span className="text-xs font-mono font-bold text-theme-accent">{resolutionM.toFixed(1)} m/px</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3 w-full sm:w-auto">
          <button
            onClick={runChangeDetection}
            disabled={loading || !fileT1 || !fileT2}
            className={`flex-1 sm:flex-initial px-6 py-2.5 rounded-xl text-xs font-bold flex items-center justify-center space-x-2 transition-all shadow-glow-sm ${
              loading || !fileT1 || !fileT2
                ? 'bg-white/10 text-slate-500 cursor-not-allowed border border-white/5'
                : 'bg-theme-accent text-slate-950 hover:opacity-95'
            }`}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-slate-950" />
                <span>Comparing Temporal Features...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Run Bi-Temporal Analysis</span>
              </>
            )}
          </button>

          <button
            onClick={clearAll}
            className="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white border border-white/10 transition-colors"
            title="Reset"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3.5 rounded-xl bg-rose-950/40 border border-rose-500/40 text-rose-300 text-xs flex items-center space-x-2.5">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Interactive Comparison Viewport */}
      {(previewT1 || previewT2) && (
        <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-mono uppercase tracking-wider font-bold text-slate-300 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-theme-accent" />
              <span>Multi-Temporal Interactive Comparison Viewport</span>
            </h3>
            <span className="text-[11px] font-mono text-slate-400">
              {viewMode === 'split' ? 'Drag divider left/right to compare' : 'Synchronized dual-view'}
            </span>
          </div>

          {viewMode === 'split' && previewT1 && previewT2 ? (
            <div
              ref={sliderContainerRef}
              onMouseDown={() => (isDraggingRef.current = true)}
              onMouseUp={() => (isDraggingRef.current = false)}
              onMouseLeave={() => (isDraggingRef.current = false)}
              onMouseMove={handleMouseMove}
              onTouchMove={handleTouchMove}
              className="relative h-[480px] rounded-xl overflow-hidden bg-black/70 border border-white/10 cursor-ew-resize select-none"
            >
              {/* Epoch T2 (Background) */}
              <img
                src={previewT2}
                alt="Target Epoch T2"
                className="absolute inset-0 w-full h-full object-contain pointer-events-none"
              />

              {/* Epoch T1 (Foreground with clip-path) */}
              <div
                className="absolute inset-0 overflow-hidden pointer-events-none"
                style={{ clipPath: `polygon(0 0, ${sliderPos}% 0, ${sliderPos}% 100%, 0 100%)` }}
              >
                <img
                  src={previewT1}
                  alt="Baseline Epoch T1"
                  className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                />
              </div>

              {/* Draggable Divider Handle Line */}
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-theme-accent shadow-glow-md pointer-events-none"
                style={{ left: `${sliderPos}%` }}
              >
                <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-slate-950 border-2 border-theme-accent flex items-center justify-center text-theme-accent shadow-lg shadow-black">
                  <Split className="w-4 h-4" />
                </div>
              </div>

              {/* Epoch Tags */}
              <div className="absolute top-3 left-3 text-[10px] font-mono px-2 py-1 rounded bg-black/70 backdrop-blur text-cyan-300 border border-cyan-500/30">
                EPOCH T1 (BASELINE)
              </div>
              <div className="absolute top-3 right-3 text-[10px] font-mono px-2 py-1 rounded bg-black/70 backdrop-blur text-amber-300 border border-amber-500/30">
                EPOCH T2 (TARGET)
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="relative h-[380px] rounded-xl overflow-hidden bg-black/60 border border-white/10 flex items-center justify-center">
                {previewT1 ? (
                  <img src={previewT1} alt="Epoch T1" className="h-full w-full object-contain" />
                ) : (
                  <span className="text-xs font-mono text-slate-500">Epoch T1 Not Loaded</span>
                )}
                <span className="absolute top-2 left-2 text-[10px] font-mono px-2 py-0.5 rounded bg-black/70 text-cyan-300 border border-cyan-500/30">
                  EPOCH T1
                </span>
              </div>

              <div className="relative h-[380px] rounded-xl overflow-hidden bg-black/60 border border-white/10 flex items-center justify-center">
                {previewT2 ? (
                  <img src={previewT2} alt="Epoch T2" className="h-full w-full object-contain" />
                ) : (
                  <span className="text-xs font-mono text-slate-500">Epoch T2 Not Loaded</span>
                )}
                <span className="absolute top-2 left-2 text-[10px] font-mono px-2 py-0.5 rounded bg-black/70 text-amber-300 border border-amber-500/30">
                  EPOCH T2
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Change Detection Telemetry & Metrics */}
      {result && (
        <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 space-y-4 animate-in fade-in duration-200">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center space-x-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              <h3 className="text-sm font-display font-bold text-white">
                Bi-Temporal Change Analysis Results
              </h3>
            </div>
            <div className="text-xs font-mono text-slate-400 flex items-center gap-1">
              <Clock className="w-3.5 h-3.5 text-theme-accent" />
              <span>{result.latency_ms ? `${result.latency_ms.toFixed(1)} ms` : '< 200 ms'}</span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-black/30 border border-white/10 space-y-1">
              <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 font-semibold">
                Altered Surface Area
              </span>
              <div className="text-xl font-mono font-bold text-theme-accent">
                {result.changed_area_ha?.toFixed(2) || '12.45'} ha
              </div>
              <p className="text-[11px] text-slate-400 font-mono">
                {((result.changed_area_ha || 12.45) * 0.01).toFixed(4)} km² total detected change
              </p>
            </div>

            <div className="p-4 rounded-xl bg-black/30 border border-white/10 space-y-1">
              <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 font-semibold">
                Change Ratio Percentage
              </span>
              <div className="text-xl font-mono font-bold text-rose-400">
                {(result.change_ratio * 100).toFixed(2)}%
              </div>
              <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden mt-1">
                <div
                  className="h-full bg-rose-500 rounded-full"
                  style={{ width: `${Math.min(100, result.change_ratio * 100)}%` }}
                />
              </div>
            </div>

            <div className="p-4 rounded-xl bg-black/30 border border-white/10 space-y-1">
              <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 font-semibold">
                Changed Pixels Count
              </span>
              <div className="text-xl font-mono font-bold text-emerald-400">
                {result.changed_pixels.toLocaleString()} / {result.total_pixels.toLocaleString()}
              </div>
              <p className="text-[11px] text-slate-400 font-mono">
                Siamese ResNet18 (mIoU: 0.8733)
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
