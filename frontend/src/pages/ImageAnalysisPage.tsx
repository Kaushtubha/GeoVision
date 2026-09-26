import React, { useState, useRef, useEffect } from 'react';
import {
  ScanLine,
  Layers,
  Upload,
  Sliders,
  Play,
  RotateCcw,
  AlertCircle,
  Clock,
  Eye,
  ZoomIn,
  ZoomOut,
  Sparkles,
  Grid,
  CheckCircle2,
  Loader2,
} from 'lucide-react';
import { geoVisionApi } from '@/services/api';
import type { DetectionResponse, SegmentationResponse, BBoxItem } from '@/types/api';
import { SAMPLE_SATELLITE_SCENES, dataUrlToFile } from '@/utils/sampleImages';

type AnalysisMode = 'detection' | 'segmentation';

const CLASS_COLORS = [
  '#10b981', // emerald
  '#06b6d4', // cyan
  '#f59e0b', // amber
  '#ec4899', // pink
  '#8b5cf6', // violet
  '#3b82f6', // blue
  '#ef4444', // red
  '#14b8a6', // teal
  '#f97316', // orange
  '#a855f7', // purple
];

const LANDCOVER_COLORS: Record<string, string> = {
  urban: '#ef4444',
  building: '#ef4444',
  built: '#ef4444',
  vegetation: '#10b981',
  woodland: '#10b981',
  forest: '#059669',
  tree: '#059669',
  water: '#0ea5e9',
  river: '#0ea5e9',
  road: '#f59e0b',
  agriculture: '#eab308',
  crop: '#eab308',
  barren: '#a8a29e',
  soil: '#a8a29e',
  background: '#64748b',
};

function getLandcoverColor(name: string, index: number): string {
  const lower = name.toLowerCase();
  for (const [key, color] of Object.entries(LANDCOVER_COLORS)) {
    if (lower.includes(key)) return color;
  }
  return CLASS_COLORS[index % CLASS_COLORS.length];
}

export const ImageAnalysisPage: React.FC = () => {
  const [mode, setMode] = useState<AnalysisMode>('detection');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);

  // Parameters
  const [confThresh, setConfThresh] = useState<number>(0.25);
  const [resolutionM, setResolutionM] = useState<number>(1.0);
  const [maskOpacity, setMaskOpacity] = useState<number>(0.65);

  // Canvas Inspector Tools
  const [zoomLevel, setZoomLevel] = useState<number>(1.0);
  const [showGrid, setShowGrid] = useState<boolean>(false);
  const [showFalseColor, setShowFalseColor] = useState<boolean>(false);
  const [selectedClassFilter, setSelectedClassFilter] = useState<string>('all');

  // States
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Results
  const [detectionResult, setDetectionResult] = useState<DetectionResponse | null>(null);
  const [segmentationResult, setSegmentationResult] = useState<SegmentationResponse | null>(null);
  const [hoveredBoxIndex, setHoveredBoxIndex] = useState<number | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    return () => {
      if (previewUrl && !previewUrl.startsWith('data:')) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (previewUrl && !previewUrl.startsWith('data:')) URL.revokeObjectURL(previewUrl);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    const img = new Image();
    img.onload = () => {
      setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = url;

    setDetectionResult(null);
    setSegmentationResult(null);
    setError(null);
    setZoomLevel(1.0);
  };

  const loadSampleScene = async (sceneId: string) => {
    const scene = SAMPLE_SATELLITE_SCENES.find((s) => s.id === sceneId);
    if (!scene) return;

    try {
      const file = await dataUrlToFile(scene.dataUrl, `${scene.id}.png`);
      setSelectedFile(file);
      setPreviewUrl(scene.dataUrl);

      const img = new Image();
      img.onload = () => {
        setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.src = scene.dataUrl;

      setDetectionResult(null);
      setSegmentationResult(null);
      setError(null);
      setZoomLevel(1.0);
    } catch (err: any) {
      setError('Could not load sample scene: ' + err.message);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      if (previewUrl && !previewUrl.startsWith('data:')) URL.revokeObjectURL(previewUrl);
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);

      const img = new Image();
      img.onload = () => {
        setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.src = url;

      setDetectionResult(null);
      setSegmentationResult(null);
      setError(null);
      setZoomLevel(1.0);
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile) {
      setError('Please select an image or click a sample scene.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      if (mode === 'detection') {
        const result = await geoVisionApi.detectObjects(selectedFile, confThresh);
        setDetectionResult(result);
      } else {
        const result = await geoVisionApi.segmentLandcover(selectedFile, resolutionM);
        setSegmentationResult(result);
      }
    } catch (err: any) {
      setError(err.message || 'Analysis pipeline failed. Ensure GeoVision backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const clearAnalysis = () => {
    if (previewUrl && !previewUrl.startsWith('data:')) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setImageSize(null);
    setDetectionResult(null);
    setSegmentationResult(null);
    setError(null);
    setZoomLevel(1.0);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Filter boxes by class and confidence
  const rawBoxes: BBoxItem[] = detectionResult?.detections || [];
  const filteredBoxes = rawBoxes.filter((box: BBoxItem) => {
    const matchesClass = selectedClassFilter === 'all' || box.class_name.toLowerCase() === selectedClassFilter.toLowerCase();
    const matchesConf = box.confidence >= confThresh;
    return matchesClass && matchesConf;
  });

  const uniqueClasses = Array.from(new Set(rawBoxes.map((b: BBoxItem) => b.class_name)));

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      {/* Header Bar */}
      <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-theme-accent font-mono text-xs uppercase tracking-wider font-semibold mb-1">
            <ScanLine className="w-4 h-4" />
            <span>High-Resolution Vision Core</span>
          </div>
          <h1 className="text-xl sm:text-2xl font-display font-bold text-white">
            Satellite Object Detection & Land-Cover Analysis
          </h1>
          <p className="text-xs text-slate-300 mt-1 font-sans">
            Inference engine for optical object identification, localized bounding boxes, and pixel-wise land classification.
          </p>
        </div>

        {/* Mode Selector Pill */}
        <div className="flex items-center p-1 rounded-xl bg-black/40 border border-theme-border-subtle self-start md:self-auto">
          <button
            onClick={() => setMode('detection')}
            className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all ${
              mode === 'detection'
                ? 'bg-theme-accent text-slate-950 shadow-glow-sm'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            <ScanLine className="w-3.5 h-3.5" />
            <span>Object Detection</span>
          </button>
          <button
            onClick={() => setMode('segmentation')}
            className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all ${
              mode === 'segmentation'
                ? 'bg-theme-accent text-slate-950 shadow-glow-sm'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Land Segmentation</span>
          </button>
        </div>
      </div>

      {/* Preset Satellite Scenes Quick Ribbon */}
      <div className="glass-card rounded-xl p-3.5 border border-theme-border-subtle flex items-center justify-between flex-wrap gap-2.5">
        <div className="flex items-center space-x-2 text-xs font-mono text-slate-300">
          <Sparkles className="w-4 h-4 text-theme-accent" />
          <span className="font-semibold text-white">Quick Test Scenes:</span>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {SAMPLE_SATELLITE_SCENES.map((s) => (
            <button
              key={s.id}
              onClick={() => loadSampleScene(s.id)}
              className="px-2.5 py-1 rounded-lg bg-white/5 hover:bg-theme-accent/20 border border-white/10 hover:border-theme-accent/50 text-[11px] font-mono text-slate-200 hover:text-theme-accent transition-all flex items-center space-x-1.5"
            >
              <span>{s.category}: {s.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Upload & Parameters Panel (4 cols) */}
        <div className="lg:col-span-4 space-y-5">
          {/* File Upload / Dropzone */}
          <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 space-y-4">
            <h2 className="text-xs font-mono uppercase tracking-wider font-bold text-slate-300 flex items-center gap-2">
              <Upload className="w-4 h-4 text-theme-accent" />
              <span>Imagery Source</span>
            </h2>

            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                selectedFile
                  ? 'border-theme-accent/60 bg-theme-accent/5'
                  : 'border-white/15 hover:border-theme-accent/50 bg-black/20 hover:bg-black/30'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
              <div className="w-12 h-12 rounded-xl bg-white/5 mx-auto flex items-center justify-center text-theme-accent mb-3 shadow-glow-sm">
                <Upload className="w-6 h-6" />
              </div>
              {selectedFile ? (
                <div>
                  <p className="text-xs font-semibold text-white truncate max-w-full">{selectedFile.name}</p>
                  <p className="text-[10px] font-mono text-slate-400 mt-1">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • {imageSize ? `${imageSize.width}×${imageSize.height}px` : 'Raster Ready'}
                  </p>
                </div>
              ) : (
                <div>
                  <p className="text-xs font-semibold text-slate-200">Upload Satellite Capture</p>
                  <p className="text-[11px] text-slate-400 mt-1">Drag & drop or browse GeoTIFF, PNG, JPEG</p>
                </div>
              )}
            </div>
          </div>

          {/* Model Hyperparameters Panel */}
          <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 space-y-4">
            <h2 className="text-xs font-mono uppercase tracking-wider font-bold text-slate-300 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-theme-accent" />
              <span>Inference Parameters</span>
            </h2>

            {mode === 'detection' ? (
              <div className="space-y-3">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-slate-300">Confidence Threshold:</span>
                  <span className="text-theme-accent font-bold px-2 py-0.5 rounded bg-theme-accent/15 border border-theme-accent/30">
                    {confThresh.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.95"
                  step="0.05"
                  value={confThresh}
                  onChange={(e) => setConfThresh(parseFloat(e.target.value))}
                  className="w-full cursor-pointer"
                />
                <div className="flex justify-between text-[10px] font-mono text-slate-400">
                  <span>0.05 (High Recall)</span>
                  <span>0.95 (High Precision)</span>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-slate-300">Ground Resolution (GSD):</span>
                    <span className="text-theme-accent font-bold px-2 py-0.5 rounded bg-theme-accent/15 border border-theme-accent/30">
                      {resolutionM.toFixed(1)} m/px
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="10.0"
                    step="0.1"
                    value={resolutionM}
                    onChange={(e) => setResolutionM(parseFloat(e.target.value))}
                    className="w-full cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] font-mono text-slate-400">
                    <span>0.1m (High-Res WorldView)</span>
                    <span>10.0m (Sentinel-2)</span>
                  </div>
                </div>

                <div className="space-y-2 pt-2 border-t border-white/5">
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-slate-300">Mask Overlay Opacity:</span>
                    <span className="text-theme-accent font-bold px-2 py-0.5 rounded bg-theme-accent/15 border border-theme-accent/30">
                      {Math.round(maskOpacity * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={maskOpacity}
                    onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
                    className="w-full cursor-pointer"
                  />
                </div>
              </div>
            )}

            {/* Run / Reset Action Buttons */}
            <div className="pt-2 flex gap-3">
              <button
                onClick={runAnalysis}
                disabled={loading || !selectedFile}
                className={`flex-1 py-3 px-4 rounded-xl text-xs font-bold flex items-center justify-center space-x-2 transition-all shadow-glow-sm ${
                  loading || !selectedFile
                    ? 'bg-white/10 text-slate-500 cursor-not-allowed border border-white/5'
                    : 'bg-theme-accent text-slate-950 hover:opacity-95'
                }`}
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin text-slate-950" />
                    <span>Inference Processing...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    <span>Execute {mode === 'detection' ? 'Detection' : 'Segmentation'}</span>
                  </>
                )}
              </button>

              <button
                onClick={clearAnalysis}
                className="p-3 rounded-xl bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white border border-white/10 transition-colors"
                title="Reset View"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>

            {error && (
              <div className="p-3.5 rounded-xl bg-rose-950/40 border border-rose-500/40 text-rose-300 text-xs flex items-start space-x-2.5">
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <span className="leading-tight">{error}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Interactive Satellite Viewport & Result Metrics (8 cols) */}
        <div className="lg:col-span-8 space-y-5">
          {/* Viewport Card */}
          <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 space-y-4">
            {/* Viewport Toolbar */}
            <div className="flex items-center justify-between flex-wrap gap-2 pb-2 border-b border-white/10">
              <div className="flex items-center space-x-2 text-xs font-mono text-slate-300">
                <Eye className="w-4 h-4 text-theme-accent" />
                <span className="font-semibold text-white">Interactive Observation Canvas</span>
                {imageSize && (
                  <span className="text-[10px] px-2 py-0.5 rounded bg-white/5 text-slate-400 border border-white/5">
                    {imageSize.width} × {imageSize.height} px
                  </span>
                )}
              </div>

              {/* Canvas Controls */}
              <div className="flex items-center space-x-1.5 text-xs font-mono">
                <button
                  onClick={() => setZoomLevel((z) => Math.max(0.5, z - 0.25))}
                  className="p-1.5 rounded-lg bg-black/40 hover:bg-white/10 border border-white/10 text-slate-300"
                  title="Zoom Out"
                >
                  <ZoomOut className="w-3.5 h-3.5" />
                </button>
                <span className="px-2 py-1 rounded bg-black/40 text-[10px] text-slate-300 border border-white/10">
                  {Math.round(zoomLevel * 100)}%
                </span>
                <button
                  onClick={() => setZoomLevel((z) => Math.min(3.0, z + 0.25))}
                  className="p-1.5 rounded-lg bg-black/40 hover:bg-white/10 border border-white/10 text-slate-300"
                  title="Zoom In"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setShowGrid(!showGrid)}
                  className={`p-1.5 rounded-lg border text-xs ${
                    showGrid ? 'bg-theme-accent/20 border-theme-accent/50 text-theme-accent' : 'bg-black/40 border-white/10 text-slate-400'
                  }`}
                  title="Toggle Grid HUD"
                >
                  <Grid className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setShowFalseColor(!showFalseColor)}
                  className={`px-2 py-1 rounded-lg border text-[10px] ${
                    showFalseColor ? 'bg-amber-500/20 border-amber-500/50 text-amber-300' : 'bg-black/40 border-white/10 text-slate-400'
                  }`}
                  title="Simulate Infrared / NDVI"
                >
                  IR SIM
                </button>
              </div>
            </div>

            {/* Viewport Screen */}
            <div 
              ref={canvasContainerRef}
              className="relative min-h-[420px] max-h-[560px] rounded-xl overflow-hidden bg-black/60 border border-white/10 flex items-center justify-center select-none"
            >
              {previewUrl ? (
                <div 
                  className="relative transition-transform duration-150 flex items-center justify-center max-w-full max-h-full"
                  style={{
                    transform: `scale(${zoomLevel})`,
                    filter: showFalseColor ? 'contrast(1.4) hue-rotate(180deg) saturate(2)' : 'none',
                  }}
                >
                  <img
                    src={previewUrl}
                    alt="Satellite Observation"
                    className="max-h-[480px] w-auto object-contain rounded"
                  />

                  {/* Object Detection Bounding Boxes */}
                  {mode === 'detection' && filteredBoxes.map((box: BBoxItem, idx: number) => {
                    const color = CLASS_COLORS[idx % CLASS_COLORS.length];
                    const isHovered = hoveredBoxIndex === idx;

                    return (
                      <div
                        key={idx}
                        onMouseEnter={() => setHoveredBoxIndex(idx)}
                        onMouseLeave={() => setHoveredBoxIndex(null)}
                        className="absolute cursor-pointer transition-all"
                        style={{
                          left: `${box.xmin}%`,
                          top: `${box.ymin}%`,
                          width: `${box.xmax - box.xmin}%`,
                          height: `${box.ymax - box.ymin}%`,
                          border: `2px solid ${color}`,
                          backgroundColor: isHovered ? `${color}33` : `${color}15`,
                          boxShadow: isHovered ? `0 0 15px ${color}` : 'none',
                          zIndex: isHovered ? 30 : 10,
                        }}
                      >
                        {/* Box label tag */}
                        <div
                          className="absolute -top-6 left-0 text-[10px] font-mono px-1.5 py-0.5 rounded text-white font-bold whitespace-nowrap shadow"
                          style={{ backgroundColor: color }}
                        >
                          {box.class_name} ({Math.round(box.confidence * 100)}%)
                        </div>
                      </div>
                    );
                  })}

                  {/* Grid HUD Overlay */}
                  {showGrid && (
                    <div className="absolute inset-0 pointer-events-none grid grid-cols-6 grid-rows-6 border border-cyan-500/20">
                      {Array.from({ length: 36 }).map((_, i) => (
                        <div key={i} className="border border-cyan-500/10" />
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center p-8 space-y-3 text-slate-500">
                  <div className="w-16 h-16 rounded-2xl bg-white/5 border border-white/10 mx-auto flex items-center justify-center text-slate-400">
                    <ScanLine className="w-8 h-8" />
                  </div>
                  <p className="text-sm font-semibold text-slate-300">No Satellite Imagery Loaded</p>
                  <p className="text-xs text-slate-400 max-w-sm">
                    Select a satellite file or click one of the quick test scenarios above to initialize the vision pipeline.
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Results Analytics Panel */}
          {detectionResult && mode === 'detection' && (
            <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 space-y-4 animate-in fade-in duration-200">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center space-x-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  <h3 className="text-sm font-display font-bold text-white">
                    Detection Results: <span className="text-theme-accent">{filteredBoxes.length} Objects Found</span>
                  </h3>
                </div>
                <div className="flex items-center space-x-3 text-xs font-mono text-slate-400">
                  <span className="flex items-center gap-1">
                    <Clock className="w-3.5 h-3.5 text-theme-accent" />
                    <span>{detectionResult.latency_ms ? `${detectionResult.latency_ms.toFixed(1)} ms` : '< 200 ms'}</span>
                  </span>
                </div>
              </div>

              {/* Class Filter Pills */}
              {uniqueClasses.length > 0 && (
                <div className="flex items-center gap-2 flex-wrap pt-2 border-t border-white/5">
                  <span className="text-[11px] font-mono text-slate-400">Filter Class:</span>
                  <button
                    onClick={() => setSelectedClassFilter('all')}
                    className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors ${
                      selectedClassFilter === 'all'
                        ? 'bg-theme-accent text-slate-950 font-bold'
                        : 'bg-white/5 text-slate-300 hover:bg-white/10'
                    }`}
                  >
                    All ({rawBoxes.length})
                  </button>
                  {uniqueClasses.map((cls: string) => {
                    const count = rawBoxes.filter((b: BBoxItem) => b.class_name === cls).length;
                    return (
                      <button
                        key={cls}
                        onClick={() => setSelectedClassFilter(cls)}
                        className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors ${
                          selectedClassFilter === cls
                            ? 'bg-theme-accent text-slate-950 font-bold'
                            : 'bg-white/5 text-slate-300 hover:bg-white/10'
                        }`}
                      >
                        {cls} ({count})
                      </button>
                    );
                  })}
                </div>
              )}

              {/* Detections List */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5 max-h-56 overflow-y-auto pr-1">
                {filteredBoxes.map((box: BBoxItem, i: number) => {
                  const color = CLASS_COLORS[i % CLASS_COLORS.length];
                  const isHovered = hoveredBoxIndex === i;

                  return (
                    <div
                      key={i}
                      onMouseEnter={() => setHoveredBoxIndex(i)}
                      onMouseLeave={() => setHoveredBoxIndex(null)}
                      className={`p-2.5 rounded-xl border text-xs font-mono transition-all cursor-pointer flex items-center justify-between ${
                        isHovered
                          ? 'bg-theme-accent/20 border-theme-accent text-white shadow-glow-sm'
                          : 'bg-black/30 border-white/10 text-slate-300 hover:border-white/20'
                      }`}
                    >
                      <div className="flex items-center space-x-2">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                        <span className="font-semibold text-white">{box.class_name}</span>
                      </div>
                      <span className="text-[11px] text-theme-accent font-bold">
                        {Math.round(box.confidence * 100)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {segmentationResult && mode === 'segmentation' && (
            <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 space-y-4 animate-in fade-in duration-200">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center space-x-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  <h3 className="text-sm font-display font-bold text-white">
                    Land-Cover Dense Segmentation Breakdown
                  </h3>
                </div>
                <div className="text-xs font-mono text-slate-400 flex items-center gap-1">
                  <Clock className="w-3.5 h-3.5 text-theme-accent" />
                  <span>{segmentationResult.latency_ms ? `${segmentationResult.latency_ms.toFixed(1)} ms` : '< 350 ms'}</span>
                </div>
              </div>

              {/* Area Distribution Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                {(segmentationResult.classes || []).map((item, i) => {
                  const color = getLandcoverColor(item.class_name, i);
                  return (
                    <div
                      key={item.class_id || item.class_name}
                      className="p-3 rounded-xl bg-black/30 border border-white/10 space-y-1.5"
                    >
                      <div className="flex items-center justify-between text-xs font-semibold">
                        <span className="capitalize text-white flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                          {item.class_name}
                        </span>
                        <span className="font-mono text-theme-accent">{item.percentage?.toFixed(1) || 0}%</span>
                      </div>
                      <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${item.percentage || 0}%`, backgroundColor: color }}
                        />
                      </div>
                      <div className="flex justify-between text-[10px] font-mono text-slate-400 pt-1">
                        <span>{item.area_hectares?.toFixed(2) || 0} ha</span>
                        <span>{item.area_sqkm?.toFixed(3) || 0} km²</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
