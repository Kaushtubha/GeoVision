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
  Crosshair,
  Maximize2,
  PieChart,
  Tag,
  Loader2,
} from 'lucide-react';
import { geoVisionApi } from '@/services/api';
import type { DetectionResponse, SegmentationResponse, BBoxItem } from '@/types/api';

type AnalysisMode = 'detection' | 'segmentation';

// Color mapping for detection classes
const CLASS_COLORS = [
  '#06b6d4', // cyan-500
  '#10b981', // emerald-500
  '#f59e0b', // amber-500
  '#ec4899', // pink-500
  '#8b5cf6', // violet-500
  '#3b82f6', // blue-500
  '#ef4444', // red-500
  '#14b8a6', // teal-500
  '#f97316', // orange-500
  '#a855f7', // purple-500
];

const LANDCOVER_COLORS: Record<string, string> = {
  urban: '#ef4444',
  building: '#ef4444',
  built: '#ef4444',
  vegetation: '#10b981',
  forest: '#059669',
  tree: '#059669',
  water: '#0ea5e9',
  river: '#0ea5e9',
  agriculture: '#eab308',
  crop: '#eab308',
  barren: '#a8a29e',
  soil: '#a8a29e',
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

  // States
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Results
  const [detectionResult, setDetectionResult] = useState<DetectionResponse | null>(null);
  const [segmentationResult, setSegmentationResult] = useState<SegmentationResponse | null>(null);
  const [hoveredBoxIndex, setHoveredBoxIndex] = useState<number | null>(null);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Clean up object URLs
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    // Read image dimensions
    const img = new Image();
    img.onload = () => {
      setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = url;

    // Reset results
    setDetectionResult(null);
    setSegmentationResult(null);
    setError(null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
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
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile) {
      setError('Please select or upload a satellite image first.');
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
      setError(err.message || 'Analysis failed. Ensure the GeoVision backend is running.');
    } finally {
      setLoading(false);
    }
  };

  // Render bounding boxes onto Canvas overlay
  useEffect(() => {
    if (!canvasRef.current || !imageRef.current || !detectionResult || !showBoxes) {
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
      return;
    }

    const canvas = canvasRef.current;
    const img = imageRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match canvas coordinate size to image display size
    const displayWidth = img.clientWidth;
    const displayHeight = img.clientHeight;
    canvas.width = displayWidth;
    canvas.height = displayHeight;

    const originalH = detectionResult.image_shape[0] || imageSize?.height || displayHeight;
    const originalW = detectionResult.image_shape[1] || imageSize?.width || displayWidth;

    const scaleX = displayWidth / originalW;
    const scaleY = displayHeight / originalH;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detectionResult.detections.forEach((det: BBoxItem, index: number) => {
      const isHovered = hoveredBoxIndex === index;
      const color = CLASS_COLORS[det.class_id % CLASS_COLORS.length];

      const x = det.xmin * scaleX;
      const y = det.ymin * scaleY;
      const w = (det.xmax - det.xmin) * scaleX;
      const h = (det.ymax - det.ymin) * scaleY;

      // Box outline
      ctx.strokeStyle = color;
      ctx.lineWidth = isHovered ? 3 : 2;
      ctx.fillStyle = `${color}${isHovered ? '40' : '15'}`;

      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);

      // Label background & text
      const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
      ctx.font = isHovered ? 'bold 12px JetBrains Mono, monospace' : '11px JetBrains Mono, monospace';
      const textWidth = ctx.measureText(label).width;
      const labelHeight = 18;

      ctx.fillStyle = color;
      ctx.fillRect(x, Math.max(0, y - labelHeight), textWidth + 8, labelHeight);

      ctx.fillStyle = '#0f172a'; // dark space text
      ctx.fillText(label, x + 4, Math.max(13, y - 4));
    });
  }, [detectionResult, hoveredBoxIndex, showBoxes, imageSize]);

  // Aggregate detection count by class
  const classCounts = detectionResult?.detections.reduce<Record<string, number>>((acc, cur) => {
    acc[cur.class_name] = (acc[cur.class_name] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      {/* Header & Mode Switcher */}
      <div className="p-5 rounded-lg bg-space-900 border border-space-700/80 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-cyan-400 font-mono text-xs uppercase tracking-wider mb-1">
            <ScanLine className="w-4 h-4" />
            <span>Earth Observation Core</span>
          </div>
          <h1 className="text-xl font-bold text-white font-sans">
            Satellite Optical Image Analysis
          </h1>
          <p className="text-xs text-slate-400 mt-1 font-sans">
            Execute high-resolution optical object detection or 5-class land-cover segmentation with precision area computation.
          </p>
        </div>

        {/* Mode Selector Tabs */}
        <div className="flex items-center p-1 rounded-lg bg-space-950 border border-space-800 self-start md:self-auto">
          <button
            onClick={() => {
              setMode('detection');
              setError(null);
            }}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-md text-xs font-medium font-mono transition-all ${
              mode === 'detection'
                ? 'bg-cyan-600 text-white shadow-md shadow-cyan-900/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <ScanLine className="w-3.5 h-3.5" />
            <span>Object Detection</span>
          </button>
          <button
            onClick={() => {
              setMode('segmentation');
              setError(null);
            }}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-md text-xs font-medium font-mono transition-all ${
              mode === 'segmentation'
                ? 'bg-cyan-600 text-white shadow-md shadow-cyan-900/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Land-Cover Segmentation</span>
          </button>
        </div>
      </div>

      {/* Control Parameters Bar */}
      <div className="p-4 rounded-lg bg-space-900/80 border border-space-700/70 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-6">
          {mode === 'detection' ? (
            <div className="flex items-center space-x-3 text-xs font-mono">
              <Sliders className="w-4 h-4 text-cyan-400" />
              <span className="text-slate-300">Confidence Threshold:</span>
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.05"
                value={confThresh}
                onChange={(e) => setConfThresh(parseFloat(e.target.value))}
                className="w-28 accent-cyan-500 cursor-pointer"
              />
              <span className="text-cyan-300 px-2 py-0.5 rounded bg-space-800 border border-space-700 font-bold">
                {(confThresh * 100).toFixed(0)}%
              </span>
            </div>
          ) : (
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
          )}

          {detectionResult && mode === 'detection' && (
            <button
              onClick={() => setShowBoxes(!showBoxes)}
              className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-space-800 border border-space-700 text-xs font-mono text-slate-300 hover:text-white"
            >
              <Eye className="w-3.5 h-3.5 text-cyan-400" />
              <span>{showBoxes ? 'Hide BBoxes' : 'Show BBoxes'}</span>
            </button>
          )}
        </div>

        {/* Execution Button */}
        <div className="flex items-center space-x-3">
          {selectedFile && (
            <button
              onClick={() => {
                setSelectedFile(null);
                setPreviewUrl(null);
                setDetectionResult(null);
                setSegmentationResult(null);
                setImageSize(null);
                if (fileInputRef.current) fileInputRef.current.value = '';
              }}
              className="px-3 py-1.5 rounded bg-space-800 hover:bg-space-750 text-slate-400 hover:text-slate-200 text-xs font-mono flex items-center space-x-1.5 border border-space-700"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Clear</span>
            </button>
          )}

          <button
            onClick={runAnalysis}
            disabled={loading || !selectedFile}
            className={`px-4 py-2 rounded-md text-xs font-mono font-bold flex items-center space-x-2 transition-all shadow-md ${
              loading || !selectedFile
                ? 'bg-space-800 text-slate-500 border border-space-700 cursor-not-allowed'
                : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-cyan-900/30'
            }`}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-cyan-200" />
                <span>Processing Model Pipeline...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Run {mode === 'detection' ? 'Object Detection' : 'Land-Cover Segmentation'}</span>
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
            <p className="font-semibold font-mono">Inference Engine Error</p>
            <p className="text-rose-400 font-mono mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Image Canvas & Upload Dropzone */}
        <div className="lg:col-span-7 space-y-4">
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className={`relative rounded-lg border-2 border-dashed transition-all overflow-hidden flex flex-col items-center justify-center min-h-[420px] bg-space-900/90 ${
              previewUrl
                ? 'border-space-700 p-2'
                : 'border-space-700/80 hover:border-cyan-500/50 p-8 cursor-pointer'
            }`}
            onClick={() => !previewUrl && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,.tif,.tiff"
              onChange={handleFileChange}
              className="hidden"
            />

            {previewUrl ? (
              <div className="relative w-full flex items-center justify-center bg-black/40 rounded overflow-hidden">
                <img
                  ref={imageRef}
                  src={previewUrl}
                  alt="Satellite Observation"
                  className="max-h-[520px] w-auto object-contain rounded select-none"
                />
                {/* Canvas Overlay for Detection Boxes */}
                {mode === 'detection' && detectionResult && (
                  <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full pointer-events-none"
                  />
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center text-center space-y-3">
                <div className="w-14 h-14 rounded-full bg-space-850 border border-space-700 flex items-center justify-center text-cyan-400 shadow-inner">
                  <Upload className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-white font-sans">
                    Drop satellite image here or click to browse
                  </p>
                  <p className="text-xs text-slate-400 mt-1 font-mono">
                    Supports Optical GeoTIFF, PNG, JPEG scenes
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Image Details Pill */}
          {selectedFile && imageSize && (
            <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 px-3 py-2 rounded bg-space-900 border border-space-750">
              <span className="truncate max-w-xs">{selectedFile.name}</span>
              <span>
                {imageSize.width} × {imageSize.height} px • {(selectedFile.size / 1024).toFixed(1)} KB
              </span>
            </div>
          )}
        </div>

        {/* Right Column: Telemetry & Results Breakdown */}
        <div className="lg:col-span-5 space-y-4">
          {/* Object Detection Results View */}
          {mode === 'detection' && (
            <div className="p-5 rounded-lg bg-space-900 border border-space-700 space-y-5">
              <div className="flex items-center justify-between border-b border-space-750 pb-3">
                <div className="flex items-center space-x-2">
                  <Crosshair className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-xs font-mono uppercase tracking-wider font-bold text-white">
                    Detection Telemetry
                  </h3>
                </div>
                {detectionResult && (
                  <div className="flex items-center space-x-1.5 text-[11px] font-mono text-slate-400">
                    <Clock className="w-3.5 h-3.5 text-cyan-400" />
                    <span>{detectionResult.latency_ms.toFixed(1)} ms</span>
                  </div>
                )}
              </div>

              {detectionResult ? (
                <>
                  {/* Summary Metric Cards */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3.5 rounded bg-space-850 border border-space-750">
                      <p className="text-[10px] font-mono text-slate-400 uppercase">Total Detections</p>
                      <p className="text-2xl font-mono font-bold text-cyan-300 mt-1">
                        {detectionResult.count}
                      </p>
                    </div>
                    <div className="p-3.5 rounded bg-space-850 border border-space-750">
                      <p className="text-[10px] font-mono text-slate-400 uppercase">Input Resolution</p>
                      <p className="text-xs font-mono font-semibold text-slate-200 mt-2">
                        {detectionResult.image_shape[1]} × {detectionResult.image_shape[0]} px
                      </p>
                    </div>
                  </div>

                  {/* Class Breakdown Badges */}
                  {classCounts && Object.keys(classCounts).length > 0 && (
                    <div className="space-y-2">
                      <p className="text-[11px] font-mono uppercase text-slate-400">Detected Classes</p>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(classCounts).map(([cls, cnt], idx) => (
                          <span
                            key={cls}
                            className="px-2.5 py-1 rounded text-xs font-mono font-medium border flex items-center space-x-1.5"
                            style={{
                              backgroundColor: `${CLASS_COLORS[idx % CLASS_COLORS.length]}15`,
                              borderColor: `${CLASS_COLORS[idx % CLASS_COLORS.length]}50`,
                              color: CLASS_COLORS[idx % CLASS_COLORS.length],
                            }}
                          >
                            <Tag className="w-3 h-3" />
                            <span>{cls}</span>
                            <span className="font-bold">({cnt})</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Detected Bounding Boxes Table */}
                  <div className="space-y-2">
                    <p className="text-[11px] font-mono uppercase text-slate-400">
                      Bounding Box Registry ({detectionResult.detections.length})
                    </p>
                    <div className="max-h-56 overflow-y-auto space-y-1.5 pr-1 text-xs font-mono">
                      {detectionResult.detections.map((box: BBoxItem, index: number) => {
                        const color = CLASS_COLORS[box.class_id % CLASS_COLORS.length];
                        const isHovered = hoveredBoxIndex === index;
                        return (
                          <div
                            key={index}
                            onMouseEnter={() => setHoveredBoxIndex(index)}
                            onMouseLeave={() => setHoveredBoxIndex(null)}
                            className={`p-2 rounded border transition-colors cursor-pointer flex items-center justify-between ${
                              isHovered
                                ? 'bg-space-800 border-cyan-500/80 text-white'
                                : 'bg-space-850/60 border-space-750 text-slate-300'
                            }`}
                          >
                            <div className="flex items-center space-x-2">
                              <span
                                className="w-2.5 h-2.5 rounded-full"
                                style={{ backgroundColor: color }}
                              />
                              <span className="font-semibold text-white">{box.class_name}</span>
                              <span className="text-[10px] text-slate-400">
                                [{box.xmin.toFixed(0)}, {box.ymin.toFixed(0)}, {box.xmax.toFixed(0)}, {box.ymax.toFixed(0)}]
                              </span>
                            </div>
                            <span className="font-bold text-cyan-300">
                              {(box.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </>
              ) : (
                <div className="p-8 text-center space-y-2 text-slate-400">
                  <Maximize2 className="w-8 h-8 text-slate-600 mx-auto" />
                  <p className="text-xs font-mono">
                    Upload an optical image and click &quot;Run Object Detection&quot; to inspect detections.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Semantic Segmentation Results View */}
          {mode === 'segmentation' && (
            <div className="p-5 rounded-lg bg-space-900 border border-space-700 space-y-5">
              <div className="flex items-center justify-between border-b border-space-750 pb-3">
                <div className="flex items-center space-x-2">
                  <PieChart className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-xs font-mono uppercase tracking-wider font-bold text-white">
                    Land-Cover Distribution
                  </h3>
                </div>
                {segmentationResult && (
                  <div className="flex items-center space-x-1.5 text-[11px] font-mono text-slate-400">
                    <Clock className="w-3.5 h-3.5 text-cyan-400" />
                    <span>{segmentationResult.latency_ms.toFixed(1)} ms</span>
                  </div>
                )}
              </div>

              {segmentationResult ? (
                <>
                  {/* Total Area Statistics */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3.5 rounded bg-space-850 border border-space-750">
                      <p className="text-[10px] font-mono text-slate-400 uppercase">Total Surface Area</p>
                      <p className="text-xl font-mono font-bold text-cyan-300 mt-1">
                        {segmentationResult.total_area_hectares.toFixed(2)} <span className="text-xs text-slate-400">ha</span>
                      </p>
                    </div>
                    <div className="p-3.5 rounded bg-space-850 border border-space-750">
                      <p className="text-[10px] font-mono text-slate-400 uppercase">Area in km²</p>
                      <p className="text-xl font-mono font-bold text-emerald-300 mt-1">
                        {(segmentationResult.total_area_hectares / 100).toFixed(3)} <span className="text-xs text-slate-400">km²</span>
                      </p>
                    </div>
                  </div>

                  {/* Coverage Proportional Stacked Bar */}
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                      <span>Land-Cover Composition</span>
                      <span>100%</span>
                    </div>
                    <div className="w-full h-3 rounded-full bg-space-950 overflow-hidden flex">
                      {segmentationResult.classes.map((cls, idx) => {
                        const color = getLandcoverColor(cls.class_name, idx);
                        return (
                          <div
                            key={cls.class_id}
                            style={{
                              width: `${cls.percentage}%`,
                              backgroundColor: color,
                            }}
                            title={`${cls.class_name}: ${cls.percentage.toFixed(1)}%`}
                            className="h-full transition-all"
                          />
                        );
                      })}
                    </div>
                  </div>

                  {/* Class-wise Breakdown Table */}
                  <div className="space-y-2">
                    <p className="text-[11px] font-mono uppercase text-slate-400">Class Breakdown Table</p>
                    <div className="max-h-60 overflow-y-auto space-y-2 pr-1 text-xs font-mono">
                      {segmentationResult.classes.map((cls, idx) => {
                        const color = getLandcoverColor(cls.class_name, idx);
                        return (
                          <div
                            key={cls.class_id}
                            className="p-3 rounded bg-space-850 border border-space-750 space-y-2"
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-2">
                                <span
                                  className="w-3 h-3 rounded-sm"
                                  style={{ backgroundColor: color }}
                                />
                                <span className="font-semibold text-white capitalize">
                                  {cls.class_name}
                                </span>
                              </div>
                              <span className="font-bold text-cyan-300">
                                {cls.percentage.toFixed(1)}%
                              </span>
                            </div>

                            <div className="grid grid-cols-3 gap-2 text-[10px] text-slate-400 pt-1 border-t border-space-750/60">
                              <div>
                                <span>Pixels:</span> <span className="text-slate-200">{cls.pixel_count.toLocaleString()}</span>
                              </div>
                              <div>
                                <span>Hectares:</span> <span className="text-slate-200">{cls.area_hectares.toFixed(2)} ha</span>
                              </div>
                              <div>
                                <span>km²:</span> <span className="text-slate-200">{cls.area_sqkm.toFixed(4)}</span>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </>
              ) : (
                <div className="p-8 text-center space-y-2 text-slate-400">
                  <Layers className="w-8 h-8 text-slate-600 mx-auto" />
                  <p className="text-xs font-mono">
                    Upload a scene and click &quot;Run Land-Cover Segmentation&quot; to calculate land use.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
