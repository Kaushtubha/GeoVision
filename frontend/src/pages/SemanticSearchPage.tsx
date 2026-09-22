import React, { useState, useRef, useEffect } from 'react';
import {
  Search,
  Image as ImageIcon,
  Sliders,
  Play,
  RotateCcw,
  AlertCircle,
  Clock,
  Database,
  Tag,
  Loader2,
  Sparkles,
  Layers,
} from 'lucide-react';
import { geoVisionApi } from '@/services/api';
import type { SearchResponse, SearchResultItem } from '@/types/api';

type SearchInputMode = 'text' | 'image';

const SAMPLE_QUERIES = [
  'Commercial airport with runway and multiple aircraft',
  'Dense urban residential buildings and road networks',
  'Harbor port with container ships and dock cranes',
  'Dense green forest canopy and winding river',
  'Agricultural crop fields and circular irrigation pivots',
  'Industrial oil storage tanks and refinery facilities',
];

export const SemanticSearchPage: React.FC = () => {
  const [searchMode, setSearchMode] = useState<SearchInputMode>('text');
  const [queryText, setQueryText] = useState<string>('');
  const [queryImage, setQueryImage] = useState<File | null>(null);
  const [previewImageUrl, setPreviewImageUrl] = useState<string | null>(null);
  const [topK, setTopK] = useState<number>(5);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (previewImageUrl) URL.revokeObjectURL(previewImageUrl);
    };
  }, [previewImageUrl]);

  const handleImageSelect = (file: File) => {
    if (previewImageUrl) URL.revokeObjectURL(previewImageUrl);
    setQueryImage(file);
    setPreviewImageUrl(URL.createObjectURL(file));
    setError(null);
  };

  const handleSearch = async () => {
    if (searchMode === 'text' && !queryText.trim()) {
      setError('Please enter a natural-language satellite scene query.');
      return;
    }
    if (searchMode === 'image' && !queryImage) {
      setError('Please upload a reference satellite image to search.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await geoVisionApi.searchScenes(
        searchMode === 'text' ? queryText.trim() : undefined,
        searchMode === 'image' ? queryImage : null,
        topK
      );
      setResults(response);
    } catch (err: any) {
      setError(err.message || 'Semantic search failed. Ensure the GeoVision backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const clearQuery = () => {
    if (previewImageUrl) URL.revokeObjectURL(previewImageUrl);
    setQueryText('');
    setQueryImage(null);
    setPreviewImageUrl(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="p-5 rounded-lg bg-space-900 border border-space-700/80 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-cyan-400 font-mono text-xs uppercase tracking-wider mb-1">
            <Search className="w-4 h-4" />
            <span>Retrieval Core</span>
          </div>
          <h1 className="text-xl font-bold text-white font-sans">
            Multimodal Semantic Scene Retrieval
          </h1>
          <p className="text-xs text-slate-400 mt-1 font-sans">
            Search satellite scenes using OpenCLIP multimodal embeddings and Qdrant high-dimensional vector index.
          </p>
        </div>

        {/* Search Mode Selector */}
        <div className="flex items-center p-1 rounded-lg bg-space-950 border border-space-800 self-start md:self-auto">
          <button
            onClick={() => {
              setSearchMode('text');
              setError(null);
            }}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-md text-xs font-medium font-mono transition-all ${
              searchMode === 'text'
                ? 'bg-cyan-600 text-white shadow-md shadow-cyan-900/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Search className="w-3.5 h-3.5" />
            <span>Text-to-Scene</span>
          </button>
          <button
            onClick={() => {
              setSearchMode('image');
              setError(null);
            }}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-md text-xs font-medium font-mono transition-all ${
              searchMode === 'image'
                ? 'bg-cyan-600 text-white shadow-md shadow-cyan-900/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <ImageIcon className="w-3.5 h-3.5" />
            <span>Image-to-Scene</span>
          </button>
        </div>
      </div>

      {/* Query Formulation Input Box */}
      <div className="p-5 rounded-lg bg-space-900 border border-space-700 space-y-4">
        {searchMode === 'text' ? (
          <div className="space-y-3">
            <label className="text-xs font-mono uppercase text-slate-300 flex items-center space-x-1.5">
              <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
              <span>Natural-Language Earth Observation Query</span>
            </label>
            <div className="relative">
              <input
                type="text"
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="e.g. Commercial airport with runway and multiple parked passenger aircraft..."
                className="w-full px-4 py-3 bg-space-950 border border-space-700 rounded-md text-sm text-slate-100 placeholder-slate-500 font-sans focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/50"
              />
            </div>

            {/* Quick Sample Queries */}
            <div className="space-y-1.5 pt-1">
              <span className="text-[11px] font-mono text-slate-400">Sample Scientific Prompts:</span>
              <div className="flex flex-wrap gap-1.5">
                {SAMPLE_QUERIES.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => setQueryText(q)}
                    className="px-2.5 py-1 rounded bg-space-850 hover:bg-space-800 border border-space-750 text-[11px] text-slate-300 hover:text-cyan-300 font-sans text-left transition-colors"
                  >
                    &ldquo;{q}&rdquo;
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <label className="text-xs font-mono uppercase text-slate-300 flex items-center space-x-1.5">
              <ImageIcon className="w-3.5 h-3.5 text-cyan-400" />
              <span>Reference Satellite Query Image</span>
            </label>

            <div
              onClick={() => !previewImageUrl && fileInputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const f = e.dataTransfer.files?.[0];
                if (f) handleImageSelect(f);
              }}
              className={`rounded-lg border-2 border-dashed transition-all flex flex-col items-center justify-center min-h-[160px] bg-space-950 ${
                previewImageUrl
                  ? 'border-space-700 p-2'
                  : 'border-space-700/80 hover:border-cyan-500/50 p-6 cursor-pointer'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.tif,.tiff"
                onChange={(e) => e.target.files?.[0] && handleImageSelect(e.target.files[0])}
                className="hidden"
              />

              {previewImageUrl ? (
                <div className="relative flex items-center justify-center">
                  <img
                    src={previewImageUrl}
                    alt="Query Image"
                    className="max-h-40 w-auto object-contain rounded select-none"
                  />
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center text-center space-y-1.5">
                  <div className="w-10 h-10 rounded-full bg-space-850 border border-space-700 flex items-center justify-center text-cyan-400">
                    <ImageIcon className="w-5 h-5" />
                  </div>
                  <p className="text-xs font-semibold text-white">Drop reference scene here</p>
                  <p className="text-[10px] text-slate-400 font-mono">Embeddings will match visual similarity</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Search Parameter & Execution Bar */}
        <div className="pt-3 border-t border-space-750 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3 text-xs font-mono">
            <Sliders className="w-4 h-4 text-cyan-400" />
            <span className="text-slate-300">Top-K Results:</span>
            <input
              type="range"
              min="1"
              max="20"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-24 accent-cyan-500 cursor-pointer"
            />
            <span className="text-cyan-300 px-2 py-0.5 rounded bg-space-800 border border-space-700 font-bold">
              K = {topK}
            </span>
          </div>

          <div className="flex items-center space-x-3">
            {(queryText || queryImage) && (
              <button
                onClick={clearQuery}
                className="px-3 py-1.5 rounded bg-space-800 hover:bg-space-750 text-slate-400 hover:text-slate-200 text-xs font-mono flex items-center space-x-1.5 border border-space-700"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Clear</span>
              </button>
            )}

            <button
              onClick={handleSearch}
              disabled={loading || (searchMode === 'text' ? !queryText.trim() : !queryImage)}
              className={`px-4 py-2 rounded-md text-xs font-mono font-bold flex items-center space-x-2 transition-all shadow-md ${
                loading || (searchMode === 'text' ? !queryText.trim() : !queryImage)
                  ? 'bg-space-800 text-slate-500 border border-space-700 cursor-not-allowed'
                  : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-cyan-900/30'
              }`}
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin text-cyan-200" />
                  <span>Computing Vector Similarity...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Execute Vector Query</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-800/80 text-rose-300 text-xs flex items-center space-x-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
          <div>
            <p className="font-semibold font-mono">Retrieval Engine Notice</p>
            <p className="text-rose-400 font-mono mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Results Telemetry & Ranked Gallery */}
      {results && (
        <div className="p-5 rounded-lg bg-space-900 border border-space-700 space-y-4">
          <div className="flex items-center justify-between border-b border-space-750 pb-3">
            <div className="flex items-center space-x-2">
              <Database className="w-4 h-4 text-cyan-400" />
              <h3 className="text-xs font-mono uppercase tracking-wider font-bold text-white">
                Ranked Vector Match Results ({results.count})
              </h3>
            </div>
            <div className="flex items-center space-x-1.5 text-xs font-mono text-slate-400">
              <Clock className="w-3.5 h-3.5 text-cyan-400" />
              <span>Latency: {results.latency_ms.toFixed(1)} ms</span>
            </div>
          </div>

          {results.results.length === 0 ? (
            <div className="p-8 text-center text-slate-400 space-y-2 font-mono text-xs">
              <Layers className="w-8 h-8 text-slate-600 mx-auto" />
              <p>No matching satellite scenes found in the vector index.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {results.results.map((item: SearchResultItem, index: number) => {
                const scorePercentage = Math.max(0, Math.min(100, item.score * 100));
                return (
                  <div
                    key={item.id || index}
                    className="p-4 rounded-lg bg-space-850 border border-space-750 hover:border-cyan-500/50 transition-all space-y-3"
                  >
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <div className="flex items-center space-x-3">
                        <span className="w-7 h-7 rounded bg-space-800 border border-space-700 flex items-center justify-center text-xs font-mono font-bold text-cyan-300">
                          #{index + 1}
                        </span>
                        <div>
                          <p className="text-sm font-semibold text-white font-mono flex items-center space-x-2">
                            <span>Scene ID: {item.id}</span>
                            {item.label && (
                              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-950/60 text-cyan-300 border border-cyan-800/60">
                                {item.label}
                              </span>
                            )}
                          </p>
                        </div>
                      </div>

                      {/* Cosine Score Pill */}
                      <div className="flex items-center space-x-2">
                        <div className="text-right">
                          <span className="text-xs font-mono font-bold text-cyan-300">
                            {item.score.toFixed(4)}
                          </span>
                          <p className="text-[10px] font-mono text-slate-400">Cosine Similarity</p>
                        </div>
                        <div className="w-16 h-2 rounded-full bg-space-950 overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400"
                            style={{ width: `${scorePercentage}%` }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Metadata attributes */}
                    {item.metadata && Object.keys(item.metadata).length > 0 && (
                      <div className="pt-2 border-t border-space-750/70">
                        <div className="flex flex-wrap gap-2 text-[11px] font-mono text-slate-300">
                          {Object.entries(item.metadata).map(([key, val]) => (
                            <div
                              key={key}
                              className="px-2 py-1 rounded bg-space-900 border border-space-750 flex items-center space-x-1"
                            >
                              <Tag className="w-3 h-3 text-slate-400" />
                              <span className="text-slate-400">{key}:</span>
                              <span className="text-cyan-300 font-medium">
                                {typeof val === 'object' ? JSON.stringify(val) : String(val)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
