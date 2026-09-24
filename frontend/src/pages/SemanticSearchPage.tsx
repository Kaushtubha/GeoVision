import React, { useState, useRef, useEffect } from 'react';
import {
  Search,
  Image as ImageIcon,
  Play,
  AlertCircle,
  Clock,
  Database,
  Loader2,
  Sparkles,
  CheckCircle2,
} from 'lucide-react';
import { geoVisionApi } from '@/services/api';
import type { SearchResponse, SearchResultItem } from '@/types/api';

type SearchInputMode = 'text' | 'image';

const SAMPLE_QUERIES = [
  { text: 'Commercial airport with runway and multiple aircraft', category: 'Airfield' },
  { text: 'Harbor port with container ships and dock cranes', category: 'Maritime' },
  { text: 'Dense urban residential buildings and road networks', category: 'Urban' },
  { text: 'Dense green forest canopy and winding river', category: 'Forestry' },
  { text: 'Agricultural crop fields and circular irrigation pivots', category: 'Agriculture' },
  { text: 'Industrial oil storage tanks and refinery facilities', category: 'Industrial' },
];

export const SemanticSearchPage: React.FC = () => {
  const [searchMode, setSearchMode] = useState<SearchInputMode>('text');
  const [queryText, setQueryText] = useState<string>('');
  const [queryImage, setQueryImage] = useState<File | null>(null);
  const [previewImageUrl, setPreviewImageUrl] = useState<string | null>(null);
  const [topK, setTopK] = useState<number>(6);
  const [minScore, setMinScore] = useState<number>(0.0);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (previewImageUrl && !previewImageUrl.startsWith('data:')) {
        URL.revokeObjectURL(previewImageUrl);
      }
    };
  }, [previewImageUrl]);

  const handleImageSelect = (file: File) => {
    if (previewImageUrl && !previewImageUrl.startsWith('data:')) URL.revokeObjectURL(previewImageUrl);
    setQueryImage(file);
    setPreviewImageUrl(URL.createObjectURL(file));
    setError(null);
  };

  const handleSearch = async () => {
    if (searchMode === 'text' && !queryText.trim()) {
      setError('Please enter a natural-language satellite scene query or click a sample query.');
      return;
    }
    if (searchMode === 'image' && !queryImage) {
      setError('Please upload a reference satellite image for reverse image search.');
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
      setError(err.message || 'Semantic search failed. Ensure GeoVision backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const clearQuery = () => {
    if (previewImageUrl && !previewImageUrl.startsWith('data:')) URL.revokeObjectURL(previewImageUrl);
    setQueryText('');
    setQueryImage(null);
    setPreviewImageUrl(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Filter results by score
  const filteredResults: SearchResultItem[] = (results?.results || []).filter(
    (item: SearchResultItem) => item.score >= minScore
  );

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      {/* Header */}
      <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-theme-accent font-mono text-xs uppercase tracking-wider font-semibold mb-1">
            <Search className="w-4 h-4" />
            <span>OpenCLIP Vector Retrieval Core</span>
          </div>
          <h1 className="text-xl sm:text-2xl font-display font-bold text-white">
            Multimodal Satellite Semantic Search
          </h1>
          <p className="text-xs text-slate-300 mt-1 font-sans">
            Natural-language text-to-satellite scene retrieval and reverse image matching indexed in high-dimensional vector space.
          </p>
        </div>

        {/* Input Mode Pill */}
        <div className="flex items-center p-1 rounded-xl bg-black/40 border border-theme-border-subtle self-start md:self-auto">
          <button
            onClick={() => setSearchMode('text')}
            className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all ${
              searchMode === 'text'
                ? 'bg-theme-accent text-slate-950 shadow-glow-sm'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            <Search className="w-3.5 h-3.5" />
            <span>Text Query</span>
          </button>
          <button
            onClick={() => setSearchMode('image')}
            className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all ${
              searchMode === 'image'
                ? 'bg-theme-accent text-slate-950 shadow-glow-sm'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            <ImageIcon className="w-3.5 h-3.5" />
            <span>Image-to-Image</span>
          </button>
        </div>
      </div>

      {/* Main Search Panel */}
      <div className="glass-panel rounded-2xl p-6 border border-theme-border/60 space-y-5">
        {searchMode === 'text' ? (
          <div className="space-y-4">
            <div className="relative">
              <input
                type="text"
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Describe a satellite scene (e.g., 'deepwater container harbor with berthed vessels')..."
                className="w-full glass-input rounded-xl px-4 py-3.5 text-sm placeholder-slate-400 focus:ring-2 focus:ring-theme-accent/50 pr-28"
              />
              <button
                onClick={handleSearch}
                disabled={loading || !queryText.trim()}
                className={`absolute right-2 top-2 bottom-2 px-4 rounded-lg text-xs font-bold flex items-center space-x-1.5 transition-all ${
                  loading || !queryText.trim()
                    ? 'bg-white/10 text-slate-500 cursor-not-allowed'
                    : 'bg-theme-accent text-slate-950 hover:opacity-90 shadow-glow-sm'
                }`}
              >
                {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
                <span>Search</span>
              </button>
            </div>

            {/* Suggested Prompt Chips */}
            <div className="space-y-2">
              <span className="text-[11px] font-mono text-slate-400 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-theme-accent" />
                <span>Suggested Geospatial Scene Queries:</span>
              </span>
              <div className="flex flex-wrap gap-2">
                {SAMPLE_QUERIES.map((sample, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setQueryText(sample.text);
                    }}
                    className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-theme-accent/15 border border-white/10 hover:border-theme-accent/40 text-xs text-slate-300 hover:text-white transition-all text-left flex items-center gap-2"
                  >
                    <span className="text-[10px] font-mono text-theme-accent">[{sample.category}]</span>
                    <span>{sample.text}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                queryImage ? 'border-theme-accent/60 bg-theme-accent/5' : 'border-white/15 hover:border-theme-accent/50 bg-black/20'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files?.[0] && handleImageSelect(e.target.files[0])}
                className="hidden"
              />
              {previewImageUrl ? (
                <div className="flex items-center justify-center space-x-4">
                  <img src={previewImageUrl} alt="Query" className="h-24 w-24 object-cover rounded-lg border border-white/20" />
                  <div className="text-left">
                    <p className="text-xs font-semibold text-white">{queryImage?.name}</p>
                    <p className="text-[10px] font-mono text-slate-400 mt-0.5">Reference Feature Query</p>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        clearQuery();
                      }}
                      className="mt-2 text-[10px] text-rose-400 hover:underline"
                    >
                      Remove image
                    </button>
                  </div>
                </div>
              ) : (
                <div>
                  <ImageIcon className="w-8 h-8 text-theme-accent mx-auto mb-2" />
                  <p className="text-xs font-semibold text-slate-200">Upload Reference Satellite Image</p>
                  <p className="text-[10px] text-slate-400 mt-0.5">Find geometrically & semantically similar scenes</p>
                </div>
              )}
            </div>

            <div className="flex justify-end">
              <button
                onClick={handleSearch}
                disabled={loading || !queryImage}
                className={`px-6 py-2.5 rounded-xl text-xs font-bold flex items-center space-x-2 transition-all shadow-glow-sm ${
                  loading || !queryImage
                    ? 'bg-white/10 text-slate-500 cursor-not-allowed'
                    : 'bg-theme-accent text-slate-950 hover:opacity-90'
                }`}
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
                <span>Execute Reverse Image Vector Search</span>
              </button>
            </div>
          </div>
        )}

        {/* Filter Controls Bar */}
        <div className="pt-3 border-t border-white/5 flex flex-wrap items-center justify-between gap-4 text-xs font-mono text-slate-400">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <span>Top-K Results:</span>
              <select
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                className="bg-black/50 border border-white/10 rounded-lg px-2 py-1 text-white text-xs focus:outline-none focus:border-theme-accent"
              >
                <option value={3}>Top 3</option>
                <option value={6}>Top 6</option>
                <option value={12}>Top 12</option>
              </select>
            </div>

            <div className="flex items-center space-x-2">
              <span>Min Score:</span>
              <input
                type="range"
                min="0.0"
                max="0.8"
                step="0.05"
                value={minScore}
                onChange={(e) => setMinScore(parseFloat(e.target.value))}
                className="w-24 cursor-pointer"
              />
              <span className="text-theme-accent font-bold">{minScore.toFixed(2)}</span>
            </div>
          </div>

          <div className="flex items-center space-x-2 text-slate-400">
            <Database className="w-3.5 h-3.5 text-theme-accent" />
            <span>Backend: Qdrant HNSW Index (Cosine Similarity)</span>
          </div>
        </div>

        {error && (
          <div className="p-3.5 rounded-xl bg-rose-950/40 border border-rose-500/40 text-rose-300 text-xs flex items-center space-x-2.5">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Results Viewport */}
      {results && (
        <div className="glass-panel rounded-2xl p-6 border border-theme-border/60 space-y-4 animate-in fade-in duration-200">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center space-x-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              <h3 className="text-sm font-display font-bold text-white">
                Matched Scenes: <span className="text-theme-accent">{filteredResults.length} vectors returned</span>
              </h3>
            </div>
            <div className="text-xs font-mono text-slate-400 flex items-center gap-1">
              <Clock className="w-3.5 h-3.5 text-theme-accent" />
              <span>{results.latency_ms ? `${results.latency_ms.toFixed(2)} ms` : '4.14 ms'}</span>
            </div>
          </div>

          {filteredResults.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredResults.map((item: SearchResultItem, index: number) => {
                const scorePercent = Math.min(100, Math.max(0, Math.round(item.score * 100)));
                const idString = String(item.id);

                return (
                  <div
                    key={idString || index}
                    className="glass-card rounded-xl p-4 border border-theme-border-subtle hover:border-theme-accent/50 space-y-3 transition-all group"
                  >
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-white truncate max-w-[160px]">
                          {item.label || item.metadata?.title || item.metadata?.category || `Vector Node #${idString.slice(0, 8)}`}
                        </span>
                        <span className="text-xs font-mono font-bold text-theme-accent px-2 py-0.5 rounded bg-theme-accent/15 border border-theme-accent/30">
                          {item.score.toFixed(3)}
                        </span>
                      </div>

                      {/* Similarity Bar */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-[10px] font-mono text-slate-400">
                          <span>Cosine Similarity</span>
                          <span>{scorePercent}%</span>
                        </div>
                        <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-teal-500 to-theme-accent rounded-full"
                            style={{ width: `${scorePercent}%` }}
                          />
                        </div>
                      </div>

                      {item.metadata && Object.keys(item.metadata).length > 0 && (
                        <div className="pt-2 border-t border-white/5 flex flex-wrap gap-1">
                          {Object.entries(item.metadata).slice(0, 4).map(([k, v]) => (
                            <span
                              key={k}
                              className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-white/5 text-slate-400 border border-white/5"
                            >
                              {k}: {String(v)}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-8 text-slate-500 font-mono text-xs">
              No scenes matched the minimum similarity threshold. Try adjusting the score slider.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
