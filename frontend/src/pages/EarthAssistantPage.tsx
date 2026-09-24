import React, { useState, useRef, useEffect } from 'react';
import {
  Bot,
  Send,
  Upload,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Loader2,
  Image as ImageIcon,
  Sparkles,
  Copy,
  Check,
} from 'lucide-react';
import { geoVisionApi } from '@/services/api';
import type {
  ChatApiResponse,
  EvidenceExtractionResponse,
} from '@/types/api';
import { SAMPLE_SATELLITE_SCENES, dataUrlToFile } from '@/utils/sampleImages';

interface DisplayMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  responseMeta?: ChatApiResponse;
}

const SUGGESTED_PROMPTS = [
  'What objects are detected in this satellite scene?',
  'What is the dominant land-cover classification and total surface area?',
  'Provide a full grounded intelligence summary for this observation.',
  'What evidence supports the presence of infrastructure or maritime vessels?',
];

export const EarthAssistantPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Grounded Evidence extracted from the image
  const [evidenceData, setEvidenceData] = useState<EvidenceExtractionResponse | null>(null);
  const [extractingEvidence, setExtractingEvidence] = useState<boolean>(false);
  const [showEvidencePanel, setShowEvidencePanel] = useState<boolean>(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Chat conversation
  const [messages, setMessages] = useState<DisplayMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content:
        'Welcome to the Grounded Earth AI Assistant. Upload a satellite scene or pick a preset scenario, and I will extract computer-vision evidence (optical detections, land-cover segments, spatial areas) to answer your queries with verified, grounded spatial citations.',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [inputQuery, setInputQuery] = useState<string>('');
  const [chatLoading, setChatLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, chatLoading]);

  useEffect(() => {
    return () => {
      if (previewUrl && !previewUrl.startsWith('data:')) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleImageUpload = (file: File) => {
    if (previewUrl && !previewUrl.startsWith('data:')) URL.revokeObjectURL(previewUrl);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setEvidenceData(null);
    setError(null);
    triggerEvidenceExtraction(file);
  };

  const loadSampleSceneForAssistant = async (sceneId: string) => {
    const scene = SAMPLE_SATELLITE_SCENES.find((s) => s.id === sceneId);
    if (!scene) return;

    try {
      const file = await dataUrlToFile(scene.dataUrl, `${scene.id}.png`);
      setSelectedFile(file);
      setPreviewUrl(scene.dataUrl);
      setEvidenceData(null);
      setError(null);
      triggerEvidenceExtraction(file);
    } catch (err: any) {
      setError('Could not load sample scene: ' + err.message);
    }
  };

  const triggerEvidenceExtraction = async (file: File) => {
    setExtractingEvidence(true);
    setError(null);
    try {
      const response = await geoVisionApi.extractEvidence(file, 1.0);
      setEvidenceData(response);
    } catch (err: any) {
      setError(err.message || 'Could not extract vision evidence from satellite scene.');
    } finally {
      setExtractingEvidence(false);
    }
  };

  const handleSendMessage = async (queryText?: string) => {
    const query = (queryText || inputQuery).trim();
    if (!query || chatLoading) return;

    const userMsg: DisplayMessage = {
      id: String(Date.now()),
      role: 'user',
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputQuery('');
    setChatLoading(true);
    setError(null);

    try {
      const response = await geoVisionApi.chatAssistant({
        query: query,
        context_evidence: evidenceData?.raw_evidence || undefined,
      });

      const assistantMsg: DisplayMessage = {
        id: String(Date.now() + 1),
        role: 'assistant',
        content: response.answer || 'No analysis returned from intelligence engine.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        responseMeta: response,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      setError(err.message || 'Chat assistant engine failed. Ensure GeoVision backend is active.');
    } finally {
      setChatLoading(false);
    }
  };

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const clearChat = () => {
    setMessages([
      {
        id: 'welcome',
        role: 'assistant',
        content: 'Conversation session cleared. Ask a question about any loaded satellite scene.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      {/* Header */}
      <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-theme-accent font-mono text-xs uppercase tracking-wider font-semibold mb-1">
            <Bot className="w-4 h-4" />
            <span>Grounded VLM Intelligence Engine</span>
          </div>
          <h1 className="text-xl sm:text-2xl font-display font-bold text-white">
            Grounded Earth AI Assistant
          </h1>
          <p className="text-xs text-slate-300 mt-1 font-sans">
            Spatial reasoning conversational assistant backed by verifiable bounding box detections and land-cover calculations.
          </p>
        </div>

        <button
          onClick={clearChat}
          className="px-3.5 py-2 rounded-xl bg-white/5 hover:bg-white/10 text-slate-300 hover:text-white border border-white/10 text-xs font-mono flex items-center space-x-1.5 transition-colors self-start md:self-auto"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Clear Chat</span>
        </button>
      </div>

      {/* Main Grid: Left Evidence Viewport (5 cols) & Right Chat Stream (7 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Visual Grounding & Evidence (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Quick Scene Ribbon */}
          <div className="glass-panel rounded-2xl p-4 border border-theme-border/60 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono text-slate-300">
              <span className="font-semibold flex items-center gap-1.5 text-white">
                <Sparkles className="w-3.5 h-3.5 text-theme-accent" />
                <span>Test Satellite Capture:</span>
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {SAMPLE_SATELLITE_SCENES.slice(0, 4).map((s) => (
                <button
                  key={s.id}
                  onClick={() => loadSampleSceneForAssistant(s.id)}
                  className="p-2 rounded-lg bg-white/5 hover:bg-theme-accent/20 border border-white/10 hover:border-theme-accent/40 text-[11px] font-mono text-slate-200 text-left truncate transition-all"
                >
                  {s.name}
                </button>
              ))}
            </div>
          </div>

          {/* Satellite Viewport Card */}
          <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase tracking-wider font-bold text-slate-300 flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-theme-accent" />
                <span>Observation Context</span>
              </span>
              {extractingEvidence && (
                <span className="text-[10px] font-mono text-theme-accent flex items-center gap-1">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  <span>Extracting Evidence...</span>
                </span>
              )}
            </div>

            <div
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all ${
                selectedFile ? 'border-theme-accent/60 bg-black/40' : 'border-white/15 hover:border-theme-accent/50 bg-black/20'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files?.[0] && handleImageUpload(e.target.files[0])}
                className="hidden"
              />
              {previewUrl ? (
                <div className="space-y-2">
                  <img src={previewUrl} alt="Context" className="h-44 w-full object-contain rounded-lg" />
                  <p className="text-[11px] font-mono text-slate-300 truncate">{selectedFile?.name}</p>
                </div>
              ) : (
                <div className="py-6">
                  <Upload className="w-7 h-7 text-theme-accent mx-auto mb-2" />
                  <p className="text-xs font-semibold text-slate-200">Load Observation for Spatial Q&A</p>
                  <p className="text-[10px] text-slate-400 mt-0.5">Auto-extracts verified bounding boxes</p>
                </div>
              )}
            </div>
          </div>

          {/* Extracted Grounded Evidence Breakdown */}
          {evidenceData && (
            <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 space-y-3 animate-in fade-in duration-200">
              <div
                onClick={() => setShowEvidencePanel(!showEvidencePanel)}
                className="flex items-center justify-between cursor-pointer"
              >
                <div className="flex items-center space-x-2 text-xs font-mono font-bold text-slate-200">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  <span>Extracted Vision Evidence ({evidenceData.evidence_tags?.length || 0} Tags)</span>
                </div>
                {showEvidencePanel ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </div>

              {showEvidencePanel && (
                <div className="space-y-3 pt-2 border-t border-white/5 text-xs font-mono">
                  {evidenceData.summary_text && (
                    <p className="text-slate-300 text-[11px] leading-relaxed">
                      {evidenceData.summary_text}
                    </p>
                  )}

                  {evidenceData.evidence_tags && evidenceData.evidence_tags.length > 0 && (
                    <div className="space-y-1.5">
                      <span className="text-[10px] text-slate-400 uppercase tracking-wider">Spatial Tags:</span>
                      <div className="flex flex-wrap gap-1.5">
                        {evidenceData.evidence_tags.map((tag: string, i: number) => (
                          <span
                            key={i}
                            className="text-[10px] px-2 py-0.5 rounded bg-white/5 border border-white/10 text-slate-300"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Chat Stream & Interactive Prompting (7 cols) */}
        <div className="lg:col-span-7 flex flex-col h-[650px] glass-panel rounded-2xl border border-theme-border/60 overflow-hidden">
          {/* Chat Messages Log */}
          <div className="flex-1 overflow-y-auto p-5 space-y-4">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex items-start space-x-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-xl bg-theme-accent/20 border border-theme-accent/40 flex items-center justify-center text-theme-accent shrink-0 mt-0.5 shadow-glow-sm">
                    <Bot className="w-4 h-4" />
                  </div>
                )}

                <div
                  className={`max-w-[85%] rounded-2xl p-4 text-xs font-sans leading-relaxed relative group ${
                    msg.role === 'user'
                      ? 'bg-theme-accent text-slate-950 font-medium ml-12 rounded-tr-sm'
                      : 'bg-black/50 border border-white/10 text-slate-200 rounded-tl-sm shadow-panel'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{msg.content}</p>

                  {/* Grounded Citation Badges */}
                  {msg.responseMeta?.citations && msg.responseMeta.citations.length > 0 && (
                    <div className="mt-3 pt-2.5 border-t border-white/10 space-y-1.5">
                      <span className="text-[10px] font-mono text-emerald-400 font-semibold uppercase tracking-wider flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3" /> Verified Citations:
                      </span>
                      <div className="flex flex-wrap gap-1.5">
                        {msg.responseMeta.citations.map((c, i) => (
                          <span
                            key={i}
                            className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/50 border border-emerald-500/30 text-emerald-300"
                          >
                            {typeof c === 'string' ? c : JSON.stringify(c)}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Timestamp & Copy */}
                  <div
                    className={`mt-2 flex items-center justify-between text-[10px] font-mono ${
                      msg.role === 'user' ? 'text-slate-800' : 'text-slate-400'
                    }`}
                  >
                    <span>{msg.timestamp}</span>
                    <button
                      onClick={() => handleCopy(msg.content, msg.id)}
                      className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:text-white"
                      title="Copy response"
                    >
                      {copiedId === msg.id ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                    </button>
                  </div>
                </div>
              </div>
            ))}

            {chatLoading && (
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 rounded-xl bg-theme-accent/20 border border-theme-accent/40 flex items-center justify-center text-theme-accent shrink-0">
                  <Bot className="w-4 h-4 animate-pulse" />
                </div>
                <div className="p-3.5 rounded-2xl rounded-tl-sm bg-black/50 border border-white/10 text-xs font-mono text-theme-accent flex items-center space-x-2">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Synthesizing grounded spatial reasoning...</span>
                </div>
              </div>
            )}
            <div ref={chatBottomRef} />
          </div>

          {/* Suggested Quick Prompt Carousel */}
          <div className="px-4 py-2 border-t border-white/5 bg-black/30 overflow-x-auto flex items-center gap-2 text-xs">
            {SUGGESTED_PROMPTS.map((p, idx) => (
              <button
                key={idx}
                onClick={() => handleSendMessage(p)}
                disabled={chatLoading}
                className="px-2.5 py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-[11px] text-slate-300 hover:text-white whitespace-nowrap transition-colors"
              >
                {p}
              </button>
            ))}
          </div>

          {/* Input Box */}
          <div className="p-4 bg-black/40 border-t border-theme-border-subtle">
            {error && (
              <div className="mb-2 p-2 rounded-lg bg-rose-950/40 border border-rose-500/30 text-rose-300 text-[11px] flex items-center gap-2">
                <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <div className="flex items-center space-x-2">
              <input
                type="text"
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder="Ask Earth AI about detected infrastructure, vessels, or spatial land-cover..."
                className="flex-1 glass-input rounded-xl px-4 py-3 text-xs placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-theme-accent"
              />
              <button
                onClick={() => handleSendMessage()}
                disabled={chatLoading || !inputQuery.trim()}
                className={`p-3 rounded-xl transition-all shadow-glow-sm ${
                  chatLoading || !inputQuery.trim()
                    ? 'bg-white/10 text-slate-500 cursor-not-allowed'
                    : 'bg-theme-accent text-slate-950 hover:opacity-95'
                }`}
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
