import React, { useState, useRef, useEffect } from 'react';
import {
  Bot,
  Send,
  Upload,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  Clock,
  ChevronDown,
  ChevronUp,
  Loader2,
  Image as ImageIcon,
  Tag,
  ScanLine,
} from 'lucide-react';
import { geoVisionApi } from '@/services/api';
import type {
  ChatApiResponse,
  EvidenceExtractionResponse,
} from '@/types/api';

interface DisplayMessage {
  role: 'user' | 'assistant';
  content: string;
  responseMeta?: ChatApiResponse;
}

const SUGGESTED_PROMPTS = [
  'What objects are detected in this satellite scene?',
  'What is the dominant land-cover classification and total surface area?',
  'Provide a full grounded intelligence summary for this observation.',
  'What evidence supports the presence of infrastructure or vegetation?',
];

export const EarthAssistantPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Grounded Evidence extracted from the image
  const [evidenceData, setEvidenceData] = useState<EvidenceExtractionResponse | null>(null);
  const [extractingEvidence, setExtractingEvidence] = useState<boolean>(false);
  const [showEvidencePanel, setShowEvidencePanel] = useState<boolean>(false);

  // Chat conversation
  const [messages, setMessages] = useState<DisplayMessage[]>([
    {
      role: 'assistant',
      content:
        'Hello! I am the Grounded Earth AI Assistant. Upload a satellite scene, and I will extract computer-vision evidence (object detections, land-cover segments, spatial areas) to answer your questions with verified citations.',
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
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleImageUpload = (file: File) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setEvidenceData(null);
    setError(null);

    // Automatically trigger vision evidence extraction
    triggerEvidenceExtraction(file);
  };

  const triggerEvidenceExtraction = async (file: File) => {
    setExtractingEvidence(true);
    setError(null);
    try {
      const response = await geoVisionApi.extractEvidence(file, 1.0);
      setEvidenceData(response);
    } catch (err: any) {
      setError(err.message || 'Could not extract evidence from satellite scene.');
    } finally {
      setExtractingEvidence(false);
    }
  };

  const handleSendMessage = async () => {
    const query = inputQuery.trim();
    if (!query || chatLoading) return;

    const newDisplayMessages: DisplayMessage[] = [
      ...messages,
      { role: 'user', content: query },
    ];
    setMessages(newDisplayMessages);
    setInputQuery('');
    setChatLoading(true);
    setError(null);

    try {
      const chatResponse = await geoVisionApi.chatAssistant({
        query,
        context_evidence: evidenceData?.raw_evidence || null,
        provider: 'deterministic',
      });

      setMessages([
        ...newDisplayMessages,
        {
          role: 'assistant',
          content: chatResponse.answer,
          responseMeta: chatResponse,
        },
      ]);
    } catch (err: any) {
      setError(err.message || 'Assistant reasoning failed. Ensure the GeoVision backend is running.');
    } finally {
      setChatLoading(false);
    }
  };

  const resetChat = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setEvidenceData(null);
    setError(null);
    setMessages([
      {
        role: 'assistant',
        content:
          'Hello! I am the Grounded Earth AI Assistant. Upload a satellite scene, and I will extract computer-vision evidence to answer your questions with verified citations.',
      },
    ]);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="p-5 rounded-lg bg-space-900 border border-space-700/80 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-cyan-400 font-mono text-xs uppercase tracking-wider mb-1">
            <Bot className="w-4 h-4" />
            <span>Reasoning & VLM Core</span>
          </div>
          <h1 className="text-xl font-bold text-white font-sans">
            Grounded Earth AI Assistant
          </h1>
          <p className="text-xs text-slate-400 mt-1 font-sans">
            Conversational multimodal reasoning with automated computer vision evidence extraction and hallucination-free citation verification.
          </p>
        </div>

        {selectedFile && (
          <button
            onClick={resetChat}
            className="px-3 py-1.5 rounded bg-space-800 hover:bg-space-750 text-slate-400 hover:text-slate-200 text-xs font-mono flex items-center space-x-1.5 border border-space-700 self-start md:self-auto"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Session</span>
          </button>
        )}
      </div>

      {/* Main Grid: Scene + Evidence Inspector on Left, Chat on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Satellite Scene & Extracted Evidence */}
        <div className="lg:col-span-5 space-y-4">
          <div className="p-4 rounded-lg bg-space-900 border border-space-700 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase text-slate-300 flex items-center space-x-1.5">
                <ImageIcon className="w-3.5 h-3.5 text-cyan-400" />
                <span>Target Satellite Scene</span>
              </span>
              {selectedFile && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-space-800 text-cyan-300 border border-space-700 truncate max-w-[140px]">
                  {selectedFile.name}
                </span>
              )}
            </div>

            <div
              onClick={() => !previewUrl && fileInputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const f = e.dataTransfer.files?.[0];
                if (f) handleImageUpload(f);
              }}
              className={`rounded-lg border-2 border-dashed transition-all flex flex-col items-center justify-center min-h-[220px] bg-space-950 ${
                previewUrl
                  ? 'border-space-700 p-2'
                  : 'border-space-700/80 hover:border-cyan-500/50 p-6 cursor-pointer'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.tif,.tiff"
                onChange={(e) => e.target.files?.[0] && handleImageUpload(e.target.files[0])}
                className="hidden"
              />

              {previewUrl ? (
                <div className="relative flex items-center justify-center">
                  <img
                    src={previewUrl}
                    alt="Target Satellite Scene"
                    className="max-h-60 w-auto object-contain rounded select-none"
                  />
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center text-center space-y-2">
                  <div className="w-12 h-12 rounded-full bg-space-850 border border-space-700 flex items-center justify-center text-cyan-400">
                    <Upload className="w-5 h-5" />
                  </div>
                  <p className="text-xs font-semibold text-white">Upload Satellite Scene</p>
                  <p className="text-[10px] text-slate-400 font-mono">GeoTIFF, PNG, JPEG</p>
                </div>
              )}
            </div>
          </div>

          {/* Grounded Evidence Summary Card */}
          <div className="p-4 rounded-lg bg-space-900 border border-space-700 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase text-slate-300 flex items-center space-x-1.5">
                <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
                <span>Grounded Evidence Vault</span>
              </span>
              {extractingEvidence && (
                <div className="flex items-center space-x-1 text-[11px] font-mono text-cyan-400">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Extracting...</span>
                </div>
              )}
            </div>

            {evidenceData ? (
              <div className="space-y-3 text-xs font-mono">
                {evidenceData.summary_text && (
                  <div className="p-2.5 rounded bg-space-850 border border-space-750 text-slate-300 leading-relaxed font-sans text-xs">
                    {evidenceData.summary_text}
                  </div>
                )}

                {evidenceData.evidence_tags && evidenceData.evidence_tags.length > 0 && (
                  <div className="space-y-1.5">
                    <span className="text-[10px] text-slate-400 uppercase">Extracted CV Tags:</span>
                    <div className="flex flex-wrap gap-1.5">
                      {evidenceData.evidence_tags.map((tag, tIdx) => (
                        <span
                          key={tIdx}
                          className="px-2 py-0.5 rounded text-[10px] bg-space-800 text-cyan-300 border border-space-700 font-mono"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Collapsible raw structured evidence */}
                <button
                  type="button"
                  onClick={() => setShowEvidencePanel(!showEvidencePanel)}
                  className="w-full py-1.5 px-2 rounded bg-space-800 hover:bg-space-750 border border-space-700 text-slate-300 text-[11px] flex items-center justify-between transition-colors"
                >
                  <span>Raw Structured Evidence Inspector</span>
                  {showEvidencePanel ? (
                    <ChevronUp className="w-3.5 h-3.5 text-cyan-400" />
                  ) : (
                    <ChevronDown className="w-3.5 h-3.5 text-cyan-400" />
                  )}
                </button>

                {showEvidencePanel && (
                  <pre className="p-2.5 rounded bg-space-950 border border-space-750 text-[10px] text-cyan-300 overflow-x-auto max-h-48 font-mono">
                    {JSON.stringify(evidenceData.raw_evidence, null, 2)}
                  </pre>
                )}
              </div>
            ) : (
              <div className="p-4 text-center text-slate-400 space-y-1">
                <ScanLine className="w-6 h-6 text-slate-600 mx-auto" />
                <p className="text-xs font-mono">
                  {selectedFile
                    ? 'Extracting multi-modal evidence...'
                    : 'Upload an image to extract grounded evidence for the assistant.'}
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Right: Conversational Interface */}
        <div className="lg:col-span-7 flex flex-col h-[650px] rounded-lg bg-space-900 border border-space-700 overflow-hidden">
          {/* Messages Stream */}
          <div className="flex-1 p-4 overflow-y-auto space-y-4">
            {messages.map((msg, index) => {
              const isAssistant = msg.role === 'assistant';
              const meta = msg.responseMeta;
              return (
                <div
                  key={index}
                  className={`flex flex-col ${
                    isAssistant ? 'items-start' : 'items-end'
                  }`}
                >
                  <div
                    className={`max-w-[85%] rounded-lg p-4 space-y-2.5 ${
                      isAssistant
                        ? 'bg-space-850 border border-space-750 text-slate-100'
                        : 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-md'
                    }`}
                  >
                    <div className="flex items-center space-x-2 text-[11px] font-mono opacity-80">
                      {isAssistant ? (
                        <>
                          <Bot className="w-3.5 h-3.5 text-cyan-400" />
                          <span className="font-semibold text-cyan-300">GeoVision Earth Assistant</span>
                        </>
                      ) : (
                        <span className="font-semibold">Operator Query</span>
                      )}
                    </div>

                    {/* Text Message Content */}
                    <div className="text-xs leading-relaxed whitespace-pre-wrap font-sans">
                      {msg.content}
                    </div>

                    {/* Grounding & Citation Evidence Telemetry */}
                    {isAssistant && meta && (
                      <div className="pt-2.5 mt-2 border-t border-space-750/80 space-y-2">
                        {/* Citation Score Pill */}
                        <div className="flex items-center justify-between flex-wrap gap-2 text-[10px] font-mono">
                          <div className="flex items-center space-x-1.5">
                            {meta.is_grounded ? (
                              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                            ) : (
                              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                            )}
                            <span className="text-slate-300 font-semibold">
                              Citation Grounding:
                            </span>
                            <span
                              className={`font-bold ${
                                meta.is_grounded ? 'text-emerald-300' : 'text-amber-300'
                              }`}
                            >
                              {(meta.grounding_score * 100).toFixed(0)}% Verified
                            </span>
                          </div>

                          <div className="flex items-center space-x-1 text-slate-400">
                            <Clock className="w-3 h-3 text-cyan-400" />
                            <span>{meta.latency_ms.toFixed(1)} ms</span>
                          </div>
                        </div>

                        {/* Citation tags */}
                        {meta.citations && meta.citations.length > 0 && (
                          <div className="flex flex-wrap gap-1.5 pt-1">
                            {meta.citations.map((cite, cIdx) => (
                              <span
                                key={cIdx}
                                className="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-950/60 text-cyan-300 border border-cyan-800/60 flex items-center space-x-1"
                              >
                                <Tag className="w-2.5 h-2.5" />
                                <span>[{cite}]</span>
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {chatLoading && (
              <div className="flex items-start">
                <div className="max-w-[80%] rounded-lg p-4 bg-space-850 border border-space-750 space-y-2">
                  <div className="flex items-center space-x-2 text-xs font-mono text-cyan-400">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Grounding response in satellite vision evidence...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={chatBottomRef} />
          </div>

          {/* Quick Prompts Bar */}
          <div className="p-2 border-t border-space-800 bg-space-950/70 overflow-x-auto">
            <div className="flex items-center space-x-2 min-w-max">
              <span className="text-[10px] font-mono text-slate-400 pl-1">Suggested:</span>
              {SUGGESTED_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => setInputQuery(prompt)}
                  className="px-2.5 py-1 rounded bg-space-850 hover:bg-space-800 border border-space-750 text-[11px] text-slate-300 hover:text-cyan-300 font-sans transition-colors"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>

          {/* Error Alert */}
          {error && (
            <div className="p-3 bg-rose-950/40 border-t border-rose-800/80 text-rose-300 text-xs font-mono flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Input Box */}
          <div className="p-3 border-t border-space-750 bg-space-900 flex items-center space-x-2">
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Ask the Earth AI Assistant about this scene..."
              disabled={chatLoading}
              className="flex-1 px-4 py-2.5 bg-space-950 border border-space-700 rounded-md text-xs text-slate-100 placeholder-slate-500 font-sans focus:outline-none focus:border-cyan-500"
            />
            <button
              onClick={handleSendMessage}
              disabled={chatLoading || !inputQuery.trim()}
              className={`p-2.5 rounded-md text-xs font-mono font-bold transition-all ${
                chatLoading || !inputQuery.trim()
                  ? 'bg-space-800 text-slate-500 border border-space-700 cursor-not-allowed'
                  : 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-md shadow-cyan-900/40'
              }`}
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
