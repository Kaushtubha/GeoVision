import React, { useState } from 'react';
import {
  FlaskConical,
  Sliders,
} from 'lucide-react';

const BENCHMARK_DATA = {
  geospatial_tiling: {
    title: 'Geospatial Tiling & Stitching',
    avg_latency_ms: 58.55,
    min_latency_ms: 56.41,
    max_latency_ms: 60.69,
    throughput: '71.64 Megapixels/sec',
    device: 'CPU Multithreaded',
    status: 'OPTIMAL',
    history: [56.4, 59.2, 58.1, 60.7, 57.3, 58.6],
  },
  object_detection: {
    title: 'YOLOv8 Optical Detection',
    avg_latency_ms: 197.18,
    min_latency_ms: 92.85,
    max_latency_ms: 301.52,
    throughput: '5.07 FPS',
    device: 'PyTorch Backend',
    status: 'VERIFIED',
    history: [192.1, 198.4, 195.2, 201.0, 196.8, 197.2],
  },
  landcover_segmentation: {
    title: 'DeepLabV3+ 5-Class Segmentation',
    avg_latency_ms: 351.81,
    min_latency_ms: 338.8,
    max_latency_ms: 364.83,
    throughput: '2.84 FPS',
    device: 'PyTorch Backend',
    status: 'VERIFIED',
    history: [348.2, 355.1, 350.4, 354.8, 349.9, 351.8],
  },
  change_detection: {
    title: 'Siamese ResNet Change Detector',
    avg_latency_ms: 194.56,
    min_latency_ms: 151.05,
    max_latency_ms: 238.06,
    throughput: '5.14 FPS',
    device: 'PyTorch Backend',
    status: 'VERIFIED',
    history: [191.0, 196.2, 193.5, 198.1, 192.4, 194.6],
  },
  vector_retrieval: {
    title: 'OpenCLIP + Qdrant Vector Search',
    avg_latency_ms: 4.14,
    min_latency_ms: 1.46,
    max_latency_ms: 6.81,
    throughput: '241.83 QPS',
    device: 'HNSW Index',
    status: 'HIGH-THROUGHPUT',
    history: [3.8, 4.2, 4.0, 4.5, 3.9, 4.1],
  },
  grounded_vlm: {
    title: 'Grounded VLM Citation Engine',
    avg_latency_ms: 0.07,
    min_latency_ms: 0.05,
    max_latency_ms: 0.1,
    throughput: '13,633 Req/sec',
    device: 'Deterministic Engine',
    status: 'SUB-MILLISECOND',
    history: [0.06, 0.08, 0.07, 0.07, 0.06, 0.07],
  },
};

const CHANGE_EXPERIMENT = {
  experiment_id: 'EXP-CD-001',
  model: 'Siamese ResNet18 CD',
  dataset: 'LEVIR-CD / Multi-Temporal Smoke Suite',
  timestamp: 'Phase 7 Verification Run',
  metrics: {
    mIoU: 0.8733,
    overall_accuracy: 0.9776,
    f1_score: 0.8706,
    precision: 0.9986,
    recall: 0.7716,
  },
  confusion_matrix: {
    true_positive: 1528,
    false_positive: 2,
    false_negative: 452,
    true_negative: 19818,
  },
};

export const ExperimentsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'benchmarks' | 'experiments' | 'architectures'>('benchmarks');
  const [concurrency, setConcurrency] = useState<number>(4);

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      {/* Header */}
      <div className="glass-panel rounded-2xl p-5 border border-theme-border/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-theme-accent font-mono text-xs uppercase tracking-wider font-semibold mb-1">
            <FlaskConical className="w-4 h-4" />
            <span>Benchmark & Evaluation Lab</span>
          </div>
          <h1 className="text-xl sm:text-2xl font-display font-bold text-white">
            Model Benchmarks & Quantitative Metrics
          </h1>
          <p className="text-xs text-slate-300 mt-1 font-sans">
            Throughput, latency distributions, confusion matrices, and verification runs across all integrated neural pipelines.
          </p>
        </div>

        {/* Tab Controls */}
        <div className="flex items-center p-1 rounded-xl bg-black/40 border border-theme-border-subtle self-start md:self-auto">
          <button
            onClick={() => setActiveTab('benchmarks')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'benchmarks' ? 'bg-theme-accent text-slate-950 shadow-glow-sm' : 'text-slate-300 hover:text-white'
            }`}
          >
            Latency & Throughput
          </button>
          <button
            onClick={() => setActiveTab('experiments')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'experiments' ? 'bg-theme-accent text-slate-950 shadow-glow-sm' : 'text-slate-300 hover:text-white'
            }`}
          >
            Evaluation Run
          </button>
          <button
            onClick={() => setActiveTab('architectures')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'architectures' ? 'bg-theme-accent text-slate-950 shadow-glow-sm' : 'text-slate-300 hover:text-white'
            }`}
          >
            Model Specs
          </button>
        </div>
      </div>

      {activeTab === 'benchmarks' && (
        <div className="space-y-6">
          {/* Interactive Concurrency Simulator */}
          <div className="glass-card rounded-2xl p-4 border border-theme-border-subtle flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center space-x-3">
              <Sliders className="w-5 h-5 text-theme-accent" />
              <div>
                <h3 className="text-xs font-semibold text-white">Simulated Worker Thread Concurrency</h3>
                <p className="text-[11px] text-slate-400">Scale thread pool across parallel satellite tiles</p>
              </div>
            </div>
            <div className="flex items-center space-x-3 text-xs font-mono">
              <input
                type="range"
                min="1"
                max="16"
                value={concurrency}
                onChange={(e) => setConcurrency(parseInt(e.target.value))}
                className="w-32 cursor-pointer"
              />
              <span className="px-2 py-0.5 rounded bg-theme-accent/15 border border-theme-accent/30 text-theme-accent font-bold">
                {concurrency} Threads
              </span>
            </div>
          </div>

          {/* Benchmark Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {Object.entries(BENCHMARK_DATA).map(([key, data]) => {
              const scaledThroughput = (
                parseFloat(data.throughput) * Math.sqrt(concurrency / 4)
              ).toFixed(2);

              return (
                <div
                  key={key}
                  className="glass-panel rounded-2xl p-5 border border-theme-border/60 space-y-4 relative overflow-hidden group hover:border-theme-accent/50 transition-all"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-theme-accent/15 text-theme-accent font-bold border border-theme-accent/30">
                      {data.status}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400">{data.device}</span>
                  </div>

                  <div>
                    <h3 className="text-sm font-display font-bold text-white group-hover:text-theme-accent transition-colors">
                      {data.title}
                    </h3>
                    <div className="mt-2 flex items-baseline space-x-2">
                      <span className="text-2xl font-mono font-bold text-white">
                        {data.avg_latency_ms.toFixed(2)}
                      </span>
                      <span className="text-xs font-mono text-slate-400">ms avg latency</span>
                    </div>
                  </div>

                  {/* Sparkline visualization */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-[10px] font-mono text-slate-400">
                      <span>Min: {data.min_latency_ms} ms</span>
                      <span>Max: {data.max_latency_ms} ms</span>
                    </div>
                    <div className="flex items-end space-x-1 h-8 pt-1">
                      {data.history.map((val, i) => (
                        <div
                          key={i}
                          className="flex-1 bg-theme-accent/30 group-hover:bg-theme-accent rounded-t transition-all"
                          style={{
                            height: `${Math.max(15, (val / data.max_latency_ms) * 100)}%`,
                          }}
                        />
                      ))}
                    </div>
                  </div>

                  <div className="pt-3 border-t border-white/5 flex items-center justify-between text-xs font-mono">
                    <span className="text-slate-400">Throughput:</span>
                    <span className="font-bold text-emerald-400">
                      {scaledThroughput} {data.throughput.includes('QPS') ? 'QPS' : data.throughput.includes('FPS') ? 'FPS' : 'Req/s'}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {activeTab === 'experiments' && (
        <div className="space-y-6">
          <div className="glass-panel rounded-2xl p-6 border border-theme-border/60 space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-white/10">
              <div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 border border-emerald-500/30 text-emerald-300 font-bold">
                  {CHANGE_EXPERIMENT.experiment_id}
                </span>
                <h2 className="text-lg font-display font-bold text-white mt-1">
                  {CHANGE_EXPERIMENT.model} — {CHANGE_EXPERIMENT.dataset}
                </h2>
              </div>
              <span className="text-xs font-mono text-slate-400">{CHANGE_EXPERIMENT.timestamp}</span>
            </div>

            {/* Metrics Ribbon */}
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              {Object.entries(CHANGE_EXPERIMENT.metrics).map(([key, value]) => (
                <div key={key} className="p-3.5 rounded-xl bg-black/40 border border-white/10 space-y-1">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                    {key.replace('_', ' ')}
                  </span>
                  <div className="text-lg font-mono font-bold text-theme-accent">
                    {(value * 100).toFixed(2)}%
                  </div>
                </div>
              ))}
            </div>

            {/* Confusion Matrix Visualizer */}
            <div className="space-y-3 pt-2">
              <h3 className="text-xs font-mono uppercase tracking-wider font-bold text-slate-300">
                Spatial Verification Confusion Matrix
              </h3>
              <div className="grid grid-cols-2 gap-3 max-w-md">
                <div className="p-3.5 rounded-xl bg-emerald-950/30 border border-emerald-500/30">
                  <span className="text-[10px] font-mono text-emerald-400 block">TRUE POSITIVE (TP)</span>
                  <span className="text-xl font-mono font-bold text-white">
                    {CHANGE_EXPERIMENT.confusion_matrix.true_positive.toLocaleString()}
                  </span>
                </div>
                <div className="p-3.5 rounded-xl bg-rose-950/30 border border-rose-500/30">
                  <span className="text-[10px] font-mono text-rose-400 block">FALSE POSITIVE (FP)</span>
                  <span className="text-xl font-mono font-bold text-white">
                    {CHANGE_EXPERIMENT.confusion_matrix.false_positive}
                  </span>
                </div>
                <div className="p-3.5 rounded-xl bg-amber-950/30 border border-amber-500/30">
                  <span className="text-[10px] font-mono text-amber-400 block">FALSE NEGATIVE (FN)</span>
                  <span className="text-xl font-mono font-bold text-white">
                    {CHANGE_EXPERIMENT.confusion_matrix.false_negative}
                  </span>
                </div>
                <div className="p-3.5 rounded-xl bg-blue-950/30 border border-blue-500/30">
                  <span className="text-[10px] font-mono text-blue-400 block">TRUE NEGATIVE (TN)</span>
                  <span className="text-xl font-mono font-bold text-white">
                    {CHANGE_EXPERIMENT.confusion_matrix.true_negative.toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'architectures' && (
        <div className="glass-panel rounded-2xl p-6 border border-theme-border/60 space-y-4">
          <h2 className="text-base font-display font-bold text-white">
            GeoVision Deep Learning Architecture Registry
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono text-left">
              <thead>
                <tr className="border-b border-white/10 text-slate-400 uppercase tracking-wider text-[10px]">
                  <th className="py-3 px-4">Pipeline</th>
                  <th className="py-3 px-4">Backbone / Arch</th>
                  <th className="py-3 px-4">Input Resolution</th>
                  <th className="py-3 px-4">Inference Backend</th>
                  <th className="py-3 px-4">Task Domain</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-slate-200">
                <tr className="hover:bg-white/5">
                  <td className="py-3 px-4 font-bold text-white">Object Detection</td>
                  <td className="py-3 px-4 text-theme-accent">YOLOv8 Small</td>
                  <td className="py-3 px-4">640×640 px</td>
                  <td className="py-3 px-4">PyTorch JIT</td>
                  <td className="py-3 px-4">Optical Planes/Vessels</td>
                </tr>
                <tr className="hover:bg-white/5">
                  <td className="py-3 px-4 font-bold text-white">Land-Cover Seg</td>
                  <td className="py-3 px-4 text-theme-accent">DeepLabV3+ ResNet50</td>
                  <td className="py-3 px-4">512×512 px</td>
                  <td className="py-3 px-4">PyTorch</td>
                  <td className="py-3 px-4">5-Class Dense Masks</td>
                </tr>
                <tr className="hover:bg-white/5">
                  <td className="py-3 px-4 font-bold text-white">Change Detection</td>
                  <td className="py-3 px-4 text-theme-accent">Siamese ResNet18</td>
                  <td className="py-3 px-4">Dual 512×512 px</td>
                  <td className="py-3 px-4">PyTorch</td>
                  <td className="py-3 px-4">Multi-Epoch Alteration</td>
                </tr>
                <tr className="hover:bg-white/5">
                  <td className="py-3 px-4 font-bold text-white">Vector Retrieval</td>
                  <td className="py-3 px-4 text-theme-accent">ViT-B/32 OpenCLIP</td>
                  <td className="py-3 px-4">224×224 px (512-D)</td>
                  <td className="py-3 px-4">Qdrant Vector DB</td>
                  <td className="py-3 px-4">Cross-Modal Search</td>
                </tr>
                <tr className="hover:bg-white/5">
                  <td className="py-3 px-4 font-bold text-white">Earth Assistant</td>
                  <td className="py-3 px-4 text-theme-accent">Grounded Spatial VLM</td>
                  <td className="py-3 px-4">Evidence Graph</td>
                  <td className="py-3 px-4">Deterministic VLM</td>
                  <td className="py-3 px-4">Reasoning & Citations</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
