import React, { useState } from 'react';
import {
  FlaskConical,
  GitCompare,
  Layers,
  Search,
  CheckCircle2,
  Gauge,
} from 'lucide-react';

// Real benchmark data from benchmark_report.json
const BENCHMARK_DATA = {
  geospatial_tiling: {
    title: 'Geospatial Tiling & Stitching',
    avg_latency_ms: 58.55,
    min_latency_ms: 56.41,
    max_latency_ms: 60.69,
    throughput: '71.64 Megapixels/sec',
    device: 'CPU Multithreaded',
    status: 'OPTIMAL',
  },
  object_detection: {
    title: 'YOLOv8 Optical Detection',
    avg_latency_ms: 197.18,
    min_latency_ms: 92.85,
    max_latency_ms: 301.52,
    throughput: '5.07 FPS',
    device: 'PyTorch Backend',
    status: 'VERIFIED',
  },
  landcover_segmentation: {
    title: 'DeepLabV3+ 5-Class Segmentation',
    avg_latency_ms: 351.81,
    min_latency_ms: 338.8,
    max_latency_ms: 364.83,
    throughput: '2.84 FPS',
    device: 'PyTorch Backend',
    status: 'VERIFIED',
  },
  change_detection: {
    title: 'Siamese ResNet Change Detector',
    avg_latency_ms: 194.56,
    min_latency_ms: 151.05,
    max_latency_ms: 238.06,
    throughput: '5.14 FPS',
    device: 'PyTorch Backend',
    status: 'VERIFIED',
  },
  vector_retrieval: {
    title: 'OpenCLIP + Qdrant Vector Search',
    avg_latency_ms: 4.14,
    min_latency_ms: 1.46,
    max_latency_ms: 6.81,
    throughput: '241.83 QPS',
    device: 'HNSW Index',
    status: 'HIGH-THROUGHPUT',
  },
  grounded_vlm: {
    title: 'Grounded VLM Citation Engine',
    avg_latency_ms: 0.07,
    min_latency_ms: 0.05,
    max_latency_ms: 0.1,
    throughput: '13,633 Req/sec',
    device: 'Deterministic Engine',
    status: 'SUB-MILLISECOND',
  },
};

// Real change detection experiment evaluation from experiments/runs/change/smoke_eval.json
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
    change_iou: 0.7708,
    no_change_iou: 0.9758,
  },
  confusion_matrix: {
    tp: 98769,
    fp: 140,
    tn: 1182580,
    fn: 29231,
  },
};

// Real segmentation experiment evaluation from experiments/runs/segmentation/smoke_eval.json
const SEGMENTATION_EXPERIMENT = {
  experiment_id: 'EXP-SEG-001',
  model: 'DeepLabV3+ ResNet34',
  dataset: 'LandCover.ai 5-Class Benchmark',
  timestamp: 'Phase 7 Verification Run',
  metrics: {
    mIoU: 0.0229,
    mean_dice: 0.0411,
    overall_accuracy: 0.1144,
    loss: 1.2357,
  },
  per_class_iou: {
    Woodland: 0.1144,
    Background: 0.0,
    Building: 0.0,
    Water: 0.0,
    Road: 0.0,
  },
  per_class_dice: {
    Woodland: 0.2054,
    Background: 0.0,
    Building: 0.0,
    Water: 0.0,
    Road: 0.0,
  },
};

// Real retrieval experiment evaluation from experiments/runs/retrieval/smoke_eval.json
const RETRIEVAL_EXPERIMENT = {
  experiment_id: 'EXP-RET-001',
  model: 'OpenCLIP ViT-B/32 Satellite Embedder',
  dataset: 'EuroSAT Multimodal Evaluation (50 Scenes)',
  timestamp: 'Phase 7 Verification Run',
  metrics: {
    'Recall@1': 0.06,
    'Recall@5': 0.34,
    'Recall@10': 0.66,
    'Zero-Shot Top-1': 0.16,
    'Zero-Shot Top-5': 0.54,
    MRR: 0.2038,
    mAP: 0.14,
  },
};

type ActiveTab = 'benchmarks' | 'change' | 'segmentation' | 'retrieval';

export const ExperimentsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('benchmarks');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="p-5 rounded-lg bg-space-900 border border-space-700/80 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-cyan-400 font-mono text-xs uppercase tracking-wider mb-1">
            <FlaskConical className="w-4 h-4" />
            <span>Evaluation & Benchmark Telemetry</span>
          </div>
          <h1 className="text-xl font-bold text-white font-sans">
            Experiments & Model Metrics
          </h1>
          <p className="text-xs text-slate-400 mt-1 font-sans">
            Real benchmark results, latency distributions, IoU/Dice evaluation curves, and retrieval recall from verified training runs.
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center p-1 rounded-lg bg-space-950 border border-space-800 self-start md:self-auto overflow-x-auto">
          <button
            onClick={() => setActiveTab('benchmarks')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all shrink-0 ${
              activeTab === 'benchmarks'
                ? 'bg-cyan-600 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Gauge className="w-3.5 h-3.5" />
            <span>Pipeline Benchmarks</span>
          </button>
          <button
            onClick={() => setActiveTab('change')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all shrink-0 ${
              activeTab === 'change'
                ? 'bg-cyan-600 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <GitCompare className="w-3.5 h-3.5" />
            <span>Change Detection</span>
          </button>
          <button
            onClick={() => setActiveTab('segmentation')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all shrink-0 ${
              activeTab === 'segmentation'
                ? 'bg-cyan-600 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Segmentation</span>
          </button>
          <button
            onClick={() => setActiveTab('retrieval')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all shrink-0 ${
              activeTab === 'retrieval'
                ? 'bg-cyan-600 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Search className="w-3.5 h-3.5" />
            <span>Vector Retrieval</span>
          </button>
        </div>
      </div>

      {/* Tab 1: System Benchmarks */}
      {activeTab === 'benchmarks' && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(BENCHMARK_DATA).map(([key, item]) => (
              <div
                key={key}
                className="p-4 rounded-lg bg-space-900 border border-space-700 hover:border-cyan-500/40 transition-colors space-y-3"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-semibold text-white">
                    {item.title}
                  </span>
                  <span className="px-2 py-0.5 rounded text-[9px] font-mono bg-emerald-950/60 text-emerald-300 border border-emerald-800/60 font-bold">
                    {item.status}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 pt-1 border-t border-space-800">
                  <div className="p-2 rounded bg-space-850 border border-space-750">
                    <p className="text-[10px] font-mono text-slate-400 uppercase">Avg Latency</p>
                    <p className="text-base font-mono font-bold text-cyan-300 mt-0.5">
                      {item.avg_latency_ms} <span className="text-[10px] text-slate-400">ms</span>
                    </p>
                  </div>
                  <div className="p-2 rounded bg-space-850 border border-space-750">
                    <p className="text-[10px] font-mono text-slate-400 uppercase">Throughput</p>
                    <p className="text-xs font-mono font-bold text-emerald-300 mt-1 truncate">
                      {item.throughput}
                    </p>
                  </div>
                </div>

                <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 pt-1">
                  <span>Range: {item.min_latency_ms} - {item.max_latency_ms} ms</span>
                  <span className="text-slate-300">{item.device}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab 2: Change Detection Experiment */}
      {activeTab === 'change' && (
        <div className="p-5 rounded-lg bg-space-900 border border-space-700 space-y-5">
          <div className="flex items-center justify-between border-b border-space-750 pb-3 flex-wrap gap-2">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                  {CHANGE_EXPERIMENT.experiment_id}
                </span>
                <h3 className="text-sm font-semibold text-white">
                  {CHANGE_EXPERIMENT.model}
                </h3>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-1">
                Dataset: {CHANGE_EXPERIMENT.dataset} • {CHANGE_EXPERIMENT.timestamp}
              </p>
            </div>
            <span className="px-2.5 py-1 rounded text-xs font-mono bg-emerald-950/60 text-emerald-300 border border-emerald-800/60 flex items-center space-x-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>VALIDATED RUN</span>
            </span>
          </div>

          {/* Metric Cards Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3.5 rounded bg-space-850 border border-space-750">
              <p className="text-[10px] font-mono text-slate-400 uppercase">mIoU Score</p>
              <p className="text-xl font-mono font-bold text-cyan-300 mt-1">
                {(CHANGE_EXPERIMENT.metrics.mIoU * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-3.5 rounded bg-space-850 border border-space-750">
              <p className="text-[10px] font-mono text-slate-400 uppercase">F1-Score</p>
              <p className="text-xl font-mono font-bold text-emerald-300 mt-1">
                {(CHANGE_EXPERIMENT.metrics.f1_score * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-3.5 rounded bg-space-850 border border-space-750">
              <p className="text-[10px] font-mono text-slate-400 uppercase">Precision</p>
              <p className="text-xl font-mono font-bold text-white mt-1">
                {(CHANGE_EXPERIMENT.metrics.precision * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-3.5 rounded bg-space-850 border border-space-750">
              <p className="text-[10px] font-mono text-slate-400 uppercase">Overall Accuracy</p>
              <p className="text-xl font-mono font-bold text-white mt-1">
                {(CHANGE_EXPERIMENT.metrics.overall_accuracy * 100).toFixed(2)}%
              </p>
            </div>
          </div>

          {/* Confusion Matrix Table */}
          <div className="space-y-2">
            <p className="text-xs font-mono uppercase text-slate-300">
              Binary Change Confusion Matrix (Pixel Counts)
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
              <div className="p-3 rounded bg-space-950 border border-space-800">
                <span className="text-slate-400">True Positives (TP):</span>
                <p className="text-emerald-300 font-bold mt-0.5">
                  {CHANGE_EXPERIMENT.confusion_matrix.tp.toLocaleString()}
                </p>
              </div>
              <div className="p-3 rounded bg-space-950 border border-space-800">
                <span className="text-slate-400">False Positives (FP):</span>
                <p className="text-rose-400 font-bold mt-0.5">
                  {CHANGE_EXPERIMENT.confusion_matrix.fp.toLocaleString()}
                </p>
              </div>
              <div className="p-3 rounded bg-space-950 border border-space-800">
                <span className="text-slate-400">True Negatives (TN):</span>
                <p className="text-slate-200 font-bold mt-0.5">
                  {CHANGE_EXPERIMENT.confusion_matrix.tn.toLocaleString()}
                </p>
              </div>
              <div className="p-3 rounded bg-space-950 border border-space-800">
                <span className="text-slate-400">False Negatives (FN):</span>
                <p className="text-amber-400 font-bold mt-0.5">
                  {CHANGE_EXPERIMENT.confusion_matrix.fn.toLocaleString()}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: Segmentation Experiment */}
      {activeTab === 'segmentation' && (
        <div className="p-5 rounded-lg bg-space-900 border border-space-700 space-y-5">
          <div className="flex items-center justify-between border-b border-space-750 pb-3 flex-wrap gap-2">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                  {SEGMENTATION_EXPERIMENT.experiment_id}
                </span>
                <h3 className="text-sm font-semibold text-white">
                  {SEGMENTATION_EXPERIMENT.model}
                </h3>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-1">
                Dataset: {SEGMENTATION_EXPERIMENT.dataset} • {SEGMENTATION_EXPERIMENT.timestamp}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3.5 rounded bg-space-850 border border-space-750">
              <p className="text-[10px] font-mono text-slate-400 uppercase">mIoU</p>
              <p className="text-xl font-mono font-bold text-cyan-300 mt-1">
                {(SEGMENTATION_EXPERIMENT.metrics.mIoU * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-3.5 rounded bg-space-850 border border-space-750">
              <p className="text-[10px] font-mono text-slate-400 uppercase">Mean Dice</p>
              <p className="text-xl font-mono font-bold text-emerald-300 mt-1">
                {(SEGMENTATION_EXPERIMENT.metrics.mean_dice * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-3.5 rounded bg-space-850 border border-space-750">
              <p className="text-[10px] font-mono text-slate-400 uppercase">Overall Accuracy</p>
              <p className="text-xl font-mono font-bold text-white mt-1">
                {(SEGMENTATION_EXPERIMENT.metrics.overall_accuracy * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-3.5 rounded bg-space-850 border border-space-750">
              <p className="text-[10px] font-mono text-slate-400 uppercase">Loss</p>
              <p className="text-xl font-mono font-bold text-slate-200 mt-1">
                {SEGMENTATION_EXPERIMENT.metrics.loss.toFixed(4)}
              </p>
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-xs font-mono uppercase text-slate-300">
              Class-Wise IoU & Dice Breakdown
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
              {Object.entries(SEGMENTATION_EXPERIMENT.per_class_iou).map(([cls, iou]) => (
                <div key={cls} className="p-3 rounded bg-space-850 border border-space-750">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-white">{cls}</span>
                    <span className="text-cyan-300">{(iou * 100).toFixed(1)}% IoU</span>
                  </div>
                  <p className="text-[10px] text-slate-400">
                    Dice: {((SEGMENTATION_EXPERIMENT.per_class_dice[cls as keyof typeof SEGMENTATION_EXPERIMENT.per_class_dice] || 0) * 100).toFixed(1)}%
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Tab 4: Vector Retrieval Experiment */}
      {activeTab === 'retrieval' && (
        <div className="p-5 rounded-lg bg-space-900 border border-space-700 space-y-5">
          <div className="flex items-center justify-between border-b border-space-750 pb-3 flex-wrap gap-2">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                  {RETRIEVAL_EXPERIMENT.experiment_id}
                </span>
                <h3 className="text-sm font-semibold text-white">
                  {RETRIEVAL_EXPERIMENT.model}
                </h3>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-1">
                Dataset: {RETRIEVAL_EXPERIMENT.dataset} • {RETRIEVAL_EXPERIMENT.timestamp}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {Object.entries(RETRIEVAL_EXPERIMENT.metrics).map(([name, val]) => (
              <div key={name} className="p-3.5 rounded bg-space-850 border border-space-750">
                <p className="text-[10px] font-mono text-slate-400 uppercase">{name}</p>
                <p className="text-xl font-mono font-bold text-cyan-300 mt-1">
                  {typeof val === 'number' ? (val < 1 ? (val * 100).toFixed(1) + '%' : val) : val}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
