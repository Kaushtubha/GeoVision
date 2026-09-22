import React, { useEffect, useState } from 'react';
import { 
  ScanLine, 
  Layers, 
  GitCompare, 
  Search, 
  Bot, 
  Activity, 
  ShieldCheck, 
  Cpu, 
  Database,
  ArrowRight,
  HardDrive
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { geoVisionApi } from '@/services/api';
import type { HealthResponse } from '@/types/api';

export const DashboardPage: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    const loadHealth = async () => {
      try {
        setLoading(true);
        const data = await geoVisionApi.getHealth();
        if (isMounted) {
          setHealth(data);
          setError(null);
        }
      } catch (err: any) {
        if (isMounted) {
          setError(err.message || 'Failed to connect to GeoVision backend');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };
    loadHealth();
    return () => {
      isMounted = false;
    };
  }, []);

  const capabilities = [
    {
      title: 'Object Detection',
      desc: 'Satellite optical object detection for aircraft, vessels, and infrastructure.',
      route: '/analysis',
      icon: ScanLine,
      status: health?.models_status?.detection ? 'Active' : 'Standby',
      tag: 'YOLO / Vision'
    },
    {
      title: 'Land-Cover Segmentation',
      desc: '5-Class pixel-level segmentation with area calculation in hectares and km².',
      route: '/analysis',
      icon: Layers,
      status: health?.models_status?.segmentation ? 'Active' : 'Standby',
      tag: 'DeepLabV3+'
    },
    {
      title: 'Bi-Temporal Change Detection',
      desc: 'Compare multi-temporal satellite captures (T1 vs T2) to detect structural alterations.',
      route: '/change',
      icon: GitCompare,
      status: (health?.models_status?.change || health?.models_status?.change_detection) ? 'Active' : 'Standby',
      tag: 'Siamese ResNet'
    },
    {
      title: 'Semantic Vector Retrieval',
      desc: 'Multimodal cross-modal text-to-image and image-to-image similarity search.',
      route: '/search',
      icon: Search,
      status: (health?.models_status?.retrieval || health?.models_status?.qdrant) ? 'Active' : 'Standby',
      tag: 'OpenCLIP + Qdrant'
    },
    {
      title: 'Grounded Earth AI Assistant',
      desc: 'VLM reasoning engine with verified bounding box and segmentation citations.',
      route: '/assistant',
      icon: Bot,
      status: health?.models_status?.vlm ? 'Active' : 'Standby',
      tag: 'Grounded VLM'
    },
  ];

  const isHealthy = health?.status === 'ok' || health?.status === 'healthy';

  return (
    <div className="space-y-6">
      {/* Top Banner / Mission Statement */}
      <div className="p-6 rounded-lg bg-space-900 border border-space-700/80 shadow-md relative overflow-hidden">
        <div className="absolute -right-10 -bottom-10 w-60 h-60 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2 text-cyan-400 font-mono text-xs tracking-wider uppercase mb-1">
              <ShieldCheck className="w-4 h-4" />
              <span>Operational Intelligence System</span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white font-sans">
              GeoVision Earth Observation Platform
            </h1>
            <p className="text-sm text-slate-400 mt-1 max-w-2xl font-sans">
              High-throughput multimodal satellite imagery analysis, automated land-use classification, bi-temporal change monitoring, and grounded VLM intelligence.
            </p>
          </div>

          <div className="flex items-center space-x-3 shrink-0">
            <Link
              to="/analysis"
              className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold rounded-md flex items-center space-x-2 transition-colors shadow-lg shadow-cyan-900/30"
            >
              <span>Launch Analysis</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </div>

      {/* System Telemetry Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 rounded-lg bg-space-900/90 border border-space-700">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-mono uppercase tracking-wider">System Status</span>
            <Activity className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="flex items-center space-x-2">
            <span className={`w-2.5 h-2.5 rounded-full ${isHealthy ? 'bg-emerald-400' : 'bg-amber-400'}`} />
            <span className="text-lg font-bold font-mono text-white capitalize">
              {loading ? 'Polling...' : (isHealthy ? 'Operational' : (health?.status || 'Offline'))}
            </span>
          </div>
          <p className="text-[11px] text-slate-400 mt-1 font-mono">
            Backend API: {health?.version ? `v${health.version}` : 'Connecting...'}
          </p>
        </div>

        <div className="p-4 rounded-lg bg-space-900/90 border border-space-700">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-mono uppercase tracking-wider">Compute Target</span>
            <Cpu className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-lg font-bold font-mono text-white uppercase">
            {loading ? 'Checking...' : (health?.device || 'CPU')}
          </div>
          <p className="text-[11px] text-slate-400 mt-1 font-mono">
            PyTorch Accelerator Core
          </p>
        </div>

        <div className="p-4 rounded-lg bg-space-900/90 border border-space-700">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-mono uppercase tracking-wider">Vector Database</span>
            <Database className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-lg font-bold font-mono text-white">
            {loading ? 'Checking...' : ((health?.models_status?.retrieval || health?.models_status?.qdrant) ? 'CONNECTED' : 'STANDBY')}
          </div>
          <p className="text-[11px] text-slate-400 mt-1 font-mono">
            Qdrant Engine Instance
          </p>
        </div>

        <div className="p-4 rounded-lg bg-space-900/90 border border-space-700">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-mono uppercase tracking-wider">Active Models</span>
            <HardDrive className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-lg font-bold font-mono text-white">
            {health?.models_status ? `${Object.values(health.models_status).filter(Boolean).length} Online` : '0 Active'}
          </div>
          <p className="text-[11px] text-slate-400 mt-1 font-mono">
            Inference Pipelines Loaded
          </p>
        </div>
      </div>

      {/* Module Overview Cards */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-slate-300 uppercase font-mono tracking-wider">
          Intelligence Capabilities
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {capabilities.map((cap) => {
            const Icon = cap.icon;
            return (
              <Link
                key={cap.title}
                to={cap.route}
                className="group p-5 rounded-lg bg-space-900/70 border border-space-700/70 hover:border-cyan-500/50 hover:bg-space-850/80 transition-all flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div className="p-2 rounded bg-space-800 border border-space-700 text-cyan-400 group-hover:text-cyan-300 transition-colors">
                      <Icon className="w-5 h-5" />
                    </div>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-space-800 text-slate-400 border border-space-700">
                      {cap.tag}
                    </span>
                  </div>
                  <h3 className="text-sm font-semibold text-white group-hover:text-cyan-300 transition-colors">
                    {cap.title}
                  </h3>
                  <p className="text-xs text-slate-400 mt-1.5 leading-relaxed font-sans">
                    {cap.desc}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-space-750 flex items-center justify-between text-xs font-mono text-cyan-400">
                  <span>Launch Engine</span>
                  <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
};
