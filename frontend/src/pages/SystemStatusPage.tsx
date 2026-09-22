import React, { useEffect, useState } from 'react';
import { 
  Activity, 
  Cpu, 
  ShieldCheck, 
  Server, 
  RefreshCw, 
  CheckCircle2, 
  XCircle,
  AlertCircle
} from 'lucide-react';
import { geoVisionApi } from '@/services/api';
import type { HealthResponse } from '@/types/api';

export const SystemStatusPage: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date>(new Date());

  const fetchHealth = async () => {
    try {
      setLoading(true);
      const data = await geoVisionApi.getHealth();
      setHealth(data);
      setError(null);
      setLastRefreshed(new Date());
    } catch (err: any) {
      setError(err.message || 'Failed to connect to backend service.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    const timer = setInterval(fetchHealth, 10000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="p-5 rounded-lg bg-space-900 border border-space-700 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-cyan-400 font-mono text-xs uppercase tracking-wider mb-1">
            <Activity className="w-4 h-4" />
            <span>Diagnostics & Health</span>
          </div>
          <h1 className="text-xl font-bold text-white font-sans">
            System & Model Status
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Real-time telemetry, model readiness, compute target, and backend engine status.
          </p>
        </div>

        <button
          onClick={fetchHealth}
          disabled={loading}
          className="px-3 py-1.5 rounded bg-space-800 hover:bg-space-750 text-slate-300 hover:text-white border border-space-700 text-xs font-mono flex items-center space-x-2 transition-colors self-start sm:self-auto"
        >
          <RefreshCw className={`w-3.5 h-3.5 text-cyan-400 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Status</span>
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-800/80 text-rose-300 text-xs flex items-center space-x-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
          <div>
            <p className="font-semibold">Backend Unreachable</p>
            <p className="text-rose-400 font-mono mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Grid of Diagnostics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-5 rounded-lg bg-space-900 border border-space-700 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400 uppercase">Core Status</span>
            <Server className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="flex items-center space-x-2">
            {(health?.status === 'ok' || health?.status === 'healthy') ? (
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
            ) : (
              <XCircle className="w-5 h-5 text-rose-400" />
            )}
            <span className="text-lg font-bold font-mono text-white capitalize">
              {(health?.status === 'ok' || health?.status === 'healthy') ? 'Operational' : (health?.status || 'Offline')}
            </span>
          </div>
          <p className="text-xs text-slate-400 font-mono">
            API Version: {health?.version ? `v${health.version}` : 'Unavailable'}
          </p>
        </div>

        <div className="p-5 rounded-lg bg-space-900 border border-space-700 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400 uppercase">Compute Target</span>
            <Cpu className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-lg font-bold font-mono text-cyan-300 uppercase">
            {health?.device || 'CPU'}
          </div>
          <p className="text-xs text-slate-400 font-mono">
            PyTorch Accelerator Device
          </p>
        </div>

        <div className="p-5 rounded-lg bg-space-900 border border-space-700 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400 uppercase">Telemetry Update</span>
            <Activity className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-sm font-mono text-slate-200">
            {lastRefreshed.toLocaleTimeString()}
          </div>
          <p className="text-xs text-slate-400 font-mono">
            Polling Frequency: 10s
          </p>
        </div>
      </div>

      {/* Models Status Breakdown */}
      <div className="p-5 rounded-lg bg-space-900 border border-space-700 space-y-4">
        <h2 className="text-sm font-semibold text-white font-mono uppercase tracking-wider flex items-center space-x-2">
          <ShieldCheck className="w-4 h-4 text-cyan-400" />
          <span>Subsystem & Inference Engine Status</span>
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {health?.models_status && Object.entries(health.models_status).map(([subsystem, isLoaded]) => (
            <div
              key={subsystem}
              className="p-3.5 rounded bg-space-850 border border-space-750 flex items-center justify-between"
            >
              <div>
                <p className="text-xs font-mono font-medium text-slate-200 capitalize">
                  {subsystem.replace(/_/g, ' ')}
                </p>
                <p className="text-[10px] text-slate-400 font-mono">
                  {isLoaded ? 'Pipeline Ready' : 'Standby / Lazy-Loaded'}
                </p>
              </div>
              <div className={`px-2 py-0.5 rounded text-[10px] font-mono border ${
                isLoaded 
                  ? 'bg-emerald-950/60 text-emerald-300 border-emerald-800/60' 
                  : 'bg-space-800 text-slate-400 border-space-700'
              }`}>
                {isLoaded ? 'ONLINE' : 'STANDBY'}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
