import React, { useEffect, useState } from 'react';
import { Globe, Cpu, ShieldCheck, Wifi, WifiOff } from 'lucide-react';
import { geoVisionApi } from '@/services/api';
import type { HealthResponse } from '@/types/api';

export const AppHeader: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isOnline, setIsOnline] = useState<boolean>(false);

  useEffect(() => {
    let isMounted = true;
    const checkStatus = async () => {
      try {
        const data = await geoVisionApi.getHealth();
        if (isMounted) {
          setHealth(data);
          setIsOnline(true);
        }
      } catch {
        if (isMounted) {
          setIsOnline(false);
        }
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="h-14 border-b border-space-700/60 bg-space-900/90 backdrop-blur px-5 flex items-center justify-between z-30 sticky top-0">
      {/* Brand & Mission Identifier */}
      <div className="flex items-center space-x-3">
        <div className="w-8 h-8 rounded bg-gradient-to-tr from-cyan-600 to-blue-500 flex items-center justify-center shadow-lg shadow-cyan-500/10 border border-cyan-400/30">
          <Globe className="w-5 h-5 text-white animate-pulse" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <span className="font-bold text-sm tracking-wider uppercase bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
              GeoVision
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-space-800 text-cyan-300 border border-space-700">
              v{health?.version || '1.0.0'}
            </span>
          </div>
          <p className="text-[10px] text-slate-400 font-mono tracking-tight hidden sm:block">
            Multimodal Earth Observation Intelligence Platform
          </p>
        </div>
      </div>

      {/* Telemetry & Telemetry Pills */}
      <div className="flex items-center space-x-3 text-xs font-mono">
        {/* Device compute indicator */}
        <div className="hidden md:flex items-center space-x-1.5 px-2.5 py-1 rounded bg-space-850 border border-space-700 text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span>DEVICE:</span>
          <span className="text-cyan-300 uppercase font-semibold">
            {health?.device || 'CPU'}
          </span>
        </div>

        {/* Subsystems indicator */}
        <div className="hidden lg:flex items-center space-x-1.5 px-2.5 py-1 rounded bg-space-850 border border-space-700 text-slate-300">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          <span>ENGINES:</span>
          <span className="text-emerald-300">
            {health?.models_status ? `${Object.keys(health.models_status).length}/5 ACTIVE` : 'READY'}
          </span>
        </div>

        {/* Live Backend Connection Status */}
        <div className={`flex items-center space-x-1.5 px-2.5 py-1 rounded border ${
          isOnline 
            ? 'bg-emerald-950/40 border-emerald-800/60 text-emerald-300' 
            : 'bg-rose-950/40 border-rose-800/60 text-rose-300'
        }`}>
          {isOnline ? (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
              <Wifi className="w-3.5 h-3.5 text-emerald-400" />
              <span className="hidden sm:inline">ONLINE</span>
            </>
          ) : (
            <>
              <WifiOff className="w-3.5 h-3.5 text-rose-400" />
              <span>OFFLINE</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
};
