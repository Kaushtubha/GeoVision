import React, { useEffect, useState, useRef } from 'react';
import { 
  Globe2, 
  Cpu, 
  WifiOff, 
  Palette, 
  Command, 
  ChevronDown,
  Clock,
  Sparkles,
} from 'lucide-react';
import { geoVisionApi } from '@/services/api';
import type { HealthResponse } from '@/types/api';
import { useTheme, THEMES } from '@/context/ThemeContext';
import { CommandPalette } from '@/components/common/CommandPalette';


export const AppHeader: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isOnline, setIsOnline] = useState<boolean>(false);
  const [utcTime, setUtcTime] = useState<string>('');
  const [showThemeMenu, setShowThemeMenu] = useState<boolean>(false);
  const [showCommandPalette, setShowCommandPalette] = useState<boolean>(false);
  const { theme, setTheme, currentTheme } = useTheme();
  const themeMenuRef = useRef<HTMLDivElement>(null);

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

    // Live UTC Clock updater
    const timeInterval = setInterval(() => {
      const now = new Date();
      setUtcTime(now.toUTCString().slice(17, 25) + ' UTC');
    }, 1000);

    return () => {
      isMounted = false;
      clearInterval(interval);
      clearInterval(timeInterval);
    };
  }, []);

  // Close theme menu on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (themeMenuRef.current && !themeMenuRef.current.contains(event.target as Node)) {
        setShowThemeMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <>
      <header className="h-16 border-b border-theme-border/60 bg-theme-panel/90 backdrop-blur-xl px-5 flex items-center justify-between z-30 sticky top-0 shadow-panel">
        {/* Brand & Mission Identifier */}
        <div className="flex items-center space-x-3.5">
          <div className="relative group">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-theme-accent to-theme-accent-secondary flex items-center justify-center shadow-glow-sm border border-white/20 transition-transform group-hover:scale-105">
              <Globe2 className="w-5 h-5 text-slate-950 font-bold animate-pulse-subtle" />
            </div>
            <div className="absolute -inset-1 rounded-xl bg-theme-accent/20 blur-sm -z-10 group-hover:opacity-100 transition-opacity opacity-50" />
          </div>

          <div>
            <div className="flex items-center space-x-2">
              <span className="font-display font-bold text-base tracking-wider uppercase text-white">
                Geo<span className="text-theme-accent">Vision</span>
              </span>
              <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-theme-accent/15 text-theme-accent border border-theme-accent/30 shadow-sm">
                MISSION OS
              </span>
            </div>
            <p className="text-[10px] text-slate-400 font-mono tracking-tight hidden sm:flex items-center gap-1.5">
              <span>Earth Observation Core</span>
              <span className="text-slate-600">•</span>
              <span className="text-theme-accent">v{health?.version || '1.0.0'}</span>
            </p>
          </div>
        </div>

        {/* Global Quick Command Bar Trigger */}
        <button
          onClick={() => setShowCommandPalette(true)}
          className="hidden md:flex items-center space-x-2.5 px-3.5 py-1.5 rounded-lg bg-black/40 border border-theme-border-subtle hover:border-theme-border/80 text-slate-400 hover:text-slate-200 transition-all text-xs font-mono group"
        >
          <Command className="w-3.5 h-3.5 text-theme-accent" />
          <span>Quick search or command...</span>
          <kbd className="px-1.5 py-0.5 rounded bg-white/10 text-[10px] text-slate-300 border border-white/10 group-hover:border-theme-accent/40">
            ⌘K
          </kbd>
        </button>

        {/* Telemetry, UTC Clock, Theme Switcher & Status */}
        <div className="flex items-center space-x-3 text-xs font-mono">
          {/* UTC Satellite Clock */}
          <div className="hidden xl:flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-black/30 border border-theme-border-subtle text-slate-300">
            <Clock className="w-3.5 h-3.5 text-theme-accent" />
            <span>{utcTime || 'ORBITAL TIME'}</span>
          </div>

          {/* Compute Accelerator Indicator */}
          <div className="hidden lg:flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-black/30 border border-theme-border-subtle text-slate-300">
            <Cpu className="w-3.5 h-3.5 text-theme-accent" />
            <span className="text-slate-400">ENGINE:</span>
            <span className="text-white uppercase font-semibold">
              {health?.device || 'CPU'}
            </span>
          </div>

          {/* Theme Selector Dropdown */}
          <div className="relative" ref={themeMenuRef}>
            <button
              onClick={() => setShowThemeMenu(!showThemeMenu)}
              className="flex items-center space-x-2 px-2.5 py-1.5 rounded-lg bg-black/40 border border-theme-border-subtle hover:border-theme-border/80 text-slate-200 transition-all text-xs"
              title="Change Visual Theme"
            >
              <Palette className="w-3.5 h-3.5 text-theme-accent" />
              <span className="hidden sm:inline font-sans text-xs font-medium">{currentTheme.name}</span>
              <div className="flex -space-x-1">
                {currentTheme.previewColors.slice(1).map((c, i) => (
                  <span
                    key={i}
                    className="w-2.5 h-2.5 rounded-full border border-black"
                    style={{ backgroundColor: c }}
                  />
                ))}
              </div>
              <ChevronDown className={`w-3 h-3 text-slate-400 transition-transform ${showThemeMenu ? 'rotate-180' : ''}`} />
            </button>

            {showThemeMenu && (
              <div className="absolute right-0 mt-2 w-64 rounded-xl glass-panel shadow-2xl p-2 z-50 border border-theme-border/60 animate-in fade-in slide-in-from-top-2 duration-150">
                <div className="px-2.5 py-1.5 border-b border-white/10 mb-1.5 flex items-center justify-between">
                  <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold">
                    Visual Color Themes
                  </span>
                  <Sparkles className="w-3.5 h-3.5 text-theme-accent" />
                </div>
                <div className="space-y-1">
                  {THEMES.map((t) => (
                    <button
                      key={t.id}
                      onClick={() => {
                        setTheme(t.id);
                        setShowThemeMenu(false);
                      }}
                      className={`w-full flex items-center justify-between p-2 rounded-lg text-left transition-all ${
                        theme === t.id
                          ? 'bg-theme-accent/20 border border-theme-accent/50 text-white'
                          : 'hover:bg-white/5 text-slate-300 hover:text-white border border-transparent'
                      }`}
                    >
                      <div className="flex items-center space-x-2.5">
                        <div className="flex -space-x-1">
                          {t.previewColors.map((color, i) => (
                            <span
                              key={i}
                              className="w-3 h-3 rounded-full border border-black/50"
                              style={{ backgroundColor: color }}
                            />
                          ))}
                        </div>
                        <div>
                          <p className="text-xs font-semibold">{t.name}</p>
                          <p className="text-[10px] text-slate-400">{t.badge}</p>
                        </div>
                      </div>
                      {theme === t.id && (
                        <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-theme-accent text-slate-950 font-bold">
                          ACTIVE
                        </span>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Live Backend Connection Status */}
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-lg border transition-all ${
            isOnline 
              ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-300 shadow-sm shadow-emerald-900/20' 
              : 'bg-rose-950/40 border-rose-500/30 text-rose-300 shadow-sm shadow-rose-900/20'
          }`}>
            {isOnline ? (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                </span>
                <span className="hidden sm:inline font-mono font-semibold text-[11px]">TELEMETRY LIVE</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3.5 h-3.5 text-rose-400" />
                <span className="font-mono font-semibold text-[11px]">DISCONNECTED</span>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Global Keyboard Command Palette */}
      <CommandPalette 
        isOpen={showCommandPalette} 
        onClose={() => setShowCommandPalette(false)} 
      />
    </>
  );
};
