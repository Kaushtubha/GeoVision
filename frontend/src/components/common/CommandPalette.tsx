import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search,
  LayoutDashboard,
  ScanLine,
  GitCompare,
  Bot,
  FlaskConical,
  Activity,
  Palette,
  X,
  Sparkles,
  ArrowRight,
  Zap,
} from 'lucide-react';
import { useTheme, THEMES } from '@/context/ThemeContext';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ isOpen, onClose }) => {
  const [query, setQuery] = useState('');
  const navigate = useNavigate();
  const { setTheme, theme: currentThemeId } = useTheme();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        if (isOpen) onClose();
        else {
          // Open handled by parent or state
        }
      }
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const navActions = [
    {
      title: 'Mission Dashboard',
      description: 'Overview of Earth Observation operations & telemetry',
      path: '/',
      icon: LayoutDashboard,
      badge: 'Main',
    },
    {
      title: 'Optical Detection & Segmentation',
      description: 'Aircraft/Vessel detection & 5-class land-cover analysis',
      path: '/analysis',
      icon: ScanLine,
      badge: 'Vision',
    },
    {
      title: 'Bi-Temporal Change Detection',
      description: 'Multi-epoch surface transformation & wipe slider comparison',
      path: '/change',
      icon: GitCompare,
      badge: 'Temporal',
    },
    {
      title: 'Multimodal Semantic Search',
      description: 'Cross-modal text-to-satellite scene vector retrieval',
      path: '/search',
      icon: Search,
      badge: 'Vector',
    },
    {
      title: 'Grounded Earth AI Assistant',
      description: 'Spatial reasoning chat engine with verified bounding box evidence',
      path: '/assistant',
      icon: Bot,
      badge: 'VLM AI',
    },
    {
      title: 'Benchmark Experiments & Metrics',
      description: 'Throughput benchmarks, mIoU curves, and confusion matrix',
      path: '/experiments',
      icon: FlaskConical,
      badge: 'Metrics',
    },
    {
      title: 'System Diagnostics & Telemetry',
      description: 'Accelerator status, API latency, and model registry',
      path: '/status',
      icon: Activity,
      badge: 'Health',
    },
  ];

  const filteredNav = navActions.filter(
    (item) =>
      item.title.toLowerCase().includes(query.toLowerCase()) ||
      item.description.toLowerCase().includes(query.toLowerCase()) ||
      item.badge.toLowerCase().includes(query.toLowerCase())
  );

  const filteredThemes = THEMES.filter(
    (t) =>
      t.name.toLowerCase().includes(query.toLowerCase()) ||
      t.description.toLowerCase().includes(query.toLowerCase()) ||
      query.toLowerCase().includes('theme') ||
      query.toLowerCase().includes('color')
  );

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-4 bg-black/70 backdrop-blur-md animate-in fade-in duration-150">
      <div 
        className="w-full max-w-2xl rounded-xl glass-panel shadow-2xl overflow-hidden border border-theme-border/60"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Input Bar */}
        <div className="flex items-center px-4 py-3.5 border-b border-theme-border-subtle bg-black/30">
          <Search className="w-5 h-5 text-theme-accent mr-3 shrink-0" />
          <input
            type="text"
            placeholder="Type a command, module, or search theme..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoFocus
            className="w-full bg-transparent text-sm text-white placeholder-slate-400 focus:outline-none font-sans"
          />
          <button
            onClick={onClose}
            className="p-1 rounded-md text-slate-400 hover:text-white hover:bg-white/10 transition-colors ml-2"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results List */}
        <div className="max-h-[60vh] overflow-y-auto p-2 space-y-4">
          {/* Navigation Section */}
          {filteredNav.length > 0 && (
            <div>
              <p className="px-3 py-1.5 text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold flex items-center gap-1.5">
                <Zap className="w-3 h-3 text-theme-accent" /> Intelligence Modules
              </p>
              <div className="space-y-1">
                {filteredNav.map((item) => {
                  const Icon = item.icon;
                  return (
                    <button
                      key={item.path}
                      onClick={() => {
                        navigate(item.path);
                        onClose();
                      }}
                      className="w-full flex items-center justify-between p-2.5 rounded-lg hover:bg-theme-hover text-left transition-all group border border-transparent hover:border-theme-border/50"
                    >
                      <div className="flex items-center space-x-3">
                        <div className="p-2 rounded-md bg-white/5 group-hover:bg-theme-accent/20 text-slate-300 group-hover:text-theme-accent transition-colors">
                          <Icon className="w-4 h-4" />
                        </div>
                        <div>
                          <div className="text-sm font-medium text-slate-200 group-hover:text-white flex items-center gap-2">
                            <span>{item.title}</span>
                            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white/5 text-slate-400 border border-white/10">
                              {item.badge}
                            </span>
                          </div>
                          <p className="text-xs text-slate-400 line-clamp-1">{item.description}</p>
                        </div>
                      </div>
                      <ArrowRight className="w-4 h-4 text-slate-500 opacity-0 group-hover:opacity-100 group-hover:translate-x-0.5 text-theme-accent transition-all" />
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Theme Switcher Section */}
          {filteredThemes.length > 0 && (
            <div>
              <p className="px-3 py-1.5 text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold flex items-center gap-1.5">
                <Palette className="w-3 h-3 text-theme-accent" /> Visual Themes & Palettes
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                {filteredThemes.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => {
                      setTheme(t.id);
                      onClose();
                    }}
                    className={`flex items-center justify-between p-2.5 rounded-lg text-left transition-all border ${
                      currentThemeId === t.id
                        ? 'bg-theme-accent/15 border-theme-accent/60 text-white shadow-glow-sm'
                        : 'bg-white/5 border-transparent hover:border-theme-border/40 hover:bg-theme-hover text-slate-300'
                    }`}
                  >
                    <div className="flex items-center space-x-2.5">
                      <div className="flex -space-x-1">
                        {t.previewColors.map((color, i) => (
                          <span
                            key={i}
                            className="w-3.5 h-3.5 rounded-full border border-black/40"
                            style={{ backgroundColor: color }}
                          />
                        ))}
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-slate-200">{t.name}</p>
                        <p className="text-[10px] text-slate-400">{t.badge}</p>
                      </div>
                    </div>
                    {currentThemeId === t.id && (
                      <span className="text-[10px] font-mono font-bold text-theme-accent uppercase px-1.5 py-0.5 rounded bg-theme-accent/20">
                        Active
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Empty State */}
          {filteredNav.length === 0 && filteredThemes.length === 0 && (
            <div className="py-12 text-center text-slate-400 space-y-2">
              <Search className="w-8 h-8 text-slate-500 mx-auto" />
              <p className="text-sm font-medium text-slate-300">No matching commands or themes found</p>
              <p className="text-xs text-slate-500 font-mono">Try searching for "vision", "change", "search", or "theme"</p>
            </div>
          )}
        </div>

        {/* Footer shortcuts */}
        <div className="px-4 py-2.5 bg-black/40 border-t border-theme-border-subtle flex items-center justify-between text-[11px] font-mono text-slate-400">
          <div className="flex items-center space-x-3">
            <span>Navigation: <kbd className="px-1 py-0.5 rounded bg-white/10 text-slate-300">↑</kbd> <kbd className="px-1 py-0.5 rounded bg-white/10 text-slate-300">↓</kbd></span>
            <span>Select: <kbd className="px-1 py-0.5 rounded bg-white/10 text-slate-300">↵</kbd></span>
            <span>Close: <kbd className="px-1 py-0.5 rounded bg-white/10 text-slate-300">ESC</kbd></span>
          </div>
          <div className="flex items-center space-x-1 text-slate-400">
            <Sparkles className="w-3 h-3 text-theme-accent" />
            <span>GeoVision Pro Intelligence</span>
          </div>
        </div>
      </div>
    </div>
  );
};
