import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  ScanLine,
  GitCompare,
  Search,
  Bot,
  FlaskConical,
  Activity,
  ChevronRight,
  Radio,
} from 'lucide-react';


interface NavGroup {
  label: string;
  items: {
    name: string;
    path: string;
    icon: React.ComponentType<{ className?: string }>;
    badge: string;
    badgeColor?: string;
  }[];
}

const navGroups: NavGroup[] = [
  {
    label: 'Command & Analytics',
    items: [
      { name: 'Mission Dashboard', path: '/', icon: LayoutDashboard, badge: 'LIVE' },
      { name: 'Image Analysis', path: '/analysis', icon: ScanLine, badge: 'YOLO+SEG' },
      { name: 'Change Detection', path: '/change', icon: GitCompare, badge: 'BI-TEMPORAL' },
    ],
  },
  {
    label: 'Neural Intelligence',
    items: [
      { name: 'Semantic Search', path: '/search', icon: Search, badge: 'VECTOR' },
      { name: 'Earth AI Assistant', path: '/assistant', icon: Bot, badge: 'GROUNDED' },
    ],
  },
  {
    label: 'System & Benchmarks',
    items: [
      { name: 'Experiments & Metrics', path: '/experiments', icon: FlaskConical, badge: '98.2% mIoU' },
      { name: 'System Diagnostics', path: '/status', icon: Activity, badge: 'HEALTH' },
    ],
  },
];

export const AppSidebar: React.FC = () => {
  return (
    <aside className="w-64 border-r border-theme-border/60 bg-theme-panel/70 backdrop-blur-xl flex flex-col justify-between shrink-0 select-none">
      {/* Navigation Sections */}
      <div className="py-5 px-3.5 space-y-6 overflow-y-auto">
        {navGroups.map((group, groupIdx) => (
          <div key={groupIdx}>
            <div className="px-3 flex items-center justify-between mb-2">
              <span className="text-[10px] font-mono uppercase tracking-widest text-slate-400 font-bold">
                {group.label}
              </span>
              <span className="w-1.5 h-1.5 rounded-full bg-theme-accent/60" />
            </div>
            <nav className="space-y-1">
              {group.items.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    end={item.path === '/'}
                    className={({ isActive }) =>
                      `relative flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-medium transition-all group ${
                        isActive
                          ? 'bg-theme-accent/15 text-white border border-theme-accent/40 shadow-glow-sm'
                          : 'text-slate-300 hover:text-white hover:bg-white/5 border border-transparent'
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        {/* Active Left Indicator Pill */}
                        {isActive && (
                          <span className="absolute left-0 top-2 bottom-2 w-1 rounded-r bg-theme-accent shadow-glow-sm" />
                        )}

                        <div className="flex items-center space-x-3 pl-1">
                          <Icon
                            className={`w-4 h-4 transition-transform group-hover:scale-110 ${
                              isActive
                                ? 'text-theme-accent'
                                : 'text-slate-400 group-hover:text-slate-200'
                            }`}
                          />
                          <span className="tracking-wide font-sans">{item.name}</span>
                        </div>

                        <div className="flex items-center space-x-1.5">
                          <span
                            className={`text-[9px] font-mono px-1.5 py-0.5 rounded border transition-colors ${
                              isActive
                                ? 'bg-theme-accent/25 text-theme-accent border-theme-accent/40 font-semibold'
                                : 'bg-white/5 text-slate-400 border-white/5 group-hover:border-white/10'
                            }`}
                          >
                            {item.badge}
                          </span>
                          <ChevronRight
                            className={`w-3.5 h-3.5 transition-transform ${
                              isActive
                                ? 'text-theme-accent translate-x-0.5 opacity-100'
                                : 'text-slate-600 opacity-0 group-hover:opacity-100 group-hover:text-slate-300'
                            }`}
                          />
                        </div>
                      </>
                    )}
                  </NavLink>
                );
              })}
            </nav>
          </div>
        ))}
      </div>

      {/* Footer telemetry card */}
      <div className="p-3.5 border-t border-theme-border/40 bg-black/40">
        <div className="p-3 rounded-xl bg-theme-card/80 border border-theme-border-subtle text-[11px] font-mono text-slate-400 relative overflow-hidden">
          <div className="absolute right-2 top-2 opacity-20 text-theme-accent pointer-events-none">
            <Radio className="w-8 h-8 animate-pulse-subtle" />
          </div>
          <div className="flex items-center justify-between text-slate-200 mb-1">
            <div className="flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-theme-accent animate-ping" />
              <span className="font-bold text-white tracking-wider">GEOVISION NODE</span>
            </div>
            <span className="text-[9px] px-1.5 py-0.5 bg-theme-accent/20 rounded text-theme-accent font-bold">
              EO-ONLINE
            </span>
          </div>
          <p className="text-[10px] text-slate-400 leading-relaxed font-sans mt-0.5">
            Grounded Multimodal Earth Observation Subsystem Active
          </p>
        </div>
      </div>
    </aside>
  );
};
