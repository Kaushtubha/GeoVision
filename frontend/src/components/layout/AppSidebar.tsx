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
} from 'lucide-react';

interface NavItem {
  name: string;
  path: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
}

const navItems: NavItem[] = [
  { name: 'Mission Dashboard', path: '/', icon: LayoutDashboard },
  { name: 'Image Analysis', path: '/analysis', icon: ScanLine },
  { name: 'Change Detection', path: '/change', icon: GitCompare },
  { name: 'Semantic Search', path: '/search', icon: Search },
  { name: 'Earth AI Assistant', path: '/assistant', icon: Bot },
  { name: 'Experiments & Metrics', path: '/experiments', icon: FlaskConical },
  { name: 'System & Models', path: '/status', icon: Activity },
];

export const AppSidebar: React.FC = () => {
  return (
    <aside className="w-64 border-r border-space-700/60 bg-space-900/60 backdrop-blur flex flex-col justify-between shrink-0 select-none">
      {/* Navigation Sections */}
      <div className="py-4 px-3 space-y-6">
        <div>
          <p className="px-3 text-[10px] font-mono uppercase tracking-widest text-slate-400 font-semibold mb-2">
            Intelligence Modules
          </p>
          <nav className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  end={item.path === '/'}
                  className={({ isActive }) =>
                    `flex items-center justify-between px-3 py-2.5 rounded-md text-xs font-medium transition-all group ${
                      isActive
                        ? 'bg-cyan-950/40 text-cyan-300 border border-cyan-800/60 shadow-sm shadow-cyan-950/30'
                        : 'text-slate-300 hover:text-slate-100 hover:bg-space-800/60 border border-transparent'
                    }`
                  }
                >
                  {({ isActive }) => (
                    <>
                      <div className="flex items-center space-x-2.5">
                        <Icon
                          className={`w-4 h-4 transition-colors ${
                            isActive
                              ? 'text-cyan-400'
                              : 'text-slate-400 group-hover:text-slate-200'
                          }`}
                        />
                        <span>{item.name}</span>
                      </div>
                      <ChevronRight
                        className={`w-3.5 h-3.5 transition-transform ${
                          isActive
                            ? 'text-cyan-400 translate-x-0.5'
                            : 'text-slate-500 opacity-0 group-hover:opacity-100'
                        }`}
                      />
                    </>
                  )}
                </NavLink>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Footer telemetry notice */}
      <div className="p-3 border-t border-space-700/40 bg-space-950/40">
        <div className="p-2.5 rounded bg-space-850/70 border border-space-700/50 text-[11px] font-mono text-slate-400">
          <div className="flex items-center justify-between text-slate-300 mb-1">
            <span className="font-semibold text-cyan-400">GEOVISION EO</span>
            <span className="text-[9px] px-1 py-0.2 bg-space-750 rounded text-slate-300">GEO-AI</span>
          </div>
          <p className="text-[10px] text-slate-400 leading-tight">
            Standard Earth Observation & Grounded Analysis Node
          </p>
        </div>
      </div>
    </aside>
  );
};
