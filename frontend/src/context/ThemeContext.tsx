import React, { createContext, useContext, useEffect, useState } from 'react';

export type ThemeId = 'emerald' | 'amber' | 'cyan' | 'violet';

export interface ThemeConfig {
  id: ThemeId;
  name: string;
  badge: string;
  description: string;
  previewColors: string[];
  primaryColor: string;
  accentGlow: string;
}

export const THEMES: ThemeConfig[] = [
  {
    id: 'emerald',
    name: 'Earth Borealis',
    badge: 'Earth Observation',
    description: 'Deep Obsidian Slate with Emerald & Electric Teal Glow',
    previewColors: ['#070a11', '#10b981', '#06b6d4'],
    primaryColor: '#10b981',
    accentGlow: 'rgba(16, 185, 129, 0.35)',
  },
  {
    id: 'cyan',
    name: 'Orbital Cyan',
    badge: 'Deep Space',
    description: 'Cosmic Space Navy with Polar Ice Cyan & Indigo',
    previewColors: ['#060b14', '#38bdf8', '#6366f1'],
    primaryColor: '#38bdf8',
    accentGlow: 'rgba(56, 189, 248, 0.35)',
  },
  {
    id: 'amber',
    name: 'Tactical FLIR',
    badge: 'Thermal & SAR',
    description: 'Titanium Graphite with Hyper Amber & Solar Gold',
    previewColors: ['#090a0f', '#f59e0b', '#f43f5e'],
    primaryColor: '#f59e0b',
    accentGlow: 'rgba(245, 158, 11, 0.35)',
  },
  {
    id: 'violet',
    name: 'Nebula Amethyst',
    badge: 'Spectral AI',
    description: 'Deep Amethyst Slate with Electric Violet & Neon Cyan',
    previewColors: ['#0b0914', '#a855f7', '#06b6d4'],
    primaryColor: '#a855f7',
    accentGlow: 'rgba(168, 85, 247, 0.35)',
  },
];

interface ThemeContextType {
  theme: ThemeId;
  currentTheme: ThemeConfig;
  setTheme: (theme: ThemeId) => void;
  cycleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const STORAGE_KEY = 'geovision_theme_mode';

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setThemeState] = useState<ThemeId>(() => {
    const saved = localStorage.getItem(STORAGE_KEY) as ThemeId | null;
    return (saved && THEMES.some((t) => t.id === saved)) ? saved : 'emerald';
  });

  const currentTheme = THEMES.find((t) => t.id === theme) || THEMES[0];

  useEffect(() => {
    const root = document.documentElement;
    THEMES.forEach((t) => root.classList.remove(`theme-${t.id}`));
    root.classList.add(`theme-${theme}`);
    localStorage.setItem(STORAGE_KEY, theme);
  }, [theme]);

  const setTheme = (newTheme: ThemeId) => {
    setThemeState(newTheme);
  };

  const cycleTheme = () => {
    const currentIndex = THEMES.findIndex((t) => t.id === theme);
    const nextIndex = (currentIndex + 1) % THEMES.length;
    setThemeState(THEMES[nextIndex].id);
  };

  return (
    <ThemeContext.Provider value={{ theme, currentTheme, setTheme, cycleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};
