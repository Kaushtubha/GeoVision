/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        theme: {
          bg: 'var(--bg-main)',
          panel: 'var(--bg-panel)',
          card: 'var(--bg-card)',
          hover: 'var(--bg-hover)',
          border: 'var(--border-main)',
          'border-subtle': 'var(--border-subtle)',
          accent: 'var(--accent-primary)',
          'accent-secondary': 'var(--accent-secondary)',
          'accent-glow': 'var(--accent-glow)',
          text: 'var(--text-main)',
          'text-muted': 'var(--text-muted)',
        },
        space: {
          950: '#06090e',
          900: '#0b0f17',
          850: '#101622',
          800: '#161f2e',
          750: '#1c283c',
          700: '#222f44',
          600: '#334460',
          500: '#475e82',
        },
        satellite: {
          cyan: '#00f2fe',
          teal: '#06b6d4',
          emerald: '#10b981',
          amber: '#f59e0b',
          rose: '#f43f5e',
          indigo: '#6366f1',
          violet: '#8b5cf6',
        },
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        display: ['"Space Grotesk"', '"Plus Jakarta Sans"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'Fira Code', 'SF Mono', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        'glow-sm': '0 0 15px -3px var(--accent-glow)',
        'glow-md': '0 0 25px -4px var(--accent-glow)',
        'glow-lg': '0 0 35px -5px var(--accent-glow)',
        'panel': '0 8px 32px 0 rgba(0, 0, 0, 0.45)',
      },
      animation: {
        'pulse-subtle': 'pulseSubtle 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'radar-sweep': 'radarSweep 4s linear infinite',
        'shimmer': 'shimmer 2.5s infinite linear',
      },
      keyframes: {
        pulseSubtle: {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.6 },
        },
        radarSweep: {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
    },
  },
  plugins: [],
}

