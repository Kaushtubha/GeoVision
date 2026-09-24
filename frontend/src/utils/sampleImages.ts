// High-fidelity procedural satellite scene generators for immediate 1-click test drives

export interface SampleScene {
  id: string;
  name: string;
  category: 'Airfield' | 'Maritime' | 'Forestry' | 'Urban' | 'Agriculture';
  description: string;
  resolution: string;
  coords: string;
  dataUrl: string;
  dataUrlT2?: string; // For bi-temporal change detection
}

// Generate high quality SVG rasterized to DataURL
function createAirportScene(): string {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="640" viewBox="0 0 640 640">
    <defs>
      <pattern id="grass" width="20" height="20" patternUnits="userSpaceOnUse">
        <rect width="20" height="20" fill="#2d4a2d"/>
        <rect width="10" height="10" fill="#325432"/>
      </pattern>
      <linearGradient id="tarmac" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#374151"/>
        <stop offset="100%" stop-color="#1f2937"/>
      </linearGradient>
    </defs>
    <!-- Background field -->
    <rect width="640" height="640" fill="url(#grass)"/>
    
    <!-- Taxiways & Runways -->
    <rect x="280" y="0" width="80" height="640" fill="url(#tarmac)"/>
    <line x1="320" y1="20" x2="320" y2="620" stroke="#facc15" stroke-width="4" stroke-dasharray="25,25"/>
    <rect x="0" y="240" width="640" height="60" fill="url(#tarmac)"/>
    <line x1="20" y1="270" x2="620" y2="270" stroke="#ffffff" stroke-width="3" stroke-dasharray="20,20"/>
    
    <!-- Apron -->
    <polygon points="100,100 280,100 280,240 100,240" fill="#4b5563"/>
    <polygon points="360,300 540,300 540,480 360,480" fill="#4b5563"/>
    
    <!-- Aircraft 1 -->
    <g transform="translate(160, 160) rotate(45)">
      <path d="M 0,-30 L 6,-8 L 30,10 L 30,16 L 6,8 L 4,32 L 14,40 L 14,44 L 0,40 L -14,44 L -14,40 L -4,32 L -6,8 L -30,16 L -30,10 L -6,-8 Z" fill="#f8fafc" stroke="#64748b" stroke-width="1.5"/>
    </g>
    <!-- Aircraft 2 -->
    <g transform="translate(220, 180) rotate(45)">
      <path d="M 0,-24 L 5,-6 L 24,8 L 24,13 L 5,6 L 3,25 L 11,32 L 11,35 L 0,32 L -11,35 L -11,32 L -3,25 L -5,6 L -24,13 L -24,8 L -5,-6 Z" fill="#e2e8f0" stroke="#475569" stroke-width="1.5"/>
    </g>
    <!-- Aircraft 3 on Runway -->
    <g transform="translate(320, 420) rotate(0)">
      <path d="M 0,-35 L 7,-10 L 36,12 L 36,18 L 7,10 L 5,36 L 16,45 L 16,50 L 0,46 L -16,50 L -16,45 L -5,36 L -7,10 L -36,18 L -36,12 L -7,-10 Z" fill="#f1f5f9" stroke="#334155" stroke-width="2"/>
    </g>
    <!-- Aircraft 4 -->
    <g transform="translate(450, 390) rotate(-135)">
      <path d="M 0,-26 L 5,-7 L 26,9 L 26,14 L 5,7 L 3,27 L 12,35 L 12,38 L 0,35 L -12,38 L -12,35 L -3,27 L -5,7 L -26,14 L -26,9 L -5,-7 Z" fill="#ffffff" stroke="#64748b" stroke-width="1.5"/>
    </g>
    
    <!-- Hangars / Terminal buildings -->
    <rect x="80" y="40" width="120" height="50" fill="#94a3b8" stroke="#1e293b" stroke-width="2"/>
    <rect x="420" y="500" width="140" height="60" fill="#64748b" stroke="#0f172a" stroke-width="2"/>
  </svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

function createMaritimePortScene(): string {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="640" viewBox="0 0 640 640">
    <defs>
      <linearGradient id="ocean" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#0e3a53"/>
        <stop offset="100%" stop-color="#082234"/>
      </linearGradient>
    </defs>
    <!-- Deep Water -->
    <rect width="640" height="640" fill="url(#ocean)"/>
    
    <!-- Pier / Docks -->
    <polygon points="0,0 240,0 240,400 120,400 120,640 0,640" fill="#475569" stroke="#1e293b" stroke-width="3"/>
    <polygon points="240,160 480,160 480,240 240,240" fill="#334155" stroke="#0f172a" stroke-width="3"/>
    
    <!-- Cargo Container stacks -->
    <g fill="#ef4444"><rect x="30" y="40" width="40" height="15"/><rect x="30" y="60" width="40" height="15"/><rect x="80" y="40" width="40" height="15"/></g>
    <g fill="#3b82f6"><rect x="30" y="100" width="40" height="15"/><rect x="80" y="80" width="40" height="15"/><rect x="130" y="40" width="40" height="15"/></g>
    <g fill="#10b981"><rect x="80" y="120" width="40" height="15"/><rect x="130" y="80" width="40" height="15"/><rect x="30" y="140" width="40" height="15"/></g>
    <g fill="#f59e0b"><rect x="130" y="120" width="40" height="15"/><rect x="80" y="160" width="40" height="15"/><rect x="130" y="160" width="40" height="15"/></g>

    <!-- Cargo Vessel 1 Berthed -->
    <g transform="translate(360, 110) rotate(0)">
      <path d="M -90,-18 L 60,-18 Q 90,-18 100,0 Q 90,18 60,18 L -90,18 Z" fill="#991b1b" stroke="#450a0a" stroke-width="2"/>
      <rect x="-80" y="-12" width="120" height="24" fill="#1e293b"/>
      <!-- Cargo cells -->
      <rect x="-70" y="-10" width="20" height="9" fill="#0284c7"/>
      <rect x="-45" y="-10" width="20" height="9" fill="#d97706"/>
      <rect x="-20" y="-10" width="20" height="9" fill="#16a34a"/>
      <rect x="-70" y="1" width="20" height="9" fill="#dc2626"/>
      <rect x="-45" y="1" width="20" height="9" fill="#2563eb"/>
      <rect x="-20" y="1" width="20" height="9" fill="#d97706"/>
    </g>

    <!-- Large Container Ship in Channel -->
    <g transform="translate(440, 480) rotate(-35)">
      <path d="M -110,-24 L 80,-24 Q 120,-24 135,0 Q 120,24 80,24 L -110,24 Z" fill="#1e3a8a" stroke="#0f172a" stroke-width="2"/>
      <rect x="-95" y="-18" width="160" height="36" fill="#0f172a"/>
      <rect x="-85" y="-14" width="25" height="12" fill="#ef4444"/>
      <rect x="-55" y="-14" width="25" height="12" fill="#3b82f6"/>
      <rect x="-25" y="-14" width="25" height="12" fill="#10b981"/>
      <rect x="5" y="-14" width="25" height="12" fill="#f59e0b"/>
      <rect x="-85" y="2" width="25" height="12" fill="#8b5cf6"/>
      <rect x="-55" y="2" width="25" height="12" fill="#06b6d4"/>
      <rect x="-25" y="2" width="25" height="12" fill="#ef4444"/>
      <rect x="5" y="2" width="25" height="12" fill="#10b981"/>
    </g>

    <!-- Small Patrol Boats -->
    <g transform="translate(180, 520) rotate(60)">
      <path d="M -20,-6 L 15,-6 Q 25,-6 30,0 Q 25,6 15,6 L -20,6 Z" fill="#ffffff" stroke="#334155" stroke-width="1.5"/>
    </g>
  </svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

function createRainforestScenes(): { t1: string; t2: string } {
  // Epoch T1: Dense Pristine Canopy with River
  const svgT1 = `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="640" viewBox="0 0 640 640">
    <rect width="640" height="640" fill="#14532d"/>
    <!-- River -->
    <path d="M 0,200 Q 160,260 320,180 T 640,240 L 640,320 Q 480,260 320,280 T 0,300 Z" fill="#0284c7" stroke="#0369a1" stroke-width="2"/>
    <!-- Canopy textures -->
    <circle cx="120" cy="100" r="70" fill="#166534" opacity="0.8"/>
    <circle cx="260" cy="80" r="80" fill="#15803d" opacity="0.7"/>
    <circle cx="480" cy="120" r="90" fill="#166534" opacity="0.8"/>
    <circle cx="100" cy="460" r="85" fill="#15803d" opacity="0.7"/>
    <circle cx="340" cy="500" r="95" fill="#14532d" opacity="0.9"/>
    <circle cx="520" cy="440" r="80" fill="#166534" opacity="0.8"/>
  </svg>`;

  // Epoch T2: Same region with visible deforestation & new roads
  const svgT2 = `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="640" viewBox="0 0 640 640">
    <rect width="640" height="640" fill="#14532d"/>
    <!-- River -->
    <path d="M 0,200 Q 160,260 320,180 T 640,240 L 640,320 Q 480,260 320,280 T 0,300 Z" fill="#0284c7" stroke="#0369a1" stroke-width="2"/>
    <!-- Canopy textures -->
    <circle cx="120" cy="100" r="70" fill="#166534" opacity="0.8"/>
    <circle cx="480" cy="120" r="90" fill="#166534" opacity="0.8"/>
    <circle cx="100" cy="460" r="85" fill="#15803d" opacity="0.7"/>
    
    <!-- Logging Access Roads (Fishbone Pattern) -->
    <path d="M 320,0 L 320,640" stroke="#a8a29e" stroke-width="12"/>
    <line x1="320" y1="420" x2="560" y2="390" stroke="#a8a29e" stroke-width="6"/>
    <line x1="320" y1="490" x2="580" y2="470" stroke="#a8a29e" stroke-width="6"/>
    <line x1="320" y1="560" x2="540" y2="550" stroke="#a8a29e" stroke-width="6"/>
    
    <!-- Deforested Clear-cut Parcels -->
    <rect x="360" y="380" width="160" height="180" fill="#78716c" stroke="#ca8a04" stroke-width="2"/>
    <polygon points="180,60 300,50 300,160 160,140" fill="#ca8a04" opacity="0.9"/>
  </svg>`;

  return {
    t1: `data:image/svg+xml;utf8,${encodeURIComponent(svgT1)}`,
    t2: `data:image/svg+xml;utf8,${encodeURIComponent(svgT2)}`,
  };
}

function createUrbanGridScene(): string {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="640" viewBox="0 0 640 640">
    <rect width="640" height="640" fill="#334155"/>
    
    <!-- Road Grid -->
    <line x1="0" y1="160" x2="640" y2="160" stroke="#0f172a" stroke-width="22"/>
    <line x1="0" y1="360" x2="640" y2="360" stroke="#0f172a" stroke-width="26"/>
    <line x1="0" y1="520" x2="640" y2="520" stroke="#0f172a" stroke-width="18"/>
    <line x1="160" y1="0" x2="160" y2="640" stroke="#0f172a" stroke-width="22"/>
    <line x1="400" y1="0" x2="400" y2="640" stroke="#0f172a" stroke-width="26"/>
    <line x1="560" y1="0" x2="560" y2="640" stroke="#0f172a" stroke-width="18"/>

    <!-- City Blocks & High-rise Roofs -->
    <g fill="#94a3b8" stroke="#1e293b" stroke-width="2">
      <rect x="25" y="25" width="110" height="110"/>
      <rect x="185" y="25" width="90" height="110" fill="#cbd5e1"/>
      <rect x="290" y="25" width="85" height="110"/>
      <rect x="425" y="25" width="110" height="110" fill="#64748b"/>

      <rect x="25" y="185" width="110" height="150" fill="#e2e8f0"/>
      <rect x="185" y="185" width="190" height="150" fill="#64748b"/>
      <rect x="425" y="185" width="110" height="150" fill="#94a3b8"/>

      <rect x="25" y="385" width="110" height="110"/>
      <rect x="185" y="385" width="190" height="110" fill="#e2e8f0"/>
      <rect x="425" y="385" width="110" height="110"/>
    </g>

    <!-- Green Park & Water Basin -->
    <rect x="25" y="545" width="110" height="70" fill="#15803d"/>
    <polygon points="185,545 375,545 375,615 185,615" fill="#0284c7"/>
  </svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export const SAMPLE_SATELLITE_SCENES: SampleScene[] = [
  {
    id: 'airport-hub',
    name: 'International Airfield Hub',
    category: 'Airfield',
    description: 'Multi-runway tactical airfield with aircraft on aprons and taxiways.',
    resolution: '0.5m GSD Optical',
    coords: '25°15\'10"N 55°21\'52"E',
    dataUrl: createAirportScene(),
  },
  {
    id: 'deepwater-port',
    name: 'Maritime Container Port',
    category: 'Maritime',
    description: 'Deepwater harbor with berthed container ships, loading cranes and berths.',
    resolution: '0.3m GSD Super-Res',
    coords: '31°13\'48"N 121°30\'29"E',
    dataUrl: createMaritimePortScene(),
  },
  {
    id: 'rainforest-bi-temporal',
    name: 'Amazon Rainforest Epochs (T1 vs T2)',
    category: 'Forestry',
    description: 'Bi-temporal deforestation monitoring and access road emergence.',
    resolution: '1.0m Sentinel-2 / Landsat',
    coords: '09°45\'22"S 63°08\'14"W',
    dataUrl: createRainforestScenes().t1,
    dataUrlT2: createRainforestScenes().t2,
  },
  {
    id: 'urban-metropolis',
    name: 'Metropolitan High-Rise District',
    category: 'Urban',
    description: 'Dense commercial grid with road networks, infrastructure, and parks.',
    resolution: '0.3m WorldView-3',
    coords: '37°46\'26"N 122°25\'10"W',
    dataUrl: createUrbanGridScene(),
  },
];

// Helper to convert DataURL to standard File object for GeoVision API requests
export async function dataUrlToFile(dataUrl: string, filename: string): Promise<File> {
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  return new File([blob], filename, { type: blob.type || 'image/png' });
}
