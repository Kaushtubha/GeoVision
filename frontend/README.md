# GeoVision Earth Observation Intelligence Platform — Frontend

Production React & TypeScript frontend for the GeoVision Multimodal Earth Observation platform.

## Architecture

- **Framework**: React 18 + Vite + TypeScript
- **Styling**: Tailwind CSS (Mission Control Dark Palette) + Lucide Icons
- **Routing**: React Router DOM v6
- **API Layer**: Typed fetch client (`src/services/api.ts`) matching FastAPI Pydantic schemas (`src/types/api.ts`)

## Directory Layout

```
frontend/
├── src/
│   ├── components/
│   │   └── layout/
│   │       ├── AppHeader.tsx
│   │       ├── AppSidebar.tsx
│   │       └── AppLayout.tsx
│   ├── pages/
│   │   ├── DashboardPage.tsx
│   │   ├── ImageAnalysisPage.tsx
│   │   ├── ChangeDetectionPage.tsx
│   │   ├── SemanticSearchPage.tsx
│   │   ├── EarthAssistantPage.tsx
│   │   ├── ExperimentsPage.tsx
│   │   └── SystemStatusPage.tsx
│   ├── services/
│   │   └── api.ts
│   ├── types/
│   │   └── api.ts
│   ├── App.tsx
│   ├── index.css
│   └── main.tsx
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

## Running Locally

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start development server:
   ```bash
   npm run dev
   ```

3. Build production bundle:
   ```bash
   npm run build
   ```
