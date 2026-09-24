import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { AppLayout } from './components/layout/AppLayout';
import { DashboardPage } from './pages/DashboardPage';
import { ImageAnalysisPage } from './pages/ImageAnalysisPage';
import { ChangeDetectionPage } from './pages/ChangeDetectionPage';
import { SemanticSearchPage } from './pages/SemanticSearchPage';
import { EarthAssistantPage } from './pages/EarthAssistantPage';
import { ExperimentsPage } from './pages/ExperimentsPage';
import { SystemStatusPage } from './pages/SystemStatusPage';

export const App: React.FC = () => {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/analysis" element={<ImageAnalysisPage />} />
            <Route path="/change" element={<ChangeDetectionPage />} />
            <Route path="/search" element={<SemanticSearchPage />} />
            <Route path="/assistant" element={<EarthAssistantPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
            <Route path="/status" element={<SystemStatusPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
};

export default App;
