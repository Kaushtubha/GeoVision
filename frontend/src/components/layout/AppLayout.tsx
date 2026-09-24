import React from 'react';
import { Outlet } from 'react-router-dom';
import { AppHeader } from './AppHeader';
import { AppSidebar } from './AppSidebar';

export const AppLayout: React.FC = () => {
  return (
    <div className="min-h-screen bg-theme-bg text-slate-100 flex flex-col font-sans antialiased selection:bg-theme-accent/30 selection:text-white transition-colors duration-300">
      <AppHeader />
      <div className="flex flex-1 overflow-hidden">
        <AppSidebar />
        <main className="flex-1 overflow-y-auto p-6 md:p-8 bg-transparent">
          <div className="max-w-7xl mx-auto pb-12">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};
