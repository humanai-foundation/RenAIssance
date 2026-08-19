import { ChevronLeft, Sparkles, CheckCircle2, AlertCircle, Check, FileText } from 'lucide-react';
import ResizablePanels from './ResizablePanels';

// Full-viewport 3-column shell for the OCR workspace.
export default function OCRLayout({
  onBack,
  onComplete,
  onHome,
  processedCount,
  totalPages,
  hasAnyTranscript,
  backendOnline,
  leftSidebar,
  centerPanel,
  rightPanel,
}) {
  const progressPercent = totalPages > 0 ? (processedCount / totalPages) * 100 : 0;
  const isComplete = processedCount === totalPages && totalPages > 0;

  return (
    <div className="h-full w-full flex flex-col bg-gradient-to-br from-slate-50 via-blue-50/30 to-indigo-50/40 overflow-hidden">
      {/* Header - Fixed height with truly centered progress */}
      <header className="h-12 bg-white/95 backdrop-blur-md border-b border-gray-200 shrink-0 relative z-10">
        {/* Left: Back button and title - absolutely positioned */}
        <div className="absolute left-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
          <button
            onClick={onBack}
            className="flex items-center gap-1 text-gray-600 hover:text-blue-600 hover:bg-blue-50 px-2 py-1 rounded-lg transition-all text-sm"
          >
            <ChevronLeft size={16} />
            <span className="font-medium hidden sm:inline">Back</span>
          </button>
          <div className="h-4 w-px bg-gray-200 hidden sm:block" />
          <div className="flex items-center gap-1.5 hidden sm:flex">
            <div className="p-1 bg-gradient-to-br from-blue-500 to-indigo-600 rounded text-white">
              <Sparkles size={14} />
            </div>
            <span className="text-sm font-bold text-gray-700">RenAIssance OCR</span>
          </div>
        </div>

        {/* Center: Step Progress - truly centered */}
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
          <div className="flex items-center gap-3 px-4 py-1.5 bg-gradient-to-r from-blue-50 via-white to-indigo-50 rounded-full border border-blue-100/60 shadow-sm">
            <div className={`w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-bold ${
              isComplete 
                ? 'bg-gradient-to-br from-green-500 to-emerald-600' 
                : 'bg-gradient-to-br from-blue-500 to-indigo-600'
            }`}>
              {isComplete ? <Check size={12} /> : <FileText size={12} />}
            </div>
            <div className="flex items-center gap-2">
              <div className="w-40 h-1.5 bg-blue-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isComplete 
                      ? 'bg-gradient-to-r from-green-500 to-emerald-500' 
                      : 'bg-gradient-to-r from-blue-500 to-indigo-500'
                  }`}
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
              <span className={`text-xs font-bold ${isComplete ? 'text-green-600' : 'text-blue-700'}`}>
                {processedCount}/{totalPages}
              </span>
            </div>
          </div>
        </div>

        {/* Right: Complete button - absolutely positioned */}
        <div className="absolute right-3 top-1/2 -translate-y-1/2">
          <div className="flex items-center gap-2">
            {onHome && (
              <button
                onClick={onHome}
                className="px-3 py-1.5 rounded-lg font-semibold flex items-center gap-1.5 transition-all text-sm bg-white text-gray-600 border border-gray-200 hover:bg-gray-50"
              >
                <ChevronLeft size={14} />
                <span className="hidden sm:inline">Home</span>
              </button>
            )}
            <button
              onClick={onComplete}
              disabled={!hasAnyTranscript}
              className={`px-3 py-1.5 rounded-lg font-semibold flex items-center gap-1.5 transition-all text-sm ${
                hasAnyTranscript
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700 shadow-sm'
                  : 'bg-gray-100 text-gray-400 cursor-not-allowed'
              }`}
            >
              <CheckCircle2 size={14} />
              <span className="hidden sm:inline">Complete</span>
            </button>
          </div>
        </div>
      </header>

      {/* Backend offline warning */}
      {backendOnline === false && (
        <div className="px-3 py-1.5 shrink-0">
          <div className="px-3 py-2 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-red-700 text-sm">
            <AlertCircle size={14} />
            <span>
              Backend offline. Run:{' '}
              <code className="bg-red-100 px-1.5 py-0.5 rounded font-mono text-xs">
                uvicorn main:app --reload
              </code>
            </span>
          </div>
        </div>
      )}

      {/* Main 3-column layout - Takes remaining viewport height */}
      <main className="flex-1 min-h-0 p-2 overflow-hidden">
        {/* Resizable 3-column layout */}
        <div className="h-full hidden lg:block">
          <ResizablePanels
            leftPanel={leftSidebar}
            centerPanel={centerPanel}
            rightPanel={rightPanel}
            defaultLeftWidth={260}
            defaultRightWidth={340}
            minLeftWidth={200}
            maxLeftWidth={400}
            minRightWidth={280}
            maxRightWidth={500}
            minCenterWidth={350}
          />
        </div>
        
        {/* Mobile fallback - stacked layout */}
        <div className="h-full lg:hidden flex flex-col gap-2 overflow-hidden">
          <div className="flex-1 min-h-0 overflow-hidden">
            {centerPanel}
          </div>
        </div>
      </main>
    </div>
  );
}
