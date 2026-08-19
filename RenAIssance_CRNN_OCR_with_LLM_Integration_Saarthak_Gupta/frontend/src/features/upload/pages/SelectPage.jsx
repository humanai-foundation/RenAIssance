import React from 'react';
import { ArrowRight, ArrowLeft, FileImage, Layers } from 'lucide-react';
import PdfPreviewGrid from '../../../components/PdfPreviewGrid';

export default function SelectPage({
  pages,
  selectedPages,
  onSelectionChange,
  onBack,
  onNext,
  isLoading,
}) {
  const hasSelection = selectedPages.length > 0;

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg text-white shadow-md shadow-blue-500/20">
              <Layers className="w-5 h-5" />
            </div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
              Select Pages
            </h1>
          </div>
          <p className="text-gray-500 mt-1 ml-11">
            Choose which pages to include in preprocessing. Click to select, Shift+click for range selection.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="btn-ghost"
          >
            <ArrowLeft className="w-5 h-5" />
            Back
          </button>

          <button
            onClick={onNext}
            disabled={!hasSelection || isLoading}
            className={`btn-primary ${hasSelection && !isLoading
                ? ''
                : 'opacity-50 cursor-not-allowed hover:translate-y-0'
              }`}
          >
            Continue
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Loading state */}
      {isLoading ? (
        <div className="flex flex-col items-center justify-center py-20 bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg">
          <div className="relative">
            <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
            <div className="absolute inset-0 blur-xl bg-blue-400/30 animate-pulse" />
          </div>
          <p className="mt-6 text-blue-600 font-semibold text-lg">Extracting pages...</p>
          <p className="text-sm text-gray-500 mt-1">This may take a moment for large PDFs</p>
        </div>
      ) : pages.length > 0 ? (
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-gray-100 p-6">
          <PdfPreviewGrid
            pages={pages}
            selectedPages={selectedPages}
            onSelectionChange={onSelectionChange}
          />
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-20 bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-gray-100">
          <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-2xl flex items-center justify-center">
            <FileImage className="w-10 h-10 text-blue-400" />
          </div>
          <h3 className="mt-6 text-lg font-semibold text-gray-700">No pages extracted</h3>
          <p className="text-gray-500 mt-2">Please go back and upload a PDF document.</p>
        </div>
      )}

    </div>
  );
}
