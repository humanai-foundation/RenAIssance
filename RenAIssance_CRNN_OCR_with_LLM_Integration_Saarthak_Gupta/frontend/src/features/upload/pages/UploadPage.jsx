import React from 'react';
import { FileImage, FileText, ArrowRight, Zap } from 'lucide-react';
import UploadZone from '../../../components/UploadZone';

export default function UploadPage({ onFilesSelected, onNext, isLoading, files }) {
  const hasFiles = files && files.length > 0;
  const isPdf = hasFiles && files[0].type === 'application/pdf';
  const isImages = hasFiles && files[0].type.startsWith('image/');

  return (
    <div className="max-w-4xl mx-auto animate-fade-in">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-full text-sm font-semibold text-blue-600 mb-4 shadow-sm">
          <Zap className="w-4 h-4" />
          Step 1 of 5
        </div>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-gray-800 via-gray-700 to-gray-600 bg-clip-text text-transparent mb-3">
          Upload Your Documents
        </h1>
        <p className="text-gray-500 max-w-xl mx-auto text-lg">
          Upload a PDF document or multiple images to begin preprocessing for OCR.
          We'll help you prepare your documents for optimal text extraction.
        </p>
      </div>

      {/* Upload zone */}
      <UploadZone onFilesSelected={onFilesSelected} isLoading={isLoading} />

      {/* File type info cards */}
      <div className="grid md:grid-cols-2 gap-5 mt-10">
        <div
          className={`p-6 rounded-2xl border-2 transition-all duration-300 ${isPdf
              ? 'border-blue-500 bg-gradient-to-br from-blue-50 to-indigo-50 shadow-lg shadow-blue-500/10'
              : 'border-gray-200 bg-white/70 backdrop-blur-sm hover:border-blue-300 hover:shadow-md hover:-translate-y-0.5'
            }`}
        >
          <div className="flex items-start gap-4">
            <div
              className={`p-3.5 rounded-xl transition-all duration-300 ${isPdf
                  ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-lg shadow-blue-500/30'
                  : 'bg-gradient-to-br from-red-50 to-red-100 text-red-500'
                }`}
            >
              <FileText className="w-6 h-6" />
            </div>
            <div className="flex-1">
              <h3 className="font-bold text-gray-800 text-lg">PDF Document</h3>
              <p className="text-sm text-gray-500 mt-1.5 leading-relaxed">
                Upload a PDF and we'll extract pages as images. You can then
                select which pages to process.
              </p>
              {isPdf && (
                <div className="mt-3 inline-flex items-center gap-2 px-3 py-1.5 bg-green-100 text-green-700 rounded-full text-sm font-semibold">
                  <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                  PDF detected - Ready to extract
                </div>
              )}
            </div>
          </div>
        </div>

        <div
          className={`p-6 rounded-2xl border-2 transition-all duration-300 ${isImages
              ? 'border-blue-500 bg-gradient-to-br from-blue-50 to-indigo-50 shadow-lg shadow-blue-500/10'
              : 'border-gray-200 bg-white/70 backdrop-blur-sm hover:border-blue-300 hover:shadow-md hover:-translate-y-0.5'
            }`}
        >
          <div className="flex items-start gap-4">
            <div
              className={`p-3.5 rounded-xl transition-all duration-300 ${isImages
                  ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-lg shadow-blue-500/30'
                  : 'bg-gradient-to-br from-blue-50 to-blue-100 text-blue-500'
                }`}
            >
              <FileImage className="w-6 h-6" />
            </div>
            <div className="flex-1">
              <h3 className="font-bold text-gray-800 text-lg">Image Files</h3>
              <p className="text-sm text-gray-500 mt-1.5 leading-relaxed">
                Upload multiple images directly. All images will be added to
                your preprocessing queue.
              </p>
              {isImages && (
                <div className="mt-3 inline-flex items-center gap-2 px-3 py-1.5 bg-green-100 text-green-700 rounded-full text-sm font-semibold">
                  <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                  {files.length} image{files.length > 1 ? 's' : ''} selected
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Next button */}
      {hasFiles && (
        <div className="flex justify-center mt-10 animate-slide-up">
          <button
            onClick={onNext}
            disabled={isLoading}
            className="btn-gradient flex items-center gap-3 px-10 py-4 text-lg"
          >
            {isPdf ? 'Extract Pages' : 'Continue to Preprocessing'}
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>
      )}
    </div>
  );
}
