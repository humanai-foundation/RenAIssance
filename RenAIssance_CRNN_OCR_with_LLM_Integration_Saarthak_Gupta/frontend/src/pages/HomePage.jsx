import React from 'react';
import { FileText, Database, ArrowRight, BookOpen, Archive, LogOut, UserCircle } from 'lucide-react';
import homeImage from '../assets/home-image.png';

// Landing hero plus the two workflow cards.
export default function HomePage({ onSelectMode, user, onLogout, onProfile }) {
  return (
    <div className="relative h-screen w-screen overflow-hidden flex flex-col bg-gradient-to-br from-slate-50 via-white to-blue-50">
      {/* Decorative background blobs */}
      <div className="pointer-events-none absolute -top-40 -left-40 w-[34rem] h-[34rem] rounded-full bg-blue-200/40 blur-3xl" aria-hidden="true" />
      <div className="pointer-events-none absolute -bottom-40 -right-40 w-[36rem] h-[36rem] rounded-full bg-indigo-200/40 blur-3xl" aria-hidden="true" />

      {/* Header */}
      <header className="relative z-10 bg-white/80 backdrop-blur-md border-b border-gray-100/60 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-center relative">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 bg-gradient-to-br from-blue-500 via-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/30">
              <BookOpen className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
                RenAIssance OCR
              </h1>
              <p className="text-xs text-gray-500">Historical document processing</p>
            </div>
          </div>

          {/* Signed-in user + profile + logout */}
          {user && (
            <div className="absolute right-6 top-1/2 -translate-y-1/2 flex items-center gap-1.5">
              <button
                onClick={onProfile}
                className="flex items-center gap-1.5 text-sm font-medium text-gray-600 hover:text-blue-600 hover:bg-blue-50 px-3 py-2 rounded-lg transition-all duration-200"
                title="Edit your profile"
              >
                <UserCircle className="w-4 h-4" />
                <span className="hidden sm:inline font-semibold text-gray-800">
                  {user.name || user.username}
                </span>
              </button>
              <button
                onClick={onLogout}
                className="flex items-center gap-1.5 text-sm font-medium text-gray-500 hover:text-blue-600 hover:bg-blue-50 px-3 py-2 rounded-lg transition-all duration-200"
              >
                <LogOut className="w-4 h-4" />
                <span className="hidden sm:inline">Log out</span>
              </button>
            </div>
          )}
        </div>
      </header>

      {/* Main */}
      <main className="relative z-10 flex-1 min-h-0 grid lg:grid-cols-2 gap-8 items-stretch px-6 lg:px-16 py-6 overflow-hidden">
        {/* Left: decorative image only */}
        <div className="flex flex-col min-h-0 h-full justify-start">
          <div className="relative flex-1 min-h-[18rem] rounded-2xl overflow-hidden border border-blue-100 shadow-xl shadow-blue-500/10">
            <img
              src={homeImage}
              alt="Historical manuscript"
              className="absolute inset-0 w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-blue-900/30 via-transparent to-transparent" />
          </div>
        </div>

        {/* Right: mode cards */}
        <div className="flex flex-col min-h-0 h-full justify-start gap-5">
          {/* Dataset */}
          <button
            onClick={() => onSelectMode('dataset')}
            className="group relative text-left p-6 rounded-2xl border border-gray-200 bg-white/90 backdrop-blur-sm hover:border-blue-400 hover:shadow-2xl hover:shadow-blue-500/15 hover:-translate-y-0.5 transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-blue-100"
          >
            <div className="flex items-start gap-5">
              <div className="p-3.5 rounded-xl bg-gradient-to-br from-blue-50 to-indigo-50 text-blue-600 group-hover:from-blue-500 group-hover:to-indigo-600 group-hover:text-white group-hover:shadow-lg group-hover:shadow-blue-500/30 transition-all duration-300 flex-shrink-0">
                <Database className="w-7 h-7" strokeWidth={2} />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-xl font-bold text-gray-800 mb-1">Generate OCR Dataset</h3>
                <p className="text-sm text-gray-500 leading-relaxed">
                  Build line-level training data from book pages and transcripts.
                  Export in TrOCR, CRNN, YOLO, or COCO formats.
                </p>
                <div className="mt-4 flex items-center gap-1.5 text-blue-600 font-semibold text-sm group-hover:gap-3 transition-all">
                  Start dataset workflow
                  <ArrowRight className="w-4 h-4" />
                </div>
              </div>
            </div>
          </button>

          {/* OCR */}
          <button
            onClick={() => onSelectMode('ocr')}
            className="group relative text-left p-6 rounded-2xl border border-gray-200 bg-white/90 backdrop-blur-sm hover:border-indigo-400 hover:shadow-2xl hover:shadow-indigo-500/15 hover:-translate-y-0.5 transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-indigo-100"
          >
            <div className="flex items-start gap-5">
              <div className="p-3.5 rounded-xl bg-gradient-to-br from-indigo-50 to-blue-50 text-indigo-600 group-hover:from-indigo-500 group-hover:to-blue-600 group-hover:text-white group-hover:shadow-lg group-hover:shadow-indigo-500/30 transition-all duration-300 flex-shrink-0">
                <FileText className="w-7 h-7" strokeWidth={2} />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-xl font-bold text-gray-800 mb-1">Transcribe Pages</h3>
                <p className="text-sm text-gray-500 leading-relaxed">
                  Extract text using Gemini, ChatGPT, or local PaddleOCR / TrOCR
                  models. Export as TXT, DOCX, or PDF.
                </p>
                <div className="mt-4 flex items-center gap-1.5 text-indigo-600 font-semibold text-sm group-hover:gap-3 transition-all">
                  Start OCR workflow
                  <ArrowRight className="w-4 h-4" />
                </div>
              </div>
            </div>
          </button>

          {/* My Files */}
          <button
            onClick={() => onSelectMode('files')}
            className="group relative text-left p-6 rounded-2xl border border-gray-200 bg-white/90 backdrop-blur-sm hover:border-cyan-400 hover:shadow-2xl hover:shadow-cyan-500/15 hover:-translate-y-0.5 transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-cyan-100"
          >
            <div className="flex items-start gap-5">
              <div className="p-3.5 rounded-xl bg-gradient-to-br from-cyan-50 to-blue-50 text-cyan-700 group-hover:from-cyan-500 group-hover:to-blue-600 group-hover:text-white group-hover:shadow-lg group-hover:shadow-cyan-500/30 transition-all duration-300 flex-shrink-0">
                <Archive className="w-7 h-7" strokeWidth={2} />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-xl font-bold text-gray-800 mb-1">My Files</h3>
                <p className="text-sm text-gray-500 leading-relaxed">
                  Browse saved transcription sessions and generated datasets.
                  View, download, or delete persistent artifacts.
                </p>
                <div className="mt-4 flex items-center gap-1.5 text-cyan-700 font-semibold text-sm group-hover:gap-3 transition-all">
                  Open file manager
                  <ArrowRight className="w-4 h-4" />
                </div>
              </div>
            </div>
          </button>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 bg-white/60 backdrop-blur-sm border-t border-gray-100 py-3">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <p className="text-sm text-gray-500 flex items-center justify-center gap-2">
            <span className="font-semibold text-gray-600">RenAIssance OCR</span>
            <span className="text-gray-300">|</span>
            <span className="text-blue-600 font-medium">Historical Document Processing</span>
          </p>
        </div>
      </footer>
    </div>
  );
}
