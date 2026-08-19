import React, { useCallback, useState, useRef } from 'react';
import {
  Upload,
  FileText,
  FileImage,
  BookOpen,
  ArrowRight,
  Check,
  AlertTriangle,
  Loader2,
  UploadCloud,
  X,
  FileScan,
  ScrollText,
  ScanSearch,
} from 'lucide-react';
import { API_ORIGIN } from '../config';

const API_BASE = API_ORIGIN;

// Book (PDF or images) and transcript uploaded together, before matching.
export default function CombinedUploadPage({
  onNext,
  isProcessingBook,
  initialBookFiles,
  initialTranscript,
  initialDatasetMode = 'recognition',
}) {
  // 'recognition' needs a transcript; 'detection' is boxes only.
  const [datasetMode, setDatasetMode] = useState(initialDatasetMode);
  const isDetection = datasetMode === 'detection';
  // ── Book upload ──
  const [bookFiles, setBookFiles] = useState(initialBookFiles || null);
  const [isDraggingBook, setIsDraggingBook] = useState(false);
  const bookInputRef = useRef(null);

  // ── Transcript upload ──
  const [transcriptFile, setTranscriptFile] = useState(null);
  const [parsedTranscript, setParsedTranscript] = useState(initialTranscript || null);
  const [isDraggingTranscript, setIsDraggingTranscript] = useState(false);
  const [isParsing, setIsParsing] = useState(false);
  const [transcriptError, setTranscriptError] = useState(null);
  const transcriptInputRef = useRef(null);

  // ── Book handlers ──
  const processBookFiles = (files) => {
    const valid = files.filter((f) => {
      const ext = f.name.split('.').pop().toLowerCase();
      return (
        f.type === 'application/pdf' ||
        f.type.startsWith('image/') ||
        f.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
        ext === 'docx'
      );
    });
    if (valid.length > 0) setBookFiles(valid);
  };

  const handleBookDrop = useCallback((e) => {
    e.preventDefault();
    setIsDraggingBook(false);
    processBookFiles(Array.from(e.dataTransfer.files));
  }, []);

  const handleBookInput = useCallback((e) => {
    processBookFiles(Array.from(e.target.files));
  }, []);

  // ── Transcript handlers ──
  const processTranscriptFile = async (file) => {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    const validExts = ['.txt', '.md', '.markdown', '.pdf', '.docx'];
    if (!validExts.includes(ext)) {
      setTranscriptError('Unsupported file type. Please upload TXT, DOCX, PDF, or Markdown.');
      return;
    }
    setTranscriptFile(file);
    setTranscriptError(null);
    setIsParsing(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_BASE}/api/dataset/parse-transcript`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('Server error: ' + res.statusText);
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Failed to parse transcript');
      setParsedTranscript(data.pages);
    } catch (err) {
      setTranscriptError(err.message);
      setParsedTranscript(null);
    } finally {
      setIsParsing(false);
    }
  };

  const handleTranscriptDrop = useCallback((e) => {
    e.preventDefault();
    setIsDraggingTranscript(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) processTranscriptFile(files[0]);
  }, []);

  const handleTranscriptInput = useCallback((e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) processTranscriptFile(files[0]);
  }, []);

  // ── Helpers ──
  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const bookType = bookFiles
    ? bookFiles[0].type === 'application/pdf'
      ? 'pdf'
      : bookFiles[0].type.startsWith('image/')
      ? 'images'
      : 'docx'
    : null;

  const transcriptPageCount = parsedTranscript ? Object.keys(parsedTranscript).length : 0;
  const totalTranscriptLines = parsedTranscript
    ? Object.values(parsedTranscript).reduce((s, l) => s + l.length, 0)
    : 0;

  // Detection mode needs no transcript.
  const canContinue = isDetection
    ? bookFiles && !isProcessingBook
    : bookFiles && parsedTranscript && !isProcessingBook && !isParsing;

  const handleContinue = () => {
    if (!canContinue) return;
    onNext(bookFiles, isDetection ? null : parsedTranscript, datasetMode);
  };

  return (
    <div className="flex flex-col h-full w-full overflow-hidden">
      {/* ── Page header ───────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-8 py-5 border-b border-gray-100/80 bg-white/60 backdrop-blur-sm flex-shrink-0 gap-6">
        <div className="flex items-center gap-4 min-w-0 flex-1">
          <div className="p-2.5 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl text-white shadow-lg shadow-emerald-500/30 flex-shrink-0">
            <Upload className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <h1 className="text-xl font-bold text-gray-800 truncate">
              {isDetection ? 'Upload Book' : 'Upload Book \u0026 Transcript'}
            </h1>
            <p className="text-sm text-gray-500 mt-0.5 truncate">
              {isDetection
                ? 'Detection mode: only images are needed — line bounding boxes will be exported, no transcript required'
                : 'Upload both files together — only pages with matching transcripts will be processed'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4 flex-shrink-0">
        {/* Dataset-type toggle */}
        <div className="flex bg-gray-100 rounded-lg p-1 flex-shrink-0">
          <button
            onClick={() => setDatasetMode('recognition')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
              !isDetection
                ? 'bg-white text-emerald-700 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            }`}
            title="Aligned line images + transcript labels"
          >
            <FileText className="w-3.5 h-3.5" />
            Recognition
          </button>
          <button
            onClick={() => setDatasetMode('detection')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
              isDetection
                ? 'bg-white text-indigo-700 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            }`}
            title="Pages + bounding boxes only (no transcript)"
          >
            <ScanSearch className="w-3.5 h-3.5" />
            Detection
          </button>
        </div>

        <button
          onClick={handleContinue}
          disabled={!canContinue}
          className={`flex items-center justify-center gap-2.5 px-7 py-3 rounded-xl font-bold text-sm transition-all duration-200 flex-shrink-0 whitespace-nowrap min-w-[210px] ${
            canContinue
              ? 'bg-gradient-to-r from-emerald-500 to-teal-600 text-white shadow-lg shadow-emerald-500/30 hover:shadow-xl hover:-translate-y-0.5'
              : 'bg-gray-100 text-gray-400 cursor-not-allowed'
          }`}
        >
          {isProcessingBook ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Processing…
            </>
          ) : (
            <>
              {isDetection ? 'Continue' : 'Review \u0026 Match Pages'}
              <ArrowRight className="w-4 h-4" />
            </>
          )}
        </button>
        </div>
      </div>

      {/* ── Main layout: two columns in recognition mode, single column in detection ── */}
      <div className={`flex-1 grid ${isDetection ? 'grid-cols-1' : 'grid-cols-2'} gap-0 overflow-hidden`}>

        {/* ── LEFT: Book Upload ──────────────────────────────────── */}
        <div className={`flex flex-col ${isDetection ? '' : 'border-r border-gray-100'} overflow-hidden bg-gradient-to-br from-slate-50 to-white`}>
          {/* Section header */}
          <div className="flex items-center gap-3 px-8 pt-7 pb-4 flex-shrink-0">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-100 to-indigo-100 flex items-center justify-center">
              <BookOpen className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <h2 className="font-bold text-gray-800">Book / Document</h2>
              <p className="text-xs text-gray-500">PDF, images, or DOCX of your book</p>
            </div>
            {bookFiles && (
              <div className="ml-auto flex items-center gap-1.5 px-2.5 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-semibold">
                <Check className="w-3.5 h-3.5" />
                Ready
              </div>
            )}
          </div>

          <div className="flex-1 px-8 pb-8 flex flex-col gap-5 overflow-y-auto">
            {/* Drop zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDraggingBook(true); }}
              onDragLeave={(e) => { e.preventDefault(); setIsDraggingBook(false); }}
              onDrop={handleBookDrop}
              onClick={() => !bookFiles && bookInputRef.current?.click()}
              className={`relative border-2 border-dashed rounded-2xl transition-all duration-300 cursor-pointer flex-shrink-0 ${
                isDraggingBook
                  ? 'border-blue-500 bg-blue-50/80 scale-[1.01] shadow-xl shadow-blue-500/20'
                  : bookFiles
                  ? 'border-blue-300 bg-blue-50/40 cursor-default'
                  : 'border-gray-300 bg-white/60 hover:border-blue-400 hover:bg-blue-50/20 hover:shadow-md'
              } ${isProcessingBook ? 'opacity-60 pointer-events-none' : ''}`}
            >
              <input
                ref={bookInputRef}
                type="file"
                multiple
                accept=".pdf,image/*,.docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                onChange={handleBookInput}
                className="hidden"
              />
              <div className="p-10 flex flex-col items-center text-center">
                {isProcessingBook ? (
                  <>
                    <Loader2 className="w-14 h-14 text-blue-500 animate-spin" />
                    <p className="mt-4 font-semibold text-blue-600">Extracting pages…</p>
                  </>
                ) : bookFiles ? (
                  <>
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/30">
                      {bookType === 'pdf' ? (
                        <FileText className="w-8 h-8 text-white" />
                      ) : (
                        <FileImage className="w-8 h-8 text-white" />
                      )}
                    </div>
                    <p className="mt-4 font-bold text-gray-800">
                      {bookFiles.length === 1 ? bookFiles[0].name : `${bookFiles.length} files selected`}
                    </p>
                    <p className="text-sm text-gray-500 mt-1">
                      {bookFiles.length === 1 ? formatFileSize(bookFiles[0].size) : ''}
                    </p>
                    <button
                      onClick={(e) => { e.stopPropagation(); setBookFiles(null); bookInputRef.current.value = ''; }}
                      className="mt-3 text-xs text-gray-400 hover:text-red-500 flex items-center gap-1 transition-colors"
                    >
                      <X className="w-3.5 h-3.5" /> Remove
                    </button>
                  </>
                ) : (
                  <>
                    <div className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-all duration-500 ${
                      isDraggingBook
                        ? 'bg-gradient-to-br from-blue-500 to-indigo-600 shadow-xl'
                        : 'bg-gradient-to-br from-blue-100 to-indigo-100'
                    }`}>
                      <UploadCloud className={`w-8 h-8 transition-all duration-300 ${isDraggingBook ? 'text-white animate-bounce' : 'text-blue-500'}`} />
                    </div>
                    <p className="mt-4 font-bold text-gray-700">
                      {isDraggingBook ? 'Drop here!' : 'Drag & drop book files'}
                    </p>
                    <p className="text-sm text-gray-400 mt-1">or click to browse</p>
                    <div className="mt-4 flex flex-wrap justify-center gap-2">
                      {['PDF', 'Images', 'DOCX'].map((type) => (
                        <span key={type} className="px-2.5 py-1 bg-gray-100 text-gray-500 text-xs rounded-full font-medium">
                          {type}
                        </span>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Book info cards */}
            {bookFiles && !isProcessingBook && (
              <div className="grid grid-cols-2 gap-3 animate-fade-in">
                <div className="p-4 bg-white rounded-xl border border-gray-100 shadow-sm text-center">
                  <p className="text-xs text-gray-500 uppercase tracking-wide">Type</p>
                  <p className="text-lg font-bold text-blue-600 mt-1 capitalize">
                    {bookType === 'pdf' ? 'PDF' : bookType === 'images' ? 'Images' : 'DOCX'}
                  </p>
                </div>
                <div className="p-4 bg-white rounded-xl border border-gray-100 shadow-sm text-center">
                  <p className="text-xs text-gray-500 uppercase tracking-wide">Files</p>
                  <p className="text-lg font-bold text-blue-600 mt-1">{bookFiles.length}</p>
                </div>
              </div>
            )}

            {/* Accepted formats hint */}
            <div className="p-4 bg-white/70 rounded-xl border border-gray-100 text-sm text-gray-500">
              <p className="font-semibold text-gray-700 mb-2 flex items-center gap-2">
                <FileScan className="w-4 h-4 text-blue-500" />
                Accepted formats
              </p>
              <ul className="space-y-1 text-xs">
                <li className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-blue-400" />PDF — pages extracted automatically</li>
                <li className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-blue-400" />Images (JPG, PNG, TIFF, WebP) — one image = one page</li>
                <li className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-blue-400" />DOCX — converted to images</li>
              </ul>
            </div>
          </div>
        </div>

        {/* ── RIGHT: Transcript Upload (recognition mode only) ───── */}
        {!isDetection && (
        <div className="flex flex-col overflow-hidden bg-gradient-to-br from-emerald-50/30 to-white">
          {/* Section header */}
          <div className="flex items-center gap-3 px-8 pt-7 pb-4 flex-shrink-0">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-100 to-teal-100 flex items-center justify-center">
              <ScrollText className="w-5 h-5 text-emerald-600" />
            </div>
            <div>
              <h2 className="font-bold text-gray-800">Transcript File</h2>
              <p className="text-xs text-gray-500">TXT, DOCX, PDF, or Markdown with page markers</p>
            </div>
            {parsedTranscript && (
              <div className="ml-auto flex items-center gap-1.5 px-2.5 py-1 bg-emerald-100 text-emerald-700 rounded-full text-xs font-semibold">
                <Check className="w-3.5 h-3.5" />
                Parsed
              </div>
            )}
          </div>

          <div className="flex-1 px-8 pb-8 flex flex-col gap-5 overflow-y-auto">
            {/* Drop zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDraggingTranscript(true); }}
              onDragLeave={(e) => { e.preventDefault(); setIsDraggingTranscript(false); }}
              onDrop={handleTranscriptDrop}
              onClick={() => !transcriptFile && transcriptInputRef.current?.click()}
              className={`relative border-2 border-dashed rounded-2xl transition-all duration-300 cursor-pointer flex-shrink-0 ${
                isDraggingTranscript
                  ? 'border-emerald-500 bg-emerald-50/80 scale-[1.01] shadow-xl shadow-emerald-500/20'
                  : transcriptFile
                  ? 'border-emerald-300 bg-emerald-50/40 cursor-default'
                  : 'border-gray-300 bg-white/60 hover:border-emerald-400 hover:bg-emerald-50/20 hover:shadow-md'
              } ${isParsing ? 'opacity-60 pointer-events-none' : ''}`}
            >
              <input
                ref={transcriptInputRef}
                type="file"
                accept=".txt,.md,.markdown,.pdf,.docx"
                onChange={handleTranscriptInput}
                className="hidden"
              />
              <div className="p-10 flex flex-col items-center text-center">
                {isParsing ? (
                  <>
                    <Loader2 className="w-14 h-14 text-emerald-500 animate-spin" />
                    <p className="mt-4 font-semibold text-emerald-600">Parsing transcript…</p>
                  </>
                ) : parsedTranscript ? (
                  <>
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/30">
                      <FileText className="w-8 h-8 text-white" />
                    </div>
                    <p className="mt-4 font-bold text-gray-800">{transcriptFile?.name}</p>
                    <p className="text-sm text-emerald-600 mt-1 font-semibold">
                      {transcriptPageCount} pages · {totalTranscriptLines} lines
                    </p>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setTranscriptFile(null);
                        setParsedTranscript(null);
                        transcriptInputRef.current.value = '';
                      }}
                      className="mt-3 text-xs text-gray-400 hover:text-red-500 flex items-center gap-1 transition-colors"
                    >
                      <X className="w-3.5 h-3.5" /> Remove
                    </button>
                  </>
                ) : (
                  <>
                    <div className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-all duration-500 ${
                      isDraggingTranscript
                        ? 'bg-gradient-to-br from-emerald-500 to-teal-600 shadow-xl'
                        : 'bg-gradient-to-br from-emerald-100 to-teal-100'
                    }`}>
                      <UploadCloud className={`w-8 h-8 transition-all duration-300 ${isDraggingTranscript ? 'text-white animate-bounce' : 'text-emerald-500'}`} />
                    </div>
                    <p className="mt-4 font-bold text-gray-700">
                      {isDraggingTranscript ? 'Drop here!' : 'Drag & drop transcript file'}
                    </p>
                    <p className="text-sm text-gray-400 mt-1">or click to browse</p>
                    <div className="mt-4 flex flex-wrap justify-center gap-2">
                      {['TXT', 'DOCX', 'PDF', 'MD'].map((type) => (
                        <span key={type} className="px-2.5 py-1 bg-gray-100 text-gray-500 text-xs rounded-full font-medium">
                          {type}
                        </span>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Transcript error */}
            {transcriptError && (
              <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-xl text-red-600 text-sm animate-fade-in">
                <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold">Parse Error</p>
                  <p className="text-xs mt-0.5">{transcriptError}</p>
                </div>
              </div>
            )}

            {/* Parsed transcript stats */}
            {parsedTranscript && (
              <div className="grid grid-cols-3 gap-3 animate-fade-in">
                <div className="p-4 bg-white rounded-xl border border-gray-100 shadow-sm text-center">
                  <p className="text-xs text-gray-500 uppercase tracking-wide">Pages</p>
                  <p className="text-lg font-bold text-emerald-600 mt-1">{transcriptPageCount}</p>
                </div>
                <div className="p-4 bg-white rounded-xl border border-gray-100 shadow-sm text-center">
                  <p className="text-xs text-gray-500 uppercase tracking-wide">Total Lines</p>
                  <p className="text-lg font-bold text-emerald-600 mt-1">{totalTranscriptLines}</p>
                </div>
                <div className="p-4 bg-white rounded-xl border border-gray-100 shadow-sm text-center">
                  <p className="text-xs text-gray-500 uppercase tracking-wide">Avg/Page</p>
                  <p className="text-lg font-bold text-emerald-600 mt-1">
                    {transcriptPageCount > 0 ? Math.round(totalTranscriptLines / transcriptPageCount) : 0}
                  </p>
                </div>
              </div>
            )}

            {/* Transcript page preview */}
            {parsedTranscript && (
              <div className="space-y-2 animate-fade-in">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide px-1">Preview</p>
                <div className="space-y-2 max-h-52 overflow-y-auto pr-1">
                  {Object.entries(parsedTranscript).slice(0, 8).map(([pageKey, lines]) => (
                    <div key={pageKey} className="p-3 bg-white rounded-xl border border-gray-100 shadow-sm">
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full">
                          Page {pageKey}
                        </span>
                        <span className="text-xs text-gray-400">{lines.length} lines</span>
                      </div>
                      <p className="text-xs text-gray-600 truncate leading-relaxed">
                        {lines[0] || '—'}
                      </p>
                    </div>
                  ))}
                  {transcriptPageCount > 8 && (
                    <p className="text-xs text-gray-400 text-center italic py-1">
                      +{transcriptPageCount - 8} more pages…
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Format guide */}
            <details className="bg-white/70 rounded-xl border border-gray-100 group">
              <summary className="px-4 py-3 cursor-pointer text-xs font-semibold text-gray-600 hover:text-emerald-600 transition-colors flex items-center gap-2 select-none">
                <ScrollText className="w-3.5 h-3.5" />
                How to format page markers
              </summary>
              <div className="px-4 pb-4 text-xs text-gray-500 space-y-2">
                <p>Use page markers on separate lines to split pages:</p>
                <div className="bg-gray-50 rounded-lg p-3 font-mono text-xs space-y-0.5 border border-gray-100">
                  <p className="text-emerald-600 font-bold">PDF p1</p>
                  <p className="text-gray-600">First line of page 1…</p>
                  <p className="text-emerald-600 font-bold mt-1.5">PDF p2</p>
                  <p className="text-gray-600">First line of page 2…</p>
                </div>
                <p className="text-gray-400">Also supports: <code className="bg-gray-100 px-1 rounded">--- Page N ---</code>, <code className="bg-gray-100 px-1 rounded">[Page N]</code></p>
              </div>
            </details>
          </div>
        </div>
        )}
      </div>

      {/* ── Bottom status bar ─────────────────────────────────────── */}
      {(bookFiles || (!isDetection && parsedTranscript)) && (
        <div className="flex items-center justify-between px-8 py-3 border-t border-gray-100 bg-white/80 backdrop-blur-sm flex-shrink-0 animate-slide-up">
          <div className="flex items-center gap-6">
            <StatusDot
              label="Book uploaded"
              active={!!bookFiles}
              activeColor="blue"
            />
            {!isDetection && (
              <StatusDot
                label="Transcript parsed"
                active={!!parsedTranscript}
                activeColor="emerald"
              />
            )}
          </div>
          {canContinue && !isDetection && (
            <p className="text-xs text-gray-500">
              Both files ready — click <span className="font-semibold text-emerald-600">Review &amp; Match Pages</span> to continue
            </p>
          )}
          {canContinue && isDetection && (
            <p className="text-xs text-gray-500">
              Book ready — click <span className="font-semibold text-indigo-600">Continue</span> to proceed
            </p>
          )}
          {!canContinue && bookFiles && !isDetection && !parsedTranscript && (
            <p className="text-xs text-amber-600 font-medium">Upload a transcript to continue</p>
          )}
          {!canContinue && !bookFiles && !isDetection && parsedTranscript && (
            <p className="text-xs text-amber-600 font-medium">Upload the book to continue</p>
          )}
        </div>
      )}
    </div>
  );
}

function StatusDot({ label, active, activeColor }) {
  const colors = {
    blue: active ? 'bg-blue-500' : 'bg-gray-200',
    emerald: active ? 'bg-emerald-500' : 'bg-gray-200',
  };
  return (
    <div className="flex items-center gap-2">
      <span className={`w-2 h-2 rounded-full transition-colors duration-300 ${colors[activeColor]}`} />
      <span className={`text-xs font-medium transition-colors duration-300 ${active ? 'text-gray-700' : 'text-gray-400'}`}>
        {label}
      </span>
    </div>
  );
}
