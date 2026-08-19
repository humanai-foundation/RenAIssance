import React, { useState, useCallback, useEffect } from 'react';
import { FileText, Database, BookOpen } from 'lucide-react';
import Stepper from './components/Stepper';
import DatasetStepper from './components/DatasetStepper';
import HomePage from './pages/HomePage';
import MyFilesPage from './pages/MyFilesPage';
import CombinedUploadPage from './pages/CombinedUploadPage';
import PageMatchReviewPage from './pages/PageMatchReviewPage';
import DatasetGenerationPage from './pages/DatasetGenerationPage';
import UploadPage from './features/upload/pages/UploadPage';
import SelectPage from './features/upload/pages/SelectPage';
import PreprocessPage from './features/preprocess/pages/PreprocessPage';
import TextDetectionPage from './features/ocr/pages/TextDetectionPage';
import TextRecognitionPage from './features/ocr/pages/TextRecognitionPage';
import LayoutAwareDetectionPage from './components/LayoutAwareDetectionPanel';
import { usePdfPreview } from './hooks/usePdfPreview';
import { revokeObjectUrls } from './utils/imageUrl';
import AuthPage from './features/auth/AuthPage';
import ProfilePage from './features/auth/ProfilePage';
import { fetchMe, logout as apiLogout } from './features/auth/authApi';


function App() {
  // ── Auth gate ──
  const [user, setUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [showProfile, setShowProfile] = useState(false);

  // null (home) | 'ocr' | 'dataset' | 'files'
  const [mode, setMode] = useState(null);

  // ── Shared state ──
  const [currentStep, setCurrentStep] = useState(1);
  const [files, setFiles] = useState(null);
  const [selectedPages, setSelectedPages] = useState([]);
  const [processedImages, setProcessedImages] = useState({});
  const [detectionMethod, setDetectionMethod] = useState(null);
  const [detectionProvider, setDetectionProvider] = useState(null);

  // ── Dataset state ──
  // 'recognition' needs a transcript and runs match review; 'detection' is
  // boxes only, so it skips both.
  const [datasetMode, setDatasetMode] = useState('recognition');
  const [parsedTranscript, setParsedTranscript] = useState(null);
  const [allPagesBoxes, setAllPagesBoxes] = useState({});
  const [alignedTranscriptByPage, setAlignedTranscriptByPage] = useState({});
  const [isProcessingBook, setIsProcessingBook] = useState(false);
  // Survives step 4 <-> 5 navigation.
  const [detectionCache, setDetectionCache] = useState({ pages: {}, alignment: {} });

  const [ocrDetectionCache, setOcrDetectionCache] = useState({ pages: {}, alignment: {} });

  // Both end up as My Files metadata/tags.
  const [preprocessing, setPreprocessing] = useState([]);
  const [detectionModelInfo, setDetectionModelInfo] = useState({});

  const {
    pages,
    isLoading,
    error,
    progress,
    extractPages,
    loadImages,
    reset: resetPdfPreview,
  } = usePdfPreview();

  const handleFilesSelected = useCallback((selectedFiles) => {
    setFiles(selectedFiles);
    setSelectedPages([]);
  }, []);

  // ── Dataset: combined upload ──
  const handleCombinedUploadNext = useCallback(async (bookFiles, transcript, mode = 'recognition') => {
    if (!bookFiles || bookFiles.length === 0) return;
    setIsProcessingBook(true);
    setDatasetMode(mode);
    setParsedTranscript(mode === 'detection' ? null : transcript);
    try {
      const isPdf = bookFiles[0].type === 'application/pdf';
      let loadedPages;
      if (isPdf) {
        loadedPages = await extractPages(bookFiles[0]);
      } else {
        loadedPages = await loadImages(bookFiles);
      }
      // Detection mode uses step 2 as a Select-Pages step, so preselect
      // everything and let the user deselect. Recognition mode picks in review.
      if (mode === 'detection' || !isPdf) {
        setSelectedPages((loadedPages || []).map((p) => p.pageNumber));
      }
      setFiles(bookFiles);
      setCurrentStep(2);
    } catch (err) {
      console.error('Failed to process book files:', err);
    } finally {
      setIsProcessingBook(false);
    }
  }, [extractPages, loadImages]);

  // ── Dataset: after match review ──
  const handleMatchReviewNext = useCallback((matchedPageNumbers, updatedTranscript) => {
    setSelectedPages(matchedPageNumbers);
    if (updatedTranscript) setParsedTranscript(updatedTranscript);
    setCurrentStep(3);
  }, []);


  const handleUploadNext = useCallback(async () => {
    if (!files || files.length === 0) return;

    const isPdf = files[0].type === 'application/pdf';

    if (isPdf) {
      try {
        await extractPages(files[0]);
        setCurrentStep(2);
      } catch (err) {
        console.error('Failed to extract PDF:', err);
      }
    } else {
      try {
        const loadedPages = await loadImages(files);
        setSelectedPages(loadedPages.map((p) => p.pageNumber));
        setCurrentStep(2);
      } catch (err) {
        console.error('Failed to load images:', err);
      }
    }
  }, [files, extractPages, loadImages, mode]);

  const handleSelectionChange = useCallback((newSelection) => {
    setSelectedPages(newSelection);
  }, []);

  const handleStepClick = useCallback((stepId) => {
    if (stepId <= currentStep) {
      setCurrentStep(stepId);
    }
  }, [currentStep]);

  const goToStep = useCallback((step) => {
    setCurrentStep(step);
  }, []);

  const handleReset = useCallback(() => {
    setMode(null);
    setCurrentStep(1);
    setFiles(null);
    setSelectedPages([]);
    // Object URLs leak unless we revoke them before the next book.
    revokeObjectUrls(Object.values(processedImages));
    setProcessedImages({});
    setDetectionMethod(null);
    setDetectionProvider(null);
    setDatasetMode('recognition');
    setParsedTranscript(null);
    setAllPagesBoxes({});
    setAlignedTranscriptByPage({});
    setIsProcessingBook(false);
    setOcrDetectionCache({ pages: {}, alignment: {} });
    setDetectionCache({ pages: {}, alignment: {} });
    setPreprocessing([]);
    setDetectionModelInfo({});
    resetPdfPreview();
  }, [resetPdfPreview, processedImages]);

  // ── Mode selection ──
  const handleSelectMode = useCallback((selectedMode) => {
    setMode(selectedMode);
    setCurrentStep(1);
  }, []);

  // ── Check for an existing session on load ──
  useEffect(() => {
    let active = true;
    fetchMe().then((u) => {
      if (active) {
        setUser(u);
        setAuthChecked(true);
      }
    });
    return () => {
      active = false;
    };
  }, []);

  const handleLogout = useCallback(async () => {
    try {
      await apiLogout();
    } finally {
      handleReset();
      setUser(null);
    }
  }, [handleReset]);

  // Must be logged in before anything else renders.
  if (!authChecked) {
    return (
      <div className="min-h-screen w-screen flex items-center justify-center bg-gradient-to-br from-slate-50 via-white to-blue-50">
        <div className="h-10 w-10 rounded-full border-4 border-blue-200 border-t-blue-600 animate-spin" />
      </div>
    );
  }

  if (!user) {
    return <AuthPage onAuthSuccess={setUser} />;
  }

  // No mode picked yet -> home.
  if (!mode) {
    return (
      <>
        <HomePage
          onSelectMode={handleSelectMode}
          user={user}
          onLogout={handleLogout}
          onProfile={() => setShowProfile(true)}
        />
        {showProfile && (
          <ProfilePage
            user={user}
            onClose={() => setShowProfile(false)}
            onUpdated={setUser}
          />
        )}
      </>
    );
  }

  if (mode === 'files') {
    return <MyFilesPage onBack={handleReset} />;
  }

  // ── Dataset mode: upload -> match review -> preprocess -> detect -> export ──
  if (mode === 'dataset') {

    // ── Full-screen steps ──
    if (currentStep === 3) {
      const backStep = 2;
      return (
        <div className="h-screen w-screen overflow-hidden flex flex-col">
          <PreprocessPage
            pages={pages}
            selectedPages={selectedPages}
            initialProcessedImages={processedImages}
            onBack={() => goToStep(backStep)}
            onNext={() => goToStep(4)}
            onProcessedImagesChange={setProcessedImages}
            onPipelineChange={setPreprocessing}
          />
        </div>
      );
    }

    if (currentStep === 4) {
      return (
        <div className="h-screen w-screen overflow-hidden flex flex-col">
          <LayoutAwareDetectionPage
            pages={pages}
            selectedPages={selectedPages}
            processedImages={processedImages}
            transcript={parsedTranscript}
            preprocessing={preprocessing}
            onBack={() => goToStep(3)}
            onHome={handleReset}
            datasetMode={true}
            initialDetectedPages={detectionCache.pages}
            initialAlignmentByPage={detectionCache.alignment}
            onStateChange={(cache) => setDetectionCache(cache)}
            onDatasetNext={({ boxesByPage, alignedTranscriptByPage: alignedMap, modelInfo }) => {
              setAllPagesBoxes(boxesByPage || {});
              setAlignedTranscriptByPage(alignedMap || {});
              if (modelInfo) setDetectionModelInfo(modelInfo);
              goToStep(5);
            }}
          />
        </div>
      );
    }

    if (currentStep === 5) {
      return (
        <div className="h-screen w-screen overflow-hidden flex flex-col bg-gradient-to-br from-slate-50 via-white to-emerald-50">
          <DatasetGenerationPage
            pages={pages}
            selectedPages={selectedPages}
            processedImages={processedImages}
            transcript={Object.keys(alignedTranscriptByPage).length > 0 ? alignedTranscriptByPage : parsedTranscript}
            allPagesBoxes={allPagesBoxes}
            preprocessing={preprocessing}
            modelInfo={detectionModelInfo}
            onBack={() => goToStep(4)}
            onHome={handleReset}
            bookName={files?.[0]?.name?.replace(/\.[^.]+$/, '') || 'dataset'}
            forceMode={datasetMode === 'detection' ? 'detection' : null}
          />
        </div>
      );
    }

    // ── Steps 1 & 2 share a layout ──
    return (
      <div className="h-screen w-screen flex flex-col overflow-hidden bg-gradient-to-br from-slate-50 via-white to-emerald-50/30">
        {/* ── Compact header ─────────────────────────────────────── */}
        <header className="flex-shrink-0 bg-white/90 backdrop-blur-md border-b border-gray-100/70 shadow-sm z-40">
          <div className="flex items-center justify-between px-6 py-3 gap-6">
            {/* Logo */}
            <div className="flex items-center gap-3 flex-shrink-0">
              <div className="w-9 h-9 bg-gradient-to-br from-emerald-500 via-teal-500 to-emerald-600 rounded-xl flex items-center justify-center shadow-md shadow-emerald-500/25">
                <Database className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-sm font-bold text-gray-800 leading-none">Dataset Generator</p>
                <p className="text-[10px] text-gray-400 mt-0.5">OCR training data pipeline</p>
              </div>
            </div>

            {/* Stepper — centered, full width */}
            <div className="flex-1">
              <DatasetStepper currentStep={currentStep} onStepClick={handleStepClick} compact datasetMode={datasetMode} />
            </div>

            {/* Home */}
            <button
              onClick={handleReset}
              className="flex-shrink-0 text-sm font-medium text-gray-500 hover:text-emerald-600 hover:bg-emerald-50 px-4 py-2 rounded-lg transition-all duration-200"
            >
              Home
            </button>
          </div>
        </header>

        {/* ── Error/progress banner ──────────────────────────────── */}
        {(error || (isLoading && progress > 0)) && (
          <div className="flex-shrink-0 px-6 pt-3">
            {error && (
              <div className="p-3 bg-red-50/90 border border-red-200 rounded-xl text-red-600 text-sm animate-fade-in shadow-sm mb-2">
                <span className="font-semibold">Error: </span>{error}
              </div>
            )}
            {isLoading && progress > 0 && (
              <div className="p-3 bg-white/80 backdrop-blur-sm border border-gray-100 rounded-xl animate-fade-in shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-gray-600">Extracting pages…</span>
                  <span className="text-xs font-bold text-emerald-600">{progress}%</span>
                </div>
                <div className="h-1.5 bg-emerald-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Page content (full remaining height) ──────────────── */}
        <main className="flex-1 overflow-hidden">
          {currentStep === 1 && (
            <CombinedUploadPage
              onNext={handleCombinedUploadNext}
              isProcessingBook={isProcessingBook || isLoading}
              initialBookFiles={files}
              initialTranscript={parsedTranscript}
              initialDatasetMode={datasetMode}
            />
          )}
          {currentStep === 2 && datasetMode === 'detection' && (
            <SelectPage
              pages={pages}
              selectedPages={selectedPages}
              onSelectionChange={handleSelectionChange}
              onBack={() => goToStep(1)}
              onNext={() => goToStep(3)}
              isLoading={isLoading}
            />
          )}
          {currentStep === 2 && datasetMode !== 'detection' && (
            <PageMatchReviewPage
              pages={pages}
              parsedTranscript={parsedTranscript}
              onBack={() => goToStep(1)}
              onNext={handleMatchReviewNext}
            />
          )}
        </main>
      </div>
    );
  }

  // ── OCR mode ──
  return (
    <div className="min-h-screen flex flex-col">
      {currentStep === 3 ? (
        <div className="h-screen w-screen overflow-hidden flex flex-col">
          <PreprocessPage
            pages={pages}
            selectedPages={selectedPages}
            initialProcessedImages={processedImages}
            onBack={() => goToStep(2)}
            onNext={() => goToStep(4)}
            onProcessedImagesChange={setProcessedImages}
            onPipelineChange={setPreprocessing}
          />
        </div>
      ) : currentStep === 6 ? (
        <div className="h-screen w-screen overflow-hidden flex flex-col">
          <LayoutAwareDetectionPage
            pages={pages}
            selectedPages={selectedPages}
            processedImages={processedImages}
            preprocessing={preprocessing}
            bookName={files?.[0]?.name?.replace(/\.[^.]+$/, '') || 'transcript'}
            onBack={() => goToStep(4)}
            onHome={handleReset}
            initialDetectedPages={ocrDetectionCache.pages}
            initialAlignmentByPage={ocrDetectionCache.alignment}
            onStateChange={(cache) => {
              setOcrDetectionCache(cache);
            }}
          />
        </div>
      ) : currentStep === 5 ? (
        <div className="h-screen w-screen overflow-hidden">
          <TextRecognitionPage
            provider={detectionProvider || 'gemini'}
            bookName={files?.[0]?.name?.replace(/\.[^.]+$/, '') || 'transcript'}
            processedImages={
              selectedPages.map((pageNum) => {
                const original = pages.find(p => p.pageNumber === pageNum);
                const preprocessed = processedImages[pageNum];
                return {
                  pageNumber: pageNum,
                  originalPageNumber: original?.originalPageNumber || null,
                  isSplit: original?.isSplit || false,
                  splitSide: original?.splitSide || null,
                  original: original?.thumbnail || original,
                  processed: preprocessed || original?.thumbnail || original,
                };
              })
            }
            onBack={() => goToStep(4)}
            onHome={handleReset}
            onComplete={(message) => {
              alert(message || 'OCR processing complete! Transcripts have been saved.');
            }}
          />
        </div>
      ) : (
        <>
          {/* Header */}
          <header className="bg-white/80 backdrop-blur-md border-b border-gray-100/50 sticky top-0 z-40 shadow-sm">
            <div className="px-6 lg:px-10 xl:px-16 py-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-11 h-11 bg-gradient-to-br from-blue-500 via-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/30">
                  <FileText className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
                    OCR Preprocess Studio
                  </h1>
                  <p className="text-xs text-gray-500">
                    Prepare documents for text extraction
                  </p>
                </div>
              </div>

              <button
                onClick={handleReset}
                className="text-sm font-medium text-gray-500 hover:text-blue-600 hover:bg-blue-50 px-4 py-2 rounded-lg transition-all duration-200"
              >
                Home
              </button>
            </div>
          </header>

          {/* Main content */}
          <main className="flex-1 w-full px-6 lg:px-10 xl:px-16 py-6">
            <Stepper currentStep={currentStep} onStepClick={handleStepClick} />

            {error && (
              <div className="mb-6 p-4 bg-red-50/80 backdrop-blur-sm border border-red-200 rounded-xl text-red-600 animate-fade-in shadow-sm">
                <p className="font-semibold">Error</p>
                <p className="text-sm mt-1">{error}</p>
              </div>
            )}

            {isLoading && progress > 0 && (
              <div className="mb-6 animate-fade-in">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-gray-700">
                    Processing...
                  </span>
                  <span className="text-sm font-bold text-blue-600">{progress}%</span>
                </div>
                <div className="h-2.5 bg-blue-100 rounded-full overflow-hidden shadow-inner">
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full transition-all duration-300 shadow-sm"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}

            <div className="min-h-[600px]">
              {currentStep === 1 && (
                <UploadPage
                  files={files}
                  onFilesSelected={handleFilesSelected}
                  onNext={handleUploadNext}
                  isLoading={isLoading}
                />
              )}

              {currentStep === 2 && (
                <SelectPage
                  pages={pages}
                  selectedPages={selectedPages}
                  onSelectionChange={handleSelectionChange}
                  onBack={() => goToStep(1)}
                  onNext={() => goToStep(3)}
                  isLoading={isLoading}
                />
              )}

              {currentStep === 4 && (
                <TextDetectionPage
                  pages={pages}
                  selectedPages={selectedPages}
                  processedImages={processedImages}
                  onBack={() => goToStep(3)}
                  onNext={(method, provider) => {
                    setDetectionMethod(method);
                    setDetectionProvider(provider);
                    if (method === 'layout-aware') {
                      goToStep(6);
                    } else {
                      goToStep(5);
                    }
                  }}
                />
              )}
            </div>
          </main>

          {/* Footer */}
          <footer className="bg-white/60 backdrop-blur-sm border-t border-gray-100 py-4">
            <div className="px-6 lg:px-10 xl:px-16 text-center">
              <p className="text-sm text-gray-500 flex items-center justify-center gap-2">
                <span className="font-semibold text-gray-600">OCR Preprocess Studio</span>
                <span className="text-gray-300">|</span>
                <span className="text-blue-600 font-medium">RenAIssance Project</span>
              </p>
            </div>
          </footer>
        </>
      )}
    </div>
  );
}

export default App;
