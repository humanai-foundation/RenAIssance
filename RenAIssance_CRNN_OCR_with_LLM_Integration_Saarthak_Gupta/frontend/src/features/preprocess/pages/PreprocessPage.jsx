import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  Crop,
  Play,
  SkipForward,
  Sparkles,
  Layers,
  Search,
  Undo2,
  Redo2,
  RotateCcw,
  Info,
  X,
  AlertTriangle,
  Eraser,
  BookOpen,
  ChevronDown,
} from 'lucide-react';

import { getBookTypePresets, getBookTypePipeline } from '../../../config/preprocessOperations';

import {
  OperationsSidebar,
  PipelineStack,
  BeforeAfterViewer,
  ZoomLensPreview,
  ProgressBarLabeled,
  ImageEraser,
} from '../../../components/preprocess';

import ImageCropper from '../../../components/ImageCropper';

import { usePipeline } from '../../../hooks/usePipeline';
import { useHistoryStore, HistoryOps } from '../../../hooks/useHistoryStore';
import { preprocessImage } from '../../../services/api';

// "1_left" -> "1L", "2_right" -> "2R"; plain numbers pass through.
function shortPageLabel(pageNumber) {
  const s = String(pageNumber);
  return s.replace('_left', 'L').replace('_right', 'R');
}

// ── Dismissible tip banner ──
function InfoBanner({ message, onDismiss, variant = 'info' }) {
  const variants = {
    info: {
      bg: 'bg-blue-50',
      border: 'border-blue-200',
      text: 'text-blue-800',
      icon: 'text-blue-500',
    },
    warning: {
      bg: 'bg-amber-50',
      border: 'border-amber-200',
      text: 'text-amber-800',
      icon: 'text-amber-500',
    },
  };

  const style = variants[variant] || variants.info;
  const Icon = variant === 'warning' ? AlertTriangle : Info;

  return (
    <div className={`${style.bg} ${style.border} border rounded-lg px-4 py-3 flex items-start gap-3`}>
      <Icon className={`w-5 h-5 ${style.icon} flex-shrink-0 mt-0.5`} />
      <p className={`text-sm ${style.text} flex-1`}>{message}</p>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className={`${style.text} hover:opacity-70 transition-opacity p-0.5`}
          aria-label="Dismiss"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
}

// The preprocessing editor. Ops chain onto each page's current working image,
// and every edit goes through the shared undo/redo store.
export default function PreprocessPage({
  pages,
  selectedPages,
  initialProcessedImages,
  onBack,
  onNext,
  onProcessedImagesChange,
  onPipelineChange,
}) {
  // ── State ──
  const [currentPageIndex, setCurrentPageIndex] = useState(0);

  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [currentProcessingStep, setCurrentProcessingStep] = useState(null);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });

  // { pageNumber: { current, original } }. Undo history lives in `history`.
  const [imageStates, setImageStates] = useState({});

  // Seeded from the parent so leaving and returning keeps progress.
  const [processedImages, setProcessedImages] = useState(() => initialProcessedImages || {});

  // One timeline for every page's edits.
  const history = useHistoryStore();

  const [showCropper, setShowCropper] = useState(false);
  const [showEraser, setShowEraser] = useState(false);
  const [showZoomLens, setShowZoomLens] = useState(false);
  const [showPipelinePanel, setShowPipelinePanel] = useState(true);
  const [showTipBanner, setShowTipBanner] = useState(true);

  // ── Derived data ──

  const selectedPageData = useMemo(() =>
    selectedPages.map((pageNum) => ({
      pageNumber: pageNum,
      ...(pages.find(p => p.pageNumber === pageNum) || {}),
    })),
    [selectedPages, pages]
  );

  const currentPage = selectedPageData[currentPageIndex];

  const getOrCreateImageState = useCallback((pageNumber, originalImage) => {
    if (!imageStates[pageNumber]) {
      return {
        current: originalImage,
        original: originalImage,
        history: [],
      };
    }
    return imageStates[pageNumber];
  }, [imageStates]);

  const currentImageState = useMemo(() => {
    if (!currentPage) return null;
    const state = imageStates[currentPage.pageNumber];
    return state?.current || currentPage.thumbnail;
  }, [currentPage, imageStates]);

  const originalImageState = useMemo(() => {
    if (!currentPage) return null;
    return currentPage.thumbnail;
  }, [currentPage]);

  // history.version is in the deps so the memo invalidates on timeline changes.
  const canUndo = useMemo(() => (
    currentPage ? history.canUndoPage(currentPage.pageNumber) : false
  ), [currentPage, history, history.version]); // eslint-disable-line react-hooks/exhaustive-deps

  const canRedo = useMemo(() => (
    currentPage ? history.canRedoPage(currentPage.pageNumber) : false
  ), [currentPage, history, history.version]); // eslint-disable-line react-hooks/exhaustive-deps

  const isModified = useMemo(() => {
    if (!currentPage) return false;
    const state = imageStates[currentPage.pageNumber];
    return state?.current && state.current !== currentPage.thumbnail;
  }, [currentPage, imageStates]);

  // ── Pipeline ──

  const {
    pipeline,
    enabledOperations,
    toggleOperation,
    updateOperationParams,
    togglePipelineStep,
    updatePipelineStepParams,
    removePipelineStep,
    reorderPipeline,
    applyRecommendedPipeline,
    applyPipeline,
    resetPipeline,
    getActivePipeline,
    buildPipelineConfig,
  } = usePipeline({
    debounceMs: 600,
    onPreviewRequest: () => {
      // Auto-preview off — too slow on full-res scans.
    },
  });

  // ── Book-type presets (Recommendation dropdown) ──
  const [showBookTypeMenu, setShowBookTypeMenu] = useState(false);
  const bookTypePresets = getBookTypePresets();

  const handleSelectBookType = useCallback((bookTypeId) => {
    // Select-only: fill the pipeline, don't run. User reviews then processes.
    applyPipeline(getBookTypePipeline(bookTypeId));
    setShowBookTypeMenu(false);
  }, [applyPipeline]);

  // ── Effects ──
  // Push processed AND crop-only images up. The callback is pinned to a ref so
  // a new prop identity doesn't re-fire this — the data maps are the real deps.
  const onProcessedImagesChangeRef = useRef(onProcessedImagesChange);
  useEffect(() => { onProcessedImagesChangeRef.current = onProcessedImagesChange; }, [onProcessedImagesChange]);

  useEffect(() => {
    if (!onProcessedImagesChangeRef.current) return;
    const combinedImages = { ...processedImages };
    selectedPageData.forEach(page => {
      const pageNum = page.pageNumber;
      if (!combinedImages[pageNum]) {
        const state = imageStates[pageNum];
        if (state?.current && state.current !== page.thumbnail) {
          combinedImages[pageNum] = state.current;
        }
      }
    });
    onProcessedImagesChangeRef.current(combinedImages);
  }, [processedImages, imageStates, selectedPageData]);

  // Report the applied op list up for the My Files tags. Pinned like above.
  const onPipelineChangeRef = useRef(onPipelineChange);
  useEffect(() => { onPipelineChangeRef.current = onPipelineChange; }, [onPipelineChange]);

  useEffect(() => {
    if (!onPipelineChangeRef.current) return;
    onPipelineChangeRef.current(buildPipelineConfig().map((s) => s.op));
  }, [pipeline, buildPipelineConfig]);

  // ── Image state ──

  // Read a page's working image without touching state. Falls back to its thumbnail.
  const getCurrentImageFor = useCallback((pageNumber) => {
    const state = imageStates[pageNumber];
    if (state?.current) return state.current;
    const page = pages.find((p) => p.pageNumber === pageNumber);
    return page?.thumbnail;
  }, [imageStates, pages]);

  // Write a page's current image. Does NOT push history — callers that want
  // the edit undoable must add the op themselves.
  const setCurrentImageFor = useCallback((pageNumber, newImageData) => {
    setImageStates((prev) => {
      const pageData = pages.find((p) => p.pageNumber === pageNumber);
      const entry = prev[pageNumber] || {
        current: pageData?.thumbnail,
        original: pageData?.thumbnail,
      };
      if (entry.current === newImageData) return prev;
      return { ...prev, [pageNumber]: { ...entry, current: newImageData } };
    });
  }, [pages]);

  // Restore a processedImages snapshot, dropping nulls to keep the map small.
  const replaceProcessedImages = useCallback((nextMap) => {
    setProcessedImages(() => {
      const out = {};
      Object.entries(nextMap || {}).forEach(([k, v]) => {
        if (v != null) out[k] = v;
      });
      return out;
    });
  }, []);

  // Redo: replay an op's `after` side through the normal setters.
  const applyForward = useCallback((op) => {
    if (!op) return;
    switch (op.kind) {
      case 'crop':
        setCurrentImageFor(op.pageNumber, op.after.image);
        setProcessedImages((prev) => {
          const { [op.pageNumber]: _removed, ...rest } = prev;
          return rest;
        });
        break;
      case 'crop-batch':
        Object.entries(op.after.images).forEach(([pn, url]) => {
          setCurrentImageFor(pn in imageStates ? pn : Number(pn), url);
        });
        setProcessedImages((prev) => {
          const next = { ...prev };
          Object.keys(op.after.images).forEach((pn) => { delete next[pn]; });
          return next;
        });
        break;
      case 'erase':
        setCurrentImageFor(op.pageNumber, op.after.image);
        setProcessedImages((prev) => {
          const { [op.pageNumber]: _removed, ...rest } = prev;
          return rest;
        });
        break;
      case 'erase-batch':
        Object.entries(op.after.images).forEach(([pn, url]) => {
          const key = isNaN(Number(pn)) ? pn : Number(pn);
          setCurrentImageFor(key, url);
        });
        setProcessedImages((prev) => {
          const next = { ...prev };
          Object.keys(op.after.images).forEach((pn) => { delete next[pn]; });
          return next;
        });
        break;
      case 'pipeline':
        setProcessedImages((prev) => ({ ...prev, [op.pageNumber]: op.after.processed }));
        break;
      case 'pipeline-batch':
        replaceProcessedImages(op.after.processed);
        break;
      case 'reset-page': {
        const page = pages.find((p) => p.pageNumber === op.pageNumber);
        setCurrentImageFor(op.pageNumber, page?.thumbnail);
        setProcessedImages((prev) => {
          const { [op.pageNumber]: _removed, ...rest } = prev;
          return rest;
        });
        break;
      }
      default:
        console.warn('applyForward: unknown op kind', op.kind);
    }
  }, [imageStates, pages, replaceProcessedImages, setCurrentImageFor]);

  // Undo: walk one op's state changes backwards. The store already moved its cursor.
  const applyRevert = useCallback((op) => {
    if (!op) return;
    switch (op.kind) {
      case 'crop':
      case 'erase':
        setCurrentImageFor(op.pageNumber, op.before.image);
        // The forward op cleared it and we can't re-derive it.
        break;
      case 'crop-batch':
      case 'erase-batch':
        Object.entries(op.before.images).forEach(([pn, url]) => {
          const key = isNaN(Number(pn)) ? pn : Number(pn);
          setCurrentImageFor(key, url);
        });
        break;
      case 'pipeline':
        if (op.before.processed == null) {
          setProcessedImages((prev) => {
            const { [op.pageNumber]: _removed, ...rest } = prev;
            return rest;
          });
        } else {
          setProcessedImages((prev) => ({ ...prev, [op.pageNumber]: op.before.processed }));
        }
        break;
      case 'pipeline-batch':
        replaceProcessedImages(op.before.processed);
        break;
      case 'reset-page':
        if (op.before.image != null) setCurrentImageFor(op.pageNumber, op.before.image);
        if (op.before.processed != null) {
          setProcessedImages((prev) => ({ ...prev, [op.pageNumber]: op.before.processed }));
        }
        break;
      default:
        console.warn('applyRevert: unknown op kind', op.kind);
    }
  }, [replaceProcessedImages, setCurrentImageFor]);

  // Undo the latest op touching this page.
  const handleUndo = useCallback(() => {
    if (!currentPage) return;
    const op = history.undoPage(currentPage.pageNumber);
    if (op) applyRevert(op);
  }, [currentPage, history, applyRevert]);

  // Redo the next op. Global on purpose — the store's branching already keeps
  // the next op coherent.
  const handleRedo = useCallback(() => {
    const op = history.redoGlobal();
    if (op) applyForward(op);
  }, [history, applyForward]);

  // Ctrl/Cmd+Z and Shift+Z. Skipped while a modal is open — cropper and eraser
  // handle their own keys.
  useEffect(() => {
    if (showCropper || showEraser) return;
    const onKey = (e) => {
      const meta = e.ctrlKey || e.metaKey;
      if (!meta) return;
      const tag = document.activeElement?.tagName?.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
      if (e.key === 'z' || e.key === 'Z') {
        e.preventDefault();
        if (e.shiftKey) handleRedo();
        else handleUndo();
      } else if (e.key === 'y' || e.key === 'Y') {
        e.preventDefault();
        handleRedo();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [handleUndo, handleRedo, showCropper, showEraser]);

  // Reset the page, pushed as an op so the reset itself is undoable.
  const handleResetImage = useCallback(() => {
    if (!currentPage) return;
    const pn = currentPage.pageNumber;
    const beforeImage = getCurrentImageFor(pn);
    const beforeProcessed = processedImages[pn] ?? null;
    if (beforeImage === currentPage.thumbnail && beforeProcessed == null) return;

    history.push(HistoryOps.resetPage({
      pageNumber: pn,
      beforeImage,
      beforeProcessed,
    }));

    setImageStates((prev) => ({
      ...prev,
      [pn]: { current: currentPage.thumbnail, original: currentPage.thumbnail },
    }));
    setProcessedImages((prev) => {
      const { [pn]: _removed, ...rest } = prev;
      return rest;
    });
  }, [currentPage, getCurrentImageFor, history, processedImages]);

  // ── Crop ──

  // Single-page crop. The op captures before/after so it can be undone.
  const handleCropComplete = useCallback((croppedDataUrl, cropData) => {
    if (!currentPage) return;
    const pn = currentPage.pageNumber;
    const before = getCurrentImageFor(pn);

    history.push(HistoryOps.crop({ pageNumber: pn, before, after: croppedDataUrl, cropData }));
    setCurrentImageFor(pn, croppedDataUrl);
    setProcessedImages((prev) => {
      const { [pn]: _removed, ...rest } = prev;
      return rest;
    });
    setShowCropper(false);
  }, [currentPage, getCurrentImageFor, history, setCurrentImageFor]);

  // The eraser hands back every page it touched. Folded into one op so a single
  // undo rewinds the whole save.
  const handleEraserSave = useCallback((results) => {
    const pageNumbers = [];
    const beforeMap = {};
    const afterMap = {};

    Object.entries(results).forEach(([pageNumStr, dataUrl]) => {
      const page = selectedPageData.find((p) => String(p.pageNumber) === pageNumStr);
      const pn = page ? page.pageNumber : (isNaN(Number(pageNumStr)) ? pageNumStr : Number(pageNumStr));
      pageNumbers.push(pn);
      beforeMap[pn] = getCurrentImageFor(pn);
      afterMap[pn] = dataUrl;
    });

    if (pageNumbers.length === 1) {
      const pn = pageNumbers[0];
      history.push(HistoryOps.erase({ pageNumber: pn, before: beforeMap[pn], after: afterMap[pn] }));
    } else if (pageNumbers.length > 1) {
      history.push(HistoryOps.eraseBatch({ pageNumbers, before: beforeMap, after: afterMap }));
    }

    pageNumbers.forEach((pn) => setCurrentImageFor(pn, afterMap[pn]));
    setProcessedImages((prev) => {
      const next = { ...prev };
      pageNumbers.forEach((pn) => { delete next[pn]; });
      return next;
    });
    setShowEraser(false);
  }, [getCurrentImageFor, history, selectedPageData, setCurrentImageFor]);

  // One op spanning the current page plus every batch target.
  const handleBatchCropComplete = useCallback((currentCroppedUrl, cropData, batchResults) => {
    if (!currentPage) return;
    const pageNumbers = [currentPage.pageNumber];
    const beforeMap = { [currentPage.pageNumber]: getCurrentImageFor(currentPage.pageNumber) };
    const afterMap = { [currentPage.pageNumber]: currentCroppedUrl };

    batchResults.forEach(({ pageNumber, croppedDataUrl }) => {
      pageNumbers.push(pageNumber);
      beforeMap[pageNumber] = getCurrentImageFor(pageNumber);
      afterMap[pageNumber] = croppedDataUrl;
    });

    history.push(HistoryOps.cropBatch({ pageNumbers, before: beforeMap, after: afterMap }));
    pageNumbers.forEach((pn) => setCurrentImageFor(pn, afterMap[pn]));
    setProcessedImages((prev) => {
      const next = { ...prev };
      pageNumbers.forEach((pn) => { delete next[pn]; });
      return next;
    });
    setShowCropper(false);
  }, [currentPage, getCurrentImageFor, history, setCurrentImageFor]);

  // ── Preprocessing ──

  // Runs on the page's working image, which may already be cropped.
  const handleApplyToCurrentPage = useCallback(async () => {
    if (!currentPage || isProcessing) return;

    const activePipeline = getActivePipeline();
    if (activePipeline.length === 0) return;

    setIsProcessing(true);
    setProcessingProgress(0);

    try {
      const pipelineConfig = buildPipelineConfig();

      // Working image, not the original — crops must survive.
      const sourceImage = currentImageState;

      // Fake progress; the backend doesn't stream any.
      for (let i = 0; i < pipelineConfig.length; i++) {
        setCurrentProcessingStep(pipelineConfig[i].op);
        setProcessingProgress(((i + 0.5) / pipelineConfig.length) * 100);
        await new Promise(r => setTimeout(r, 100));
      }

      const result = await preprocessImage(sourceImage, pipelineConfig);
      const pn = currentPage.pageNumber;
      const beforeProcessed = processedImages[pn] ?? null;

      history.push(HistoryOps.pipeline({
        pageNumber: pn,
        beforeProcessed,
        afterProcessed: result,
        pipelineConfig,
      }));

      setProcessedImages((prev) => ({ ...prev, [pn]: result }));
      setProcessingProgress(100);
    } catch (error) {
      console.error('Processing failed:', error);
    } finally {
      setIsProcessing(false);
      setCurrentProcessingStep(null);
      setProcessingProgress(0);
    }
  }, [currentPage, isProcessing, getActivePipeline, buildPipelineConfig, currentImageState, history, processedImages]);

  // Same, for every selected page, each from its own working image.
  const handleApplyToAllPages = useCallback(async () => {
    if (isProcessing) return;

    const activePipeline = getActivePipeline();
    if (activePipeline.length === 0) return;

    setIsProcessing(true);
    setBatchProgress({ current: 0, total: selectedPageData.length });

    try {
      const pipelineConfig = buildPipelineConfig();
      const results = {};

      for (let i = 0; i < selectedPageData.length; i++) {
        const page = selectedPageData[i];
        setBatchProgress({ current: i + 1, total: selectedPageData.length });
        setProcessingProgress(((i + 0.5) / selectedPageData.length) * 100);

        const pageState = imageStates[page.pageNumber];
        const sourceImage = pageState?.current || page.thumbnail;

        const result = await preprocessImage(sourceImage, pipelineConfig);
        results[page.pageNumber] = result;
      }

      const pageNumbers = Object.keys(results).map((k) => (isNaN(Number(k)) ? k : Number(k)));
      const beforeProcessed = {};
      pageNumbers.forEach((pn) => { beforeProcessed[pn] = processedImages[pn] ?? null; });

      history.push(HistoryOps.pipelineBatch({
        pageNumbers,
        beforeProcessed,
        afterProcessed: results,
        pipelineConfig,
      }));

      setProcessedImages(results);
      setProcessingProgress(100);
    } catch (error) {
      console.error('Batch processing failed:', error);
    } finally {
      setIsProcessing(false);
      setBatchProgress({ current: 0, total: 0 });
      setProcessingProgress(0);
    }
  }, [isProcessing, getActivePipeline, buildPipelineConfig, selectedPageData, imageStates, processedImages, history]);

  // ── Navigation ──

  const handlePrevPage = () => {
    setCurrentPageIndex((prev) => Math.max(0, prev - 1));
  };

  const handleNextPage = () => {
    setCurrentPageIndex((prev) => Math.min(selectedPageData.length - 1, prev + 1));
  };

  // ── Reset ──

  // Pipeline config only — page images stay put.
  const handleResetOperations = useCallback(() => {
    resetPipeline();
  }, [resetPipeline]);

  // Current page only, via the undoable handler.
  const handleResetCurrentPage = handleResetImage;

  // Everything. History goes too — every op's before-state is about to vanish.
  const handleResetAll = useCallback(() => {
    resetPipeline();
    setProcessedImages({});
    setImageStates({});
    history.clear();
  }, [resetPipeline, history]);

  // "Undo All" is just the global undo — the last op is usually an apply-all.
  const handleGlobalUndo = useCallback(() => {
    const op = history.undoGlobal();
    if (op) applyRevert(op);
  }, [history, applyRevert]);

  // ── Computed ──

  const activePipelineCount = getActivePipeline().length;
  const hasProcessedImages = Object.keys(processedImages).length > 0;
  const currentPageProcessed = currentPage && processedImages[currentPage.pageNumber];

  const hasModifiedImages = useMemo(() => {
    return selectedPageData.some(page => {
      const state = imageStates[page.pageNumber];
      return state?.current && state.current !== page.thumbnail;
    });
  }, [selectedPageData, imageStates]);

  // Cropped-only pages count as progress too.
  const canProceed = hasProcessedImages || hasModifiedImages;

  const canGlobalUndo = history.canUndoGlobal;
  const canGlobalRedo = history.canRedoGlobal;

  // ── Render ──

  return (
    <div className="h-full w-full flex overflow-hidden bg-gray-50">

      {/* ========== LEFT SIDEBAR ========== */}
      <aside className="flex-shrink-0 w-[272px] flex flex-col border-r border-gray-200 bg-white overflow-hidden">

        {/* Sidebar header: step badge + title + description + quick actions */}
        <div className="flex-shrink-0 px-4 pt-4 pb-3 border-b border-gray-100 space-y-3">
          {/* Step badge */}
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-blue-50 border border-blue-100 rounded-full text-xs font-semibold text-blue-600">
              <Sparkles className="w-3 h-3" />
              Step 3 of 5
            </span>
          </div>

          {/* Title + description */}
          <div>
            <h2 className="text-base font-bold text-gray-900 leading-tight">Pipeline</h2>
            <p className="text-xs text-gray-400 mt-0.5">Toggle and tune operations. Preview updates live.</p>
          </div>

          {/* Quick actions */}
          <div className="flex gap-2">
            {/* Recommendation: pick a book type -> pre-fill a pre-tested
                pipeline (does NOT run; the user reviews and processes). */}
            <div className="relative flex-1">
              <button
                onClick={() => setShowBookTypeMenu(v => !v)}
                disabled={isProcessing}
                className="w-full flex items-center justify-center gap-1.5 px-3 py-2
                           bg-gradient-to-r from-blue-500 to-blue-600 text-white text-xs font-medium
                           rounded-lg hover:from-blue-600 hover:to-blue-700
                           disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm"
                title="Load a pre-tested pipeline for a known book type"
              >
                <BookOpen size={13} />
                Recommendation
                <ChevronDown size={13} className={`transition-transform ${showBookTypeMenu ? 'rotate-180' : ''}`} />
              </button>

              {showBookTypeMenu && (
                <>
                  {/* click-away */}
                  <div className="fixed inset-0 z-10" onClick={() => setShowBookTypeMenu(false)} />
                  <div className="absolute left-0 right-0 mt-1 z-20 bg-white border border-gray-200 rounded-lg shadow-lg overflow-hidden">
                    <p className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wide text-gray-400 border-b border-gray-100">
                      Select book type
                    </p>
                    {bookTypePresets.map(preset => (
                      <button
                        key={preset.id}
                        onClick={() => handleSelectBookType(preset.id)}
                        disabled={isProcessing}
                        className="w-full text-left px-3 py-2 hover:bg-blue-50 disabled:opacity-50
                                   disabled:cursor-not-allowed transition-colors"
                      >
                        <span className="block text-xs font-semibold text-gray-800">{preset.label}</span>
                        {preset.description && (
                          <span className="block text-[10px] text-gray-400 mt-0.5">{preset.description}</span>
                        )}
                      </button>
                    ))}
                    <p className="px-3 py-2 text-[10px] text-gray-400 border-t border-gray-100">
                      Fills the pipeline for review. Nothing runs until you process.
                    </p>
                  </div>
                </>
              )}
            </div>

            <button
              onClick={handleResetOperations}
              disabled={isProcessing}
              className="px-3 py-2 bg-gray-100 text-gray-600 text-xs font-medium rounded-lg
                         hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              title="Reset all operations"
            >
              <RotateCcw size={14} />
            </button>
          </div>
        </div>

        {/* Operations list or Pipeline Order (toggled by showPipelinePanel) */}
        <div className="flex-1 min-h-0 overflow-y-auto">
          {!showPipelinePanel ? (
            <OperationsSidebar
              enabledOperations={enabledOperations}
              onOperationToggle={toggleOperation}
              onOperationParamsChange={updateOperationParams}
              onApplyRecommended={applyRecommendedPipeline}
              onReset={handleResetOperations}
              isProcessing={isProcessing}
              hideHeader
              className="h-full"
            />
          ) : (
            <div className="p-3 space-y-3">
              {pipeline.length === 0 ? (
                <div className="mt-2 p-3 bg-blue-50 rounded-lg border border-blue-100">
                  <div className="flex items-start gap-2">
                    <Info className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-xs font-medium text-blue-700">No operations in pipeline</p>
                      <p className="text-[10px] text-blue-600 mt-1">
                        Switch back to Operations view and enable some operations to add them here.
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <PipelineStack
                  pipeline={pipeline}
                  onReorder={reorderPipeline}
                  onToggle={togglePipelineStep}
                  onUpdateParams={updatePipelineStepParams}
                  onRemove={removePipelineStep}
                  isProcessing={isProcessing}
                  currentStepId={currentProcessingStep}
                />
              )}
            </div>
          )}
        </div>

        {/* Sidebar action bar */}
        <div className="flex-shrink-0 border-t border-gray-100 p-3 space-y-2 bg-gray-50/60">

          {/* Apply buttons */}
          <div className="flex gap-2">
            <button
              onClick={handleApplyToCurrentPage}
              disabled={isProcessing || activePipelineCount === 0}
              className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3
                         bg-blue-600 text-white text-xs font-semibold rounded-lg
                         hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              title="Apply pipeline to current page"
            >
              <Play className="w-3.5 h-3.5" />
              Apply
            </button>
            <button
              onClick={handleApplyToAllPages}
              disabled={isProcessing || activePipelineCount === 0}
              className="flex items-center gap-1.5 py-2 px-3
                         bg-gray-100 text-gray-700 text-xs font-semibold rounded-lg
                         hover:bg-gray-200 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              title="Apply pipeline to all selected pages"
            >
              <Layers className="w-3.5 h-3.5" />
              All
            </button>
          </div>

          {/* Edit tools row */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <button
              onClick={() => setShowCropper(true)}
              disabled={!currentPage}
              className={`flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all
                ${isModified ? 'bg-orange-100 text-orange-700 hover:bg-orange-200' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}
                disabled:opacity-40 disabled:cursor-not-allowed`}
              title="Crop image"
            >
              <Crop className="w-3.5 h-3.5" />
              Crop
            </button>
            <button
              onClick={() => setShowEraser(true)}
              disabled={!currentPage}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium
                         bg-gray-100 text-gray-700 hover:bg-gray-200
                         disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              title="Erase artifacts"
            >
              <Eraser className="w-3.5 h-3.5" />
              Erase
            </button>

            {/* Undo / Redo / Undo-All */}
            <button
              onClick={handleUndo}
              disabled={!canUndo}
              className={`p-1.5 rounded-lg transition-all
                ${canUndo ? 'bg-gray-100 text-gray-700 hover:bg-gray-200' : 'text-gray-300 cursor-not-allowed'}`}
              title="Undo (Ctrl+Z)"
            >
              <Undo2 className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleRedo}
              disabled={!canGlobalRedo}
              className={`p-1.5 rounded-lg transition-all
                ${canGlobalRedo ? 'bg-gray-100 text-gray-700 hover:bg-gray-200' : 'text-gray-300 cursor-not-allowed'}`}
              title="Redo (Ctrl+Shift+Z)"
            >
              <Redo2 className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleGlobalUndo}
              disabled={!canGlobalUndo}
              className={`flex items-center gap-0.5 p-1.5 rounded-lg transition-all
                ${canGlobalUndo ? 'bg-gray-100 text-gray-700 hover:bg-gray-200' : 'text-gray-300 cursor-not-allowed'}`}
              title="Undo last global operation (e.g. Apply All)"
            >
              <Undo2 className="w-3.5 h-3.5" />
              <Layers className="w-2.5 h-2.5" />
            </button>

            {/* Reset page when modified */}
            {isModified && (
              <button
                onClick={handleResetImage}
                className="flex items-center gap-1 px-2 py-1.5 rounded-lg text-xs font-medium text-red-600 hover:bg-red-50 transition-colors ml-auto"
                title="Reset to original image"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Reset
              </button>
            )}
          </div>

          {/* Pipeline order toggle + pipeline count */}
          <div className="flex items-center justify-between pt-0.5">
            <button
              onClick={() => setShowPipelinePanel(!showPipelinePanel)}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all
                ${showPipelinePanel ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
              title="Toggle pipeline order view"
            >
              <Layers className="w-3.5 h-3.5" />
              {showPipelinePanel ? 'Operations' : 'Reorder Pipeline'}
            </button>
            {activePipelineCount > 0 && (
              <span className="text-[10px] text-gray-400 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                {activePipelineCount} active
              </span>
            )}
          </div>
        </div>
      </aside>

      {/* ========== MAIN CONTENT ========== */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Preview header */}
        <div className="flex-shrink-0 px-5 py-3 bg-white border-b border-gray-100 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <div className="flex-shrink-0 p-1.5 bg-blue-500 rounded-lg text-white">
              <Sparkles className="w-4 h-4" />
            </div>
            <div className="min-w-0">
              <h3 className="text-sm font-semibold text-gray-800 leading-tight">Preprocess preview</h3>
              <p className="text-xs text-gray-400">Active page · before / after</p>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-shrink-0">
            {/* Modified badge */}
            {isModified && (
              <span className="hidden sm:inline text-xs text-orange-600 bg-orange-50 border border-orange-100 px-2 py-1 rounded-lg">
                Modified
              </span>
            )}

            {/* Page counter badge */}
            <span className="px-3 py-1.5 bg-blue-50 border border-blue-100 text-blue-700 text-xs font-semibold rounded-lg whitespace-nowrap">
              Page {currentPageIndex + 1} of {selectedPageData.length} selected
            </span>

            {/* Zoom lens */}
            <button
              onClick={() => setShowZoomLens(!showZoomLens)}
              className={`p-2 rounded-lg transition-colors flex-shrink-0
                ${showZoomLens ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
              title="Toggle zoom lens"
            >
              <Search className="w-4 h-4" />
            </button>

            {/* Skip */}
            <button
              onClick={onNext}
              disabled={isProcessing}
              className="hidden md:flex items-center gap-1 px-3 py-1.5 text-xs font-medium text-gray-500 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-40"
              title="Skip preprocessing"
            >
              <SkipForward className="w-3.5 h-3.5" />
              Skip
            </button>
          </div>
        </div>

        {/* Tip banner */}
        {showTipBanner && (
          <div className="flex-shrink-0 px-5 pt-3">
            <InfoBanner
              message="Tip: Crop images first before applying preprocessing for best results. Cropping after preprocessing may change output quality."
              onDismiss={() => setShowTipBanner(false)}
              variant="info"
            />
          </div>
        )}

        {/* Processing progress bar */}
        {isProcessing && (
          <div className="flex-shrink-0 px-5 pt-3">
            <ProgressBarLabeled
              label={
                batchProgress.total > 0
                  ? `Processing page ${batchProgress.current} of ${batchProgress.total}`
                  : currentProcessingStep
                    ? `Running: ${currentProcessingStep}`
                    : 'Processing...'
              }
              progress={processingProgress}
              variant="primary"
              size="md"
              isActive
            />
          </div>
        )}

        {/* Preview area — fills remaining height */}
        <div className="flex-1 min-h-0 p-4">
          <BeforeAfterViewer
            originalImage={originalImageState}
            processedImage={currentPageProcessed || currentImageState}
            isProcessing={isProcessing}
            processingLabel={currentProcessingStep ? `Running: ${currentProcessingStep}` : 'Processing...'}
            onOpenZoomLens={() => setShowZoomLens(true)}
            className="h-full"
          />
        </div>

        {/* ===== BOTTOM BAR: thumbnail strip + back / next ===== */}
        <div className="flex-shrink-0 border-t border-gray-100 bg-white px-4 py-2.5">
          <div className="flex items-center gap-3">

            {/* Prev page arrow */}
            <button
              onClick={handlePrevPage}
              disabled={currentPageIndex === 0}
              className={`p-1.5 rounded-lg transition-colors flex-shrink-0
                ${currentPageIndex === 0 ? 'text-gray-200 cursor-not-allowed' : 'text-gray-500 hover:bg-gray-100'}`}
            >
              <ChevronLeft className="w-5 h-5" />
            </button>

            {/* Page thumbnails */}
            <div className="flex-1 flex items-end gap-2 overflow-x-auto min-w-0 py-0.5">
              {selectedPageData.map((page, idx) => (
                <button
                  key={page.pageNumber}
                  onClick={() => setCurrentPageIndex(idx)}
                  className="relative flex-shrink-0 flex flex-col items-center gap-1 group"
                >
                  <div className={`w-[52px] h-[66px] rounded-lg overflow-hidden border-2 transition-all
                    ${idx === currentPageIndex
                      ? 'border-blue-500 shadow-md shadow-blue-100'
                      : 'border-gray-200 hover:border-gray-300 group-hover:shadow-sm'
                    }`}
                  >
                    <img
                      src={processedImages[page.pageNumber] || imageStates[page.pageNumber]?.current || page.thumbnail}
                      alt={`Page ${page.pageNumber}`}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  {/* Status dot */}
                  {processedImages[page.pageNumber] && (
                    <span className="absolute top-0.5 right-0.5 w-2 h-2 bg-green-500 rounded-full border border-white shadow-sm" />
                  )}
                  {!processedImages[page.pageNumber] && imageStates[page.pageNumber]?.current !== page.thumbnail && (
                    <span className="absolute top-0.5 right-0.5 w-2 h-2 bg-orange-400 rounded-full border border-white shadow-sm" />
                  )}
                  <span className={`text-[10px] font-medium leading-none
                    ${idx === currentPageIndex ? 'text-blue-600' : 'text-gray-400'}`}
                  >
                    Page {shortPageLabel(page.pageNumber)}
                  </span>
                </button>
              ))}
            </div>

            {/* Next page arrow */}
            <button
              onClick={handleNextPage}
              disabled={currentPageIndex === selectedPageData.length - 1}
              className={`p-1.5 rounded-lg transition-colors flex-shrink-0
                ${currentPageIndex === selectedPageData.length - 1 ? 'text-gray-200 cursor-not-allowed' : 'text-gray-500 hover:bg-gray-100'}`}
            >
              <ChevronRight className="w-5 h-5" />
            </button>

            {/* Status indicators */}
            <div className="flex-shrink-0 flex items-center gap-2 text-xs text-gray-400 border-l border-gray-100 pl-3">
              <span className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                {Object.keys(processedImages).length}/{selectedPages.length}
              </span>
              {Object.keys(imageStates).filter(k => {
                const pg = pages.find(p => String(p.pageNumber) === String(k));
                return imageStates[k]?.current !== pg?.thumbnail;
              }).length > 0 && (
                <span className="flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-orange-400" />
                  {Object.keys(imageStates).filter(k => {
                    const pg = pages.find(p => String(p.pageNumber) === String(k));
                    return imageStates[k]?.current !== pg?.thumbnail;
                  }).length}
                </span>
              )}
            </div>

            {/* Divider */}
            <div className="flex-shrink-0 w-px h-8 bg-gray-100" />

            {/* Back */}
            <button
              onClick={onBack}
              className="flex-shrink-0 flex items-center gap-1.5 px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-100 rounded-xl transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </button>

            {/* Continue */}
            <button
              onClick={onNext}
              disabled={!canProceed}
              className={`flex-shrink-0 flex items-center gap-2 px-5 py-2 text-sm font-semibold rounded-xl transition-all
                ${canProceed
                  ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-sm hover:shadow-md hover:-translate-y-0.5'
                  : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                }`}
            >
              Continue to Text Detection
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </main>

      {/* ========== MODALS ========== */}

      {/* Image Cropper Modal */}
      {showCropper && currentPage && (
        <ImageCropper
          imageSrc={currentImageState}
          onCropComplete={handleCropComplete}
          onCancel={() => setShowCropper(false)}
          availablePages={selectedPageData.map(page => ({
            pageNumber: page.pageNumber,
            thumbnail: imageStates[page.pageNumber]?.current || page.thumbnail,
          }))}
          currentPageNumber={currentPage.pageNumber}
          onBatchCropComplete={handleBatchCropComplete}
        />
      )}

      {/* Image Eraser Modal */}
      {showEraser && currentPage && (
        <ImageEraser
          pages={selectedPageData.map(page => ({
            pageNumber: page.pageNumber,
            imageSrc: processedImages[page.pageNumber] || imageStates[page.pageNumber]?.current || page.thumbnail,
          }))}
          initialPageIndex={currentPageIndex}
          onSaveAll={handleEraserSave}
          onCancel={() => setShowEraser(false)}
        />
      )}

      {/* Zoom Lens Preview */}
      <ZoomLensPreview
        originalImage={originalImageState}
        processedImage={currentPageProcessed || currentImageState}
        isOpen={showZoomLens}
        onClose={() => setShowZoomLens(false)}
      />
    </div>
  );
}
