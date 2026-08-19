import { useState, useEffect, useCallback, useRef } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  Play,
  Pause,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Zap,
  Image as ImageIcon,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize2,
  Move,
  Minus,
  Plus,
  Clock,
} from 'lucide-react';

// Full-res scans blow up browser memory when displayed. Downscale for the
// preview only — OCR still gets the original.
function useDownscaledImage(imageSrc, maxWidth = 1200, maxHeight = 1600) {
  const [previewSrc, setPreviewSrc] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(false);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!imageSrc) {
      setPreviewSrc(null);
      setError(false);
      setDimensions({ width: 0, height: 0 });
      return;
    }

    if (imageSrc.startsWith('blob:') || imageSrc.length < 100000) {
      setPreviewSrc(imageSrc);
      return;
    }

    setIsLoading(true);
    setError(false);

    const img = new Image();
    
    img.onload = () => {
      try {
        setDimensions({ width: img.width, height: img.height });
        
            if (img.width <= maxWidth && img.height <= maxHeight) {
          setPreviewSrc(imageSrc);
          setIsLoading(false);
          return;
        }

            let newWidth = img.width;
        let newHeight = img.height;
        
        if (newWidth > maxWidth) {
          newHeight = (maxWidth / newWidth) * newHeight;
          newWidth = maxWidth;
        }
        if (newHeight > maxHeight) {
          newWidth = (maxHeight / newHeight) * newWidth;
          newHeight = maxHeight;
        }

            const canvas = document.createElement('canvas');
        canvas.width = Math.round(newWidth);
        canvas.height = Math.round(newHeight);
        
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Blob URL, not data URL — much easier on memory.
        canvas.toBlob((blob) => {
          if (blob) {
            const blobUrl = URL.createObjectURL(blob);
            setPreviewSrc(blobUrl);
          } else {
            setPreviewSrc(canvas.toDataURL('image/jpeg', 0.85));
          }
          setIsLoading(false);
        }, 'image/jpeg', 0.85);
      } catch (err) {
        console.error('Failed to downscale image:', err);
        setError(true);
        setIsLoading(false);
      }
    };

    img.onerror = () => {
      console.error('Failed to load image for preview');
      setError(true);
      setIsLoading(false);
    };

    img.src = imageSrc;

    return () => {
      if (previewSrc && previewSrc.startsWith('blob:')) {
        URL.revokeObjectURL(previewSrc);
      }
    };
  }, [imageSrc, maxWidth, maxHeight]);

  return { previewSrc, isLoading, error, dimensions };
}

// Big page preview with zoom, pan and navigation. Zoom can be controlled by
// the parent so it survives page changes.
export default function PagePreview({
  currentPage,
  currentIndex,
  totalPages,
  isProcessing,
  isPageProcessed,
  isAutoProcessing,
  isWaitingForRateLimit,
  rateLimitReady,
  waitSeconds,
  error,
  canProcess,
  onPrevPage,
  onNextPage,
  onProcess,
  onToggleAutoProcess,
  zoomLevel: controlledZoom,
  onZoomChange,
  processingPageIndex,
  pageLabel,
  pageLabels = null,
}) {
  const pageNumber = pageLabel || currentIndex + 1;
  const processingPageNumber = processingPageIndex !== null
    ? (pageLabels ? pageLabels[processingPageIndex] : processingPageIndex + 1)
    : null;
  const imageSrc = currentPage?.processed || currentPage?.original;

  const { previewSrc, isLoading: isDownscaling, error: downscaleError, dimensions } = useDownscaledImage(imageSrc);
  
  const [imageLoaded, setImageLoaded] = useState(false);
  const [internalZoom, setInternalZoom] = useState(1);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [scrollStart, setScrollStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);
  const imageRef = useRef(null);
  const lastClickTime = useRef(0);

  const isControlled = controlledZoom !== undefined && onZoomChange !== undefined;
  const zoomLevel = isControlled ? controlledZoom : internalZoom;
  const setZoomLevel = isControlled ? onZoomChange : setInternalZoom;

  const MIN_ZOOM = 0.25;
  const MAX_ZOOM = 5.0;
  const ZOOM_STEP = 0.1;

  // Only the loading state — zoom is the parent's business.
  useEffect(() => {
    setImageLoaded(false);
  }, [currentIndex, previewSrc]);

  const smoothZoom = useCallback((targetZoom, instant = false) => {
    const clampedZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, targetZoom));
    const roundedZoom = Math.round(clampedZoom * 100) / 100;
    setZoomLevel(roundedZoom);
  }, [setZoomLevel]);

  const handleZoomIn = useCallback(() => {
    smoothZoom(zoomLevel * 1.25); // Multiplicative zoom feels more natural
  }, [zoomLevel, smoothZoom]);

  const handleZoomOut = useCallback(() => {
    smoothZoom(zoomLevel / 1.25);
  }, [zoomLevel, smoothZoom]);

  const handleResetZoom = useCallback(() => {
    smoothZoom(1);
    if (containerRef.current) {
      containerRef.current.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
    }
  }, [smoothZoom]);

  const handleFitToPanel = useCallback(() => {
    if (!containerRef.current || !dimensions.width || !dimensions.height) {
      smoothZoom(1);
      return;
    }
    const container = containerRef.current;
    const containerWidth = container.clientWidth - 32;
    const containerHeight = container.clientHeight - 32;
    const widthRatio = containerWidth / dimensions.width;
    const heightRatio = containerHeight / dimensions.height;
    const fitZoom = Math.min(widthRatio, heightRatio, 1);
    smoothZoom(Math.max(MIN_ZOOM, fitZoom));
  }, [dimensions, smoothZoom]);

  const handleSliderChange = useCallback((e) => {
    smoothZoom(parseFloat(e.target.value));
  }, [smoothZoom]);

  const handlePresetZoom = useCallback((preset) => {
    smoothZoom(preset);
  }, [smoothZoom]);

  // Multiplicative, so each tick feels the same at any zoom level.
  const handleWheel = useCallback((e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.9 : 1.1; // 10% zoom per scroll
      smoothZoom(zoomLevel * factor);
    }
  }, [zoomLevel, smoothZoom]);

  const handleDoubleClick = useCallback((e) => {
    e.preventDefault();
    if (zoomLevel <= 1.1) {
      smoothZoom(2);
    } else {
      smoothZoom(1);
      if (containerRef.current) {
        containerRef.current.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
      }
    }
  }, [zoomLevel, smoothZoom]);

  const handleMouseDown = useCallback((e) => {
    const now = Date.now();
    if (now - lastClickTime.current < 300) {
      handleDoubleClick(e);
      lastClickTime.current = 0;
      return;
    }
    lastClickTime.current = now;

    if (zoomLevel > 1 && e.button === 0) {
      setIsPanning(true);
      setPanStart({ x: e.clientX, y: e.clientY });
      setScrollStart({ 
        x: containerRef.current?.scrollLeft || 0, 
        y: containerRef.current?.scrollTop || 0 
      });
      e.preventDefault();
    }
  }, [zoomLevel, handleDoubleClick]);

  const handleMouseMove = useCallback((e) => {
    if (isPanning && containerRef.current) {
      const dx = e.clientX - panStart.x;
      const dy = e.clientY - panStart.y;
      containerRef.current.scrollLeft = scrollStart.x - dx;
      containerRef.current.scrollTop = scrollStart.y - dy;
    }
  }, [isPanning, panStart, scrollStart]);

  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

  // On window, so releasing outside the image still ends the pan.
  useEffect(() => {
    if (isPanning) {
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('mousemove', handleMouseMove);
      return () => {
        window.removeEventListener('mouseup', handleMouseUp);
        window.removeEventListener('mousemove', handleMouseMove);
      };
    }
  }, [isPanning, handleMouseUp, handleMouseMove]);

  const showLoading = isDownscaling || (!imageLoaded && previewSrc && !downscaleError);
  const showError = downscaleError;
  const showPlaceholder = !imageSrc;
  const zoomPercent = Math.round(zoomLevel * 100);

  return (
    <div className="bg-white/95 backdrop-blur-sm rounded-xl border border-gray-200/80 shadow-sm overflow-hidden flex flex-col h-full relative">
      {/* Header with navigation */}
      <div className="px-3 py-2 bg-gradient-to-r from-blue-50/80 to-white border-b border-gray-100 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <button
            onClick={onPrevPage}
            disabled={currentIndex === 0}
            className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronLeft size={18} />
          </button>
          <span className="font-semibold text-gray-800 text-sm min-w-[80px] text-center">
            {pageNumber} / {totalPages}
          </span>
          <button
            onClick={onNextPage}
            disabled={currentIndex === totalPages - 1}
            className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronRight size={18} />
          </button>
        </div>

        {/* Status badges */}
        <div className="flex items-center gap-2">
          {/* Auto-processing indicator - shows which page is being processed */}
          {isAutoProcessing && processingPageNumber && processingPageNumber !== pageNumber && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-amber-50 text-amber-600 text-xs font-medium rounded-full animate-pulse">
              <Loader2 size={12} className="animate-spin" />
              Processing #{processingPageNumber}
            </span>
          )}
          
          {/* Current page processed badge */}
          {isPageProcessed && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-green-50 text-green-600 text-xs font-medium rounded-full">
              <CheckCircle2 size={12} />
              Done
            </span>
          )}
        </div>
      </div>

      {/* Zoom Toolbar - Enhanced with slider and presets */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-slate-50/80 border-b border-gray-100 shrink-0">
        <div className="flex items-center gap-1.5">
          {/* Zoom out button */}
          <button
            onClick={handleZoomOut}
            disabled={zoomLevel <= MIN_ZOOM}
            className="p-1.5 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            title="Zoom out (Ctrl+Scroll)"
          >
            <Minus size={14} />
          </button>
          
          {/* Zoom slider */}
          <div className="flex items-center gap-1.5">
            <input
              type="range"
              min={MIN_ZOOM}
              max={MAX_ZOOM}
              step={0.01}
              value={zoomLevel}
              onChange={handleSliderChange}
              className="w-24 h-1.5 bg-gray-200 rounded-full appearance-none cursor-pointer accent-blue-600
                [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5 
                [&::-webkit-slider-thumb]:bg-blue-600 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer
                [&::-webkit-slider-thumb]:shadow-md [&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:hover:scale-125
                [&::-moz-range-thumb]:w-3.5 [&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:bg-blue-600 
                [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:cursor-pointer"
              title="Drag to zoom"
            />
          </div>
          
          {/* Zoom in button */}
          <button
            onClick={handleZoomIn}
            disabled={zoomLevel >= MAX_ZOOM}
            className="p-1.5 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            title="Zoom in (Ctrl+Scroll)"
          >
            <Plus size={14} />
          </button>
          
          <div className="w-px h-4 bg-gray-200 mx-0.5" />
          
          {/* Zoom presets dropdown-style buttons */}
          <div className="flex items-center gap-0.5">
            {[0.5, 1, 1.5, 2].map((preset) => (
              <button
                key={preset}
                onClick={() => handlePresetZoom(preset)}
                className={`px-1.5 py-0.5 text-xs font-medium rounded transition-all ${
                  Math.abs(zoomLevel - preset) < 0.05
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-white'
                }`}
                title={`Zoom to ${preset * 100}%`}
              >
                {preset === 1 ? '1×' : preset < 1 ? `${preset * 100}%` : `${preset}×`}
              </button>
            ))}
          </div>
          
          <div className="w-px h-4 bg-gray-200 mx-0.5" />
          
          {/* Fit and reset buttons */}
          <button
            onClick={handleFitToPanel}
            className="p-1.5 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-white transition-all"
            title="Fit to panel"
          >
            <Maximize2 size={14} />
          </button>
          <button
            onClick={handleResetZoom}
            className="p-1.5 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-white transition-all"
            title="Reset to 100%"
          >
            <RotateCcw size={14} />
          </button>
        </div>
        
        {/* Pan hint when zoomed + zoom percentage */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-gray-600 bg-white px-2 py-1 rounded border border-gray-200 min-w-[50px] text-center">
            {zoomPercent}%
          </span>
          {zoomLevel > 1 && (
            <div className="flex items-center gap-1.5 text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded-full">
              <Move size={12} />
              <span>Drag / Double-click</span>
            </div>
          )}
          {zoomLevel <= 1 && (
            <div className="text-xs text-gray-400">
              Double-click to zoom
            </div>
          )}
        </div>
      </div>

      {/* Image Container with zoom and pan support */}
      <div
        ref={containerRef}
        className={`flex-1 overflow-auto bg-gradient-to-br from-gray-100 to-gray-50 min-h-0 relative ${
          zoomLevel > 1 ? (isPanning ? 'cursor-grabbing' : 'cursor-grab') : 'cursor-zoom-in'
        }`}
        onMouseDown={handleMouseDown}
      >
        {/* Loading skeleton */}
        {showLoading && (
          <div className="absolute inset-0 bg-gray-50 flex items-center justify-center z-10">
            <div className="text-center text-gray-400">
              <Loader2 size={24} className="animate-spin mx-auto mb-2" />
              <p className="text-xs">{isDownscaling ? 'Preparing...' : 'Loading...'}</p>
            </div>
          </div>
        )}

        {/* Error state */}
        {showError && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-gray-400">
              <ImageIcon size={32} className="mx-auto mb-2 opacity-50" />
              <p className="text-sm">Failed to load image</p>
            </div>
          </div>
        )}

        {/* No image placeholder */}
        {showPlaceholder && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-gray-400">
              <ImageIcon size={32} className="mx-auto mb-2 opacity-50" />
              <p className="text-sm">No image selected</p>
            </div>
          </div>
        )}

        {/* Zoomable Image - optimized for smooth zoom with high quality rendering */}
        {previewSrc && !showError && (
          <div 
            className="min-w-full min-h-full flex items-center justify-center p-4"
            style={{
              width: zoomLevel > 1 ? `${100 * zoomLevel}%` : '100%',
              height: zoomLevel > 1 ? `${100 * zoomLevel}%` : '100%',
            }}
          >
            <img
              ref={imageRef}
              src={previewSrc}
              alt={`Page ${pageNumber}`}
              className={`rounded-lg shadow-xl select-none ${
                imageLoaded ? 'opacity-100' : 'opacity-0'
              }`}
              onLoad={() => setImageLoaded(true)}
              draggable={false}
              style={{
                transform: `scale(${zoomLevel})`,
                transformOrigin: 'center center',
                transition: isPanning ? 'none' : 'transform 0.1s ease-out, opacity 0.2s ease-out',
                maxWidth: zoomLevel <= 1 ? '100%' : 'none',
                maxHeight: zoomLevel <= 1 ? '100%' : 'none',
                objectFit: 'contain',
                willChange: 'transform',
                imageRendering: zoomLevel > 1.5 ? 'auto' : 'auto',
                backfaceVisibility: 'hidden',
                WebkitBackfaceVisibility: 'hidden',
              }}
            />
          </div>
        )}

      </div>

      {/* Processing overlay - positioned relative to viewport, not scrollable content */}
      {isProcessing && (
        <div className="absolute inset-0 bg-blue-500/10 backdrop-blur-sm flex items-center justify-center z-20 pointer-events-none">
          <div className="bg-white rounded-xl shadow-lg px-5 py-3 text-center pointer-events-auto">
            <Loader2 size={28} className="animate-spin text-blue-600 mx-auto mb-2" />
            <p className="text-sm font-medium text-gray-700">Processing...</p>
          </div>
        </div>
      )}

      {/* Process Controls - Compact */}
      <div className="px-3 py-2 border-t border-gray-100 bg-white shrink-0">
        <div className="flex items-center gap-2">
          {/* Main Process Button - single page */}
          <button
            onClick={onProcess}
            disabled={!canProcess || isProcessing || isPageProcessed}
            className={`
              flex-1 py-2.5 rounded-lg font-semibold text-sm flex items-center justify-center gap-2
              transition-all duration-200
              ${!canProcess || isProcessing || isPageProcessed
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700 shadow-sm hover:shadow-md active:scale-[0.98]'
              }
            `}
          >
            {isProcessing ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Processing...
              </>
            ) : isPageProcessed ? (
              <>
                <CheckCircle2 size={16} />
                Done
              </>
            ) : (
              <>
                <Play size={16} />
                This Page
              </>
            )}
          </button>

          {/* Auto Process All Pages Button */}
          <button
            onClick={onToggleAutoProcess}
            disabled={!canProcess}
            title={isAutoProcessing ? 'Stop auto-processing' : 'Process all pages automatically'}
            className={`
              flex-1 py-2.5 rounded-lg font-semibold text-sm flex items-center justify-center gap-2
              transition-all duration-200
              ${isAutoProcessing
                ? 'bg-amber-500 text-white hover:bg-amber-600'
                : !canProcess
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-emerald-500 to-teal-600 text-white hover:from-emerald-600 hover:to-teal-700 shadow-sm hover:shadow-md active:scale-[0.98]'
              }
            `}
          >
            {isAutoProcessing ? (
              <>
                <Pause size={16} />
                Stop Processing
              </>
            ) : (
              <>
                <Zap size={16} />
                Process All Pages
              </>
            )}
          </button>
        </div>

        {/* Auto-processing waiting for rate limit - show prominent timer */}
        {isAutoProcessing && isWaitingForRateLimit && waitSeconds > 0 && (
          <div className="mt-2 px-3 py-2 rounded-lg bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 flex items-center justify-between">
            <div className="flex items-center gap-2 text-blue-700">
              <Clock size={16} className="animate-pulse" />
              <span className="text-sm font-medium">Waiting for rate limit to reset...</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="text-lg font-bold text-blue-600 tabular-nums min-w-[40px] text-center">
                {waitSeconds}s
              </div>
              <div className="w-16 h-1.5 bg-blue-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-blue-500 rounded-full transition-all duration-1000"
                  style={{ width: `${Math.max(0, (60 - waitSeconds) / 60 * 100)}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Rate limit warning (when not in auto-processing waiting state) */}
        {!rateLimitReady && !isWaitingForRateLimit && (
          <div className="mt-2 px-3 py-1.5 rounded-lg text-xs flex items-center gap-2 bg-amber-50 text-amber-700 border border-amber-100">
            <AlertCircle size={14} />
            Rate limited. Wait {waitSeconds}s...
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="mt-2 px-3 py-1.5 rounded-lg text-xs flex items-center gap-2 bg-red-50 text-red-700 border border-red-100">
            <AlertCircle size={14} />
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
