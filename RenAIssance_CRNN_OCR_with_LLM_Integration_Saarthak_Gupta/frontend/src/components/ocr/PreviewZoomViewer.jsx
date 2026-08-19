import { useState, useRef, useEffect, useCallback } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize2,
  Move,
  Loader2,
  Image as ImageIcon,
} from 'lucide-react';

// Zoom + pan viewer for preview images.
export default function PreviewZoomViewer({
  imageSrc,
  isLoading,
  hasError,
  pageNumber,
  className = '',
}) {
  const [zoomLevel, setZoomLevel] = useState(1);
  const [isPanning, setIsPanning] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const containerRef = useRef(null);
  const imageRef = useRef(null);

  const MIN_ZOOM = 0.5;
  const MAX_ZOOM = 4.0;
  const ZOOM_STEP = 0.25;

  useEffect(() => {
    setZoomLevel(1);
    setImageLoaded(false);
  }, [imageSrc]);

  const handleZoomIn = useCallback(() => {
    setZoomLevel((prev) => Math.min(prev + ZOOM_STEP, MAX_ZOOM));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoomLevel((prev) => Math.max(prev - ZOOM_STEP, MIN_ZOOM));
  }, []);

  const handleReset = useCallback(() => {
    setZoomLevel(1);
    if (containerRef.current) {
      containerRef.current.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
    }
  }, []);

  const handleFitToPanel = useCallback(() => {
    if (!containerRef.current || !imageDimensions.width || !imageDimensions.height) {
      setZoomLevel(1);
      return;
    }

    const container = containerRef.current;
    const containerWidth = container.clientWidth - 32; // Account for padding
    const containerHeight = container.clientHeight - 32;

    const widthRatio = containerWidth / imageDimensions.width;
    const heightRatio = containerHeight / imageDimensions.height;
    const fitZoom = Math.min(widthRatio, heightRatio, 1);

    setZoomLevel(Math.max(MIN_ZOOM, Math.min(fitZoom, MAX_ZOOM)));
  }, [imageDimensions]);

  const handleWheel = useCallback((e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
      setZoomLevel((prev) => Math.max(MIN_ZOOM, Math.min(prev + delta, MAX_ZOOM)));
    }
  }, []);

  // passive: false so preventDefault actually works.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

  const handleImageLoad = useCallback((e) => {
    setImageLoaded(true);
    setImageDimensions({
      width: e.target.naturalWidth,
      height: e.target.naturalHeight,
    });
  }, []);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      if ((e.ctrlKey || e.metaKey) && e.key === '=') {
        e.preventDefault();
        handleZoomIn();
      } else if ((e.ctrlKey || e.metaKey) && e.key === '-') {
        e.preventDefault();
        handleZoomOut();
      } else if ((e.ctrlKey || e.metaKey) && e.key === '0') {
        e.preventDefault();
        handleReset();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleZoomIn, handleZoomOut, handleReset]);

  const showLoading = isLoading || (!imageLoaded && imageSrc && !hasError);
  const showPlaceholder = !imageSrc && !isLoading;
  const showError = hasError;

  const zoomPercent = Math.round(zoomLevel * 100);

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-3 py-2 bg-gradient-to-r from-slate-50 to-gray-50 border-b border-gray-100 shrink-0">
        {/* Zoom Controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={handleZoomOut}
            disabled={zoomLevel <= MIN_ZOOM}
            className="p-1.5 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-white hover:shadow-sm disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            title="Zoom out (Ctrl+-)"
          >
            <ZoomOut size={18} />
          </button>
          
          <div className="px-2 py-1 min-w-[60px] text-center">
            <span className="text-sm font-semibold text-gray-700">{zoomPercent}%</span>
          </div>

          <button
            onClick={handleZoomIn}
            disabled={zoomLevel >= MAX_ZOOM}
            className="p-1.5 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-white hover:shadow-sm disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            title="Zoom in (Ctrl++)"
          >
            <ZoomIn size={18} />
          </button>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-1">
          <button
            onClick={handleFitToPanel}
            className="p-1.5 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-white hover:shadow-sm transition-all"
            title="Fit to panel"
          >
            <Maximize2 size={18} />
          </button>

          <button
            onClick={handleReset}
            className="p-1.5 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-white hover:shadow-sm transition-all"
            title="Reset zoom (Ctrl+0)"
          >
            <RotateCcw size={18} />
          </button>
        </div>

        {/* Pan indicator */}
        {zoomLevel > 1 && (
          <div className="flex items-center gap-1.5 px-2 py-1 bg-blue-50 text-blue-600 rounded-lg text-xs font-medium">
            <Move size={14} />
            <span>Scroll to pan</span>
          </div>
        )}
      </div>

      {/* Image Container with zoom/pan */}
      <div
        ref={containerRef}
        className={`flex-1 overflow-auto bg-gradient-to-br from-gray-100 to-slate-100 min-h-0 relative ${
          zoomLevel > 1 ? 'cursor-grab active:cursor-grabbing' : ''
        }`}
        style={{
          display: 'flex',
          alignItems: zoomLevel <= 1 ? 'center' : 'flex-start',
          justifyContent: zoomLevel <= 1 ? 'center' : 'flex-start',
        }}
      >
        {/* Loading State */}
        {showLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-50/80 backdrop-blur-sm z-10">
            <div className="text-center">
              <Loader2 size={32} className="animate-spin text-blue-500 mx-auto mb-2" />
              <p className="text-sm text-gray-500">Loading preview...</p>
            </div>
          </div>
        )}

        {/* Error State */}
        {showError && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-gray-400">
              <ImageIcon size={48} className="mx-auto mb-2 opacity-50" />
              <p className="text-sm font-medium">Failed to load image</p>
              <p className="text-xs mt-1">Image may be too large or corrupted</p>
            </div>
          </div>
        )}

        {/* Placeholder */}
        {showPlaceholder && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-gray-400">
              <ImageIcon size={48} className="mx-auto mb-2 opacity-50" />
              <p className="text-sm font-medium">No image selected</p>
            </div>
          </div>
        )}

        {/* Zoomable Image */}
        {imageSrc && !showError && (
          <div
            className="p-4"
            style={{
              transform: `scale(${zoomLevel})`,
              transformOrigin: zoomLevel <= 1 ? 'center center' : 'top left',
              transition: 'transform 0.15s ease-out',
              minWidth: zoomLevel > 1 ? `${imageDimensions.width * zoomLevel}px` : 'auto',
              minHeight: zoomLevel > 1 ? `${imageDimensions.height * zoomLevel}px` : 'auto',
            }}
          >
            <img
              ref={imageRef}
              src={imageSrc}
              alt={`Page ${pageNumber}`}
              className={`max-w-full h-auto rounded-lg shadow-lg transition-opacity duration-300 ${
                imageLoaded ? 'opacity-100' : 'opacity-0'
              }`}
              onLoad={handleImageLoad}
              draggable={false}
              style={{
                maxHeight: zoomLevel <= 1 ? 'calc(100% - 2rem)' : 'none',
              }}
            />
          </div>
        )}
      </div>

      {/* Zoom hint */}
      <div className="px-3 py-1.5 bg-gray-50 border-t border-gray-100 text-xs text-gray-400 text-center shrink-0">
        <span>Ctrl + Scroll to zoom • Scroll to pan when zoomed</span>
      </div>
    </div>
  );
}
