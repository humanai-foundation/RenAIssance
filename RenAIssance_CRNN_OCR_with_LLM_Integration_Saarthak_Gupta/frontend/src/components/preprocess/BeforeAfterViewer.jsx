import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Eye, 
  EyeOff, 
  Maximize2, 
  Minimize2, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  GripVertical,
  Move,
  Search
} from 'lucide-react';

// Split-view comparison: draggable divider, before/after toggle, synced
// zoom and pan across both images.
export default function BeforeAfterViewer({
  originalImage,
  processedImage,
  isProcessing = false,
  processingLabel = 'Processing...',
  onOpenZoomLens,
  className = '',
}) {
  const [mode, setMode] = useState('split'); // 'split', 'before', 'after', 'toggle'
  const [splitPosition, setSplitPosition] = useState(50);
  const [zoom, setZoom] = useState('fit'); // 'fit', 50, 75, 100, 150, 200, 300
  const [isDraggingSplit, setIsDraggingSplit] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  
  const containerRef = useRef(null);
  const imageRef = useRef(null);

  const zoomLevels = ['fit', 50, 75, 100, 150, 200, 300];

  const handleMouseMove = useCallback((e) => {
    if (isDraggingSplit && mode === 'split' && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percentage = Math.max(5, Math.min(95, (x / rect.width) * 100));
      setSplitPosition(percentage);
    }
    
    if (isPanning && zoom !== 'fit') {
      const deltaX = e.clientX - panStart.x;
      const deltaY = e.clientY - panStart.y;
      setPanOffset({
        x: panOffset.x + deltaX,
        y: panOffset.y + deltaY,
      });
      setPanStart({ x: e.clientX, y: e.clientY });
    }
  }, [isDraggingSplit, isPanning, mode, zoom, panStart, panOffset]);

  const handleMouseUp = useCallback(() => {
    setIsDraggingSplit(false);
    setIsPanning(false);
  }, []);

  const handleMouseDown = useCallback((e) => {
    if (zoom !== 'fit' && e.button === 0) {
      setIsPanning(true);
      setPanStart({ x: e.clientX, y: e.clientY });
    }
  }, [zoom]);

  useEffect(() => {
    if (isDraggingSplit || isPanning) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDraggingSplit, isPanning, handleMouseMove, handleMouseUp]);

  useEffect(() => {
    if (zoom === 'fit') {
      setPanOffset({ x: 0, y: 0 });
    }
  }, [zoom]);

  const handleZoomIn = () => {
    const currentIndex = zoomLevels.indexOf(zoom);
    if (currentIndex < zoomLevels.length - 1) {
      setZoom(zoomLevels[currentIndex + 1]);
    }
  };

  const handleZoomOut = () => {
    const currentIndex = zoomLevels.indexOf(zoom);
    if (currentIndex > 0) {
      setZoom(zoomLevels[currentIndex - 1]);
    }
  };

  const handleResetView = () => {
    setZoom('fit');
    setPanOffset({ x: 0, y: 0 });
    setSplitPosition(50);
  };

  const getImageStyle = () => {
    if (zoom === 'fit') {
      return { 
        width: '100%', 
        height: '100%', 
        objectFit: 'contain',
        transform: 'none',
      };
    }
    return { 
      transform: `scale(${zoom / 100}) translate(${panOffset.x / (zoom / 100)}px, ${panOffset.y / (zoom / 100)}px)`,
      transformOrigin: 'center center',
      cursor: isPanning ? 'grabbing' : 'grab',
    };
  };

  const imageStyle = getImageStyle();

  return (
    <div
      className={`bg-white rounded-xl shadow-card overflow-hidden flex flex-col transition-all ${
        isFullscreen ? 'fixed inset-4 z-50' : ''
      } ${className}`}
    >
      {/* Controls toolbar */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-100 bg-gray-50/50">
        {/* View mode toggle */}
        <div className="flex items-center bg-white rounded-lg border border-gray-200 p-0.5">
          <button
            onClick={() => setMode('before')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              mode === 'before'
                ? 'bg-blue-500 text-white shadow-sm'
                : 'text-gray-600 hover:text-blue-600 hover:bg-blue-50'
            }`}
          >
            <Eye className="w-3.5 h-3.5 inline mr-1" />
            Before
          </button>
          <button
            onClick={() => setMode('split')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              mode === 'split'
                ? 'bg-blue-500 text-white shadow-sm'
                : 'text-gray-600 hover:text-blue-600 hover:bg-blue-50'
            }`}
          >
            Split
          </button>
          <button
            onClick={() => setMode('after')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              mode === 'after'
                ? 'bg-blue-500 text-white shadow-sm'
                : 'text-gray-600 hover:text-blue-600 hover:bg-blue-50'
            }`}
          >
            <EyeOff className="w-3.5 h-3.5 inline mr-1" />
            After
          </button>
        </div>

        {/* Center - Zoom controls */}
        <div className="flex items-center gap-1 bg-white rounded-lg border border-gray-200 p-0.5">
          <button
            onClick={handleZoomOut}
            disabled={zoom === 'fit' || zoomLevels.indexOf(zoom) === 0}
            className="p-1.5 rounded-md hover:bg-gray-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            <ZoomOut className="w-4 h-4 text-gray-600" />
          </button>
          
          <select
            value={zoom}
            onChange={(e) => setZoom(e.target.value === 'fit' ? 'fit' : parseInt(e.target.value))}
            className="px-2 py-1 text-xs font-medium text-gray-600 bg-transparent border-0 
                       min-w-[4.5rem] text-center focus:outline-none focus:ring-0"
          >
            <option value="fit">Fit</option>
            <option value="50">50%</option>
            <option value="75">75%</option>
            <option value="100">100%</option>
            <option value="150">150%</option>
            <option value="200">200%</option>
            <option value="300">300%</option>
          </select>
          
          <button
            onClick={handleZoomIn}
            disabled={zoomLevels.indexOf(zoom) === zoomLevels.length - 1}
            className="p-1.5 rounded-md hover:bg-gray-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            <ZoomIn className="w-4 h-4 text-gray-600" />
          </button>
          
          <div className="w-px h-4 bg-gray-200 mx-0.5" />
          
          <button
            onClick={handleResetView}
            className="p-1.5 rounded-md hover:bg-gray-100 transition-colors"
            title="Reset view"
          >
            <RotateCcw className="w-4 h-4 text-gray-600" />
          </button>
        </div>

        {/* Right - Actions */}
        <div className="flex items-center gap-1">
          {/* Zoom lens button */}
          {onOpenZoomLens && (
            <button
              onClick={onOpenZoomLens}
              className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
              title="Open zoom lens"
            >
              <Search className="w-4 h-4 text-gray-600" />
            </button>
          )}
          
          {/* Pan indicator */}
          {zoom !== 'fit' && (
            <div className="flex items-center gap-1 px-2 py-1 bg-blue-50 rounded-lg text-xs text-blue-600">
              <Move size={12} />
              <span>Pan</span>
            </div>
          )}

          {/* Fullscreen toggle */}
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          >
            {isFullscreen ? (
              <Minimize2 className="w-4 h-4 text-gray-600" />
            ) : (
              <Maximize2 className="w-4 h-4 text-gray-600" />
            )}
          </button>
        </div>
      </div>

      {/* Image container */}
      <div
        ref={containerRef}
        className={`relative flex-1 bg-gray-100 overflow-hidden ${
          zoom !== 'fit' ? 'cursor-grab active:cursor-grabbing' : ''
        }`}
        onMouseDown={zoom !== 'fit' ? handleMouseDown : undefined}
        style={{ minHeight: isFullscreen ? 'auto' : '400px' }}
      >
        {/* Processing overlay */}
        {isProcessing && (
          <div className="absolute inset-0 z-30 flex items-center justify-center bg-white/80 backdrop-blur-sm">
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 border-3 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
              <p className="mt-3 text-sm font-medium text-blue-600">{processingLabel}</p>
            </div>
          </div>
        )}

        {/* Original image (shown in before and split modes) */}
        {(mode === 'before' || mode === 'split') && originalImage && (
          <div
            className={`absolute inset-0 flex items-center justify-center ${
              mode === 'split' ? 'overflow-hidden' : ''
            }`}
            style={
              mode === 'split'
                ? { clipPath: `inset(0 ${100 - splitPosition}% 0 0)` }
                : {}
            }
          >
            <img
              ref={imageRef}
              src={originalImage}
              alt="Original"
              className={zoom === 'fit' ? 'max-w-full max-h-full object-contain' : 'max-w-none'}
              style={imageStyle}
              draggable={false}
            />
          </div>
        )}

        {/* Processed image (shown in after and split modes) */}
        {(mode === 'after' || mode === 'split') && (
          <div
            className={`absolute inset-0 flex items-center justify-center ${
              mode === 'split' ? 'overflow-hidden' : ''
            }`}
            style={
              mode === 'split'
                ? { clipPath: `inset(0 0 0 ${splitPosition}%)` }
                : {}
            }
          >
            {processedImage ? (
              <img
                src={processedImage}
                alt="Processed"
                className={zoom === 'fit' ? 'max-w-full max-h-full object-contain' : 'max-w-none'}
                style={imageStyle}
                draggable={false}
              />
            ) : (
              <div className="text-gray-400 text-center px-4">
                <div className="w-16 h-16 mx-auto mb-3 rounded-full bg-gray-200 flex items-center justify-center">
                  <EyeOff className="w-8 h-8 text-gray-400" />
                </div>
                <p className="text-sm font-medium">No processed image</p>
                <p className="text-xs mt-1">Apply preprocessing to see results</p>
              </div>
            )}
          </div>
        )}

        {/* Split slider handle */}
        {mode === 'split' && (
          <div
            className="absolute top-0 bottom-0 w-1 cursor-ew-resize z-20 group"
            style={{ left: `calc(${splitPosition}% - 2px)` }}
            onMouseDown={(e) => {
              e.preventDefault();
              setIsDraggingSplit(true);
            }}
          >
            {/* Visible line */}
            <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-0.5 bg-white shadow-lg" />
            
            {/* Handle circle */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 
                            w-10 h-10 bg-white rounded-full flex items-center justify-center 
                            shadow-lg border-2 border-blue-500 cursor-grab active:cursor-grabbing
                            group-hover:scale-110 transition-transform">
              <GripVertical className="w-4 h-4 text-blue-500" />
            </div>

            {/* Labels */}
            <div className="absolute top-3 -translate-x-full -left-2 
                            bg-gray-900/80 text-white text-[10px] font-medium 
                            px-2 py-1 rounded shadow-md pointer-events-none">
              Before
            </div>
            <div className="absolute top-3 left-2 
                            bg-blue-600/90 text-white text-[10px] font-medium 
                            px-2 py-1 rounded shadow-md pointer-events-none">
              After
            </div>
          </div>
        )}

        {/* No image placeholder */}
        {!originalImage && !isProcessing && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-gray-400">
              <div className="w-16 h-16 mx-auto mb-3 rounded-full bg-gray-200 flex items-center justify-center">
                <Eye className="w-8 h-8 text-gray-400" />
              </div>
              <p className="text-sm font-medium">No image selected</p>
              <p className="text-xs mt-1">Select a page to preview</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
