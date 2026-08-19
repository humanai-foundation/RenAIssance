import React, { useState, useRef, useEffect, useCallback } from 'react';
import { X, ZoomIn, ZoomOut, Move } from 'lucide-react';

// Floating magnifier for inspecting text detail, original beside processed.
// Both the lens and the window itself are draggable.
export default function ZoomLensPreview({
  originalImage,
  processedImage,
  isOpen = false,
  onClose,
  initialPosition = { x: 20, y: 20 },
  className = '',
}) {
  const [position, setPosition] = useState(initialPosition);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  
  const [zoomLevel, setZoomLevel] = useState(200); // percentage
  const [lensPosition, setLensPosition] = useState({ x: 50, y: 50 }); // percentage of image
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  
  const [showBoth, setShowBoth] = useState(true);
  
  const windowRef = useRef(null);
  const imageContainerRef = useRef(null);

  const handleWindowMouseDown = useCallback((e) => {
    if (e.target.closest('.lens-content')) return;
    setIsDragging(true);
    setDragStart({
      x: e.clientX - position.x,
      y: e.clientY - position.y,
    });
  }, [position]);

  const handleWindowMouseMove = useCallback((e) => {
    if (isDragging) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  }, [isDragging, dragStart]);

  const handleWindowMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleLensMouseDown = useCallback((e) => {
    setIsPanning(true);
    setPanStart({ x: e.clientX, y: e.clientY });
  }, []);

  const handleLensMouseMove = useCallback((e) => {
    if (isPanning && imageContainerRef.current) {
      const rect = imageContainerRef.current.getBoundingClientRect();
      const deltaX = ((e.clientX - panStart.x) / rect.width) * 100;
      const deltaY = ((e.clientY - panStart.y) / rect.height) * 100;
      
      setLensPosition(prev => ({
        x: Math.max(0, Math.min(100, prev.x - deltaX)),
        y: Math.max(0, Math.min(100, prev.y - deltaY)),
      }));
      setPanStart({ x: e.clientX, y: e.clientY });
    }
  }, [isPanning, panStart]);

  const handleLensMouseUp = useCallback(() => {
    setIsPanning(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleWindowMouseMove);
      window.addEventListener('mouseup', handleWindowMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleWindowMouseMove);
        window.removeEventListener('mouseup', handleWindowMouseUp);
      };
    }
  }, [isDragging, handleWindowMouseMove, handleWindowMouseUp]);

  useEffect(() => {
    if (isPanning) {
      window.addEventListener('mousemove', handleLensMouseMove);
      window.addEventListener('mouseup', handleLensMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleLensMouseMove);
        window.removeEventListener('mouseup', handleLensMouseUp);
      };
    }
  }, [isPanning, handleLensMouseMove, handleLensMouseUp]);

  if (!isOpen) return null;

  const zoomLevels = [100, 150, 200, 300, 400, 500];

  const getBackgroundStyle = (imageSrc) => ({
    backgroundImage: `url(${imageSrc})`,
    backgroundSize: `${zoomLevel}%`,
    backgroundPosition: `${lensPosition.x}% ${lensPosition.y}%`,
    backgroundRepeat: 'no-repeat',
  });

  return (
    <div
      ref={windowRef}
      className={`fixed z-50 bg-white rounded-xl shadow-2xl border border-gray-200 overflow-hidden ${className}`}
      style={{
        left: position.x,
        top: position.y,
        width: showBoth ? 500 : 260,
      }}
      onMouseDown={handleWindowMouseDown}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-gray-50 border-b border-gray-200 cursor-move">
        <div className="flex items-center gap-2">
          <Move size={14} className="text-gray-400" />
          <span className="text-xs font-semibold text-gray-700">Zoom Lens</span>
        </div>
        
        <div className="flex items-center gap-2">
          {/* Show both toggle */}
          {processedImage && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowBoth(!showBoth);
              }}
              className={`px-2 py-0.5 text-[10px] font-medium rounded transition-colors ${
                showBoth 
                  ? 'bg-blue-100 text-blue-600' 
                  : 'bg-gray-200 text-gray-600'
              }`}
            >
              Compare
            </button>
          )}
          
          {/* Close button */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onClose();
            }}
            className="p-1 rounded hover:bg-gray-200 transition-colors"
          >
            <X size={14} className="text-gray-500" />
          </button>
        </div>
      </div>

      {/* Zoom controls */}
      <div className="flex items-center justify-center gap-2 px-3 py-1.5 bg-gray-50/50 border-b border-gray-100">
        <button
          onClick={() => setZoomLevel(prev => Math.max(100, prev - 50))}
          disabled={zoomLevel <= 100}
          className="p-1 rounded hover:bg-gray-200 disabled:opacity-40 transition-colors"
        >
          <ZoomOut size={14} className="text-gray-600" />
        </button>
        
        <select
          value={zoomLevel}
          onChange={(e) => setZoomLevel(parseInt(e.target.value))}
          onClick={(e) => e.stopPropagation()}
          className="text-xs font-medium text-gray-600 bg-white border border-gray-200 
                     rounded px-2 py-0.5 focus:outline-none"
        >
          {zoomLevels.map(level => (
            <option key={level} value={level}>{level}%</option>
          ))}
        </select>
        
        <button
          onClick={() => setZoomLevel(prev => Math.min(500, prev + 50))}
          disabled={zoomLevel >= 500}
          className="p-1 rounded hover:bg-gray-200 disabled:opacity-40 transition-colors"
        >
          <ZoomIn size={14} className="text-gray-600" />
        </button>
      </div>

      {/* Lens content */}
      <div 
        ref={imageContainerRef}
        className={`lens-content flex ${showBoth ? 'gap-1 p-1' : 'p-1'}`}
      >
        {/* Original image lens */}
        <div className="flex-1">
          {showBoth && (
            <div className="text-[10px] font-medium text-gray-500 text-center mb-1">
              Original
            </div>
          )}
          <div
            className={`h-48 rounded-lg border border-gray-200 cursor-crosshair ${
              isPanning ? 'cursor-grabbing' : ''
            }`}
            style={originalImage ? getBackgroundStyle(originalImage) : { backgroundColor: '#f3f4f6' }}
            onMouseDown={handleLensMouseDown}
          >
            {!originalImage && (
              <div className="h-full flex items-center justify-center text-xs text-gray-400">
                No image
              </div>
            )}
          </div>
        </div>

        {/* Processed image lens */}
        {showBoth && processedImage && (
          <div className="flex-1">
            <div className="text-[10px] font-medium text-blue-500 text-center mb-1">
              Processed
            </div>
            <div
              className={`h-48 rounded-lg border border-blue-200 cursor-crosshair ${
                isPanning ? 'cursor-grabbing' : ''
              }`}
              style={getBackgroundStyle(processedImage)}
              onMouseDown={handleLensMouseDown}
            />
          </div>
        )}

        {/* Show processed only when not in compare mode */}
        {!showBoth && processedImage && (
          <div
            className={`flex-1 h-48 rounded-lg border border-blue-200 cursor-crosshair ${
              isPanning ? 'cursor-grabbing' : ''
            }`}
            style={getBackgroundStyle(processedImage)}
            onMouseDown={handleLensMouseDown}
          />
        )}
      </div>

      {/* Position indicator */}
      <div className="px-3 py-1.5 bg-gray-50 border-t border-gray-100 text-[10px] text-gray-400 text-center">
        Position: {Math.round(lensPosition.x)}%, {Math.round(lensPosition.y)}% — Drag to pan
      </div>
    </div>
  );
}
