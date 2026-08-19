import { useState, useCallback, useRef, useEffect } from 'react';
import { GripVertical } from 'lucide-react';

// Three columns with draggable dividers.
export default function ResizablePanels({
  leftPanel,
  centerPanel,
  rightPanel = null,
  defaultLeftWidth = 260,
  defaultRightWidth = 340,
  minLeftWidth = 200,
  maxLeftWidth = 400,
  minRightWidth = 280,
  maxRightWidth = 500,
  minCenterWidth = 300,
}) {
  const hasRight = Boolean(rightPanel);
  const containerRef = useRef(null);
  const [leftWidth, setLeftWidth] = useState(defaultLeftWidth);
  const [rightWidth, setRightWidth] = useState(defaultRightWidth);
  const [isDraggingLeft, setIsDraggingLeft] = useState(false);
  const [isDraggingRight, setIsDraggingRight] = useState(false);
  const [containerWidth, setContainerWidth] = useState(0);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const updateWidth = () => {
      setContainerWidth(container.offsetWidth);
    };

    updateWidth();
    const resizeObserver = new ResizeObserver(updateWidth);
    resizeObserver.observe(container);

    return () => resizeObserver.disconnect();
  }, []);

  // Center takes whatever the two side panels leave.
  const gapWidth = 8; // gap-2 = 0.5rem = 8px
  const resizerWidth = 12; // width of resizer handles
  const rightTotalWidth = hasRight ? rightWidth + gapWidth + resizerWidth : 0;
  const centerWidth = containerWidth - leftWidth - rightTotalWidth - gapWidth - resizerWidth;

  const handleLeftMouseDown = useCallback((e) => {
    e.preventDefault();
    setIsDraggingLeft(true);
  }, []);

  const handleRightMouseDown = useCallback((e) => {
    e.preventDefault();
    setIsDraggingRight(true);
  }, []);

  // On window, so dragging past the divider still tracks.
  useEffect(() => {
    if (!isDraggingLeft && !isDraggingRight) return;

    const handleMouseMove = (e) => {
      const container = containerRef.current;
      if (!container) return;

      const containerRect = container.getBoundingClientRect();
      
      if (isDraggingLeft) {
        const newLeftWidth = e.clientX - containerRect.left;
        const maxAllowed = containerWidth - rightWidth - minCenterWidth - (gapWidth * 2) - (resizerWidth * 2);
        setLeftWidth(Math.max(minLeftWidth, Math.min(maxAllowed, Math.min(maxLeftWidth, newLeftWidth))));
      }
      
      if (isDraggingRight) {
        const newRightWidth = containerRect.right - e.clientX;
        const maxAllowed = containerWidth - leftWidth - minCenterWidth - (gapWidth * 2) - (resizerWidth * 2);
        setRightWidth(Math.max(minRightWidth, Math.min(maxAllowed, Math.min(maxRightWidth, newRightWidth))));
      }
    };

    const handleMouseUp = () => {
      setIsDraggingLeft(false);
      setIsDraggingRight(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDraggingLeft, isDraggingRight, containerWidth, leftWidth, rightWidth, minLeftWidth, maxLeftWidth, minRightWidth, maxRightWidth, minCenterWidth]);

  const handleLeftDoubleClick = useCallback(() => {
    setLeftWidth(defaultLeftWidth);
  }, [defaultLeftWidth]);

  const handleRightDoubleClick = useCallback(() => {
    setRightWidth(defaultRightWidth);
  }, [defaultRightWidth]);

  return (
    <div 
      ref={containerRef}
      className="h-full flex items-stretch select-none"
      style={{ cursor: isDraggingLeft || isDraggingRight ? 'col-resize' : 'default' }}
    >
      {/* Left Panel */}
      <div 
        className="flex-shrink-0 min-h-0 overflow-hidden flex flex-col gap-2"
        style={{ width: leftWidth }}
      >
        {leftPanel}
      </div>

      {/* Left Resizer */}
      <div
        onMouseDown={handleLeftMouseDown}
        onDoubleClick={handleLeftDoubleClick}
        className={`
          flex-shrink-0 w-3 flex items-center justify-center cursor-col-resize
          group hover:bg-blue-100 transition-colors rounded mx-0.5
          ${isDraggingLeft ? 'bg-blue-200' : 'bg-transparent'}
        `}
        title="Drag to resize, double-click to reset"
      >
        <div className={`
          w-1 h-12 rounded-full transition-all
          ${isDraggingLeft ? 'bg-blue-500' : 'bg-gray-300 group-hover:bg-blue-400'}
        `}>
          <GripVertical 
            size={12} 
            className={`
              mt-4 -ml-0.5 transition-colors
              ${isDraggingLeft ? 'text-blue-600' : 'text-gray-400 group-hover:text-blue-500'}
            `} 
          />
        </div>
      </div>

      {/* Center Panel */}
      <div 
        className="flex-1 min-w-0 min-h-0 overflow-hidden"
        style={{ minWidth: minCenterWidth }}
      >
        {centerPanel}
      </div>

      {/* Right Resizer — only when right panel exists */}
      {hasRight && (
        <div
          onMouseDown={handleRightMouseDown}
          onDoubleClick={handleRightDoubleClick}
          className={`
            flex-shrink-0 w-3 flex items-center justify-center cursor-col-resize
            group hover:bg-blue-100 transition-colors rounded mx-0.5
            ${isDraggingRight ? 'bg-blue-200' : 'bg-transparent'}
          `}
          title="Drag to resize, double-click to reset"
        >
          <div className={`
            w-1 h-12 rounded-full transition-all
            ${isDraggingRight ? 'bg-blue-500' : 'bg-gray-300 group-hover:bg-blue-400'}
          `}>
            <GripVertical 
              size={12} 
              className={`
                mt-4 -ml-0.5 transition-colors
                ${isDraggingRight ? 'text-blue-600' : 'text-gray-400 group-hover:text-blue-500'}
              `} 
            />
          </div>
        </div>
      )}

      {/* Right Panel — only when present */}
      {hasRight && (
        <div 
          className="flex-shrink-0 min-h-0 overflow-hidden"
          style={{ width: rightWidth }}
        >
          {rightPanel}
        </div>
      )}

      {/* Drag overlay to prevent selection */}
      {(isDraggingLeft || isDraggingRight) && (
        <div className="fixed inset-0 z-50 cursor-col-resize" />
      )}
    </div>
  );
}
