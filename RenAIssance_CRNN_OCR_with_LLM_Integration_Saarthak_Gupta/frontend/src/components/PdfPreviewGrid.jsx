import React, { useState, useCallback, useEffect, useRef } from 'react';
import { CheckSquare, Square, Grid3X3, LayoutGrid, X, ZoomIn } from 'lucide-react';
import PageCard from './PageCard';

// Rect intersection, both in { left, top, right, bottom }.
function rectsIntersect(a, b) {
  return !(a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom);
}

export default function PdfPreviewGrid({
  pages,
  selectedPages,
  onSelectionChange,
  onPagePreview,
}) {
  const [gridSize, setGridSize] = useState('medium'); // small, medium, large
  const [lastSelectedIndex, setLastSelectedIndex] = useState(null);
  const [previewPage, setPreviewPage] = useState(null);

  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState(null);
  const [dragEnd, setDragEnd] = useState(null);
  const dragModifierRef = useRef({ ctrl: false, shift: false });
  const preSelectionRef = useRef([]);
  const pageRefs = useRef({});
  const gridRef = useRef(null);

  const gridClasses = {
    small: 'grid-cols-4 md:grid-cols-6 lg:grid-cols-8 2xl:grid-cols-10',
    medium: 'grid-cols-3 md:grid-cols-4 lg:grid-cols-6 2xl:grid-cols-8',
    large: 'grid-cols-2 md:grid-cols-3 lg:grid-cols-4 2xl:grid-cols-5',
  };

  const handleSelectPage = useCallback(
    (pageNumber, isRangeSelect) => {
      const pageIndex = pages.findIndex(p => p.pageNumber === pageNumber);

      if (isRangeSelect && lastSelectedIndex !== null) {
        const start = Math.min(lastSelectedIndex, pageIndex);
        const end = Math.max(lastSelectedIndex, pageIndex);
        const rangePageNumbers = pages.slice(start, end + 1).map(p => p.pageNumber);
        const newSelection = Array.from(new Set([...selectedPages, ...rangePageNumbers]));
        onSelectionChange(newSelection);
      } else {
        const newSelection = selectedPages.includes(pageNumber)
          ? selectedPages.filter((p) => p !== pageNumber)
          : [...selectedPages, pageNumber];
        onSelectionChange(newSelection);
        setLastSelectedIndex(pageIndex);
      }
    },
    [pages, selectedPages, lastSelectedIndex, onSelectionChange]
  );

  const handleSelectAll = () => {
    if (selectedPages.length === pages.length) {
      onSelectionChange([]);
    } else {
      onSelectionChange(pages.map((p) => p.pageNumber));
    }
  };

  const handlePreview = (pageNumber) => {
    setPreviewPage(pageNumber);
    if (onPagePreview) {
      onPagePreview(pageNumber);
    }
  };

  const isAllSelected = selectedPages.length === pages.length && pages.length > 0;

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && previewPage) {
        setPreviewPage(null);
      }
      if (e.key === 'a' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        handleSelectAll();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [previewPage, pages.length]);

  // ── Rubber-band selection ──

  const computeDragSelection = useCallback((start, end) => {
    if (!start || !end) return;

    const bandRect = {
      left: Math.min(start.x, end.x),
      top: Math.min(start.y, end.y),
      right: Math.max(start.x, end.x),
      bottom: Math.max(start.y, end.y),
    };

    const intersectedPages = [];
    for (const page of pages) {
      const el = pageRefs.current[page.pageNumber];
      if (!el) continue;
      const r = el.getBoundingClientRect();
      const cardRect = { left: r.left, top: r.top, right: r.right, bottom: r.bottom };
      if (rectsIntersect(bandRect, cardRect)) {
        intersectedPages.push(page.pageNumber);
      }
    }

    const { ctrl, shift } = dragModifierRef.current;
    let newSelection;

    if (ctrl || shift) {
      newSelection = Array.from(new Set([...preSelectionRef.current, ...intersectedPages]));
    } else {
      newSelection = intersectedPages;
    }

    onSelectionChange(newSelection);
  }, [pages, onSelectionChange]);

  const handleGridMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    // Clicks on a card or control aren't the start of a selection drag.
    if (e.target.closest('[data-page-card]')) return;
    if (e.target.closest('button')) return;

    e.preventDefault();
    const pos = { x: e.clientX, y: e.clientY };
    setDragStart(pos);
    setDragEnd(pos);
    setIsDragging(true);
    dragModifierRef.current = { ctrl: e.ctrlKey || e.metaKey, shift: e.shiftKey };
    preSelectionRef.current = [...selectedPages];
  }, [selectedPages]);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e) => {
      const pos = { x: e.clientX, y: e.clientY };
      setDragEnd(pos);
      computeDragSelection(dragStart, pos);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      setDragStart(null);
      setDragEnd(null);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, dragStart, computeDragSelection]);

  const rubberBandStyle = isDragging && dragStart && dragEnd ? {
    position: 'fixed',
    left: Math.min(dragStart.x, dragEnd.x),
    top: Math.min(dragStart.y, dragEnd.y),
    width: Math.abs(dragEnd.x - dragStart.x),
    height: Math.abs(dragEnd.y - dragStart.y),
    zIndex: 50,
    pointerEvents: 'none',
  } : null;

  return (
    <div className="animate-fade-in">
      {/* Toolbar */}
      <div className="flex items-center justify-between mb-6 p-4 bg-white rounded-xl shadow-card">
        <div className="flex items-center gap-4">
          <button
            onClick={handleSelectAll}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${isAllSelected
                ? 'bg-blue-600 text-white'
                : 'bg-blue-50 text-blue-600 hover:bg-blue-100'
              }`}
          >
            {isAllSelected ? (
              <CheckSquare className="w-5 h-5" />
            ) : (
              <Square className="w-5 h-5" />
            )}
            {isAllSelected ? 'Deselect All' : 'Select All'}
          </button>

          <span className="text-sm text-gray-500">
            {selectedPages.length} of {pages.length} pages selected
          </span>
        </div>

        {/* Grid size controls */}
        <div className="flex items-center gap-2 bg-blue-50 rounded-lg p-1">
          <button
            onClick={() => setGridSize('small')}
            className={`p-2 rounded-md transition-colors ${gridSize === 'small'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-500 hover:text-blue-600'
              }`}
            title="Small grid"
          >
            <Grid3X3 className="w-5 h-5" />
          </button>
          <button
            onClick={() => setGridSize('medium')}
            className={`p-2 rounded-md transition-colors ${gridSize === 'medium'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-500 hover:text-blue-600'
              }`}
            title="Medium grid"
          >
            <LayoutGrid className="w-5 h-5" />
          </button>
          <button
            onClick={() => setGridSize('large')}
            className={`p-2 rounded-md transition-colors ${gridSize === 'large'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-500 hover:text-blue-600'
              }`}
            title="Large grid"
          >
            <ZoomIn className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Grid */}
      <div
        ref={gridRef}
        className={`grid gap-4 ${gridClasses[gridSize]} ${isDragging ? 'select-none' : ''}`}
        onMouseDown={handleGridMouseDown}
      >
        {pages.map((page, index) => (
          <PageCard
            key={page.pageNumber}
            ref={(el) => { if (el) pageRefs.current[page.pageNumber] = el; }}
            pageNumber={page.pageNumber}
            thumbnail={page.thumbnail}
            isSelected={selectedPages.includes(page.pageNumber)}
            onSelect={handleSelectPage}
            onPreview={handlePreview}
          />
        ))}
      </div>

      {/* Rubber band overlay */}
      {rubberBandStyle && (
        <div
          className="border-2 border-blue-500 bg-blue-200/20 rounded-sm"
          style={rubberBandStyle}
        />
      )}

      {/* Lightbox preview */}
      {previewPage && (
        <div
          className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-8 animate-fade-in"
          onClick={() => setPreviewPage(null)}
        >
          <button
            onClick={() => setPreviewPage(null)}
            className="absolute top-4 right-4 p-2 bg-white/20 rounded-full hover:bg-white/30 transition-colors"
          >
            <X className="w-6 h-6 text-white" />
          </button>

          <div className="max-w-4xl max-h-full overflow-auto">
            <img
              src={pages.find(p => p.pageNumber === previewPage)?.thumbnail}
              alt={`Page ${previewPage}`}
              className="max-w-full max-h-[85vh] object-contain rounded-lg shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            />
          </div>

          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-white/20 backdrop-blur-sm px-4 py-2 rounded-full">
            <span className="text-white font-medium">
              Page {typeof previewPage === 'string' ? previewPage.replace('_left', 'L').replace('_right', 'R') : previewPage} of {pages.length}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
