import React, { useState } from 'react';
import { Check, ZoomIn } from 'lucide-react';

const PageCard = React.forwardRef(function PageCard({
  pageNumber,
  thumbnail,
  isSelected,
  onSelect,
  onPreview,
  isShiftHeld,
}, ref) {
  const [isHovered, setIsHovered] = useState(false);

  const handleClick = (e) => {
    if (e.shiftKey || isShiftHeld) {
      onSelect(pageNumber, true); // Range select
    } else {
      onSelect(pageNumber, false);
    }
  };

  return (
    <div
      ref={ref}
      data-page-card
      className={`relative group cursor-pointer rounded-xl overflow-hidden transition-all duration-200 ${isSelected
          ? 'ring-4 ring-blue-500 shadow-card-hover scale-[1.02]'
          : 'ring-1 ring-blue-100 hover:ring-blue-300 hover:shadow-card'
        }`}
      onClick={handleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Thumbnail container */}
      <div className="relative aspect-[3/4] bg-gray-100 overflow-hidden">
        {thumbnail ? (
          <img
            src={thumbnail}
            alt={`Page ${pageNumber}`}
            className={`w-full h-full object-contain transition-transform duration-300 ${isHovered ? 'scale-110' : 'scale-100'
              }`}
          />
        ) : (
          <div className="w-full h-full shimmer" />
        )}

        {/* Overlay on hover */}
        <div
          className={`absolute inset-0 bg-blue-900/40 flex items-center justify-center gap-2 transition-opacity duration-200 ${isHovered || isSelected ? 'opacity-100' : 'opacity-0'
            }`}
        >
          {/* Preview zoom button */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onPreview(pageNumber);
            }}
            className="p-2 bg-white rounded-full shadow-lg hover:bg-blue-50 transition-colors"
          >
            <ZoomIn className="w-5 h-5 text-blue-600" />
          </button>
        </div>

        {/* Selection checkbox */}
        <div
          className={`absolute top-3 left-3 transition-all duration-200 ${isHovered || isSelected
              ? 'opacity-100 translate-y-0'
              : 'opacity-0 -translate-y-2'
            }`}
        >
          <div
            className={`w-6 h-6 rounded-md border-2 flex items-center justify-center transition-all ${isSelected
                ? 'bg-blue-600 border-blue-600'
                : 'bg-white/90 border-blue-300 hover:border-blue-500'
              }`}
          >
            {isSelected && <Check className="w-4 h-4 text-white" />}
          </div>
        </div>
      </div>

      {/* Page number badge */}
      <div
        className={`absolute bottom-0 inset-x-0 py-2 text-center text-sm font-medium transition-colors ${isSelected ? 'bg-blue-600 text-white' : 'bg-white text-gray-700'
          }`}
      >
        {typeof pageNumber === 'string'
          ? pageNumber.replace('_left', 'L').replace('_right', 'R')
          : `Page ${pageNumber}`}
      </div>
    </div>
  );
});

// The grid can be 100+ cards; memo stops one selection toggle from
// re-rendering every thumbnail.
export default React.memo(PageCard);
