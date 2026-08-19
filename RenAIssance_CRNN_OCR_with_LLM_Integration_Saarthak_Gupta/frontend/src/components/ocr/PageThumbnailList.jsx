import { useState, useEffect, useRef, useCallback, memo } from 'react';
import { CheckCircle2, Image as ImageIcon, Loader2 } from 'lucide-react';


function createThumbnail(imageSrc, maxSize = 200) {
  return new Promise((resolve, reject) => {
    if (imageSrc.startsWith('blob:') || imageSrc.length < 50000) {
      resolve(imageSrc);
      return;
    }

    const img = new Image();
    img.onload = () => {
      try {
        let width = img.width;
        let height = img.height;
        
        if (width > height) {
          if (width > maxSize) {
            height = (maxSize / width) * height;
            width = maxSize;
          }
        } else {
          if (height > maxSize) {
            width = (maxSize / height) * width;
            height = maxSize;
          }
        }

        const canvas = document.createElement('canvas');
        canvas.width = Math.round(width);
        canvas.height = Math.round(height);
        
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'medium';
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Blob URL, not data URL — easier on memory.
        canvas.toBlob((blob) => {
          if (blob) {
            resolve(URL.createObjectURL(blob));
          } else {
            resolve(canvas.toDataURL('image/jpeg', 0.7));
          }
        }, 'image/jpeg', 0.7);
      } catch (err) {
        reject(err);
      }
    };
    img.onerror = () => reject(new Error('Failed to load image'));
    img.src = imageSrc;
  });
}

// Scrollable page thumbnails, lazily rendered.
function PageThumbnailList({
  images,
  currentIndex,
  processedPages,
  processingPageIndex,
  processingPageIndices = new Set(), // Support batch processing
  onPageSelect,
  pageLabels = null,
}) {
  return (
    <div className="bg-white/95 backdrop-blur-sm rounded-xl border border-gray-200/80 shadow-sm overflow-hidden flex flex-col flex-1 min-h-0">
      {/* Header */}
      <div className="px-3 py-2 bg-gradient-to-r from-blue-50/80 to-white border-b border-gray-100 flex items-center justify-between shrink-0">
        <h3 className="font-semibold text-gray-800 flex items-center gap-2 text-sm">
          <ImageIcon size={16} className="text-blue-600" />
          Pages
        </h3>
        <span className="text-[10px] text-gray-500 bg-gray-100 px-1.5 py-0.5 rounded-full font-medium">
          {images.length}
        </span>
      </div>

      {/* Thumbnails - scrollable */}
      <div className="flex-1 overflow-y-auto p-2 min-h-0">
        <div className="grid grid-cols-2 gap-1.5">
          {images.map((img, index) => (
            <ThumbnailItem
              key={index}
              image={img}
              index={index}
              isActive={index === currentIndex}
              isProcessed={processedPages.has(pageLabels ? pageLabels[index] : index + 1)}
              isProcessing={index === processingPageIndex || processingPageIndices.has(index)}
              onClick={() => onPageSelect(index)}
              pageLabel={pageLabels ? pageLabels[index] : index + 1}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// The parent re-renders on every zoom/hover/progress tick; memo skips the work
// when nothing here actually changed.
export default memo(PageThumbnailList);


function ThumbnailItem({ image, index, isActive, isProcessed, isProcessing, onClick, pageLabel }) {
  const [thumbnailSrc, setThumbnailSrc] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(false);
  const containerRef = useRef(null);
  const hasStartedLoading = useRef(false);

  const imageSrc = image.processed || image.original;

  useEffect(() => {
    if (!imageSrc || hasStartedLoading.current) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !hasStartedLoading.current) {
            hasStartedLoading.current = true;
            setIsLoading(true);

            createThumbnail(imageSrc, 200)
              .then((thumb) => {
                setThumbnailSrc(thumb);
                setIsLoading(false);
              })
              .catch((err) => {
                console.error('Thumbnail creation failed:', err);
                setError(true);
                setIsLoading(false);
              });
          }
        });
      },
      { threshold: 0.1, rootMargin: '50px' }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, [imageSrc]);

  useEffect(() => {
    return () => {
      if (thumbnailSrc && thumbnailSrc.startsWith('blob:')) {
        URL.revokeObjectURL(thumbnailSrc);
      }
    };
  }, [thumbnailSrc]);

  const pageNumber = pageLabel !== undefined ? pageLabel : index + 1;

  return (
    <button
      ref={containerRef}
      onClick={onClick}
      className={`
        relative aspect-[3/4] rounded-lg overflow-hidden transition-all duration-200
        border-2 group
        ${isActive
          ? 'border-blue-500 ring-2 ring-blue-200 shadow-md'
          : isProcessing
            ? 'border-amber-400 ring-2 ring-amber-200'
            : isProcessed
              ? 'border-green-300 hover:border-green-400'
              : 'border-gray-200 hover:border-gray-300 hover:shadow-sm'
        }
      `}
    >
      {/* Loading state */}
      {(isLoading || (!thumbnailSrc && !error)) && (
        <div className="absolute inset-0 bg-gray-100 animate-pulse flex items-center justify-center">
          {isLoading ? (
            <Loader2 size={16} className="text-gray-400 animate-spin" />
          ) : (
            <ImageIcon size={20} className="text-gray-300" />
          )}
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="absolute inset-0 bg-gray-100 flex items-center justify-center">
          <ImageIcon size={20} className="text-gray-400" />
        </div>
      )}

      {/* Image */}
      {thumbnailSrc && !error && (
        <img
          src={thumbnailSrc}
          alt={`Page ${pageNumber}`}
          className="w-full h-full object-cover"
        />
      )}

      {/* Processing indicator - shows spinner on the page being processed */}
      {isProcessing && (
        <div className="absolute inset-0 bg-amber-500/20 flex items-center justify-center">
          <div className="bg-white rounded-full p-1 shadow-md">
            <Loader2 size={16} className="text-amber-500 animate-spin" />
          </div>
        </div>
      )}

      {/* Processed indicator */}
      {isProcessed && !isProcessing && (
        <div className="absolute top-1.5 right-1.5">
          <CheckCircle2
            size={16}
            className="text-green-500 bg-white rounded-full shadow-sm"
          />
        </div>
      )}

      {/* Page number label */}
      <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/70 to-transparent pt-4 pb-1">
        <span className="block text-center text-white text-xs font-medium">
          {pageNumber}
        </span>
      </div>

      {/* Active indicator ring */}
      {isActive && (
        <div className="absolute inset-0 border-2 border-blue-500 rounded-lg pointer-events-none" />
      )}
    </button>
  );
}
