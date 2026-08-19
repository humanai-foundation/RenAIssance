import React, { useState, useRef, useEffect } from 'react';
import { Info } from 'lucide-react';

// Info icon with a hover tooltip.
export default function TooltipInfo({ 
  text, 
  position = 'top', 
  maxWidth = 250,
  iconSize = 14,
  className = ''
}) {
  const [isVisible, setIsVisible] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
  const triggerRef = useRef(null);
  const tooltipRef = useRef(null);

  useEffect(() => {
    if (isVisible && triggerRef.current && tooltipRef.current) {
      const trigger = triggerRef.current.getBoundingClientRect();
      const tooltip = tooltipRef.current.getBoundingClientRect();
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      
      let top, left;
      
      switch (position) {
        case 'top':
          top = -tooltip.height - 8;
          left = (trigger.width - tooltip.width) / 2;
          break;
        case 'bottom':
          top = trigger.height + 8;
          left = (trigger.width - tooltip.width) / 2;
          break;
        case 'left':
          top = (trigger.height - tooltip.height) / 2;
          left = -tooltip.width - 8;
          break;
        case 'right':
          top = (trigger.height - tooltip.height) / 2;
          left = trigger.width + 8;
          break;
        default:
          top = -tooltip.height - 8;
          left = (trigger.width - tooltip.width) / 2;
      }
      
      const absoluteLeft = trigger.left + left;
      const absoluteTop = trigger.top + top;
      
      if (absoluteLeft < 8) {
        left = -trigger.left + 8;
      } else if (absoluteLeft + tooltip.width > viewportWidth - 8) {
        left = viewportWidth - trigger.left - tooltip.width - 8;
      }
      
      if (absoluteTop < 8) {
        top = trigger.height + 8;
      } else if (absoluteTop + tooltip.height > viewportHeight - 8) {
        top = -tooltip.height - 8;
      }
      
      setTooltipPosition({ top, left });
    }
  }, [isVisible, position]);

  if (!text) return null;

  return (
    <div 
      className={`relative inline-flex ${className}`}
      ref={triggerRef}
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      <Info 
        className="text-gray-400 hover:text-blue-500 cursor-help transition-colors"
        size={iconSize}
      />
      
      {isVisible && (
        <div
          ref={tooltipRef}
          className="absolute z-50 animate-fade-in"
          style={{
            top: tooltipPosition.top,
            left: tooltipPosition.left,
            maxWidth: maxWidth,
          }}
        >
          <div className="bg-gray-900 text-white text-xs leading-relaxed px-3 py-2 rounded-lg shadow-lg">
            {text}
            {/* Arrow */}
            <div 
              className={`absolute w-2 h-2 bg-gray-900 transform rotate-45 ${
                position === 'top' ? 'bottom-[-4px] left-1/2 -translate-x-1/2' :
                position === 'bottom' ? 'top-[-4px] left-1/2 -translate-x-1/2' :
                position === 'left' ? 'right-[-4px] top-1/2 -translate-y-1/2' :
                'left-[-4px] top-1/2 -translate-y-1/2'
              }`}
            />
          </div>
        </div>
      )}
    </div>
  );
}
