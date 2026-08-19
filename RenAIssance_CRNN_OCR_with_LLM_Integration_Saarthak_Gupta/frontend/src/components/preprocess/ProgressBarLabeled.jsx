import React from 'react';

// Progress bar with a label, percentage and optional detail line.
export default function ProgressBarLabeled({
  label,
  progress = 0, // 0-100
  subtitle = '',
  showPercentage = true,
  variant = 'primary', // 'primary', 'success', 'warning'
  size = 'md', // 'sm', 'md', 'lg'
  isActive = false,
  className = '',
}) {
  const clampedProgress = Math.max(0, Math.min(100, progress));

  const variantClasses = {
    primary: 'bg-blue-500',
    success: 'bg-green-500',
    warning: 'bg-amber-500',
  };

  const sizeClasses = {
    sm: 'h-1.5',
    md: 'h-2',
    lg: 'h-3',
  };

  const trackBg = {
    primary: 'bg-blue-100',
    success: 'bg-green-100',
    warning: 'bg-amber-100',
  };

  return (
    <div className={`${className}`}>
      {/* Label row */}
      {(label || showPercentage) && (
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2 min-w-0">
            {isActive && (
              <span className="flex-shrink-0 w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
            )}
            {label && (
              <span className="text-xs font-medium text-gray-700 truncate">
                {label}
              </span>
            )}
          </div>
          {showPercentage && (
            <span className="text-xs font-semibold text-gray-500 tabular-nums ml-2">
              {Math.round(clampedProgress)}%
            </span>
          )}
        </div>
      )}

      {/* Progress bar */}
      <div className={`w-full ${trackBg[variant]} rounded-full overflow-hidden ${sizeClasses[size]}`}>
        <div
          className={`h-full ${variantClasses[variant]} rounded-full transition-all duration-300 ease-out ${
            isActive ? 'animate-pulse' : ''
          }`}
          style={{ width: `${clampedProgress}%` }}
        />
      </div>

      {/* Subtitle */}
      {subtitle && (
        <p className="text-[10px] text-gray-400 mt-0.5 truncate">
          {subtitle}
        </p>
      )}
    </div>
  );
}
