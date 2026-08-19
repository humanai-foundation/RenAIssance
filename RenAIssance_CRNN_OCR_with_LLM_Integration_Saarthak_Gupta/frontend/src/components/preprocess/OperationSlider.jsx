import React from 'react';

// Compact slider for a single operation parameter.
export default function OperationSlider({
  id,
  label,
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  unit = '',
  labels = null, // { min: 'Label', max: 'Label' }
  disabled = false,
  className = '',
}) {
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className={`space-y-1 ${className}`}>
      {/* Label and value row */}
      <div className="flex items-center justify-between text-xs">
        <label 
          htmlFor={id}
          className="font-medium text-gray-600"
        >
          {label}
        </label>
        <span className="font-semibold text-blue-600 tabular-nums">
          {typeof value === 'number' ? (
            step < 1 ? value.toFixed(1) : value
          ) : value}
          {unit && <span className="text-gray-400 ml-0.5">{unit}</span>}
        </span>
      </div>

      {/* Slider */}
      <div className="relative">
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          disabled={disabled}
          className="slider-compact w-full"
          style={{
            '--slider-progress': `${percentage}%`
          }}
        />
      </div>

      {/* Min/Max labels if provided */}
      {labels && (
        <div className="flex items-center justify-between text-[10px] text-gray-400 -mt-0.5">
          <span>{labels.min}</span>
          <span>{labels.max}</span>
        </div>
      )}
    </div>
  );
}
