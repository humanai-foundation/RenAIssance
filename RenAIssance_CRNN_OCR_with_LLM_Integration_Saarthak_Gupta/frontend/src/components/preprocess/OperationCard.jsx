import React, { useState } from 'react';
import { ChevronDown, Settings2 } from 'lucide-react';
import TooltipInfo from './TooltipInfo';
import OperationSlider from './OperationSlider';
import { shouldShowControl } from '../../config/preprocessOperations';

// One operation in the sidebar: toggle, tooltip, expandable params.
export default function OperationCard({
  operation,
  enabled = false,
  params = {},
  onToggle,
  onParamsChange,
  isInPipeline = false,
  compact = false,
  className = '',
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const hasControls = operation.controls && operation.controls.length > 0;
  const visibleControls = hasControls 
    ? operation.controls.filter(c => shouldShowControl(c, params))
    : [];

  const handleToggle = (e) => {
    e.stopPropagation();
    onToggle(operation.id, !enabled);
  };

  const handleParamChange = (controlId, value) => {
    onParamsChange(operation.id, {
      ...params,
      [controlId]: value,
    });
  };

  const handleExpandClick = (e) => {
    e.stopPropagation();
    if (hasControls && enabled) {
      setIsExpanded(!isExpanded);
    }
  };

  return (
    <div
      className={`
        rounded-lg border transition-all duration-200
        ${enabled 
          ? 'border-blue-300 bg-blue-50/50' 
          : 'border-gray-200 bg-white hover:border-blue-200'
        }
        ${compact ? 'p-2' : 'p-3'}
        ${className}
      `}
    >
      {/* Header row */}
      <div className="flex items-center gap-2">
        {/* Checkbox - styled via index.css */}
        <input
          type="checkbox"
          checked={enabled}
          onChange={handleToggle}
        />

        {/* Operation name */}
        <span 
          className={`flex-1 text-sm font-medium cursor-pointer select-none ${
            enabled ? 'text-gray-800' : 'text-gray-600'
          }`}
          onClick={handleToggle}
        >
          {operation.name}
        </span>

        {/* Tooltip */}
        <TooltipInfo 
          text={operation.tooltip}
          position="left"
          iconSize={14}
        />

        {/* Expand button (only if has controls and enabled) */}
        {hasControls && (
          <button
            onClick={handleExpandClick}
            disabled={!enabled}
            className={`p-1 rounded transition-colors ${
              enabled 
                ? 'hover:bg-blue-100 text-blue-600' 
                : 'text-gray-300 cursor-not-allowed'
            }`}
          >
            <ChevronDown 
              size={16} 
              className={`transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            />
          </button>
        )}
      </div>

      {/* Expandable controls */}
      {enabled && hasControls && isExpanded && (
        <div className="mt-3 pt-3 border-t border-blue-200/50 space-y-3 animate-slide-up">
          {visibleControls.map((control) => (
            <ControlRenderer
              key={control.id}
              control={control}
              value={params[control.id] ?? control.default}
              onChange={(value) => handleParamChange(control.id, value)}
            />
          ))}
        </div>
      )}

      {/* Collapsed indicator showing active params */}
      {enabled && hasControls && !isExpanded && visibleControls.length > 0 && (
        <div className="mt-1.5 flex items-center gap-1 text-[10px] text-blue-600">
          <Settings2 size={10} />
          <span>
            {visibleControls.slice(0, 2).map(c => {
              const val = params[c.id] ?? c.default;
              return `${c.label}: ${val}${c.unit || ''}`;
            }).join(' • ')}
            {visibleControls.length > 2 && ` +${visibleControls.length - 2}`}
          </span>
        </div>
      )}
    </div>
  );
}

// Dispatches on control.type (slider, select, checkbox, ...).
function ControlRenderer({ control, value, onChange }) {
  switch (control.type) {
    case 'slider':
      return (
        <OperationSlider
          id={control.id}
          label={control.label}
          value={value}
          onChange={onChange}
          min={control.min}
          max={control.max}
          step={control.step}
          unit={control.unit}
          labels={control.labels}
        />
      );

    case 'select':
      return (
        <div className="space-y-1">
          <label className="text-xs font-medium text-gray-600">
            {control.label}
          </label>
          <div className="relative">
            <select
              value={value}
              onChange={(e) => onChange(e.target.value)}
              className="w-full text-sm bg-white border border-gray-200 rounded-md 
                         px-3 py-1.5 pr-8 appearance-none focus:outline-none 
                         focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {control.options.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <ChevronDown 
              size={14} 
              className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none"
            />
          </div>
        </div>
      );

    case 'toggle':
      return (
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-gray-600">
            {control.label}
          </label>
          <button
            onClick={() => onChange(!value)}
            className={`relative w-10 h-5 rounded-full transition-colors ${
              value ? 'bg-blue-500' : 'bg-gray-300'
            }`}
          >
            <span
              className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${
                value ? 'translate-x-5' : 'translate-x-0.5'
              }`}
            />
          </button>
        </div>
      );

    default:
      return null;
  }
}
