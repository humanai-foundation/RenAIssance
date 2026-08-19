import React, { useState, useRef } from 'react';
import { 
  GripVertical, 
  Trash2, 
  Settings2, 
  ChevronDown,
  Power,
  Plus
} from 'lucide-react';
import TooltipInfo from './TooltipInfo';
import OperationSlider from './OperationSlider';
import { getOperationById, shouldShowControl } from '../../config/preprocessOperations';

// The ordered operation list, reorderable by drag. Each step can be toggled,
// expanded for its params, or deleted.
export default function PipelineStack({
  pipeline = [],
  onReorder,
  onToggle,
  onUpdateParams,
  onRemove,
  onAddOperation,
  isProcessing = false,
  currentStepId = null,
  className = '',
}) {
  const [draggedIndex, setDraggedIndex] = useState(null);
  const [dragOverIndex, setDragOverIndex] = useState(null);
  const [expandedStep, setExpandedStep] = useState(null);
  const dragCounter = useRef(0);

  const handleDragStart = (e, index) => {
    if (isProcessing) return;
    setDraggedIndex(index);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragEnd = () => {
    if (draggedIndex !== null && dragOverIndex !== null && draggedIndex !== dragOverIndex) {
      onReorder(draggedIndex, dragOverIndex);
    }

    setDraggedIndex(null);
    setDragOverIndex(null);
    dragCounter.current = 0;
  };

  const handleDragEnter = (e, index) => {
    e.preventDefault();
    dragCounter.current++;
    setDragOverIndex(index);
  };

  const handleDragLeave = (e) => {
    dragCounter.current--;
    if (dragCounter.current === 0) {
      setDragOverIndex(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDrop = (e, index) => {
    e.preventDefault();
    dragCounter.current = 0;
  };

  if (pipeline.length === 0) {
    return (
      <div className={`${className}`}>
        <div className="text-center py-6 border-2 border-dashed border-gray-200 rounded-xl">
          <div className="text-gray-400 mb-2">
            <Plus size={24} className="mx-auto" />
          </div>
          <p className="text-sm text-gray-500">
            No operations in pipeline
          </p>
          <p className="text-xs text-gray-400 mt-1">
            Enable operations from the sidebar to add them
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`space-y-1 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-1 mb-2">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
          Pipeline ({pipeline.filter(s => s.enabled).length}/{pipeline.length} active)
        </span>
      </div>

      {/* Pipeline steps */}
      <div className="space-y-1">
        {pipeline.map((step, index) => {
          const operation = getOperationById(step.operationId);
          if (!operation) return null;

          const isActive = step.id === currentStepId;
          const isDraggedOver = dragOverIndex === index && draggedIndex !== index;
          const isExpanded = expandedStep === step.id;
          const hasControls = operation.controls && operation.controls.length > 0;
          const visibleControls = hasControls
            ? operation.controls.filter(c => shouldShowControl(c, step.params))
            : [];

          return (
            <div
              key={step.id}
              onDragEnter={(e) => handleDragEnter(e, index)}
              onDragLeave={handleDragLeave}
              onDragOver={handleDragOver}
              onDrop={(e) => handleDrop(e, index)}
              className={`
                group relative rounded-lg border transition-all duration-200
                ${step.enabled
                  ? 'bg-white border-blue-200 shadow-sm'
                  : 'bg-gray-50 border-gray-200 opacity-60'
                }
                ${isDraggedOver ? 'border-blue-400 border-2 bg-blue-50' : ''}
                ${isActive ? 'ring-2 ring-blue-400 ring-offset-1' : ''}
                ${draggedIndex === index ? 'opacity-50' : ''}
              `}
            >
              {/* Main row */}
              <div className="flex items-center gap-2 p-2">
                {/* Drag handle — ONLY this is draggable, so interacting with the
                    sliders/selects inside the card never starts a card drag. */}
                <div
                  draggable={!isProcessing}
                  onDragStart={(e) => handleDragStart(e, index)}
                  onDragEnd={handleDragEnd}
                  title="Drag to reorder"
                  className={`flex-shrink-0 text-gray-300 group-hover:text-gray-400 transition-colors ${
                    !isProcessing ? 'cursor-grab active:cursor-grabbing' : ''
                  }`}
                >
                  <GripVertical size={16} />
                </div>

                {/* Order number */}
                <span className={`
                  flex-shrink-0 w-5 h-5 rounded text-xs font-bold flex items-center justify-center
                  ${step.enabled 
                    ? 'bg-blue-100 text-blue-600' 
                    : 'bg-gray-200 text-gray-400'
                  }
                `}>
                  {index + 1}
                </span>

                {/* Operation name */}
                <span className={`flex-1 text-sm font-medium truncate ${
                  step.enabled ? 'text-gray-800' : 'text-gray-400'
                }`}>
                  {operation.name}
                </span>

                {/* Processing indicator */}
                {isActive && isProcessing && (
                  <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                )}

                {/* Actions */}
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  {/* Settings button */}
                  {hasControls && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setExpandedStep(isExpanded ? null : step.id);
                      }}
                      disabled={!step.enabled}
                      className={`p-1 rounded transition-colors ${
                        step.enabled 
                          ? 'hover:bg-blue-100 text-blue-500' 
                          : 'text-gray-300 cursor-not-allowed'
                      }`}
                      title="Settings"
                    >
                      <Settings2 size={14} />
                    </button>
                  )}

                  {/* Toggle enable */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggle(step.id, !step.enabled);
                    }}
                    disabled={isProcessing}
                    className={`p-1 rounded transition-colors ${
                      step.enabled 
                        ? 'hover:bg-green-100 text-green-500' 
                        : 'hover:bg-gray-200 text-gray-400'
                    }`}
                    title={step.enabled ? 'Disable' : 'Enable'}
                  >
                    <Power size={14} />
                  </button>

                  {/* Delete */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemove(step.id);
                    }}
                    disabled={isProcessing}
                    className="p-1 rounded hover:bg-red-100 text-red-400 hover:text-red-500 transition-colors"
                    title="Remove"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>

                {/* Tooltip - always visible on right */}
                <TooltipInfo 
                  text={operation.tooltip}
                  position="left"
                  iconSize={12}
                />
              </div>

              {/* Expanded settings */}
              {isExpanded && step.enabled && hasControls && (
                <div className="px-3 pb-3 pt-1 border-t border-gray-100 space-y-3 animate-slide-up">
                  {visibleControls.map((control) => (
                    <ControlRenderer
                      key={control.id}
                      control={control}
                      value={step.params[control.id] ?? control.default}
                      onChange={(value) => onUpdateParams(step.id, {
                        ...step.params,
                        [control.id]: value,
                      })}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Dispatches on control.type.
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
