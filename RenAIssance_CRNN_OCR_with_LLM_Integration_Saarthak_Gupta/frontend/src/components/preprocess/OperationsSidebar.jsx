import React, { useState } from 'react';
import {
  ChevronDown,
  Sliders,
  Sparkles,
  Contrast,
  Eraser,
  Layers,
  RotateCcw
} from 'lucide-react';
import OperationCard from './OperationCard';
import {
  OPERATION_CATEGORIES,
  OPERATIONS,
  getOperationsByCategory
} from '../../config/preprocessOperations';

// Accordion of operations grouped by category, plus reset / recommended-preset
// shortcuts.

const CATEGORY_ICONS = {
  basic: Sliders,
  enhancement: Sparkles,
  binarization: Contrast,
  cleanup: Eraser,
};

export default function OperationsSidebar({
  enabledOperations = {}, // { operationId: { enabled: boolean, params: {} } }
  onOperationToggle,
  onOperationParamsChange,
  onApplyRecommended,
  onReset,
  isProcessing = false,
  hideHeader = false,
  className = '',
}) {
  const [expandedCategories, setExpandedCategories] = useState(['basic', 'enhancement', 'binarization', 'cleanup']);
  const operationsByCategory = getOperationsByCategory();

  const toggleCategory = (categoryId) => {
    setExpandedCategories(prev =>
      prev.includes(categoryId)
        ? prev.filter(id => id !== categoryId)
        : [...prev, categoryId]
    );
  };

  const getEnabledCountForCategory = (categoryId) => {
    const operations = operationsByCategory[categoryId] || [];
    return operations.filter(op => enabledOperations[op.id]?.enabled).length;
  };

  const getTotalEnabledCount = () => {
    return Object.values(enabledOperations).filter(op => op?.enabled).length;
  };

  return (
    <div className={`flex flex-col h-full ${hideHeader ? '' : 'bg-white rounded-xl shadow-card'} ${className}`}>
      {/* Header */}
      {!hideHeader && (
      <div className="flex-shrink-0 px-4 py-3 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-gray-800">Operations</h3>
            {getTotalEnabledCount() > 0 && (
              <span className="px-2 py-0.5 bg-blue-100 text-blue-600 text-xs font-semibold rounded-full">
                {getTotalEnabledCount()}
              </span>
            )}
          </div>
        </div>

        {/* Quick actions */}
        <div className="flex gap-2 mt-3">
          <button
            onClick={onApplyRecommended}
            disabled={isProcessing}
            className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2
                       bg-gradient-to-r from-blue-500 to-blue-600 text-white text-xs font-medium
                       rounded-lg hover:from-blue-600 hover:to-blue-700
                       disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm"
          >
            <Sparkles size={14} />
            Recommended
          </button>
          <button
            onClick={onReset}
            disabled={isProcessing}
            className="px-3 py-2 bg-gray-100 text-gray-600 text-xs font-medium rounded-lg
                       hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Reset all operations"
          >
            <RotateCcw size={14} />
          </button>
        </div>
      </div>
      )}

      {/* Accordion categories */}
      <div className="flex-1 overflow-y-auto">
        {OPERATION_CATEGORIES.map((category) => {
          const Icon = CATEGORY_ICONS[category.id] || Layers;
          const isExpanded = expandedCategories.includes(category.id);
          const enabledCount = getEnabledCountForCategory(category.id);
          const operations = operationsByCategory[category.id] || [];

          return (
            <div key={category.id} className="border-b border-gray-100 last:border-b-0">
              {/* Category header */}
              <button
                onClick={() => toggleCategory(category.id)}
                className="w-full flex items-center gap-2 px-4 py-2.5 hover:bg-gray-50 transition-colors"
              >
                <Icon size={16} className="text-blue-500 flex-shrink-0" />
                <span className="flex-1 text-sm font-medium text-gray-700 text-left">
                  {category.label}
                </span>
                {enabledCount > 0 && (
                  <span className="px-1.5 py-0.5 bg-blue-100 text-blue-600 text-[10px] font-bold rounded">
                    {enabledCount}
                  </span>
                )}
                <ChevronDown
                  size={16}
                  className={`text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                />
              </button>

              {/* Category operations */}
              {isExpanded && (
                <div className="px-3 pb-3 space-y-1.5 animate-slide-up">
                  {operations.map((operation) => (
                    <OperationCard
                      key={operation.id}
                      operation={operation}
                      enabled={enabledOperations[operation.id]?.enabled || false}
                      params={enabledOperations[operation.id]?.params || operation.defaultParams}
                      onToggle={onOperationToggle}
                      onParamsChange={onOperationParamsChange}
                      compact
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer hint */}
      <div className="flex-shrink-0 px-4 py-2 bg-gray-50 border-t border-gray-100">
        <p className="text-[10px] text-gray-400 text-center">
          Enable operations to add them to the pipeline
        </p>
      </div>
    </div>
  );
}
