import { useState, useCallback, useRef, useEffect } from 'react';
import { 
  getOperationById, 
  getRecommendedPipeline,
  createPipelineStep 
} from '../config/preprocessOperations';

// Preprocessing pipeline state: the ordered step list plus the sidebar's
// enabled-operations map, kept in sync both ways. Preview calls are debounced.
export function usePipeline(options = {}) {
  const {
    onPreviewRequest,
    debounceMs = 500,
    initialPipeline = [],
  } = options;

  const [pipeline, setPipeline] = useState(initialPipeline);
  
  const [enabledOperations, setEnabledOperations] = useState({});
  
  const debounceTimer = useRef(null);

  // Rebuild the pipeline whenever the sidebar toggles change.
  const syncPipelineFromEnabled = useCallback((enabled) => {
    setPipeline(prev => {
      const enabledIds = Object.entries(enabled)
        .filter(([_, val]) => val?.enabled)
        .map(([id]) => id);

      // Existing steps keep their position and params.
      const existingSteps = prev.filter(step => enabledIds.includes(step.operationId));
      
      const existingOperationIds = existingSteps.map(s => s.operationId);
      const newOperationIds = enabledIds.filter(id => !existingOperationIds.includes(id));
      
      const newSteps = newOperationIds.map((opId, index) => ({
        id: `${opId}-${Date.now()}-${index}`,
        operationId: opId,
        enabled: true,
        params: enabled[opId]?.params || getOperationById(opId)?.defaultParams || {},
        order: existingSteps.length + index,
      }));

      const combined = [...existingSteps, ...newSteps].map((step, index) => ({
        ...step,
        order: index,
        params: enabled[step.operationId]?.params || step.params,
      }));

      return combined;
    });
  }, []);

  const toggleOperation = useCallback((operationId, enabled) => {
    setEnabledOperations(prev => {
      const operation = getOperationById(operationId);
      const newState = {
        ...prev,
        [operationId]: {
          enabled,
          params: prev[operationId]?.params || operation?.defaultParams || {},
        },
      };
      
      syncPipelineFromEnabled(newState);
      
      return newState;
    });

    triggerDebouncedPreview();
  }, [syncPipelineFromEnabled]);

  const updateOperationParams = useCallback((operationId, params) => {
    setEnabledOperations(prev => ({
      ...prev,
      [operationId]: {
        ...prev[operationId],
        params,
      },
    }));

    setPipeline(prev => prev.map(step => 
      step.operationId === operationId
        ? { ...step, params }
        : step
    ));

    triggerDebouncedPreview();
  }, []);

  const togglePipelineStep = useCallback((stepId, enabled) => {
    setPipeline(prev => prev.map(step =>
      step.id === stepId
        ? { ...step, enabled }
        : step
    ));

    triggerDebouncedPreview();
  }, []);

  const updatePipelineStepParams = useCallback((stepId, params) => {
    setPipeline(prev => {
      const updated = prev.map(step =>
        step.id === stepId
          ? { ...step, params }
          : step
      );
      
      const step = updated.find(s => s.id === stepId);
      if (step) {
        setEnabledOperations(prevEnabled => ({
          ...prevEnabled,
          [step.operationId]: {
            ...prevEnabled[step.operationId],
            params,
          },
        }));
      }
      
      return updated;
    });

    triggerDebouncedPreview();
  }, []);

  const removePipelineStep = useCallback((stepId) => {
    setPipeline(prev => {
      const step = prev.find(s => s.id === stepId);
      
      if (step) {
        setEnabledOperations(prevEnabled => ({
          ...prevEnabled,
          [step.operationId]: {
            ...prevEnabled[step.operationId],
            enabled: false,
          },
        }));
      }
      
      return prev.filter(s => s.id !== stepId).map((s, i) => ({ ...s, order: i }));
    });

    triggerDebouncedPreview();
  }, []);

  const reorderPipeline = useCallback((fromIndex, toIndex) => {
    setPipeline(prev => {
      const result = [...prev];
      const [removed] = result.splice(fromIndex, 1);
      result.splice(toIndex, 0, removed);
      return result.map((step, index) => ({ ...step, order: index }));
    });

    triggerDebouncedPreview();
  }, []);

  const addOperation = useCallback((operationId) => {
    const operation = getOperationById(operationId);
    if (!operation) return;

    const newStep = createPipelineStep(operationId, pipeline.length);
    if (!newStep) return;

    setPipeline(prev => [...prev, newStep]);
    setEnabledOperations(prev => ({
      ...prev,
      [operationId]: {
        enabled: true,
        params: operation.defaultParams,
      },
    }));

    triggerDebouncedPreview();
  }, [pipeline.length]);

  const applyRecommendedPipeline = useCallback(() => {
    const recommended = getRecommendedPipeline();
    setPipeline(recommended);

    const newEnabled = {};
    recommended.forEach(step => {
      newEnabled[step.operationId] = {
        enabled: true,
        params: step.params,
      };
    });
    setEnabledOperations(newEnabled);

    triggerDebouncedPreview();
  }, []);

  // Select a predefined set of steps WITHOUT running. Used by the book-type
  // presets: the user gets the pipeline pre-filled, then reviews and processes
  // themselves. `steps` come from createPipelineStep (id/operationId/params).
  const applyPipeline = useCallback((steps) => {
    if (!steps || steps.length === 0) return;
    setPipeline(steps);

    const newEnabled = {};
    steps.forEach(step => {
      newEnabled[step.operationId] = {
        enabled: true,
        params: step.params,
      };
    });
    setEnabledOperations(newEnabled);
    // Deliberately no preview trigger — selection only, per the preset contract.
  }, []);

  const resetPipeline = useCallback(() => {
    setPipeline([]);
    setEnabledOperations({});
    
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }
  }, []);

  const triggerDebouncedPreview = useCallback(() => {
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }

    if (onPreviewRequest) {
      debounceTimer.current = setTimeout(() => {
        onPreviewRequest();
      }, debounceMs);
    }
  }, [onPreviewRequest, debounceMs]);

  const getActivePipeline = useCallback(() => {
    return pipeline.filter(step => step.enabled);
  }, [pipeline]);

  const buildPipelineConfig = useCallback(() => {
    return getActivePipeline().map(step => ({
      op: step.operationId,
      params: step.params,
    }));
  }, [getActivePipeline]);

  useEffect(() => {
    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, []);

  return {
    // State
    pipeline,
    enabledOperations,
    
    // Operations sidebar actions
    toggleOperation,
    updateOperationParams,
    
    // Pipeline stack actions
    togglePipelineStep,
    updatePipelineStepParams,
    removePipelineStep,
    reorderPipeline,
    addOperation,
    
    // Quick actions
    applyRecommendedPipeline,
    applyPipeline,
    resetPipeline,
    
    // Utilities
    getActivePipeline,
    buildPipelineConfig,
  };
}

export default usePipeline;
