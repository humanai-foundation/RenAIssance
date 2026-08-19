import { useCallback, useRef, useState } from 'react';

const DEFAULT_MEMORY_BUDGET_BYTES = 150 * 1024 * 1024;

let _nextOpId = 1;
const nextOpId = () => _nextOpId++;

function estimateDataUrlBytes(dataUrl) {
  if (!dataUrl || typeof dataUrl !== 'string') return 0;
  if (dataUrl.startsWith('data:')) {
    const commaIdx = dataUrl.indexOf(',');
    return commaIdx >= 0 ? Math.floor((dataUrl.length - commaIdx - 1) * 0.75) : dataUrl.length;
  }
  return dataUrl.length;
}

function estimateOpBytes(op) {
  let bytes = 200;
  const walk = (v) => {
    if (!v) return;
    if (typeof v === 'string') bytes += estimateDataUrlBytes(v);
    else if (v instanceof Blob) bytes += v.size;
    else if (typeof v === 'object') Object.values(v).forEach(walk);
  };
  walk(op.before);
  walk(op.after);
  return bytes;
}

// One undo timeline for every page. Each op records which pages it touched, so
// the same array serves per-page undo (filter) and global undo (walk). Ops past
// the cursor stay around for redo until the next push truncates the branch.
//
// Trimmed by byte budget, not op count — otherwise a run of cheap crops evicts
// while a few huge erase snapshots survive.
export function useHistoryStore({ memoryBudget = DEFAULT_MEMORY_BUDGET_BYTES } = {}) {
  const opsRef = useRef([]);
  const cursorRef = useRef(-1);
  const bytesRef = useRef(0);
  const [version, setVersion] = useState(0);
  const bump = useCallback(() => setVersion((v) => v + 1), []);

  const enforceBudget = useCallback(() => {
    while (bytesRef.current > memoryBudget && opsRef.current.length > 0) {
      const dropped = opsRef.current.shift();
      bytesRef.current -= dropped._bytes || 0;
      cursorRef.current -= 1;
    }
    if (cursorRef.current < -1) cursorRef.current = -1;
  }, [memoryBudget]);

  const push = useCallback((op) => {
    if (!op || !op.kind) return;
    const pageNumbers = op.pageNumbers || (op.pageNumber != null ? [op.pageNumber] : []);
    const normalized = {
      id: nextOpId(),
      label: op.label || op.kind,
      ...op,
      pageNumbers,
    };
    normalized._bytes = estimateOpBytes(normalized);

    if (cursorRef.current < opsRef.current.length - 1) {
      const dropped = opsRef.current.splice(cursorRef.current + 1);
      dropped.forEach((o) => { bytesRef.current -= o._bytes || 0; });
    }

    opsRef.current.push(normalized);
    bytesRef.current += normalized._bytes;
    cursorRef.current = opsRef.current.length - 1;
    enforceBudget();
    bump();
  }, [bump, enforceBudget]);

  const topIndexForPage = useCallback((pageNumber) => {
    for (let i = cursorRef.current; i >= 0; i--) {
      const op = opsRef.current[i];
      if (op.pageNumbers.some((p) => String(p) === String(pageNumber))) return i;
    }
    return -1;
  }, []);

  const nextIndexForPage = useCallback((pageNumber) => {
    for (let i = cursorRef.current + 1; i < opsRef.current.length; i++) {
      const op = opsRef.current[i];
      if (op.pageNumbers.some((p) => String(p) === String(pageNumber))) return i;
    }
    return -1;
  }, []);

  const canUndoGlobal = cursorRef.current >= 0;
  const canRedoGlobal = cursorRef.current < opsRef.current.length - 1;

  const canUndoPage = useCallback((pageNumber) => topIndexForPage(pageNumber) >= 0, [topIndexForPage]);
  const canRedoPage = useCallback((pageNumber) => nextIndexForPage(pageNumber) >= 0, [nextIndexForPage]);

  const undoGlobal = useCallback(() => {
    if (cursorRef.current < 0) return null;
    const op = opsRef.current[cursorRef.current];
    cursorRef.current -= 1;
    bump();
    return op;
  }, [bump]);

  const redoGlobal = useCallback(() => {
    if (cursorRef.current >= opsRef.current.length - 1) return null;
    cursorRef.current += 1;
    const op = opsRef.current[cursorRef.current];
    bump();
    return op;
  }, [bump]);

  const undoPage = useCallback((pageNumber) => {
    const idx = topIndexForPage(pageNumber);
    if (idx < 0) return null;
    const op = opsRef.current[idx];
    opsRef.current.splice(idx, 1);
    bytesRef.current -= op._bytes || 0;
    if (idx <= cursorRef.current) cursorRef.current -= 1;
    bump();
    return op;
  }, [bump, topIndexForPage]);

  const clear = useCallback(() => {
    opsRef.current = [];
    cursorRef.current = -1;
    bytesRef.current = 0;
    bump();
  }, [bump]);

  const getTimeline = useCallback(() => ({
    ops: opsRef.current.slice(),
    cursor: cursorRef.current,
    bytes: bytesRef.current,
  }), []);

  return {
    version,
    push,
    undoGlobal,
    redoGlobal,
    undoPage,
    canUndoPage,
    canRedoPage,
    canUndoGlobal,
    canRedoGlobal,
    clear,
    getTimeline,
  };
}

export const HistoryOps = {
  crop: ({ pageNumber, before, after, cropData }) => ({
    kind: 'crop', pageNumber, label: 'Crop', before: { image: before }, after: { image: after, cropData },
  }),
  cropBatch: ({ pageNumbers, before, after }) => ({
    kind: 'crop-batch', pageNumbers, label: `Crop ${pageNumbers.length} pages`, before: { images: before }, after: { images: after },
  }),
  erase: ({ pageNumber, before, after, region }) => ({
    kind: 'erase', pageNumber, label: 'Erase', before: { image: before, region }, after: { image: after },
  }),
  eraseBatch: ({ pageNumbers, before, after }) => ({
    kind: 'erase-batch', pageNumbers, label: `Erase ${pageNumbers.length} pages`, before: { images: before }, after: { images: after },
  }),
  pipeline: ({ pageNumber, beforeProcessed, afterProcessed, pipelineConfig }) => ({
    kind: 'pipeline', pageNumber, label: 'Apply pipeline', before: { processed: beforeProcessed }, after: { processed: afterProcessed, pipelineConfig },
  }),
  pipelineBatch: ({ pageNumbers, beforeProcessed, afterProcessed, pipelineConfig }) => ({
    kind: 'pipeline-batch', pageNumbers, label: `Apply pipeline to ${pageNumbers.length} pages`,
    before: { processed: beforeProcessed }, after: { processed: afterProcessed, pipelineConfig },
  }),
  resetPage: ({ pageNumber, beforeImage, beforeProcessed }) => ({
    kind: 'reset-page', pageNumber, label: 'Reset page',
    before: { image: beforeImage, processed: beforeProcessed }, after: null,
  }),
};
