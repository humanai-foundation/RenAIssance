// Renders PDF pages on an OffscreenCanvas, off the main thread, so a hidden
// tab doesn't throttle extraction to a crawl.
//
// In:  { type: 'extract', id, fileBuffer, options }
// Out: 'page' (streamed) | 'progress' (0-100) | 'done' | 'error'

import * as pdfjsLib from 'pdfjs-dist';
// Vite resolves this URL at build time and bundles the worker.
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

// PDF.js's default canvas factory reaches for `document`, which doesn't exist
// in a worker. Back it with OffscreenCanvas instead.
class OffscreenCanvasFactory {
  create(width, height) {
    if (width <= 0 || height <= 0) {
      throw new Error('Invalid canvas size');
    }
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext('2d');
    return { canvas, context };
  }
  reset(canvasAndContext, width, height) {
    if (!canvasAndContext.canvas) throw new Error('Canvas is not specified');
    if (width <= 0 || height <= 0) throw new Error('Invalid canvas size');
    canvasAndContext.canvas.width = width;
    canvasAndContext.canvas.height = height;
  }
  destroy(canvasAndContext) {
    if (!canvasAndContext.canvas) return;
    canvasAndContext.canvas.width = 0;
    canvasAndContext.canvas.height = 0;
    canvasAndContext.canvas = null;
    canvasAndContext.context = null;
  }
}

// Blobs, not data URLs: structured-clone is cheap and the pixels stay off the
// JS heap. The hook wraps them in object URLs.

async function renderPage(pdf, pageNum, scale, splitDoublePages) {
  const page = await pdf.getPage(pageNum);
  const viewport = page.getViewport({ scale });

  const canvas = new OffscreenCanvas(viewport.width, viewport.height);
  const ctx = canvas.getContext('2d');
  await page.render({ canvasContext: ctx, viewport }).promise;

  const isDouble = splitDoublePages && viewport.width > viewport.height * 1.19;

  if (!isDouble) {
    const blob = await canvas.convertToBlob({ type: 'image/png' });
    return [{
      pageNumber: pageNum,
      originalPageNumber: pageNum,
      blob,
      width: viewport.width,
      height: viewport.height,
      isSplit: false,
    }];
  }

  // Split double-page spread into left / right halves.
  const halfWidth = Math.floor(viewport.width / 2);
  const out = [];
  for (const side of ['left', 'right']) {
    const sideCanvas = new OffscreenCanvas(halfWidth, viewport.height);
    const sideCtx = sideCanvas.getContext('2d');
    const srcX = side === 'left' ? 0 : halfWidth;
    sideCtx.drawImage(canvas, srcX, 0, halfWidth, viewport.height, 0, 0, halfWidth, viewport.height);
    const blob = await sideCanvas.convertToBlob({ type: 'image/png' });
    out.push({
      pageNumber: `${pageNum}_${side}`,
      originalPageNumber: pageNum,
      blob,
      width: halfWidth,
      height: viewport.height,
      isSplit: true,
      splitSide: side,
    });
  }
  return out;
}

self.onmessage = async (e) => {
  const msg = e.data || {};
  if (msg.type !== 'extract') return;
  const { id, fileBuffer, options = {} } = msg;
  const { scale = 1.5, pageRange = null, splitDoublePages = true } = options;

  try {
    const pdf = await pdfjsLib.getDocument({
      data: fileBuffer,
      CanvasFactory: OffscreenCanvasFactory,
    }).promise;
    const totalPages = pdf.numPages;

    let pagesToExtract = [];
    if (pageRange) {
      const [start, end] = pageRange;
      for (let i = Math.max(1, start); i <= Math.min(totalPages, end); i++) pagesToExtract.push(i);
    } else {
      for (let i = 1; i <= totalPages; i++) pagesToExtract.push(i);
    }

    for (let i = 0; i < pagesToExtract.length; i++) {
      const pageNum = pagesToExtract[i];
      const rendered = await renderPage(pdf, pageNum, scale, splitDoublePages);
      for (const page of rendered) {
        self.postMessage({ type: 'page', id, page });
      }
      const progress = Math.round(((i + 1) / pagesToExtract.length) * 100);
      self.postMessage({ type: 'progress', id, progress });
    }

    self.postMessage({ type: 'done', id });
  } catch (err) {
    self.postMessage({ type: 'error', id, error: err?.message || String(err) });
  }
};
