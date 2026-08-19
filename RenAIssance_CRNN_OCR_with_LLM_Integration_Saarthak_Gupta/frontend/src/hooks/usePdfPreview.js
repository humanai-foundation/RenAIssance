import { useState, useCallback, useEffect, useRef } from 'react';

// Turns a PDF or a pile of image files into page objects.
//
// PDF rendering happens in a Web Worker: background tabs throttle main-thread
// canvas work to ~1 Hz, which made extraction look frozen. Images can't use a
// worker (<img> decode is main-thread only) so they run with a concurrency cap
// instead. Everything comes back as blob: object URLs — see utils/imageUrl.

const IMAGE_CONCURRENCY = 4;

function makeWorker() {
  return new Worker(
    new URL('../workers/pdfExtractor.worker.js', import.meta.url),
    { type: 'module' }
  );
}

export function usePdfPreview() {
  const [pages, setPages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);

  const workerRef = useRef(null);
  // Revoked on reset/reload.
  const objectUrlsRef = useRef([]);

  const revokeAllUrls = useCallback(() => {
    objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    objectUrlsRef.current = [];
  }, []);

  const trackUrl = useCallback((url) => {
    objectUrlsRef.current.push(url);
    return url;
  }, []);

  useEffect(() => {
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
      revokeAllUrls();
    };
  }, [revokeAllUrls]);

  const extractPages = useCallback(async (file, options = {}) => {
    setIsLoading(true);
    setError(null);
    setProgress(0);
    setPages([]);
    revokeAllUrls();

    // Fresh worker each run, so a stuck one can't poison the next.
    if (workerRef.current) workerRef.current.terminate();
    const worker = makeWorker();
    workerRef.current = worker;

    const requestId = Date.now() + Math.random();
    const collected = [];
    // Batch the flushes — one setPages per page re-renders the whole grid.
    let pending = [];
    let flushTimer = null;
    const flush = () => {
      flushTimer = null;
      if (pending.length === 0) return;
      const batch = pending;
      pending = [];
      setPages((prev) => [...prev, ...batch]);
    };

    try {
      const fileBuffer = await file.arrayBuffer();

      const result = await new Promise((resolve, reject) => {
        worker.onmessage = (e) => {
          const msg = e.data || {};
          if (msg.id !== requestId) return;
          if (msg.type === 'page') {
            const { blob, ...rest } = msg.page;
            const page = { ...rest, thumbnail: trackUrl(URL.createObjectURL(blob)) };
            collected.push(page);
            pending.push(page);
            if (!flushTimer) flushTimer = setTimeout(flush, 200);
          } else if (msg.type === 'progress') {
            setProgress(msg.progress);
          } else if (msg.type === 'done') {
            flush();
            resolve(collected);
          } else if (msg.type === 'error') {
            reject(new Error(msg.error));
          }
        };
        worker.onerror = (e) => reject(new Error(e.message || 'PDF worker crashed'));

        worker.postMessage(
          { type: 'extract', id: requestId, fileBuffer, options },
          [fileBuffer]
        );
      });

      return result;
    } catch (err) {
      console.error('PDF extraction error:', err);
      setError(err.message || 'Failed to extract PDF pages');
      throw err;
    } finally {
      if (flushTimer) clearTimeout(flushTimer);
      setIsLoading(false);
      // Left alive for a likely next extraction; replaced or torn down later.
    }
  }, [revokeAllUrls, trackUrl]);

  const loadImages = useCallback(async (files, options = {}) => {
    const { splitDoublePages = true } = options;

    setIsLoading(true);
    setError(null);
    setProgress(0);
    setPages([]);
    revokeAllUrls();

    try {
      const total = files.length;
      let done = 0;
      const slots = new Array(total);

      const canvasToUrl = (canvas) => new Promise((resolve, reject) => {
        canvas.toBlob((blob) => {
          if (blob) resolve(trackUrl(URL.createObjectURL(blob)));
          else reject(new Error('Failed to encode split page'));
        }, 'image/png');
      });

      const processOne = async (file, index) => {
        const fileIndex = index + 1;
        const fileUrl = trackUrl(URL.createObjectURL(file));

        const imgData = await new Promise((resolve, reject) => {
          const img = new Image();
          img.onload = () => resolve({ width: img.width, height: img.height, img });
          img.onerror = () => reject(new Error('Failed to load image'));
          img.src = fileUrl;
        });

        let produced;
        if (splitDoublePages && imgData.width > imgData.height * 1.19) {
          const halfWidth = Math.floor(imgData.width / 2);

          const leftCanvas = document.createElement('canvas');
          leftCanvas.width = halfWidth;
          leftCanvas.height = imgData.height;
          leftCanvas
            .getContext('2d')
            .drawImage(imgData.img, 0, 0, halfWidth, imgData.height, 0, 0, halfWidth, imgData.height);

          const rightCanvas = document.createElement('canvas');
          rightCanvas.width = halfWidth;
          rightCanvas.height = imgData.height;
          rightCanvas
            .getContext('2d')
            .drawImage(imgData.img, halfWidth, 0, halfWidth, imgData.height, 0, 0, halfWidth, imgData.height);

          produced = [
            {
              pageNumber: `${fileIndex}_left`,
              thumbnail: await canvasToUrl(leftCanvas),
              width: halfWidth,
              height: imgData.height,
              fileName: file.name,
              isSplit: true,
              splitSide: 'left',
            },
            {
              pageNumber: `${fileIndex}_right`,
              thumbnail: await canvasToUrl(rightCanvas),
              width: halfWidth,
              height: imgData.height,
              fileName: file.name,
              isSplit: true,
              splitSide: 'right',
            },
          ];
        } else {
          produced = [{
            pageNumber: fileIndex,
            thumbnail: fileUrl,
            width: imgData.width,
            height: imgData.height,
            fileName: file.name,
            isSplit: false,
          }];
        }

        slots[index] = produced;
        done += 1;
        setProgress(Math.round((done / total) * 100));
      };

      // Each lane pulls the next index until the queue is empty.
      let cursor = 0;
      const lanes = Array.from({ length: Math.min(IMAGE_CONCURRENCY, total) }, async () => {
        while (true) {
          const idx = cursor++;
          if (idx >= total) return;
          await processOne(files[idx], idx);
        }
      });
      await Promise.all(lanes);

      const loadedPages = slots.flat().filter(Boolean);
      setPages(loadedPages);
      return loadedPages;
    } catch (err) {
      console.error('Image loading error:', err);
      setError(err.message || 'Failed to load images');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [revokeAllUrls, trackUrl]);

  const reset = useCallback(() => {
    setPages([]);
    setIsLoading(false);
    setError(null);
    setProgress(0);
    revokeAllUrls();
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
    }
  }, [revokeAllUrls]);

  return {
    pages,
    isLoading,
    error,
    progress,
    extractPages,
    loadImages,
    reset,
  };
}

export default usePdfPreview;
