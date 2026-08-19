import React, { useState, useMemo, useRef } from 'react';
import {
  ArrowLeft,
  Download,
  Loader2,
  AlertTriangle,
  Check,
  Eye,
  Database,
  ChevronLeft,
  ChevronRight,
  ScanSearch,
  FileText,
} from 'lucide-react';
import { saveDatasetToMyFiles, fetchStorageOverview } from '../services/storageApi';
import { API_ORIGIN } from '../config';
import { toDataUrl } from '../utils/imageUrl';

const API_BASE = API_ORIGIN;

// Alignment preview plus the export controls.
//
// "recognition" exports line crops with transcript labels; "detection" exports
// whole images with bbox annotations and needs no transcript.
export default function DatasetGenerationPage({
  pages,
  selectedPages,
  processedImages,
  transcript,
  allPagesBoxes,
  onBack,
  onHome = null,
  bookName = 'dataset',
  forceMode = null,        // 'detection' to lock the toggle to detection-only
  preprocessing = [],
  modelInfo = {},          // detection/layout models carried from step 4
}) {
  const [isExporting, setIsExporting] = useState(false);
  const [isSavingLocal, setIsSavingLocal] = useState(false);
  const [exportError, setExportError] = useState(null);
  const [previewIdx, setPreviewIdx] = useState(0);
  // If upload already committed to a mode, honour it and hide the toggle.
  const [datasetType, setDatasetType] = useState(forceMode || 'recognition');
  const [bboxFormat, setBboxFormat] = useState('txt'); // 'txt' | 'json' | 'yolo' | 'coco'
  const lastSavedSignatureRef = useRef('');
  const isModeLocked = forceMode != null;

  // ── Detection data ──
  const detectionData = useMemo(() => {
    if (!allPagesBoxes) return [];
    return selectedPages
      .filter(pageNum => (allPagesBoxes[pageNum] || []).length > 0)
      .map(pageNum => {
        const page = pages.find(p => p.pageNumber === pageNum);
        const imageSrc = processedImages[pageNum] || page?.thumbnail;
        const boxes = allPagesBoxes[pageNum] || [];
        return { pageNumber: pageNum, imageSrc, boxes, numBoxes: boxes.length };
      });
  }, [allPagesBoxes, selectedPages, pages, processedImages]);

  const totalDetectionBoxes = detectionData.reduce((s, d) => s + d.numBoxes, 0);

  // ── Recognition data ──
  const alignmentData = useMemo(() => {
    if (!transcript || !allPagesBoxes) return [];

    const tKeys = Object.keys(transcript).sort((a, b) => {
      const na = parseInt(a), nb = parseInt(b);
      if (!isNaN(na) && !isNaN(nb)) return na - nb;
      return a.localeCompare(b);
    });

    const data = [];

    for (const pageNum of selectedPages) {
      const rawBoxes = allPagesBoxes[pageNum] || [];
      const boxes = [...rawBoxes].sort((a, b) => {
        const midTopY = poly => {
          const byY = [...poly].sort((p, q) => p[1] - q[1]);
          return (byY[0][1] + byY[1][1]) / 2;
        };
        return midTopY(a) - midTopY(b);
      });
      const pageKey = tKeys.find((k) => {
        const n = parseInt(k);
        return n === pageNum || k === String(pageNum);
      });

      if (!pageKey) continue;

      const lines = transcript[pageKey] || [];
      const numPairs = Math.min(boxes.length, lines.length);

      const page = pages.find((p) => p.pageNumber === pageNum);
      const imageSrc = processedImages[pageNum] || page?.thumbnail;

      data.push({
        pageNumber: pageNum,
        pageKey,
        imageSrc,
        boxes,
        lines,
        numBoxes: boxes.length,
        numLines: lines.length,
        numPairs,
        mismatch: boxes.length !== lines.length,
      });
    }

    return data;
  }, [transcript, allPagesBoxes, selectedPages, pages, processedImages]);

  const totalPairs = alignmentData.reduce((s, d) => s + d.numPairs, 0);
  const totalMismatches = alignmentData.filter((d) => d.mismatch).length;

  // ── Preview ──
  const previewList = datasetType === 'detection' ? detectionData : alignmentData;
  const currentPreview = previewList[previewIdx] || null;

  // ── Export ──

  const handleExport = async () => {
    setIsExporting(true);
    setExportError(null);

    try {
      const pagesPayload = await Promise.all(alignmentData.map(async (d) => ({
        page_key: String(d.pageNumber),
        image_data: await toDataUrl(d.imageSrc),
        boxes: d.boxes,
        lines: d.lines,
      })));

      const res = await fetch(`${API_BASE}/api/dataset/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pages: pagesPayload,
          book_name: bookName,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Export failed' }));
        throw new Error(err.detail || 'Export failed');
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${bookName}_dataset.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setExportError(err.message);
    } finally {
      setIsExporting(false);
    }
  };

  const handleDetectionExport = async () => {
    setIsExporting(true);
    setExportError(null);

    try {
      const pagesPayload = await Promise.all(detectionData.map(async (d) => ({
        page_key: String(d.pageNumber),
        image_data: await toDataUrl(d.imageSrc),
        boxes: d.boxes,
      })));

      const res = await fetch(`${API_BASE}/api/dataset/export-detection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pages: pagesPayload,
          book_name: bookName,
          bbox_format: bboxFormat,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Export failed' }));
        throw new Error(err.detail || 'Export failed');
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${bookName}_detection_dataset.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setExportError(err.message);
    } finally {
      setIsExporting(false);
    }
  };

  const handleSaveLocal = async () => {
    // Name first, so cancelling doesn't leave a spinner up.
    const defaultName = bookName && bookName !== 'dataset' ? bookName : 'My Dataset';
    const chosenName = window.prompt('Name this dataset:', defaultName);
    if (chosenName === null) return;                       // user cancelled
    const finalName = chosenName.trim() || defaultName;

    try {
      const overview = await fetchStorageOverview();
      const existingNames = (overview.datasets || []).map((d) => (d.name || '').toLowerCase());
      if (existingNames.includes(finalName.toLowerCase())) {
        const proceed = window.confirm(`A dataset named "${finalName}" already exists in My Files. Save anyway?`);
        if (!proceed) return;
      }
    } catch {
    }

    // Carried from step 4 into the My Files metadata/tags.
    const combinedModelInfo = {
      ...modelInfo,
      preprocessing:
        modelInfo.preprocessing && modelInfo.preprocessing.length
          ? modelInfo.preprocessing
          : preprocessing,
    };

    setIsSavingLocal(true);
    setExportError(null);

    try {
      if (isDetection) {
        const pagesPayload = await Promise.all(detectionData.map(async (d) => ({
          page_key: String(d.pageNumber),
          image_data: await toDataUrl(d.imageSrc),
          boxes: d.boxes,
        })));

        const signature = JSON.stringify({
          mode: 'detection',
          bookName: finalName,
          bboxFormat,
          pages: pagesPayload.map((p) => ({ page_key: p.page_key, boxes: p.boxes.length })),
        });
        if (lastSavedSignatureRef.current === signature) {
          window.alert('Already saved — this exact dataset is already in My Files.');
          return;
        }

        await saveDatasetToMyFiles(pagesPayload, {
          source: 'dataset generation',
          bookName: finalName,
          bboxFormat,
          mode: 'detection',
          modelInfo: combinedModelInfo,
        });
        lastSavedSignatureRef.current = signature;
      } else {
        const pagesPayload = await Promise.all(alignmentData.map(async (d) => ({
          page_key: String(d.pageNumber),
          image_data: await toDataUrl(d.imageSrc),
          boxes: d.boxes,
          lines: d.lines,
        })));

        const signature = JSON.stringify({
          mode: 'recognition',
          bookName: finalName,
          pages: pagesPayload.map((p) => ({
            page_key: p.page_key,
            boxes: p.boxes.length,
            lines: p.lines.length,
            text: p.lines.join('\n'),
          })),
        });
        if (lastSavedSignatureRef.current === signature) {
          window.alert('Already saved — this exact dataset is already in My Files.');
          return;
        }

        await saveDatasetToMyFiles(pagesPayload, {
          source: 'dataset generation',
          bookName: finalName,
          mode: 'recognition',
          modelInfo: combinedModelInfo,
        });
        lastSavedSignatureRef.current = signature;
      }

      window.alert(`Saved "${finalName}" to My Files.`);
    } catch (err) {
      setExportError(`Save failed: ${err.message}`);
    } finally {
      setIsSavingLocal(false);
    }
  };

  const isDetection = datasetType === 'detection';
  const canExport = isDetection ? detectionData.length > 0 : alignmentData.length > 0;

  return (
    <div className="h-full flex flex-col animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-6 px-4 pt-4">
        <div className="flex items-center gap-4">
          <button onClick={onBack} className="btn-ghost">
            <ArrowLeft className="w-5 h-5" />
            Back
          </button>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-lg text-white shadow-md">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
                Generate Dataset
              </h1>
              <p className="text-sm text-gray-500">
                {isDetection
                  ? `${detectionData.length} pages, ${totalDetectionBoxes} bounding boxes`
                  : `${alignmentData.length} pages, ${totalPairs} aligned pairs`
                }
              </p>
            </div>

            {onHome && (
              <button
                onClick={onHome}
                className="ml-2 px-3 py-1.5 rounded-lg text-sm font-medium text-gray-600 hover:text-emerald-700 hover:bg-emerald-50 transition-all"
              >
                Home
              </button>
            )}
          </div>
        </div>

        {/* Dataset type toggle (hidden when upstream locked the mode) */}
        {!isModeLocked && (
        <div className="flex bg-gray-100 rounded-lg p-1">
          <button
            onClick={() => { setDatasetType('recognition'); setPreviewIdx(0); }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
              !isDetection
                ? 'bg-white text-emerald-700 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            Recognition
          </button>
          <button
            onClick={() => { setDatasetType('detection'); setPreviewIdx(0); }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
              isDetection
                ? 'bg-white text-emerald-700 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <ScanSearch className="w-3.5 h-3.5" />
            Detection
          </button>
        </div>
        )}
      </div>

      {/* Stats row */}
      {isDetection ? (
        <div className="grid grid-cols-2 gap-4 px-4 mb-6">
          <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-gray-100 p-4 text-center shadow-sm">
            <p className="text-2xl font-bold text-emerald-600">{detectionData.length}</p>
            <p className="text-xs text-gray-500 mt-1">Pages</p>
          </div>
          <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-gray-100 p-4 text-center shadow-sm">
            <p className="text-2xl font-bold text-indigo-600">{totalDetectionBoxes}</p>
            <p className="text-xs text-gray-500 mt-1">Bounding Boxes</p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-4 gap-4 px-4 mb-6">
          <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-gray-100 p-4 text-center shadow-sm">
            <p className="text-2xl font-bold text-emerald-600">{alignmentData.length}</p>
            <p className="text-xs text-gray-500 mt-1">Pages Matched</p>
          </div>
          <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-gray-100 p-4 text-center shadow-sm">
            <p className="text-2xl font-bold text-blue-600">{totalPairs}</p>
            <p className="text-xs text-gray-500 mt-1">Line Pairs</p>
          </div>
          <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-gray-100 p-4 text-center shadow-sm">
            <p className="text-2xl font-bold text-indigo-600">
              {alignmentData.reduce((s, d) => s + d.numBoxes, 0)}
            </p>
            <p className="text-xs text-gray-500 mt-1">Bounding Boxes</p>
          </div>
          <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-gray-100 p-4 text-center shadow-sm">
            <p className={`text-2xl font-bold ${totalMismatches > 0 ? 'text-amber-600' : 'text-green-600'}`}>
              {totalMismatches}
            </p>
            <p className="text-xs text-gray-500 mt-1">Mismatches</p>
          </div>
        </div>
      )}

      <div className="flex-1 flex gap-6 px-4 pb-4 overflow-hidden">
        {/* Left: page list + preview */}
        <div className="flex-1 flex flex-col bg-white/80 backdrop-blur-sm rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
            <h3 className="font-bold text-gray-700 flex items-center gap-2">
              <Eye className="w-4 h-4" />
              {isDetection ? 'Detection Preview' : 'Alignment Preview'}
            </h3>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPreviewIdx(Math.max(0, previewIdx - 1))}
                disabled={previewIdx === 0}
                className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="text-sm text-gray-500 min-w-[60px] text-center">
                {previewList.length > 0 ? `${previewIdx + 1} / ${previewList.length}` : '\u2014'}
              </span>
              <button
                onClick={() => setPreviewIdx(Math.min(previewList.length - 1, previewIdx + 1))}
                disabled={previewIdx >= previewList.length - 1}
                className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 transition-colors"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {currentPreview ? (
              isDetection ? (
                /* Detection: bbox count per page */
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-semibold text-gray-700">
                      Page {currentPreview.pageNumber}
                    </span>
                    <span className="text-xs text-indigo-600 bg-indigo-50 px-2.5 py-1 rounded-full">
                      {currentPreview.numBoxes} boxes
                    </span>
                  </div>
                  <div className="space-y-1.5 max-h-[60vh] overflow-y-auto">
                    {currentPreview.boxes.map((box, i) => {
                      const xs = box.map(p => p[0]);
                      const ys = box.map(p => p[1]);
                      const x1 = Math.round(Math.min(...xs));
                      const y1 = Math.round(Math.min(...ys));
                      const x2 = Math.round(Math.max(...xs));
                      const y2 = Math.round(Math.max(...ys));
                      return (
                        <div
                          key={i}
                          className="flex items-center gap-3 p-2.5 bg-gray-50 rounded-lg text-sm"
                        >
                          <span className="text-xs font-mono text-gray-400 w-8 text-right flex-shrink-0">
                            {i + 1}
                          </span>
                          <span className="flex-1 text-gray-600 font-mono text-xs">
                            [{x1}, {y1}, {x2 - x1}, {y2 - y1}]
                          </span>
                          <Check className="w-4 h-4 text-green-500 flex-shrink-0" />
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : (
                /* Recognition: alignment view */
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-semibold text-gray-700">
                      Page {currentPreview.pageNumber}
                    </span>
                    {currentPreview.mismatch && (
                      <span className="flex items-center gap-1.5 text-xs text-amber-600 bg-amber-50 px-2.5 py-1 rounded-full">
                        <AlertTriangle className="w-3.5 h-3.5" />
                        {currentPreview.numBoxes} boxes / {currentPreview.numLines} lines
                      </span>
                    )}
                  </div>

                  <div className="space-y-1.5 max-h-[60vh] overflow-y-auto">
                    {currentPreview.lines.slice(0, currentPreview.numPairs).map((line, i) => (
                      <div
                        key={i}
                        className="flex items-center gap-3 p-2.5 bg-gray-50 rounded-lg text-sm"
                      >
                        <span className="text-xs font-mono text-gray-400 w-8 text-right flex-shrink-0">
                          {i + 1}
                        </span>
                        <span className="flex-1 text-gray-700 truncate">{line}</span>
                        {i < currentPreview.numBoxes && (
                          <Check className="w-4 h-4 text-green-500 flex-shrink-0" />
                        )}
                      </div>
                    ))}
                    {currentPreview.numLines > currentPreview.numPairs && (
                      <p className="text-xs text-amber-500 italic pl-11">
                        {currentPreview.numLines - currentPreview.numPairs} unmatched transcript line(s)
                      </p>
                    )}
                    {currentPreview.numBoxes > currentPreview.numPairs && (
                      <p className="text-xs text-amber-500 italic pl-11">
                        {currentPreview.numBoxes - currentPreview.numPairs} unmatched bounding box(es)
                      </p>
                    )}
                  </div>
                </div>
              )
            ) : (
              <div className="flex items-center justify-center h-full text-gray-400">
                {isDetection ? 'No detection data available' : 'No alignment data available'}
              </div>
            )}
          </div>
        </div>

        {/* Right: export panel */}
        <div className="w-80 flex flex-col bg-white/80 backdrop-blur-sm rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-100">
            <h3 className="font-bold text-gray-700 flex items-center gap-2">
              <Download className="w-4 h-4" />
              Export
            </h3>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {isDetection ? (
              <div className="space-y-3">
                {(() => {
                  const layoutByFormat = {
                    txt:  `${bookName}/images/page_N.jpg\n${bookName}/bboxes/page_N.txt`,
                    json: `${bookName}/images/page_N.jpg\n${bookName}/bboxes/page_N.json`,
                    yolo: `${bookName}/images/page_N.jpg\n${bookName}/labels/page_N.txt\n${bookName}/classes.txt`,
                    coco: `${bookName}/images/page_N.jpg\n${bookName}/annotations.json`,
                  };
                  const sampleByFormat = {
                    txt:  'x1 y1 x2 y2 (one per line)',
                    json: '[[x1, y1, x2, y2], ...]',
                    yolo: '0 cx cy w h (normalized 0–1)',
                    coco: 'COCO JSON (single "text" category)',
                  };
                  return (
                    <>
                      <div className="p-3.5 rounded-xl border border-indigo-200 bg-indigo-50/70 text-sm text-indigo-700">
                        Downloads a ZIP with full page images and bounding boxes
                        for each detected text line.
                        <div className="text-xs text-indigo-600 mt-2 font-mono whitespace-pre-line">
                          {layoutByFormat[bboxFormat]}
                        </div>
                      </div>

                      <div>
                        <label className="block text-xs font-semibold text-gray-600 mb-1.5">
                          Annotation format
                        </label>
                        <div className="grid grid-cols-4 bg-gray-100 rounded-lg p-1 gap-1">
                          {['txt', 'json', 'yolo', 'coco'].map((f) => (
                            <button
                              key={f}
                              onClick={() => setBboxFormat(f)}
                              className={`px-2 py-1.5 rounded-md text-xs font-semibold uppercase transition-all ${
                                bboxFormat === f
                                  ? 'bg-white text-indigo-700 shadow-sm'
                                  : 'text-gray-500 hover:text-gray-700'
                              }`}
                            >
                              {f}
                            </button>
                          ))}
                        </div>
                        <p className="text-[11px] text-gray-400 mt-1.5 font-mono">
                          {sampleByFormat[bboxFormat]}
                        </p>
                      </div>
                    </>
                  );
                })()}
              </div>
            ) : (
              <div className="p-3.5 rounded-xl border border-emerald-200 bg-emerald-50/70 text-sm text-emerald-700">
                Downloads one ZIP with line images and transcript labels.
                <div className="text-xs text-emerald-600 mt-2 font-mono">
                  {bookName}/page_N/images/line_label.png
                </div>
              </div>
            )}
          </div>

          {/* Export button */}
          <div className="p-4 border-t border-gray-100">
            {exportError && (
              <div className="mb-3 p-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-600">
                {exportError}
              </div>
            )}

            <div className="flex gap-2 mb-2">
              <button
                onClick={handleSaveLocal}
                disabled={isExporting || isSavingLocal || !canExport}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-bold transition-all duration-200 border ${
                  isExporting || isSavingLocal || !canExport
                    ? 'bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed'
                    : 'bg-cyan-50 text-cyan-700 border-cyan-200 hover:bg-cyan-100'
                }`}
              >
                {isSavingLocal ? <Loader2 className="w-5 h-5 animate-spin" /> : <Download className="w-5 h-5" />}
                Save to My Files
              </button>
            </div>

            <button
              onClick={isDetection ? handleDetectionExport : handleExport}
              disabled={isExporting || !canExport}
              className={`w-full flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl font-bold text-white shadow-lg transition-all duration-200 ${
                isExporting || !canExport
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-gradient-to-r from-emerald-500 to-teal-600 hover:shadow-xl hover:-translate-y-0.5 shadow-emerald-500/30'
              }`}
            >
              {isExporting ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Download className="w-5 h-5" />
                  {isDetection ? 'Download Detection Dataset' : 'Download Dataset'}
                </>
              )}
            </button>

            <p className="text-xs text-gray-400 text-center mt-2">
              {isDetection
                ? `${detectionData.length} pages, ${totalDetectionBoxes} annotations`
                : `${totalPairs} line images will be exported`
              }
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
