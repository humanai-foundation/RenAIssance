import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
    Play,
    Loader2,
    AlertTriangle,
    CheckCircle2,
    ChevronLeft,
    ChevronRight,
    Layers,
    Image as ImageIcon,
    FileText,
    Info,
    ChevronDown,
    PenLine,
    ArrowUp,
    ArrowDown,
    Split,
    Link2,
    Sparkles,
    Cpu,
    Download,
    FileText as FileTextIcon,
    FileJson,
    FileType,
    BookOpen,
    RefreshCw,
    Home,
    Plus,
    Wand2,
} from 'lucide-react';
import ResizablePanels from './ocr/ResizablePanels';
import BBoxEditor from './BBoxEditor';
import {
    getLocalRecognitionModels,
    runLocalRecognition,
    exportCRNNResultsAsText,
    exportCRNNResultsAsJSON,
    downloadBlob,
} from '../features/ocr/services/ocrApi';
import { saveTranscriptSession, fetchStorageOverview } from '../services/storageApi';
import {
    getLLMProviders,
    getLLMTemplates,
    postProcessWithLLMProvider,
} from '../services/llmApi';
import { API_ORIGIN } from '../config';

const API_BASE = API_ORIGIN;

// ── Layout model options ──
const LAYOUT_MODELS = [
    { id: 'PP-DocLayout_plus-L', name: 'PP-DocLayout+ Large', desc: 'Best accuracy' },
    { id: 'PP-DocLayout-L', name: 'PP-DocLayout Large', desc: 'High accuracy' },
    { id: 'PP-DocLayout-M', name: 'PP-DocLayout Medium', desc: 'Balanced' },
    { id: 'PP-DocLayout-S', name: 'PP-DocLayout Small', desc: 'Fastest' },
    { id: 'PicoDet-L_layout_17cls', name: 'PicoDet-L 17cls', desc: '17-class layout' },
    { id: 'RT-DETR-H_layout_17cls', name: 'RT-DETR-H 17cls', desc: '17-class, high acc' },
    { id: 'PicoDet-L_layout_3cls', name: 'PicoDet-L 3cls', desc: '3-class layout' },
    { id: 'PicoDet-S_layout_3cls', name: 'PicoDet-S 3cls', desc: '3-class, fastest' },
];

// ── Detection model options ──
const DETECTION_MODELS = [
    { id: 'PP-OCRv5_server_det', name: 'PP-OCRv5 Server Det', desc: 'High accuracy, slower' },
    { id: 'PP-OCRv5_mobile_det', name: 'PP-OCRv5 Mobile Det', desc: 'Fast, lighter' },
    { id: 'DB', name: 'DBNet', desc: 'Classic, reliable' },
    { id: 'DB++', name: 'DB++', desc: 'Enhanced DBNet' },
    { id: 'EAST', name: 'EAST', desc: 'Efficient text detection' },
    { id: 'SAST', name: 'SAST', desc: 'Segmentation-based' },
];

// ── Default tuning parameters ──
const DEFAULT_PARAMS = {
    region_padding: 50,
    layout_expand: 2,
    score_thresh: 0.5,
    upscale_min_h: 60,
    nms_iou_thresh: 0.3,
    gap_multiplier: 2.0,
};

// ── Slider definitions for the tuning UI ──
const PARAM_DEFS = [
    { key: 'region_padding', label: 'Region Padding', unit: 'px', min: 0, max: 200, step: 5, type: 'int', tooltip: 'Padding around detected regions before OCR' },
    { key: 'layout_expand', label: 'Layout Expand', unit: 'px', min: 0, max: 50, step: 1, type: 'int', tooltip: 'Expand layout bounding boxes before cropping' },
    { key: 'score_thresh', label: 'Score Threshold', unit: '', min: 0.1, max: 1.0, step: 0.05, type: 'float', tooltip: 'Minimum confidence score for text detection' },
    { key: 'upscale_min_h', label: 'Upscale Min H', unit: 'px', min: 20, max: 200, step: 10, type: 'int', tooltip: 'Upscale crops shorter than this height' },
    { key: 'nms_iou_thresh', label: 'NMS IoU Thresh', unit: '', min: 0.1, max: 0.9, step: 0.05, type: 'float', tooltip: 'IoU threshold for non-max suppression' },
    { key: 'gap_multiplier', label: 'Gap Multiplier', unit: '×', min: 0.5, max: 5.0, step: 0.5, type: 'float', tooltip: 'Gap threshold multiplier for line merging' },
];

// Monotonic, not Date.now(): rows created in the same millisecond would collide
// as React keys and silently remount.
let _rowIdCounter = 0;
const nextRowId = () => ++_rowIdCounter;

// Word-level LCS diff for the before/after view. Whitespace splitting is fine —
// OCR corrections are small (diacritics, glyph fixes) and we only need to point
// at the words that changed.
function diffWords(before, after) {
    const a = before.split(/\s+/).filter(Boolean);
    const b = after.split(/\s+/).filter(Boolean);
    const n = a.length;
    const m = b.length;
    const lcs = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
    for (let i = n - 1; i >= 0; i -= 1) {
        for (let j = m - 1; j >= 0; j -= 1) {
            lcs[i][j] = a[i] === b[j]
                ? lcs[i + 1][j + 1] + 1
                : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
        }
    }
    const beforeTokens = [];
    const afterTokens = [];
    let i = 0;
    let j = 0;
    while (i < n && j < m) {
        if (a[i] === b[j]) {
            beforeTokens.push({ text: a[i], removed: false });
            afterTokens.push({ text: b[j], added: false });
            i += 1; j += 1;
        } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
            beforeTokens.push({ text: a[i], removed: true });
            i += 1;
        } else {
            afterTokens.push({ text: b[j], added: true });
            j += 1;
        }
    }
    while (i < n) { beforeTokens.push({ text: a[i], removed: true }); i += 1; }
    while (j < m) { afterTokens.push({ text: b[j], added: true }); j += 1; }
    return { beforeTokens, afterTokens };
}

function shortPageLabel(pageKey) {
    return String(pageKey).replace('_left', 'L').replace('_right', 'R');
}

function getPageBaseNumber(pageKey) {
    const value = String(pageKey).toLowerCase();
    const match = value.match(/(\d+)/);
    return match ? Number(match[1]) : null;
}

function getPageSide(pageKey) {
    const value = String(pageKey).toLowerCase();
    if (value.includes('left') || value.endsWith('l')) return 'left';
    if (value.includes('right') || value.endsWith('r')) return 'right';
    return null;
}

function resolveTranscriptKey(transcript, pageKey) {
    if (!transcript) return null;
    const keys = Object.keys(transcript);
    const asString = String(pageKey);
    if (asString in transcript) return asString;

    const targetNum = getPageBaseNumber(asString);
    const targetSide = getPageSide(asString);

    if (targetNum == null) return null;

    const exactSideKey = keys.find((k) => {
        const keyNum = getPageBaseNumber(k);
        const keySide = getPageSide(k);
        return keyNum === targetNum && keySide === targetSide;
    });
    if (exactSideKey) return exactSideKey;

    return keys.find((k) => getPageBaseNumber(k) === targetNum) || null;
}


// ── Thumbnail item ──
function ThumbnailItem({ image, index, isActive, isDetected, onClick, pageLabel, processedSrc }) {
    const imageSrc = processedSrc || image?.processed || image?.original || image?.thumbnail;

    return (
        <button
            onClick={onClick}
            className={`
                relative aspect-[3/4] rounded-lg overflow-hidden transition-all duration-200
                border-2 group
                ${isActive
                    ? 'border-teal-500 ring-2 ring-teal-200 shadow-md'
                    : isDetected
                        ? 'border-emerald-300 hover:border-emerald-400'
                        : 'border-gray-200 hover:border-gray-300 hover:shadow-sm'
                }
            `}
        >
            {imageSrc ? (
                <img src={imageSrc} alt={`Page ${pageLabel}`} className="w-full h-full object-cover" />
            ) : (
                <div className="absolute inset-0 bg-gray-100 flex items-center justify-center">
                    <ImageIcon size={20} className="text-gray-300" />
                </div>
            )}

            {isDetected && !isActive && (
                <div className="absolute top-1.5 right-1.5">
                    <CheckCircle2 size={16} className="text-emerald-500 bg-white rounded-full shadow-sm" />
                </div>
            )}

            <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/70 to-transparent pt-4 pb-1">
                <span className="block text-center text-white text-xs font-medium">{pageLabel}</span>
            </div>
        </button>
    );
}


// ── Main component ──
export default function LayoutAwareDetectionPage({
    pages,
    selectedPages,
    processedImages,
    transcript = {},
    bookName = 'transcript',
    preprocessing = [],
    onBack,
    onHome,
    datasetMode = false,
    onDatasetNext,
    // Restored when navigating back from export.
    initialDetectedPages = {},
    initialAlignmentByPage = {},
    onStateChange,
}) {
    // ── Model selection ──
    const [selectedDetModel, setSelectedDetModel] = useState('PP-OCRv5_server_det');
    const [selectedLayoutModel, setSelectedLayoutModel] = useState('PP-DocLayout_plus-L');

    // ── Tuning parameters ──
    const [tuningParams, setTuningParams] = useState({ ...DEFAULT_PARAMS });
    const [showAdvanced, setShowAdvanced] = useState(false);

    // ── UI state ──
    const [viewingPageIndex, setViewingPageIndex] = useState(0);
    const [loading, setLoading] = useState(false);
    // null = not checked yet, true = cached, false = needs the big download.
    const [modelsReady, setModelsReady] = useState(null);
    const [downloadingModels, setDownloadingModels] = useState(false);
    const [detectedPages, setDetectedPages] = useState(() => initialDetectedPages);
    const [error, setError] = useState(null);
    const [warning, setWarning] = useState(null);
    const [processingTime, setProcessingTime] = useState(null);
    const [imageLoaded, setImageLoaded] = useState(false);

    // ── BBox editor state ──
    const [showBBoxEditor, setShowBBoxEditor] = useState(false);
    // Drives the "edited" dot on thumbnails.
    const [editedPages, setEditedPages] = useState(new Set());

    // ── Process-all state ──
    const [processingAll, setProcessingAll] = useState(false);
    const [processAllProgress, setProcessAllProgress] = useState(null);
    const cancelProcessingRef = useRef(false);

    // ── Alignment state (dataset mode) ──
    const [alignmentByPage, setAlignmentByPage] = useState(() => initialAlignmentByPage);

    // ── Local OCR state (OCR mode) ──
    const [localModels, setLocalModels] = useState([]);
    const [localModelsLoading, setLocalModelsLoading] = useState(false);
    const [localModelsError, setLocalModelsError] = useState(null);
    const [selectedOcrModel, setSelectedOcrModel] = useState('');
    const [recognizedByPage, setRecognizedByPage] = useState({});
    const [ocrProcessing, setOcrProcessing] = useState(false);
    const [ocrProcessingAll, setOcrProcessingAll] = useState(false);
    const [ocrProgress, setOcrProgress] = useState(null);
    const cancelOcrRef = useRef(false);
    const lastSavedSignatureRef = useRef('');
    const [ocrDevice, setOcrDevice] = useState(null);
    const [ocrTimeMs, setOcrTimeMs] = useState(null);
    const [ocrError, setOcrError] = useState(null);

    // ── LLM post-processing (optional cleanup pass) ──
    const [llmEnabled, setLlmEnabled] = useState(false);
    const [llmProviders, setLlmProviders] = useState([]);
    const [llmProvider, setLlmProvider] = useState('gemini');
    const [llmModel, setLlmModel] = useState('gemini-2.5-flash');
    const [llmTemplates, setLlmTemplates] = useState([]);
    const [llmTemplate, setLlmTemplate] = useState('full_cleanup');
    // Keyed by provider so switching doesn't lose an entered key.
    const [llmApiKeys, setLlmApiKeys] = useState({});
    const [llmProcessing, setLlmProcessing] = useState(false);
    const [llmProgress, setLlmProgress] = useState(null);
    const [llmError, setLlmError] = useState(null);
    // Local corrector only: null = unknown/not applicable, false = base weights
    // still to download (~8 GB, one time), true = cached.
    const [llmModelReady, setLlmModelReady] = useState(null);
    const [downloadingLlmModel, setDownloadingLlmModel] = useState(false);
    // Stamped into the saved transcript's My Files metadata.
    const [lastLlmRun, setLastLlmRun] = useState(null);

    // { [pageNum]: { [boxIndex]: textBeforePolish } } for the changed-lines view.
    const [llmDiffByPage, setLlmDiffByPage] = useState({});
    const [showLlmDiff, setShowLlmDiff] = useState(true);

    // One source of truth for image<->transcript hover. Rows derive their
    // highlight from this, so hovering costs one state update, not two.
    const [hoveredBoxIndex, setHoveredBoxIndex] = useState(null);

    const imgRef = useRef(null);
    const imageWrapRef = useRef(null);
    // Row refs by box index — used to scroll a row into view on image hover.
    const lineRefsRef = useRef(new Map());
    const setLineRef = useCallback((boxIndex, el) => {
        if (boxIndex == null) return;
        const map = lineRefsRef.current;
        if (el) map.set(boxIndex, el);
        else map.delete(boxIndex);
    }, []);
    // Needed for the SVG viewBox, so polygon coords land in the right place.
    const [imageNaturalSize, setImageNaturalSize] = useState({ w: 0, h: 0 });
    // Stops the seed effect from clobbering alignment that already exists.
    const seededPages = useRef(new Set());

    // Push state up to the parent cache. Skipping the first pass matters —
    // otherwise we overwrite the cache with empty state on mount. The ref keeps
    // a new onStateChange identity from causing extra syncs.
    const onStateChangeRef = useRef(onStateChange);
    useEffect(() => { onStateChangeRef.current = onStateChange; }, [onStateChange]);
    const didMountStateSyncRef = useRef(false);
    useEffect(() => {
        if (!didMountStateSyncRef.current) {
            didMountStateSyncRef.current = true;
            return;
        }
        onStateChangeRef.current?.({ pages: detectedPages, alignment: alignmentByPage });
    }, [detectedPages, alignmentByPage]);

    // ── Derived data ──
    const availablePages = useMemo(() => selectedPages || [], [selectedPages]);
    const currentPageNum = availablePages[viewingPageIndex] || 1;
    const totalPages = availablePages.length;

    const currentLines = detectedPages[currentPageNum] || [];
    const isCurrentDetected = currentPageNum in detectedPages;
    const detectedCount = Object.keys(detectedPages).length;
    const transcriptKeyForCurrentPage = resolveTranscriptKey(transcript, currentPageNum);

    const currentAlignmentRows = alignmentByPage[currentPageNum] || [];
    const currentRecognition = recognizedByPage[currentPageNum] || [];
    const recognizedCount = Object.keys(recognizedByPage).length;

    const currentRecognitionByIndex = useMemo(() => {
        const map = {};
        currentRecognition.forEach((item) => {
            map[item.box_index] = item.text || '';
        });
        return map;
    }, [currentRecognition]);

    // { boxIndex: beforeText } for the current page.
    const currentDiff = llmDiffByPage[currentPageNum] || {};
    const changedLineCount = Object.keys(currentDiff).length;

    // Sort top-to-bottom by the midpoint-Y of the top edge (average of the two
    // highest vertices). More stable than min-Y once boxes are rotated.
    const sortedBoxIndices = useMemo(() => {
        return currentLines
            .map((poly, idx) => {
                const byY = [...poly].sort((a, b) => a[1] - b[1]);
                const topMidY = (byY[0][1] + byY[1][1]) / 2;
                return { idx, topMidY };
            })
            .sort((a, b) => a.topMidY - b.topMidY)
            .map(item => item.idx);
    }, [currentLines]);

    const getImageUrl = useCallback((pageNum) => {
        if (processedImages?.[pageNum]) return processedImages[pageNum];
        // find-by-pageNumber, so split pages ("1_left") resolve correctly.
        const page = pages?.find(p => p.pageNumber === pageNum);
        return page?.thumbnail || null;
    }, [pages, processedImages]);

    const currentImageUrl = getImageUrl(currentPageNum);

    const toDataUrl = useCallback(async (url) => {
        if (!url) return null;
        if (url.startsWith('data:')) return url;
        const response = await fetch(url);
        const blob = await response.blob();
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.onerror = () => reject(new Error('Failed converting image to base64'));
            reader.readAsDataURL(blob);
        });
    }, []);

    useEffect(() => {
        if (datasetMode) return;
        let cancelled = false;
        async function loadLocalModels() {
            setLocalModelsLoading(true);
            setLocalModelsError(null);
            try {
                const data = await getLocalRecognitionModels();
                const models = data.models || [];
                if (cancelled) return;
                setLocalModels(models);
                if (models.length > 0) {
                    const preferred = models.find((m) => m.model_type === 'crnn') || models[0];
                    setSelectedOcrModel((prev) => prev || preferred.id);
                }
            } catch (err) {
                if (!cancelled) {
                    setLocalModelsError(err.message || 'Failed loading local models');
                }
            } finally {
                if (!cancelled) {
                    setLocalModelsLoading(false);
                }
            }
        }
        loadLocalModels();
        return () => {
            cancelled = true;
        };
    }, [datasetMode]);

    // ── Arrow key navigation (BBox editor closed only) ──
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (showBBoxEditor) return;
            const tag = document.activeElement?.tagName?.toLowerCase();
            if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                setViewingPageIndex(prev => Math.max(0, prev - 1));
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                setViewingPageIndex(prev => Math.min(totalPages - 1, prev + 1));
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [showBBoxEditor, totalPages]);

    const runOcrForPage = useCallback(async (pageNum) => {
        const imageUrl = getImageUrl(pageNum);
        const boxes = detectedPages[pageNum] || [];
        if (!imageUrl || boxes.length === 0 || !selectedOcrModel) return null;
        const imageData = await toDataUrl(imageUrl);
        return runLocalRecognition(imageData, boxes, selectedOcrModel);
    }, [detectedPages, getImageUrl, selectedOcrModel, toDataUrl]);

    const handleRecognizeCurrentPage = useCallback(async () => {
        if (datasetMode || ocrProcessing || ocrProcessingAll) return;
        if (!selectedOcrModel || !isCurrentDetected) return;

        setOcrProcessing(true);
        setOcrError(null);
        try {
            const result = await runOcrForPage(currentPageNum);
            if (!result) {
                setOcrError('No detected boxes available for this page.');
                return;
            }
            if (!result.success) {
                setOcrError(result.error || 'Recognition failed');
                return;
            }

            setRecognizedByPage((prev) => ({ ...prev, [currentPageNum]: result.results || [] }));
            setLlmDiffByPage((prev) => ({ ...prev, [currentPageNum]: {} }));  // fresh OCR → drop stale diff
            setOcrDevice(result.device || null);
            setOcrTimeMs(result.processing_time_ms ?? null);
        } catch (err) {
            setOcrError(err.message || 'Recognition failed');
        } finally {
            setOcrProcessing(false);
        }
    }, [currentPageNum, datasetMode, isCurrentDetected, ocrProcessing, ocrProcessingAll, runOcrForPage, selectedOcrModel]);

    const handleRecognizeAllPages = useCallback(async () => {
        if (datasetMode || ocrProcessingAll || ocrProcessing) return;
        if (!selectedOcrModel) return;

        setOcrProcessingAll(true);
        setOcrError(null);
        cancelOcrRef.current = false;

        try {
            for (let i = 0; i < availablePages.length; i++) {
                if (cancelOcrRef.current) break;
                const pageNum = availablePages[i];
                if (!(pageNum in detectedPages)) continue;

                setOcrProgress({ current: i + 1, total: availablePages.length });
                const result = await runOcrForPage(pageNum);
                if (!result) continue;
                if (!result.success) {
                    setOcrError(`Page ${pageNum}: ${result.error || 'Recognition failed'}`);
                    break;
                }

                setRecognizedByPage((prev) => ({ ...prev, [pageNum]: result.results || [] }));
                setLlmDiffByPage((prev) => ({ ...prev, [pageNum]: {} }));  // fresh OCR → drop stale diff
                setOcrDevice(result.device || null);
                await new Promise((resolve) => setTimeout(resolve, 100));
            }
        } catch (err) {
            setOcrError(err.message || 'Batch recognition failed');
        } finally {
            setOcrProcessingAll(false);
            setOcrProgress(null);
        }
    }, [availablePages, datasetMode, detectedPages, ocrProcessing, ocrProcessingAll, runOcrForPage, selectedOcrModel]);

    const handleSaveTranscript = useCallback(async () => {
        if (recognizedCount === 0) return;

        const transcriptByPage = {};
        const transcriptImages = {};

        for (let i = 0; i < availablePages.length; i += 1) {
            const p = availablePages[i];
            const byIndex = {};
            (recognizedByPage[p] || []).forEach((r) => {
                byIndex[r.box_index] = r.text || '';
            });
            const sorted = (detectedPages[p] || [])
                .map((poly, idx) => {
                    const byY = [...poly].sort((a, b) => a[1] - b[1]);
                    return { idx, y: (byY[0][1] + byY[1][1]) / 2 };
                })
                .sort((a, b) => a.y - b.y)
                .map((item) => byIndex[item.idx] || '');

            const pageText = sorted.join('\n').trim();
            if (!pageText) continue;

            const pageKey = String(p);
            transcriptByPage[pageKey] = pageText;

            const imageUrl = getImageUrl(p);
            if (imageUrl) {
                try {
                    const dataUrl = await toDataUrl(imageUrl);
                    if (typeof dataUrl === 'string' && dataUrl.startsWith('data:')) {
                        transcriptImages[pageKey] = dataUrl;
                    }
                } catch {
                }
            }
        }

        if (Object.keys(transcriptByPage).length === 0) {
            setOcrError('No transcript pages to save.');
            return;
        }

        const signature = JSON.stringify(transcriptByPage);
        if (lastSavedSignatureRef.current === signature) {
            window.alert('Already saved — this exact transcript is already in My Files.');
            return;
        }

        const defaultName = bookName && bookName !== 'transcript' ? bookName : 'My Transcript';
        const chosenName = window.prompt('Name this transcript:', defaultName);
        if (chosenName === null) return;                       // user cancelled
        const finalName = chosenName.trim() || defaultName;

        try {
            const overview = await fetchStorageOverview();
            const existingNames = (overview.transcripts || []).map((t) => (t.name || '').toLowerCase());
            if (existingNames.includes(finalName.toLowerCase())) {
                const proceed = window.confirm(`A transcript named "${finalName}" already exists in My Files. Save anyway?`);
                if (!proceed) return;
            }
        } catch {
        }

        try {
            await saveTranscriptSession(
                transcriptByPage,
                'layout-aware ocr',
                'recognition',
                transcriptImages,
                finalName,
                {
                    preprocessing,
                    ocr_provider: 'local',
                    ocr_model: selectedOcrModel,
                    layout_model: selectedLayoutModel,
                    detection_model: selectedDetModel,
                    llm_postprocess: lastLlmRun || { used: false },
                },
            );
            lastSavedSignatureRef.current = signature;
            window.alert(`Saved "${finalName}" to My Files.`);
        } catch (err) {
            setOcrError(err.message || 'Failed to save transcript session');
        }
    }, [availablePages, bookName, detectedPages, getImageUrl, recognizedByPage, recognizedCount, selectedDetModel, selectedLayoutModel, selectedOcrModel, toDataUrl, preprocessing, lastLlmRun]);

    const updateRecognizedText = useCallback((lineIndex, value) => {
        const boxIndex = sortedBoxIndices[lineIndex];
        if (boxIndex === undefined) return;

        setRecognizedByPage((prev) => {
            const existing = prev[currentPageNum] || [];
            const next = [...existing];
            const foundIndex = next.findIndex((r) => r.box_index === boxIndex);
            if (foundIndex >= 0) {
                next[foundIndex] = { ...next[foundIndex], text: value };
            } else {
                next.push({ box_index: boxIndex, text: value });
            }
            return { ...prev, [currentPageNum]: next };
        });
    }, [currentPageNum, sortedBoxIndices]);

    // Autogrow each line textarea so the transcript reads as one page.
    const autoGrowLine = useCallback((el) => {
        if (!el) return;
        el.style.height = 'auto';
        el.style.height = `${el.scrollHeight}px`;
    }, []);

    // Fetch providers/templates only when the feature is first switched on.
    useEffect(() => {
        if (!llmEnabled || llmProviders.length > 0) return;
        getLLMProviders()
            .then((data) => {
                const provs = data.providers || [];
                setLlmProviders(provs);
                const firstEnabled = provs.find((p) => p.enabled);
                if (firstEnabled) {
                    setLlmProvider(firstEnabled.id);
                    if (firstEnabled.default_model) setLlmModel(firstEnabled.default_model);
                }
            })
            .catch(() => setLlmError('Could not load LLM providers.'));
        getLLMTemplates()
            .then((data) => setLlmTemplates(data.templates || []))
            .catch(() => { /* fallback <option>s are hardcoded in the UI */ });
    }, [llmEnabled, llmProviders.length]);

    const llmProviderMeta = useMemo(
        () => llmProviders.find((p) => p.id === llmProvider) || null,
        [llmProviders, llmProvider],
    );

    // Switching provider resets the model to that provider's default.
    const handleLlmProviderChange = useCallback((nextId) => {
        setLlmProvider(nextId);
        setLlmError(null);
        const meta = llmProviders.find((p) => p.id === nextId);
        if (meta && meta.default_model) setLlmModel(meta.default_model);
        else if (meta && meta.models && meta.models.length > 0) setLlmModel(meta.models[0].id);
    }, [llmProviders]);

    // Same top-to-bottom order as export/save, so polished text maps back onto
    // the same boxes.
    const buildSortedRecognition = useCallback((pageNum) => {
        const byIndex = {};
        (recognizedByPage[pageNum] || []).forEach((r) => {
            byIndex[r.box_index] = r.text || '';
        });
        const orderedIdx = (detectedPages[pageNum] || [])
            .map((poly, idx) => {
                const byY = [...poly].sort((a, b) => a[1] - b[1]);
                return { idx, y: (byY[0][1] + byY[1][1]) / 2 };
            })
            .sort((a, b) => a.y - b.y)
            .map((item) => item.idx);
        return { orderedIdx, lines: orderedIdx.map((i) => byIndex[i] || '') };
    }, [recognizedByPage, detectedPages]);

    // Whole page to the LLM, then split back onto the boxes in order. Extra
    // returned lines get appended to the last box rather than dropped.
    const polishPage = useCallback(async (pageNum, provider, apiKey) => {
        const { orderedIdx, lines } = buildSortedRecognition(pageNum);
        const text = lines.join('\n').trim();
        if (!text) return false;

        const res = await postProcessWithLLMProvider(
            provider, apiKey, text, llmModel, llmTemplate,
        );
        if (!res.success) {
            throw new Error(res.error || 'Post-processing failed');
        }
        const cleaned = (res.processed_text || '').split('\n');

        // Remember what changed so the UI can highlight it.
        const diff = {};
        orderedIdx.forEach((boxIdx, i) => {
            const before = lines[i] || '';
            const after = cleaned[i] !== undefined ? cleaned[i] : '';
            if (before.trim() !== after.trim()) {
                diff[boxIdx] = before;
            }
        });

        setRecognizedByPage((prev) => {
            const rows = orderedIdx.map((boxIdx, i) => ({
                box_index: boxIdx,
                text: cleaned[i] !== undefined ? cleaned[i] : '',
            }));
            if (cleaned.length > orderedIdx.length && rows.length > 0) {
                rows[rows.length - 1] = {
                    ...rows[rows.length - 1],
                    text: cleaned.slice(orderedIdx.length - 1).join('\n'),
                };
            }
            return { ...prev, [pageNum]: rows };
        });
        setLlmDiffByPage((prev) => ({ ...prev, [pageNum]: diff }));
        setShowLlmDiff(true);
        setLastLlmRun({ used: true, provider, model: llmModel, template: llmTemplate });
        return true;
    }, [buildSortedRecognition, llmModel, llmTemplate]);

    // The local corrector runs on our own GPU, so it's the only provider whose
    // weights we host — and its gated base model is a one-time ~8 GB pull.
    const isLocalLlm = llmProviderMeta ? llmProviderMeta.requires_key === false : false;

    // Probe the HF cache so we can warn once instead of looking hung. Same
    // contract as the detection-model probe above.
    useEffect(() => {
        if (!isLocalLlm) { setLlmModelReady(null); return undefined; }
        let active = true;
        fetch(`${API_BASE}/api/llm/local-status`)
            .then((r) => r.json())
            .then((d) => { if (active) setLlmModelReady(!!d.models_ready); })
            .catch(() => { if (active) setLlmModelReady(null); });
        return () => { active = false; };
    }, [isLocalLlm]);

    // Returns the key, '' for keyless local providers, or null if invalid.
    // Check `=== null` — '' is a valid result.
    const validateLlmReady = useCallback(() => {
        if (!llmProviderMeta || !llmProviderMeta.enabled) {
            setLlmError('Selected provider is not available yet.');
            return null;
        }
        if (llmProviderMeta.requires_key === false) {
            return '';
        }
        const key = (llmApiKeys[llmProvider] || '').trim();
        if (!key) {
            setLlmError(`Enter an API key for ${llmProviderMeta.name}.`);
            return null;
        }
        return key;
    }, [llmProviderMeta, llmApiKeys, llmProvider]);

    const handlePolishPage = useCallback(async () => {
        if (llmProcessing) return;
        const key = validateLlmReady();
        if (key === null) return;
        if (!(currentPageNum in recognizedByPage)) {
            setLlmError('Run OCR on this page first.');
            return;
        }
        setLlmProcessing(true);
        setLlmError(null);
        if (isLocalLlm && llmModelReady === false) setDownloadingLlmModel(true);
        try {
            const did = await polishPage(currentPageNum, llmProvider, key);
            if (!did) setLlmError('No recognized text on this page to polish.');
            // It answered, so the weights are on disk now — don't warn again.
            else if (isLocalLlm) setLlmModelReady(true);
        } catch (err) {
            setLlmError(err.message || 'Post-processing failed');
        } finally {
            setLlmProcessing(false);
            setDownloadingLlmModel(false);
        }
    }, [llmProcessing, validateLlmReady, currentPageNum, recognizedByPage, polishPage,
        llmProvider, isLocalLlm, llmModelReady]);

    const handlePolishAllPages = useCallback(async () => {
        if (llmProcessing) return;
        const key = validateLlmReady();
        if (key === null) return;
        const pages = availablePages.filter((p) => p in recognizedByPage);
        if (pages.length === 0) {
            setLlmError('Run OCR on at least one page first.');
            return;
        }
        setLlmProcessing(true);
        setLlmError(null);
        if (isLocalLlm && llmModelReady === false) setDownloadingLlmModel(true);
        try {
            for (let i = 0; i < pages.length; i += 1) {
                setLlmProgress({ current: i + 1, total: pages.length });
                // Sequential on purpose — provider rate limits.
                // eslint-disable-next-line no-await-in-loop
                await polishPage(pages[i], llmProvider, key);
                // First page answering means the download is done; drop the
                // banner so it doesn't sit there for the whole batch.
                if (isLocalLlm) { setLlmModelReady(true); setDownloadingLlmModel(false); }
            }
        } catch (err) {
            setLlmError(err.message || 'Post-processing failed');
        } finally {
            setLlmProcessing(false);
            setLlmProgress(null);
            setDownloadingLlmModel(false);
        }
    }, [llmProcessing, validateLlmReady, availablePages, recognizedByPage, polishPage,
        llmProvider, isLocalLlm, llmModelReady]);

    useEffect(() => {
        if (!isCurrentDetected) return;
        // Seed once per page; handleBBoxSave owns alignment after that.
        if (seededPages.current.has(String(currentPageNum))) return;
        // Rows restored from the parent cache may contain user edits — never
        // stomp them with the raw transcript.
        if ((alignmentByPage[currentPageNum] || []).length > 0) {
            seededPages.current.add(String(currentPageNum));
            return;
        }

        const transcriptKey = resolveTranscriptKey(transcript, currentPageNum);
        const transcriptLines = transcriptKey ? (transcript[transcriptKey] || []) : [];

        // One row per transcript line. boxIndex is null when lines outnumber boxes.
        const initialRows = transcriptLines.map((line, index) => ({
            id: `${currentPageNum}-${index}-${nextRowId()}`,
            text: line,
            boxIndex: index < sortedBoxIndices.length ? sortedBoxIndices[index] : null,
        }));

        seededPages.current.add(String(currentPageNum));
        setAlignmentByPage((prev) => ({
            ...prev,
            [currentPageNum]: initialRows,
        }));
    }, [isCurrentDetected, currentPageNum, sortedBoxIndices, transcript, alignmentByPage]);

    // Clear hover on page change.
    useEffect(() => {
        setHoveredBoxIndex(null);
    }, [viewingPageIndex]);

    const updateAlignmentRows = useCallback((pageNum, updater) => {
        setAlignmentByPage((prev) => {
            const oldRows = prev[pageNum] || [];
            const nextRows = updater(oldRows);
            return { ...prev, [pageNum]: nextRows };
        });
    }, []);

    const updateLineText = useCallback((lineIndex, value) => {
        updateAlignmentRows(currentPageNum, (rows) => rows.map((row, idx) => (
            idx === lineIndex ? { ...row, text: value } : row
        )));
    }, [currentPageNum, updateAlignmentRows]);

    // Re-pair text with the current box order — needed after box edits.
    const realignCurrentPage = useCallback(() => {
        setAlignmentByPage(prev => {
            const existingRows = prev[currentPageNum] || [];
            const texts = existingRows.map(r => r.text);
            const newRows = texts.map((text, i) => ({
                id: `${currentPageNum}-${i}-${nextRowId()}`,
                text,
                boxIndex: i < sortedBoxIndices.length ? sortedBoxIndices[i] : null,
            }));
            return { ...prev, [currentPageNum]: newRows };
        });
    }, [currentPageNum, sortedBoxIndices]);

    const assignBoxToLine = useCallback((lineIndex, boxIndexValue) => {
        const parsed = boxIndexValue === '' ? null : Number(boxIndexValue);
        updateAlignmentRows(currentPageNum, (rows) => rows.map((row, idx) => (
            idx === lineIndex ? { ...row, boxIndex: Number.isNaN(parsed) ? null : parsed } : row
        )));
    }, [currentPageNum, updateAlignmentRows]);

    const moveLine = useCallback((lineIndex, direction) => {
        updateAlignmentRows(currentPageNum, (rows) => {
            const targetIndex = lineIndex + direction;
            if (targetIndex < 0 || targetIndex >= rows.length) return rows;
            const next = [...rows];
            const temp = next[lineIndex];
            next[lineIndex] = next[targetIndex];
            next[targetIndex] = temp;
            return next;
        });
    }, [currentPageNum, updateAlignmentRows]);

    const mergeLineWithNext = useCallback((lineIndex) => {
        updateAlignmentRows(currentPageNum, (rows) => {
            if (lineIndex < 0 || lineIndex >= rows.length - 1) return rows;
            const current = rows[lineIndex];
            const next = rows[lineIndex + 1];
            const merged = {
                ...current,
                text: `${current.text} ${next.text}`.trim(),
                boxIndex: current.boxIndex ?? next.boxIndex ?? null,
            };
            return [...rows.slice(0, lineIndex), merged, ...rows.slice(lineIndex + 2)];
        });
    }, [currentPageNum, updateAlignmentRows]);

    const splitLine = useCallback((lineIndex) => {
        updateAlignmentRows(currentPageNum, (rows) => {
            if (lineIndex < 0 || lineIndex >= rows.length) return rows;
            const row = rows[lineIndex];
            const words = row.text.split(/\s+/).filter(Boolean);
            if (words.length < 2) return rows;

            const splitAt = Math.ceil(words.length / 2);
            const first = words.slice(0, splitAt).join(' ');
            const second = words.slice(splitAt).join(' ');

            const firstRow = { ...row, text: first };
            const secondRow = {
                id: `${row.id}-split-${nextRowId()}`,
                text: second,
                boxIndex: null,
            };

            return [...rows.slice(0, lineIndex), firstRow, secondRow, ...rows.slice(lineIndex + 1)];
        });
    }, [currentPageNum, updateAlignmentRows]);

    // Insert a blank unassigned line. atIndex is clamped, so "at top",
    // "below row N" and "append" all go through here.
    const insertLineAt = useCallback((atIndex) => {
        updateAlignmentRows(currentPageNum, (rows) => {
            const at = Math.max(0, Math.min(atIndex, rows.length));
            const newRow = {
                id: `${currentPageNum}-ins-${nextRowId()}`,
                text: '',
                boxIndex: null,
            };
            return [...rows.slice(0, at), newRow, ...rows.slice(at)];
        });
    }, [currentPageNum, updateAlignmentRows]);

    // ── Update one tuning param ──
    const updateParam = useCallback((key, value) => {
        setTuningParams(prev => ({ ...prev, [key]: value }));
    }, []);

    // The overlay's viewBox must be in source-image pixels, same space as the
    // polygons the detector returns.
    useEffect(() => {
        if (!currentImageUrl) {
            setImageNaturalSize({ w: 0, h: 0 });
            return;
        }
        let cancelled = false;
        const probe = new Image();
        probe.onload = () => {
            if (cancelled) return;
            setImageNaturalSize({ w: probe.naturalWidth, h: probe.naturalHeight });
        };
        probe.onerror = () => { if (!cancelled) setImageNaturalSize({ w: 0, h: 0 }); };
        probe.src = currentImageUrl;
        return () => { cancelled = true; };
    }, [currentImageUrl]);

    // Reset zoom/pan on page change.
    useEffect(() => {
        setImageLoaded(false);
        setHoveredBoxIndex(null);
    }, [viewingPageIndex]);

    // SVG overlay, not canvas — canvas meant a rebuild + base64 encode on
    // every mouse move.
    const displayUrl = currentImageUrl;

    // Hovering a box on the image scrolls its row into view. The microtask
    // keeps rapid hover changes from fighting.
    useEffect(() => {
        if (hoveredBoxIndex == null) return;
        const el = lineRefsRef.current.get(hoveredBoxIndex);
        if (!el) return;
        // 'nearest' avoids yanking a row that's already visible.
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, [hoveredBoxIndex]);

    // ── Navigation ──
    const goToPage = useCallback((index) => {
        if (index >= 0 && index < totalPages) {
            setViewingPageIndex(index);
            setError(null);
            setWarning(null);
            setProcessingTime(null);
        }
    }, [totalPages]);

    // ── Image URL -> blob ──
    const urlToBlob = useCallback(async (url) => {
        if (url.startsWith('data:')) {
            const [header, b64] = url.split(',');
            const mime = header.match(/:(.*?);/)?.[1] || 'image/png';
            const bin = atob(b64);
            const u8 = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
            return new Blob([u8], { type: mime });
        }
        const r = await fetch(url);
        return r.blob();
    }, []);

    // ── Build the detection FormData ──
    const buildFormData = useCallback((blob) => {
        const fd = new FormData();
        fd.append('image', blob, 'page.png');
        fd.append('use_gpu', 'true');
        fd.append('layout_model', selectedLayoutModel);
        fd.append('det_model', selectedDetModel);
        fd.append('region_padding', String(tuningParams.region_padding));
        fd.append('layout_expand', String(tuningParams.layout_expand));
        fd.append('score_thresh', String(tuningParams.score_thresh));
        fd.append('upscale_min_h', String(tuningParams.upscale_min_h));
        fd.append('nms_iou_thresh', String(tuningParams.nms_iou_thresh));
        fd.append('gap_multiplier', String(tuningParams.gap_multiplier));
        return fd;
    }, [selectedLayoutModel, selectedDetModel, tuningParams]);

    // PaddleOCR lazily downloads several GB on first detection — probe the
    // cache so we can warn once instead of looking hung.
    useEffect(() => {
        let active = true;
        fetch(`${API_BASE}/api/detect/models-status?use_gpu=true`)
            .then((r) => r.json())
            .then((d) => { if (active) setModelsReady(!!d.models_ready); })
            .catch(() => { if (active) setModelsReady(null); });
        return () => { active = false; };
    }, []);

    // ── Single page detection ──
    const handleDetect = async () => {
        if (!currentImageUrl) return;
        setLoading(true);
        setError(null);
        setWarning(null);
        setProcessingTime(null);
        // Warn about the download if the models aren't cached.
        if (modelsReady === false) setDownloadingModels(true);

        try {
            const blob = await urlToBlob(currentImageUrl);

            const formData = buildFormData(blob);

            const response = await fetch(`${API_BASE}/api/detect/layout-aware-lines`, {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();

            if (data.error) {
                setError(data.error);
            } else {
                setDetectedPages(prev => ({ ...prev, [currentPageNum]: data.lines || [] }));
                setProcessingTime(data.processing_time_ms);
                if (data.warning) setWarning(data.warning);
                setModelsReady(true);  // a successful detection means models are cached now
            }
        } catch (err) {
            console.error('[handleDetect] error', err);
            setError(`Request failed: ${err.message}`);
        } finally {
            setLoading(false);
            setDownloadingModels(false);
        }
    };

    // ── Process all pages, one at a time (memory) ──
    const handleDetectAll = async () => {
        if (processingAll || availablePages.length === 0) return;
        setProcessingAll(true);
        setError(null);
        setWarning(null);
        if (modelsReady === false) setDownloadingModels(true);

        const total = availablePages.length;
        let lastWarning = null;
        // Snapshot to dodge a stale closure. Keys stay strings — Number()
        // turns "3_left" into NaN, which would silently re-detect and wipe edits.
        const alreadyDetected = new Set(Object.keys(detectedPages));

        try {
            cancelProcessingRef.current = false;
            for (let i = 0; i < total; i++) {
                if (cancelProcessingRef.current) break;
                const pageNum = availablePages[i];
                setProcessAllProgress({ current: i + 1, total });
                // Don't force viewingPageIndex — let the user browse meanwhile.

                if (alreadyDetected.has(String(pageNum))) continue;

                const imageUrl = getImageUrl(pageNum);
                if (!imageUrl) continue;

                const blob = await urlToBlob(imageUrl);
                const formData = buildFormData(blob);

                const response = await fetch(`${API_BASE}/api/detect/layout-aware-lines`, {
                    method: 'POST',
                    body: formData,
                });
                const data = await response.json();

                if (data.error) {
                    setError(`Page ${pageNum}: ${data.error}`);
                    break;
                }

                setDetectedPages(prev => ({ ...prev, [pageNum]: data.lines || [] }));
                alreadyDetected.add(String(pageNum));
                if (data.warning) lastWarning = data.warning;
                // First page worked, so the models are cached now.
                setModelsReady(true);
                setDownloadingModels(false);

                // Breathing room for GC and a UI repaint.
                await new Promise(r => setTimeout(r, 100));
            }

            if (lastWarning) setWarning(lastWarning);
        } catch (err) {
            console.error('[handleDetectAll] error', err);
            setError(`Batch processing failed: ${err.message}`);
        } finally {
            setProcessingAll(false);
            setProcessAllProgress(null);
            setDownloadingModels(false);
        }
    };

    const isAnyLoading = loading || processingAll;

    // ── Left sidebar ──
    const leftSidebar = (
        <div className="h-full overflow-y-auto space-y-2 pr-0.5 scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-transparent">
            {/* Parameters Card */}
            <div className="bg-white/95 backdrop-blur-sm rounded-xl border border-gray-200/80 shadow-sm shrink-0">
                <div className="px-3 py-2 bg-gradient-to-r from-teal-50/80 to-white border-b border-gray-100">
                    <h3 className="font-semibold text-gray-800 flex items-center gap-2 text-sm">
                        <Layers size={16} className="text-teal-600" />
                        Parameters
                    </h3>
                </div>
                <div className="p-3 space-y-3">
                    {/* GPU Required Info */}
                    <div className="flex items-start gap-2 px-2.5 py-2 bg-blue-50 rounded-lg border border-blue-200 text-xs text-blue-700">
                        <Info size={14} className="shrink-0 mt-0.5 text-blue-500" />
                        <div>
                            <span className="font-semibold">GPU Required</span>
                            <p className="text-blue-600 mt-0.5">Models require GPU for stable inference.</p>
                        </div>
                    </div>

                    {/* Layout Model Dropdown */}
                    <div>
                        <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5 block">Layout Model</label>
                        <div className="relative">
                            <select
                                value={selectedLayoutModel}
                                onChange={(e) => setSelectedLayoutModel(e.target.value)}
                                disabled={loading}
                                className="w-full appearance-none px-2.5 py-1.5 bg-gray-50 rounded-lg text-xs text-gray-700 font-medium border border-gray-200 focus:border-teal-400 focus:ring-1 focus:ring-teal-200 outline-none pr-7 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {LAYOUT_MODELS.map(m => (
                                    <option key={m.id} value={m.id}>{m.name}</option>
                                ))}
                            </select>
                            <ChevronDown size={12} className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
                        </div>
                    </div>

                    {/* Detection Model Dropdown */}
                    <div>
                        <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5 block">Detection Model</label>
                        <div className="relative">
                            <select
                                value={selectedDetModel}
                                onChange={(e) => setSelectedDetModel(e.target.value)}
                                disabled={loading}
                                className="w-full appearance-none px-2.5 py-1.5 bg-gray-50 rounded-lg text-xs text-gray-700 font-medium border border-gray-200 focus:border-teal-400 focus:ring-1 focus:ring-teal-200 outline-none pr-7 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {DETECTION_MODELS.map(m => (
                                    <option key={m.id} value={m.id}>{m.name} — {m.desc}</option>
                                ))}
                            </select>
                            <ChevronDown size={12} className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
                        </div>
                    </div>

                    {/* Advanced Parameters Toggle */}
                    <button
                        onClick={() => setShowAdvanced(v => !v)}
                        className="w-full flex items-center justify-between px-2.5 py-1.5 bg-gray-50 hover:bg-gray-100 rounded-lg text-xs font-semibold text-gray-600 border border-gray-200 transition-colors"
                    >
                        <span className="flex items-center gap-1.5">
                            <Info size={13} className="text-gray-500" />
                            Advanced Parameters
                        </span>
                        <ChevronDown size={12} className={`text-gray-400 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
                    </button>

                    {/* Advanced Parameters Grid */}
                    {showAdvanced && (
                        <div className="space-y-2 pt-1 pb-1 px-1 bg-gray-50/50 rounded-lg border border-gray-100">
                            {PARAM_DEFS.map(p => (
                                <div key={p.key} className="flex items-center gap-2 px-1">
                                    <label
                                        className="text-[11px] text-gray-500 font-medium w-[90px] shrink-0 truncate cursor-help"
                                        title={p.tooltip}
                                    >
                                        {p.label}
                                    </label>
                                    <input
                                        type="number"
                                        value={tuningParams[p.key]}
                                        onChange={(e) => {
                                            const v = p.type === 'int' ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
                                            if (!isNaN(v)) updateParam(p.key, v);
                                        }}
                                        min={p.min}
                                        max={p.max}
                                        step={p.step}
                                        disabled={loading}
                                        className="flex-1 min-w-0 px-2 py-1 bg-white rounded border border-gray-200 text-xs text-gray-700 font-mono focus:border-teal-400 focus:ring-1 focus:ring-teal-200 outline-none disabled:opacity-50"
                                    />
                                    {p.unit && (
                                        <span className="text-[10px] text-gray-400 w-5 shrink-0">{p.unit}</span>
                                    )}
                                </div>
                            ))}
                            <button
                                onClick={() => setTuningParams({ ...DEFAULT_PARAMS })}
                                disabled={loading}
                                className="text-[11px] text-teal-600 hover:text-teal-700 font-medium px-2 py-0.5 disabled:opacity-50"
                            >
                                Reset to defaults
                            </button>
                        </div>
                    )}

                    {/* First-time model download notice */}
                    {downloadingModels && (
                        <div className="mb-2 p-3 rounded-lg border border-amber-200 bg-amber-50 animate-fade-in">
                            <div className="flex items-start gap-2">
                                <Loader2 size={16} className="animate-spin text-amber-600 mt-0.5 flex-shrink-0" />
                                <div className="flex-1 min-w-0">
                                    <p className="text-xs font-bold text-amber-800">
                                        Setting up detection models (one-time)
                                    </p>
                                    <p className="text-[11px] text-amber-700 mt-0.5 leading-snug">
                                        Downloading PaddleOCR models on first use — about <b>15–20 min</b> depending
                                        on your connection. Please keep this tab open; this only happens once.
                                    </p>
                                    <div className="mt-2 h-1.5 w-full bg-amber-200 rounded-full overflow-hidden">
                                        <div className="h-full w-1/3 bg-amber-500 rounded-full animate-indeterminate" />
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Detect This Page Button */}
                    <button
                        onClick={handleDetect}
                        disabled={loading || !currentImageUrl}
                        className={`w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-bold text-white transition-all duration-200 ${loading || !currentImageUrl
                            ? 'bg-gray-300 cursor-not-allowed'
                            : 'bg-gradient-to-r from-teal-500 to-emerald-600 hover:from-teal-600 hover:to-emerald-700 shadow-md hover:shadow-lg active:scale-[0.98]'
                            }`}
                    >
                        {loading ? (
                            <><Loader2 size={16} className="animate-spin" /> Detecting…</>
                        ) : (
                            <><Play size={16} /> Detect This Page</>
                        )}
                    </button>

                    {/* Process All Pages Button + Cancel */}
                    <div className="flex gap-1">
                    <button
                        onClick={handleDetectAll}
                        disabled={processingAll || loading || availablePages.length === 0}
                        className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all duration-200 ${processingAll || loading || availablePages.length === 0
                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200'
                            : 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white hover:from-blue-600 hover:to-indigo-700 shadow-sm hover:shadow-md active:scale-[0.98]'
                            }`}
                    >
                        {processingAll ? (
                            <><Loader2 size={14} className="animate-spin" /> {processAllProgress?.current}/{processAllProgress?.total}…</>
                        ) : (
                            <><Layers size={14} /> All ({totalPages})</>
                        )}
                    </button>
                    {processingAll && (
                        <button
                            onClick={() => { cancelProcessingRef.current = true; }}
                            className="px-2.5 py-2 rounded-lg text-xs font-bold bg-red-100 text-red-600 hover:bg-red-200 border border-red-200 transition-colors"
                            title="Cancel batch processing"
                        >
                            ✕
                        </button>
                    )}
                    </div>

                    {/* Edit All Boxes Button — available even during batch */}
                    <button
                        onClick={() => setShowBBoxEditor(true)}
                        disabled={loading || detectedCount === 0}
                        className={`w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all duration-200 ${loading || detectedCount === 0
                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200'
                            : 'bg-gradient-to-r from-purple-500 to-pink-600 text-white hover:from-purple-600 hover:to-pink-700 shadow-sm hover:shadow-md active:scale-[0.98]'
                            }`}
                    >
                        <PenLine size={14} /> Edit All Boxes ({detectedCount})
                    </button>

                    {/* Result stats */}
                    {processingTime !== null && !loading && (
                        <div className="flex items-center justify-between text-xs px-1">
                            <span className="flex items-center gap-1 text-emerald-600 font-semibold">
                                <CheckCircle2 size={13} /> {currentLines.length} lines
                            </span>
                            <span className="text-gray-400">{(processingTime / 1000).toFixed(1)}s</span>
                        </div>
                    )}

                    {warning && (
                        <div className="flex items-start gap-1.5 text-xs text-amber-600 bg-amber-50 rounded-lg px-2 py-1.5 border border-amber-200">
                            <AlertTriangle size={13} className="shrink-0 mt-0.5" /> {warning}
                        </div>
                    )}
                    {error && (
                        <div className="flex items-start gap-1.5 text-xs text-red-600 bg-red-50 rounded-lg px-2 py-1.5 border border-red-200">
                            <AlertTriangle size={13} className="shrink-0 mt-0.5" /> {error}
                        </div>
                    )}
                </div>
            </div>

            {/* Page Thumbnails */}
            <div className="bg-white/95 backdrop-blur-sm rounded-xl border border-gray-200/80 shadow-sm overflow-hidden">
                <div className="px-3 py-2 bg-gradient-to-r from-teal-50/80 to-white border-b border-gray-100 flex items-center justify-between shrink-0">
                    <h3 className="font-semibold text-gray-800 flex items-center gap-2 text-sm">
                        <ImageIcon size={16} className="text-teal-600" />
                        Pages
                    </h3>
                    <span className="text-[10px] text-gray-500 bg-gray-100 px-1.5 py-0.5 rounded-full font-medium">
                        {detectedCount}/{totalPages}
                    </span>
                </div>
                <div className="p-2">
                    <div className="grid grid-cols-2 gap-1.5">
                        {availablePages.map((pageNum, idx) => (
                            <div key={pageNum} className="relative">
                                <ThumbnailItem
                                    image={pages?.find(p => p.pageNumber === pageNum) || null}
                                    processedSrc={processedImages?.[pageNum] || null}
                                    index={idx}
                                    isActive={idx === viewingPageIndex}
                                    isDetected={pageNum in detectedPages}
                                    onClick={() => goToPage(idx)}
                                    pageLabel={shortPageLabel(pageNum)}
                                />
                                {/* Edited indicator badge */}
                                {editedPages.has(String(pageNum)) && (
                                    <span className="absolute top-1 left-1 px-1 py-0.5 bg-amber-500 text-white text-[9px] font-bold rounded shadow-sm leading-none">
                                        edited
                                    </span>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );

    // ── BBox editor ──
    const bboxEditorPages = useMemo(() =>
        availablePages
            .filter(pageNum => pageNum in detectedPages)
            .map(pageNum => ({
                pageNumber: pageNum,
                imageSrc: getImageUrl(pageNum),
                polygons: detectedPages[pageNum] || [],
            })),
        [availablePages, detectedPages, getImageUrl]
    );


    const handleBBoxSave = useCallback((results) => {
        // results: { [pageNumber]: polygon[] }
        setDetectedPages(prev => ({ ...prev, ...results }));

        // Re-sort the new polygons top-to-bottom, then re-pair them with the
        // existing text lines by position. Keeps manual text edits intact while
        // fixing the ordering.
        setAlignmentByPage(prevAlignment => {
            const next = { ...prevAlignment };

            Object.entries(results).forEach(([pageNum, newPolygons]) => {
                // Keep the string key: Number("3_left") is NaN.


                const sortedIdxs = newPolygons
                    .map((poly, idx) => {
                        const byY = [...poly].sort((a, b) => a[1] - b[1]);
                        const topMidY = (byY[0][1] + byY[1][1]) / 2;
                        return { idx, topMidY };
                    })
                    .sort((a, b) => a.topMidY - b.topMidY)
                    .map(item => item.idx);

                // Prefer the user's edited rows over the raw transcript.
                const existingRows = prevAlignment[pageNum];
                let texts;
                if (existingRows && existingRows.length > 0) {
                    texts = existingRows.map(r => r.text);
                } else {
                    const tk = resolveTranscriptKey(transcript, pageNum);
                    texts = tk ? (transcript[tk] || []) : [];
                }

                // One row per text line — the panel is text-driven, so surplus
                // boxes get no row and surplus texts get boxIndex=null.
                const newRows = texts.map((text, i) => ({
                    id: `${pageNum}-${i}-${nextRowId()}`,
                    text,
                    boxIndex: i < sortedIdxs.length ? sortedIdxs[i] : null,
                }));

                next[pageNum] = newRows;
                // Mark seeded so the seed effect leaves this alone.
                seededPages.current.add(String(pageNum));
            });

            return next;
        });

        setEditedPages(prev => {
            const next = new Set(prev);
            Object.keys(results).forEach(k => next.add(String(k)));
            return next;
        });
        setShowBBoxEditor(false);
    }, [transcript]);

    // ── Center panel: image preview ──
    const centerPanel = (
        <div className="bg-white/95 backdrop-blur-sm rounded-xl border border-gray-200/80 shadow-sm overflow-hidden flex flex-col h-full relative">
            {/* Navigation header */}
            <div className="px-3 py-2 bg-gradient-to-r from-teal-50/80 to-white border-b border-gray-100 flex items-center justify-between shrink-0">
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => goToPage(viewingPageIndex - 1)}
                        disabled={viewingPageIndex === 0}
                        className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    >
                        <ChevronLeft size={18} />
                    </button>
                    <span className="font-semibold text-gray-800 text-sm min-w-[80px] text-center">
                        {shortPageLabel(currentPageNum)} / {totalPages}
                    </span>
                    <button
                        onClick={() => goToPage(viewingPageIndex + 1)}
                        disabled={viewingPageIndex >= totalPages - 1}
                        className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    >
                        <ChevronRight size={18} />
                    </button>
                </div>

                {/* Status — simplified, no per-page edit button here anymore */}
                <div className="flex items-center gap-2">
                    {isAnyLoading && (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-teal-50 text-teal-600 text-xs font-medium rounded-full animate-pulse">
                            <Loader2 size={12} className="animate-spin" />
                            {processingAll ? `batch: ${processAllProgress?.current}/${processAllProgress?.total}` : 'Detecting…'}
                        </span>
                    )}
                    {isCurrentDetected && !loading && (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-emerald-50 text-emerald-600 text-xs font-medium rounded-full">
                            <CheckCircle2 size={12} /> {currentLines.length} lines
                            {editedPages.has(String(currentPageNum)) && (
                                <span className="ml-1 text-[9px] bg-amber-100 text-amber-700 px-1 rounded font-semibold">edited</span>
                            )}
                        </span>
                    )}
                </div>
            </div>

            {/* Scrollable image container — actual size, scroll if too large */}
            <div className="flex-1 overflow-auto bg-gradient-to-br from-gray-100 to-gray-50 min-h-0">
                {!currentImageUrl && (
                    <div className="h-full flex items-center justify-center text-gray-400">
                        <div className="text-center">
                            <ImageIcon size={32} className="mx-auto mb-2 opacity-50" />
                            <p className="text-sm">No image selected</p>
                        </div>
                    </div>
                )}

                {currentImageUrl && (
                    <div className="p-4 flex items-center justify-center min-h-full">
                        <div ref={imageWrapRef} className="relative inline-block">
                            <img
                                key={`${currentPageNum}-${displayUrl?.slice(-20)}`}
                                ref={imgRef}
                                src={displayUrl}
                                alt={`Page ${currentPageNum}`}
                                className={`rounded-lg shadow-xl select-none block ${imageLoaded ? 'opacity-100' : 'opacity-0'}`}
                                onLoad={(e) => {
                                    setImageLoaded(true);
                                    const el = e.currentTarget;
                                    setImageNaturalSize({ w: el.naturalWidth, h: el.naturalHeight });
                                }}
                                draggable={false}
                                style={{ transition: 'opacity 0.2s ease-out' }}
                            />
                            {/* SVG overlay — matches the displayed image via viewBox
                                and absolute positioning. Hover is handled by CSS +
                                a lightweight state update; no canvas re-render. */}
                            {imageLoaded && currentLines.length > 0 && imageNaturalSize.w > 0 && (
                                <svg
                                    className="absolute inset-0 w-full h-full"
                                    viewBox={`0 0 ${imageNaturalSize.w} ${imageNaturalSize.h}`}
                                    preserveAspectRatio="none"
                                    style={{ pointerEvents: 'none' }}
                                >
                                    {currentLines.map((poly, i) => {
                                        if (!poly || poly.length < 3) return null;
                                        const pts = poly.map((p) => `${p[0]},${p[1]}`).join(' ');
                                        const isHovered = hoveredBoxIndex === i;
                                        const strokeW = Math.max(2, Math.round(Math.max(imageNaturalSize.w, imageNaturalSize.h) / (isHovered ? 300 : 500)));
                                        return (
                                            <polygon
                                                key={i}
                                                points={pts}
                                                stroke={isHovered ? '#f97316' : '#22c55e'}
                                                strokeWidth={strokeW}
                                                fill={isHovered ? 'rgba(249,115,22,0.25)' : 'rgba(34,197,94,0.08)'}
                                                strokeLinejoin="round"
                                                onMouseEnter={() => setHoveredBoxIndex(i)}
                                                onMouseLeave={() => setHoveredBoxIndex(null)}
                                                style={{ pointerEvents: 'auto', cursor: 'pointer', transition: 'fill 120ms, stroke 120ms' }}
                                            />
                                        );
                                    })}
                                </svg>
                            )}
                        </div>
                    </div>
                )}
            </div>

            {/* Loading overlay — only for single-page detection; batch shows a badge in the header */}
            {loading && (
                <div className="absolute inset-0 bg-teal-500/10 backdrop-blur-sm flex items-center justify-center z-20 pointer-events-none">
                    <div className="bg-white rounded-xl shadow-lg px-5 py-3 text-center">
                        <Loader2 size={28} className="animate-spin text-teal-600 mx-auto mb-2" />
                        <p className="text-sm font-medium text-gray-700">Detecting lines…</p>
                    </div>
                </div>
            )}
            {/* Non-blocking batch progress badge */}
            {processingAll && processAllProgress && (
                <div className="absolute top-2 right-2 z-20 flex items-center gap-2 bg-blue-600/90 text-white text-xs font-semibold px-3 py-1.5 rounded-full shadow-lg backdrop-blur-sm pointer-events-none">
                    <Loader2 size={12} className="animate-spin" />
                    Processing {processAllProgress.current}/{processAllProgress.total}…
                </div>
            )}
        </div>
    );

    const alignedTranscriptByPage = useMemo(() => {
        const result = {};
        Object.keys(alignmentByPage).forEach((pageNum) => {
            const rows = alignmentByPage[pageNum] || [];
            result[pageNum] = rows.map((row) => row.text).filter((txt) => txt && txt.trim().length > 0);
        });
        return result;
    }, [alignmentByPage]);

    // ── Right panel: alignment (step 4) ──
    const rightPanel = (
        <div className="bg-white/95 backdrop-blur-sm rounded-xl border border-gray-200/80 shadow-sm overflow-hidden flex flex-col h-full">
            <div className="px-3 py-2 bg-gradient-to-r from-teal-50/80 via-white to-emerald-50/50 border-b border-gray-100/80 flex items-center justify-between shrink-0">
                <h3 className="font-bold text-gray-800 flex items-center gap-2 text-sm">
                    <div className="p-1 bg-gradient-to-br from-teal-500 to-emerald-600 rounded-lg text-white">
                        <FileText size={12} />
                    </div>
                    <span>Step 4 — Alignment</span>
                </h3>
                <span className="text-xs text-gray-500 font-medium">Page {shortPageLabel(currentPageNum)}</span>
            </div>

            <div className="px-3 py-2 border-b border-gray-100 shrink-0 flex items-center justify-between text-xs text-gray-500">
                <span>Detected: {isCurrentDetected ? `${currentLines.length} boxes` : 'Pending'}</span>
                <span>Transcript: {transcriptKeyForCurrentPage ? `${(transcript[transcriptKeyForCurrentPage] || []).length} lines` : 'Not matched'}</span>
            </div>

            <div className="flex-1 overflow-y-auto min-h-0 p-2.5 space-y-1.5 bg-gradient-to-br from-gray-50/50 to-gray-100/20">
                {!isCurrentDetected && (
                    <div className="text-xs text-gray-500 bg-white rounded-lg border border-gray-200 px-3 py-2">
                        Detect this page first to align lines.
                    </div>
                )}

                {isCurrentDetected && !transcriptKeyForCurrentPage && (
                    <div className="text-xs text-amber-700 bg-amber-50 rounded-lg border border-amber-200 px-3 py-2">
                        No transcript page matched for {shortPageLabel(currentPageNum)}.
                    </div>
                )}

                {isCurrentDetected && currentAlignmentRows.length > 0 && (
                    <button
                        onClick={() => insertLineAt(0)}
                        className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded-lg border border-dashed border-teal-300 bg-teal-50/50 text-teal-600 hover:bg-teal-100/60 text-[11px] font-semibold transition-colors"
                        title="Add a new line at the top"
                    >
                        <Plus size={12} /> Add line at top
                    </button>
                )}

                {isCurrentDetected && currentAlignmentRows.map((row, lineIndex) => {
                    const assignedBoxIdx = row.boxIndex;
                    const isHighlighted = assignedBoxIdx !== null && assignedBoxIdx !== undefined && hoveredBoxIndex === assignedBoxIdx;
                    const isUnassigned = assignedBoxIdx === null || assignedBoxIdx === undefined;

                    return (
                        <div
                            key={row.id}
                            ref={(el) => setLineRef(assignedBoxIdx, el)}
                            className={`rounded-lg border p-2.5 space-y-1.5 shadow-sm transition-colors cursor-pointer ${
                                isHighlighted
                                    ? 'bg-orange-50 border-orange-300 shadow-orange-100'
                                    : 'bg-white border-gray-200 hover:border-teal-300 hover:bg-teal-50/30'
                            }`}
                            onMouseEnter={() => {
                                if (assignedBoxIdx !== null && assignedBoxIdx !== undefined) {
                                    setHoveredBoxIndex(assignedBoxIdx);
                                }
                            }}
                            onMouseLeave={() => {
                                setHoveredBoxIndex(null);
                            }}
                        >
                            <div className="flex items-center justify-between gap-2">
                                <div className="flex items-center gap-1.5">
                                    <span className={`text-[11px] font-semibold ${isHighlighted ? 'text-orange-600' : 'text-gray-500'}`}>
                                        Line {lineIndex + 1}
                                    </span>
                                    {isUnassigned && (
                                        <span className="text-[9px] px-1 py-0.5 bg-amber-100 text-amber-600 rounded font-semibold">
                                            unassigned
                                        </span>
                                    )}
                                </div>
                                <div className="flex items-center gap-1">
                                    <button
                                        onClick={() => moveLine(lineIndex, -1)}
                                        disabled={lineIndex === 0}
                                        className="p-1 rounded border border-gray-200 text-gray-500 hover:bg-gray-50 disabled:opacity-40"
                                        title="Move line up"
                                    >
                                        <ArrowUp size={12} />
                                    </button>
                                    <button
                                        onClick={() => moveLine(lineIndex, 1)}
                                        disabled={lineIndex >= currentAlignmentRows.length - 1}
                                        className="p-1 rounded border border-gray-200 text-gray-500 hover:bg-gray-50 disabled:opacity-40"
                                        title="Move line down"
                                    >
                                        <ArrowDown size={12} />
                                    </button>
                                    <button
                                        onClick={() => mergeLineWithNext(lineIndex)}
                                        disabled={lineIndex >= currentAlignmentRows.length - 1}
                                        className="p-1 rounded border border-gray-200 text-gray-500 hover:bg-gray-50 disabled:opacity-40"
                                        title="Merge with next line"
                                    >
                                        <Link2 size={12} />
                                    </button>
                                    <button
                                        onClick={() => splitLine(lineIndex)}
                                        className="p-1 rounded border border-gray-200 text-gray-500 hover:bg-gray-50"
                                        title="Split line into two"
                                    >
                                        <Split size={12} />
                                    </button>
                                    <button
                                        onClick={() => insertLineAt(lineIndex + 1)}
                                        className="p-1 rounded border border-teal-200 bg-teal-50 text-teal-600 hover:bg-teal-100"
                                        title="Insert a new line below"
                                    >
                                        <Plus size={12} />
                                    </button>
                                </div>
                            </div>

                            <div className="flex items-center gap-2">
                                <label className="text-[11px] text-gray-500 w-14 shrink-0">Line box</label>
                                <select
                                    value={row.boxIndex ?? ''}
                                    onChange={(e) => assignBoxToLine(lineIndex, e.target.value)}
                                    className="flex-1 px-2 py-1.5 bg-gray-50 rounded border border-gray-200 text-xs text-gray-700"
                                >
                                    <option value="">Unassigned</option>
                                    {sortedBoxIndices.map((origIdx, sortedPos) => (
                                        <option key={origIdx} value={origIdx}>Box {sortedPos + 1} (row {origIdx + 1})</option>
                                    ))}
                                </select>
                            </div>

                            <textarea
                                value={row.text}
                                onChange={(e) => updateLineText(lineIndex, e.target.value)}
                                rows={2}
                                className="w-full px-2.5 py-2 bg-white border border-gray-200 rounded text-xs text-gray-700 resize-y focus:outline-none focus:ring-1 focus:ring-teal-200 focus:border-teal-400"
                            />
                        </div>
                    );
                })}

                {/* No rows yet — let the user start adding lines manually */}
                {isCurrentDetected && currentAlignmentRows.length === 0 && (
                    <div className="space-y-1.5">
                        <div className="text-xs text-gray-500 bg-white rounded-lg border border-gray-200 px-3 py-2">
                            No transcript lines for this page yet.
                        </div>
                        <button
                            onClick={() => insertLineAt(0)}
                            className="w-full flex items-center justify-center gap-1.5 py-2 rounded-lg border border-dashed border-teal-300 bg-teal-50/50 text-teal-700 hover:bg-teal-100/60 text-xs font-semibold transition-colors"
                        >
                            <Plus size={13} /> Add a line
                        </button>
                    </div>
                )}
            </div>

            <div className="px-3 py-2 border-t border-gray-100 bg-gradient-to-r from-gray-50 to-white flex items-center justify-between text-xs text-gray-500 shrink-0 gap-2">
                <span className="shrink-0">{currentAlignmentRows.length} lines · {currentAlignmentRows.filter(r => r.boxIndex !== null && r.boxIndex !== undefined).length} assigned</span>
                {currentAlignmentRows.length > 0 &&
                    currentAlignmentRows.every(r => r.boxIndex !== null && r.boxIndex !== undefined) && (
                    <span className="flex items-center gap-1 px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded-full font-semibold shrink-0">
                        <CheckCircle2 size={11} /> All matched
                    </span>
                )}
                {isCurrentDetected && currentAlignmentRows.length > 0 && (
                    <div className="ml-auto flex items-center gap-1.5 shrink-0">
                        <button
                            onClick={() => insertLineAt(currentAlignmentRows.length)}
                            title="Add a new line at the end"
                            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-teal-200 bg-teal-50 text-teal-700 hover:bg-teal-100 hover:border-teal-300 font-semibold transition-colors"
                        >
                            <Plus size={11} />
                            Add line
                        </button>
                        <button
                            onClick={realignCurrentPage}
                            title="Re-pair transcript lines with boxes in their current top-to-bottom order"
                            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-teal-200 bg-teal-50 text-teal-700 hover:bg-teal-100 hover:border-teal-300 font-semibold transition-colors"
                        >
                            <RefreshCw size={11} />
                            Realign
                        </button>
                    </div>
                )}
            </div>
        </div>
    );


    // ── Right panel: local OCR ──
    const ocrRightPanel = (
        <div className="bg-white/95 backdrop-blur-sm rounded-xl border border-gray-200/80 shadow-sm overflow-hidden flex flex-col h-full">
            <div className="px-3 py-2.5 bg-gradient-to-r from-blue-50/80 via-white to-indigo-50/50 border-b border-gray-100/80 shrink-0 flex items-center gap-2">
                <div className="p-1 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg text-white">
                    <Sparkles size={12} />
                </div>
                <h3 className="font-bold text-gray-800 text-sm">Step 4 — Local OCR</h3>
            </div>

            <div className="px-3 py-2 border-b border-gray-100 shrink-0 flex items-center justify-between text-xs text-gray-500">
                <span>Detected: {isCurrentDetected ? `${currentLines.length} boxes` : 'Pending'}</span>
                <span>Recognized: {currentRecognition.length} lines</span>
            </div>

            <div className="flex-1 overflow-y-auto min-h-0 p-3 space-y-3 bg-gradient-to-br from-gray-50/50 to-gray-100/20">
                <div className="space-y-2">
                    <label className="text-[11px] font-bold text-gray-600 uppercase tracking-wider flex items-center gap-1.5">
                        <Cpu size={12} className="text-gray-500" />
                        OCR Model
                    </label>

                    {localModelsLoading ? (
                        <div className="text-xs text-gray-500 flex items-center gap-1.5">
                            <Loader2 size={12} className="animate-spin" /> Loading local models...
                        </div>
                    ) : localModelsError ? (
                        <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg px-2 py-1.5">
                            {localModelsError}
                        </div>
                    ) : localModels.length === 0 ? (
                        <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-2 py-1.5">
                            No local weights found in backend/models/weights.
                        </div>
                    ) : (
                        <div className="relative">
                            <select
                                value={selectedOcrModel}
                                onChange={(e) => setSelectedOcrModel(e.target.value)}
                                disabled={ocrProcessing || ocrProcessingAll}
                                className="w-full appearance-none px-2.5 py-2 bg-white rounded-lg text-xs text-gray-700 font-medium border border-gray-200 focus:border-blue-400 focus:ring-1 focus:ring-blue-200 outline-none pr-7"
                            >
                                {localModels.map((model) => (
                                    <option key={model.id} value={model.id}>
                                        {model.name}
                                    </option>
                                ))}
                            </select>
                            <ChevronDown size={12} className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
                        </div>
                    )}

                    {ocrDevice && (
                        <p className="text-[10px] text-gray-500">Last run device: {ocrDevice.toUpperCase()}</p>
                    )}
                </div>

                <div className="grid grid-cols-2 gap-1.5">
                    <button
                        onClick={handleRecognizeCurrentPage}
                        disabled={!isCurrentDetected || !selectedOcrModel || ocrProcessing || ocrProcessingAll}
                        className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-bold text-white transition-all duration-200 disabled:bg-gray-300 disabled:cursor-not-allowed bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700"
                    >
                        {ocrProcessing ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
                        Run Page
                    </button>

                    <button
                        onClick={handleRecognizeAllPages}
                        disabled={!selectedOcrModel || ocrProcessing || ocrProcessingAll || detectedCount === 0}
                        className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-bold transition-all duration-200 disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed bg-gradient-to-r from-cyan-500 to-teal-600 text-white hover:from-cyan-600 hover:to-teal-700"
                    >
                        {ocrProcessingAll ? <Loader2 size={13} className="animate-spin" /> : <Layers size={13} />}
                        All Pages
                    </button>
                </div>

                {ocrProcessingAll && ocrProgress && (
                    <div className="text-[11px] text-blue-700 bg-blue-50 border border-blue-200 rounded-lg px-2 py-1.5 flex items-center justify-between">
                        <span>Recognizing {ocrProgress.current}/{ocrProgress.total}</span>
                        <button
                            onClick={() => { cancelOcrRef.current = true; }}
                            className="px-1.5 py-0.5 rounded bg-red-100 text-red-600 text-[10px] font-bold"
                        >
                            Stop
                        </button>
                    </div>
                )}

                {ocrTimeMs !== null && !ocrProcessing && (
                    <div className="text-[11px] text-gray-500">Last recognition time: {(ocrTimeMs / 1000).toFixed(2)}s</div>
                )}

                {ocrError && (
                    <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg px-2 py-1.5">
                        {ocrError}
                    </div>
                )}

                {!isCurrentDetected && (
                    <div className="text-xs text-gray-500 bg-white rounded-lg border border-gray-200 px-3 py-2">
                        Detect this page first, then run OCR with your selected model.
                    </div>
                )}

                {isCurrentDetected && sortedBoxIndices.length > 0 && (
                    // Lines flow together like a document rather than sitting in
                    // cards. Hover still drives the shared bbox highlight.
                    <div
                        className="bg-white rounded-lg border border-gray-200 shadow-inner px-3 py-3"
                        onMouseLeave={() => setHoveredBoxIndex(null)}
                    >
                        {sortedBoxIndices.map((boxIndex, lineIndex) => {
                            const isHighlighted = hoveredBoxIndex === boxIndex;
                            const changed = showLlmDiff
                                && Object.prototype.hasOwnProperty.call(currentDiff, boxIndex);
                            const beforeText = changed ? currentDiff[boxIndex] : '';
                            const afterText = currentRecognitionByIndex[boxIndex] || '';
                            const { beforeTokens, afterTokens } = changed
                                ? diffWords(beforeText, afterText)
                                : { beforeTokens: [], afterTokens: [] };
                            return (
                                <div
                                    key={`${currentPageNum}-${boxIndex}`}
                                    ref={(el) => setLineRef(boxIndex, el)}
                                    onMouseEnter={() => setHoveredBoxIndex(boxIndex)}
                                    className={`group flex items-start gap-2 -mx-2 px-2 rounded-md transition-colors ${
                                        isHighlighted
                                            ? 'bg-orange-100/70'
                                            : changed
                                                ? 'bg-amber-50/50 hover:bg-amber-100/50'
                                                : 'hover:bg-gray-50'
                                    } ${changed ? 'border-l-2 border-amber-400' : ''}`}
                                >
                                    <span
                                        className={`select-none w-6 shrink-0 text-right text-[10px] leading-6 tabular-nums transition-colors ${
                                            isHighlighted
                                                ? 'text-orange-500 font-bold'
                                                : changed
                                                    ? 'text-amber-500 font-bold'
                                                    : 'text-gray-300 group-hover:text-gray-400'
                                        }`}
                                        title={changed ? `Line ${lineIndex + 1} — edited by post-processing` : `Line ${lineIndex + 1}`}
                                    >
                                        {lineIndex + 1}
                                    </span>
                                    <div className="flex-1 min-w-0">
                                        <textarea
                                            value={afterText}
                                            onChange={(e) => {
                                                updateRecognizedText(lineIndex, e.target.value);
                                                autoGrowLine(e.target);
                                            }}
                                            ref={autoGrowLine}
                                            rows={1}
                                            spellCheck={false}
                                            className={`w-full bg-transparent border-0 outline-none resize-none overflow-hidden py-0.5 text-sm leading-6 text-gray-800 rounded focus:bg-blue-50/50 ${
                                                isHighlighted ? 'text-orange-900' : ''
                                            }`}
                                        />
                                        {changed && (
                                            <div className="mb-1 space-y-0.5 text-[11px] leading-snug border-t border-amber-100 pt-1">
                                                <div className="flex gap-1.5">
                                                    <span className="select-none text-rose-400 font-mono shrink-0" title="Before">−</span>
                                                    <span className="text-gray-400">
                                                        {beforeTokens.map((t, k) => (
                                                            <span
                                                                key={k}
                                                                className={t.removed ? 'bg-rose-100 text-rose-600 line-through rounded px-0.5' : ''}
                                                            >{t.text}{' '}</span>
                                                        ))}
                                                    </span>
                                                </div>
                                                <div className="flex gap-1.5">
                                                    <span className="select-none text-emerald-500 font-mono shrink-0" title="After">+</span>
                                                    <span className="text-gray-600">
                                                        {afterTokens.map((t, k) => (
                                                            <span
                                                                key={k}
                                                                className={t.added ? 'bg-emerald-100 text-emerald-700 rounded px-0.5' : ''}
                                                            >{t.text}{' '}</span>
                                                        ))}
                                                    </span>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            {/* ── AI Post-processing ───────────────────────────── */}
            <div className="shrink-0 border-t border-gray-100 p-3 bg-gradient-to-r from-purple-50/40 to-white space-y-2">
                <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold text-gray-600 uppercase tracking-wider flex items-center gap-1.5">
                        <div className="p-1 bg-gradient-to-br from-purple-500 to-pink-500 rounded text-white">
                            <Wand2 size={10} />
                        </div>
                        AI Post-processing
                    </span>
                    <button
                        type="button"
                        onClick={() => { setLlmEnabled((v) => !v); setLlmError(null); }}
                        title="Toggle LLM post-processing"
                        className={`relative w-9 h-5 rounded-full transition-colors shrink-0 ${llmEnabled ? 'bg-purple-500' : 'bg-gray-300'}`}
                    >
                        <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${llmEnabled ? 'translate-x-4' : ''}`} />
                    </button>
                </div>

                {llmEnabled && (
                    <div className="space-y-2">
                        <p className="text-[10px] text-gray-500 leading-tight">
                            Sends the recognized page transcript to an LLM for cleanup, then maps the result back onto the same lines.
                        </p>

                        {llmProviders.length === 0 ? (
                            <div className="text-[11px] text-gray-500 flex items-center gap-1.5">
                                <Loader2 size={12} className="animate-spin" /> Loading providers...
                            </div>
                        ) : (
                            <>
                                <div className="grid grid-cols-2 gap-1.5">
                                    <select
                                        value={llmProvider}
                                        onChange={(e) => handleLlmProviderChange(e.target.value)}
                                        disabled={llmProcessing}
                                        className="px-2 py-1.5 bg-white rounded-lg text-xs text-gray-700 border border-gray-200 focus:border-purple-400 focus:ring-1 focus:ring-purple-200 outline-none"
                                    >
                                        {llmProviders.map((p) => (
                                            <option key={p.id} value={p.id} disabled={!p.enabled}>
                                                {p.name}{p.enabled ? '' : ' (soon)'}
                                            </option>
                                        ))}
                                    </select>
                                    <select
                                        value={llmModel}
                                        onChange={(e) => setLlmModel(e.target.value)}
                                        disabled={llmProcessing || !llmProviderMeta || (llmProviderMeta.models || []).length === 0}
                                        className="px-2 py-1.5 bg-white rounded-lg text-xs text-gray-700 border border-gray-200 focus:border-purple-400 focus:ring-1 focus:ring-purple-200 outline-none disabled:bg-gray-50 disabled:text-gray-400"
                                    >
                                        {(llmProviderMeta?.models || []).length > 0 ? (
                                            llmProviderMeta.models.map((m) => (
                                                <option key={m.id} value={m.id}>{m.name}</option>
                                            ))
                                        ) : (
                                            <option value="">—</option>
                                        )}
                                    </select>
                                </div>

                                {llmProviderMeta && !llmProviderMeta.enabled ? (
                                    <div className="text-[11px] text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-2 py-1.5">
                                        {llmProviderMeta.note || 'This provider is not available yet.'}
                                    </div>
                                ) : (
                                    <>
                                        {/* The finetuned Spanish model does full cleanup only —
                                            no template choice to make, so hide the dropdown. */}
                                        {llmProvider !== 'local_es' && (
                                            <>
                                                <select
                                                    value={llmTemplate}
                                                    onChange={(e) => setLlmTemplate(e.target.value)}
                                                    disabled={llmProcessing}
                                                    className="w-full px-2 py-1.5 bg-white rounded-lg text-xs text-gray-700 border border-gray-200 focus:border-purple-400 focus:ring-1 focus:ring-purple-200 outline-none"
                                                >
                                                    {llmTemplates.length > 0 ? (
                                                        llmTemplates.map((t) => (
                                                            <option key={t.id} value={t.id}>{t.name}</option>
                                                        ))
                                                    ) : (
                                                        <>
                                                            <option value="full_cleanup">Full Cleanup</option>
                                                            <option value="spelling_correction">Spelling Correction</option>
                                                            <option value="formatting">Format & Structure</option>
                                                            <option value="historical_normalization">Historical Normalization</option>
                                                        </>
                                                    )}
                                                </select>
                                                {llmTemplates.length > 0 && (
                                                    <p className="text-[10px] text-gray-400 leading-tight">
                                                        {llmTemplates.find((t) => t.id === llmTemplate)?.description || ''}
                                                    </p>
                                                )}
                                            </>
                                        )}

                                        {llmProviderMeta?.requires_key === false ? (
                                            <p className="text-[10px] text-gray-500 bg-gray-50 border border-gray-200 rounded-lg px-2 py-1.5 leading-tight">
                                                Runs offline on the server — no API key needed.
                                                {llmProviderMeta?.note ? ` ${llmProviderMeta.note}` : ''}
                                            </p>
                                        ) : (
                                            <input
                                                type="password"
                                                value={llmApiKeys[llmProvider] || ''}
                                                onChange={(e) => setLlmApiKeys((prev) => ({ ...prev, [llmProvider]: e.target.value }))}
                                                placeholder={`${llmProviderMeta?.name || 'Provider'} API key`}
                                                autoComplete="off"
                                                className="w-full px-2 py-1.5 bg-white rounded-lg text-xs text-gray-700 border border-gray-200 focus:border-purple-400 focus:ring-1 focus:ring-purple-200 outline-none"
                                            />
                                        )}

                                        <div className="grid grid-cols-2 gap-1.5">
                                            <button
                                                onClick={handlePolishPage}
                                                disabled={llmProcessing || !(currentPageNum in recognizedByPage)}
                                                className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-bold text-white transition-all disabled:bg-gray-300 disabled:cursor-not-allowed bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600"
                                            >
                                                {llmProcessing && !llmProgress ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
                                                Polish Page
                                            </button>
                                            <button
                                                onClick={handlePolishAllPages}
                                                disabled={llmProcessing || recognizedCount === 0}
                                                className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-bold transition-all disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed bg-gradient-to-r from-fuchsia-500 to-purple-600 text-white hover:from-fuchsia-600 hover:to-purple-700"
                                            >
                                                {llmProcessing && llmProgress ? <Loader2 size={13} className="animate-spin" /> : <Layers size={13} />}
                                                Polish All
                                            </button>
                                        </div>
                                    </>
                                )}

                                {changedLineCount > 0 && (
                                    <div className="flex items-center justify-between gap-2 text-[11px] bg-amber-50 border border-amber-200 rounded-lg px-2 py-1.5">
                                        <span className="flex items-center gap-1.5 text-amber-700 font-semibold">
                                            <Wand2 size={11} />
                                            {changedLineCount} line{changedLineCount === 1 ? '' : 's'} edited on this page
                                        </span>
                                        <button
                                            type="button"
                                            onClick={() => setShowLlmDiff((v) => !v)}
                                            className="shrink-0 px-2 py-0.5 rounded border border-amber-300 text-amber-700 hover:bg-amber-100 font-semibold"
                                        >
                                            {showLlmDiff ? 'Hide changes' : 'Show changes'}
                                        </button>
                                    </div>
                                )}

                                {/* First-time base-model download notice */}
                                {downloadingLlmModel && (
                                    <div className="p-3 rounded-lg border border-amber-200 bg-amber-50 animate-fade-in">
                                        <div className="flex items-start gap-2">
                                            <Loader2 size={16} className="animate-spin text-amber-600 mt-0.5 flex-shrink-0" />
                                            <div className="flex-1 min-w-0">
                                                <p className="text-xs font-bold text-amber-800">
                                                    Setting up the local corrector (one-time)
                                                </p>
                                                <p className="text-[11px] text-amber-700 mt-0.5 leading-snug">
                                                    Downloading its base model on first use — about <b>8 GB</b>, so
                                                    expect <b>20–30 min</b> depending on your connection. Please keep
                                                    this tab open; this only happens once.
                                                </p>
                                                <div className="mt-2 h-1.5 w-full bg-amber-200 rounded-full overflow-hidden">
                                                    <div className="h-full w-1/3 bg-amber-500 rounded-full animate-indeterminate" />
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {llmProgress && (
                                    <div className="text-[11px] text-purple-700 bg-purple-50 border border-purple-200 rounded-lg px-2 py-1.5">
                                        Polishing page {llmProgress.current}/{llmProgress.total}
                                    </div>
                                )}
                                {llmError && (
                                    <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg px-2 py-1.5">
                                        {llmError}
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                )}
            </div>

            <div className="shrink-0 border-t border-gray-100 p-3 bg-gradient-to-r from-gray-50/80 to-white space-y-2">
                <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider flex items-center gap-1.5">
                    <Download size={11} /> Export Transcript
                </p>
                <button
                    disabled={recognizedCount === 0}
                    onClick={handleSaveTranscript}
                    className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg border border-cyan-200 bg-cyan-50 text-cyan-700 hover:bg-cyan-100 text-[10px] font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                >
                    Save to My Files
                </button>
                <div className="grid grid-cols-2 gap-1.5">
                    <button
                        disabled={recognizedCount === 0}
                        onClick={() => {
                            const resultsByPage = {};
                            availablePages.forEach((p) => {
                                const boxes = detectedPages[p] || [];
                                if (boxes.length === 0) return;
                                const indices = boxes
                                    .map((poly, idx) => {
                                        const byY = [...poly].sort((a, b) => a[1] - b[1]);
                                        return { idx, y: (byY[0][1] + byY[1][1]) / 2 };
                                    })
                                    .sort((a, b) => a.y - b.y)
                                    .map((item) => item.idx);
                                const byIndex = {};
                                (recognizedByPage[p] || []).forEach((r) => {
                                    byIndex[r.box_index] = r.text || '';
                                });
                                resultsByPage[p] = indices.map((idx) => byIndex[idx] || '');
                            });
                            const blob = exportCRNNResultsAsText(resultsByPage);
                            downloadBlob(blob, `layout_aware_transcript_${new Date().toISOString().slice(0, 10)}.txt`);
                        }}
                        className="flex flex-col items-center gap-1 py-2 rounded-lg border text-[10px] font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-white border-gray-200 text-gray-600 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700"
                    >
                        <FileTextIcon size={15} />
                        TXT
                    </button>
                    <button
                        disabled={recognizedCount === 0}
                        onClick={() => {
                            const resultsByPage = {};
                            availablePages.forEach((p) => {
                                if (!(p in recognizedByPage)) return;
                                const byIndex = {};
                                (recognizedByPage[p] || []).forEach((r) => { byIndex[r.box_index] = r.text || ''; });
                                const sorted = (detectedPages[p] || [])
                                    .map((poly, idx) => {
                                        const byY = [...poly].sort((a, b) => a[1] - b[1]);
                                        return { idx, y: (byY[0][1] + byY[1][1]) / 2 };
                                    })
                                    .sort((a, b) => a.y - b.y)
                                    .map((item) => byIndex[item.idx] || '');
                                resultsByPage[p] = sorted;
                            });
                            const blob = exportCRNNResultsAsJSON(resultsByPage);
                            downloadBlob(blob, `layout_aware_transcript_${new Date().toISOString().slice(0, 10)}.json`);
                        }}
                        className="flex flex-col items-center gap-1 py-2 rounded-lg border text-[10px] font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-white border-gray-200 text-gray-600 hover:border-emerald-300 hover:bg-emerald-50 hover:text-emerald-700"
                    >
                        <FileJson size={15} />
                        JSON
                    </button>
                    <button
                        disabled={recognizedCount === 0}
                        onClick={() => {
                            const rows = ['page,line_index,text'];
                            availablePages.forEach((p) => {
                                const byIndex = {};
                                (recognizedByPage[p] || []).forEach((r) => { byIndex[r.box_index] = r.text || ''; });
                                const sorted = (detectedPages[p] || [])
                                    .map((poly, idx) => {
                                        const byY = [...poly].sort((a, b) => a[1] - b[1]);
                                        return { idx, y: (byY[0][1] + byY[1][1]) / 2 };
                                    })
                                    .sort((a, b) => a.y - b.y)
                                    .map((item) => byIndex[item.idx] || '');
                                sorted.forEach((text, i) => {
                                    const escaped = `"${String(text).replace(/"/g, '""')}"`;
                                    rows.push(`${p},${i + 1},${escaped}`);
                                });
                            });
                            const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8' });
                            downloadBlob(blob, `layout_aware_transcript_${new Date().toISOString().slice(0, 10)}.csv`);
                        }}
                        className="flex flex-col items-center gap-1 py-2 rounded-lg border text-[10px] font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-white border-gray-200 text-gray-600 hover:border-violet-300 hover:bg-violet-50 hover:text-violet-700"
                    >
                        <FileType size={15} />
                        CSV
                    </button>
                    <button
                        disabled={recognizedCount === 0}
                        onClick={async () => {
                            const transcripts = {};
                            availablePages.forEach((p) => {
                                if (!(p in recognizedByPage)) return;
                                const byIndex = {};
                                (recognizedByPage[p] || []).forEach((r) => { byIndex[r.box_index] = r.text || ''; });
                                const sorted = (detectedPages[p] || [])
                                    .map((poly, idx) => {
                                        const byY = [...poly].sort((a, b) => a[1] - b[1]);
                                        return { idx, y: (byY[0][1] + byY[1][1]) / 2 };
                                    })
                                    .sort((a, b) => a.y - b.y)
                                    .map((item) => byIndex[item.idx] || '');
                                const text = sorted.filter((t) => t.trim()).join('\n');
                                if (text) transcripts[p] = text;
                            });
                            if (Object.keys(transcripts).length === 0) return;
                            try {
                                const response = await fetch(`${API_BASE}/api/export/pdf`, {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ transcripts, format: 'pdf' }),
                                });
                                if (!response.ok) throw new Error('Export failed');
                                const blob = await response.blob();
                                downloadBlob(blob, `layout_aware_transcript_${new Date().toISOString().slice(0, 10)}.pdf`);
                            } catch (err) {
                                console.error('PDF export failed:', err);
                            }
                        }}
                        className="flex flex-col items-center gap-1 py-2 rounded-lg border text-[10px] font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-white border-gray-200 text-gray-600 hover:border-rose-300 hover:bg-rose-50 hover:text-rose-700"
                    >
                        <BookOpen size={15} />
                        PDF
                    </button>
                </div>
                {recognizedCount === 0 && (
                    <p className="text-[10px] text-gray-400 text-center">Run OCR on at least one page to export transcripts.</p>
                )}
                {onHome && (
                    <button
                        onClick={onHome}
                        className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg border border-gray-200 bg-white text-gray-600 hover:bg-gray-50 hover:text-teal-700 hover:border-teal-200 text-[10px] font-semibold transition-all"
                    >
                        <Home size={13} />
                        Back to Home
                    </button>
                )}
            </div>
        </div>
    );

    // ── Render ──
    return (
        <div className="h-full w-full flex flex-col bg-gradient-to-br from-slate-50 via-teal-50/20 to-emerald-50/30 overflow-hidden">

            {/* BBox Editor Modal */}
            {showBBoxEditor && bboxEditorPages.length > 0 && (
                <BBoxEditor
                    pages={bboxEditorPages}
                    initialPageNumber={currentPageNum}
                    onSave={handleBBoxSave}
                    onCancel={() => setShowBBoxEditor(false)}
                />
            )}

            {/* Header */}
            <header className="h-12 bg-white/95 backdrop-blur-md border-b border-gray-200 shrink-0 relative z-10">
                <div className="absolute left-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
                    <button onClick={onBack} className="flex items-center gap-1 text-gray-600 hover:text-teal-600 hover:bg-teal-50 px-2 py-1 rounded-lg transition-all text-sm">
                        <ChevronLeft size={16} />
                        <span className="font-medium hidden sm:inline">Back</span>
                    </button>
                    {onHome && (
                        <button onClick={onHome} className="flex items-center gap-1 text-gray-600 hover:text-teal-600 hover:bg-teal-50 px-2 py-1 rounded-lg transition-all text-sm">
                            <Home size={16} />
                            <span className="font-medium hidden sm:inline">Home</span>
                        </button>
                    )}
                    <div className="h-4 w-px bg-gray-200 hidden sm:block" />
                    <div className="flex items-center gap-1.5 hidden sm:flex">
                        <div className="p-1 bg-gradient-to-br from-teal-500 to-emerald-600 rounded text-white">
                            <Layers size={14} />
                        </div>
                        <span className="text-sm font-bold text-gray-700">Layout-Aware Detection</span>
                    </div>
                </div>

                {/* Center progress */}
                <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
                    <div className="flex items-center gap-3 px-4 py-1.5 bg-gradient-to-r from-teal-50 via-white to-emerald-50 rounded-full border border-teal-100/60 shadow-sm">
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-bold ${detectedCount === totalPages && totalPages > 0
                            ? 'bg-gradient-to-br from-emerald-500 to-green-600'
                            : 'bg-gradient-to-br from-teal-500 to-emerald-600'
                            }`}>
                            <Layers size={12} />
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-40 h-1.5 bg-teal-100 rounded-full overflow-hidden">
                                <div className="h-full rounded-full transition-all duration-500 bg-gradient-to-r from-teal-500 to-emerald-500"
                                    style={{ width: `${totalPages > 0 ? (detectedCount / totalPages) * 100 : 0}%` }} />
                            </div>
                            <span className="text-xs font-bold text-teal-700">{detectedCount}/{totalPages}</span>
                        </div>
                    </div>
                </div>

                {/* Dataset mode: Continue to Export button */}
                {datasetMode && detectedCount > 0 && (
                    <div className="absolute right-3 top-1/2 -translate-y-1/2">
                        <button
                            onClick={() => {
                                if (onDatasetNext) {
                                    onDatasetNext({
                                        boxesByPage: detectedPages,
                                        alignedTranscriptByPage,
                                        modelInfo: {
                                            preprocessing,
                                            detection_model: selectedDetModel,
                                            layout_model: selectedLayoutModel,
                                        },
                                    });
                                }
                            }}
                            className="flex items-center gap-1.5 px-4 py-1.5 bg-gradient-to-r from-emerald-500 to-teal-600 text-white text-sm font-bold rounded-lg shadow-md hover:shadow-lg hover:-translate-y-0.5 transition-all duration-200"
                        >
                            Continue to Export
                            <ChevronRight size={16} />
                        </button>
                    </div>
                )}

            </header>

            {/* Main 3-column layout */}
            <main className="flex-1 min-h-0 p-2 overflow-hidden">
<div className="h-full hidden lg:block">
                    <ResizablePanels
                        leftPanel={leftSidebar}
                        centerPanel={centerPanel}
                        rightPanel={
                            datasetMode
                                ? (transcript && Object.keys(transcript).length > 0 ? rightPanel : null)
                                : ocrRightPanel
                        }
                        defaultLeftWidth={280}
                        defaultRightWidth={300}
                        minLeftWidth={240}
                        maxLeftWidth={420}
                        minRightWidth={240}
                        maxRightWidth={450}
                        minCenterWidth={350}
                    />
                </div>
                {/* Mobile fallback */}
                <div className="h-full lg:hidden flex flex-col gap-2 overflow-hidden">
                    <div className="flex-1 min-h-0 overflow-hidden">
                        {centerPanel}
                    </div>
                </div>
            </main>
        </div>
    );
}
