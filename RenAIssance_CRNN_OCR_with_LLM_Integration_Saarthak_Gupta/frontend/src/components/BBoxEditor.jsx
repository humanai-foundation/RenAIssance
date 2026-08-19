import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
    X,
    Check,
    Undo2,
    Redo2,
    ZoomIn,
    ZoomOut,
    Maximize2,
    Plus,
    MousePointer2,
    Trash2,
    ChevronLeft,
    ChevronRight,
    Copy,
    RotateCw,
} from 'lucide-react';

// ── Constants ──
const HANDLE_RADIUS = 6;        // corner handle radius in canvas coords
const MIN_BOX_SIZE = 10;
const MAX_UNDO = 40;

// ── Helpers ──
let _uid = 0;
function uid() { return `bb-${Date.now()}-${_uid++}`; }

// "2_left" -> "2L", "3_right" -> "3R".
function shortPageLabel(pageNumber) {
    return String(pageNumber).replace('_left', 'L').replace('_right', 'R');
}

// The resize/rotate math assumes [TL, TR, BR, BL] vertex order.
function orderQuad(poly) {
    const byY = [...poly].sort((a, b) => a[1] - b[1]);
    const [tl, tr] = byY.slice(0, 2).sort((a, b) => a[0] - b[0]);   // top: left, right
    const [bl, br] = byY.slice(2, 4).sort((a, b) => a[0] - b[0]);   // bottom: left, right
    return [[tl[0], tl[1]], [tr[0], tr[1]], [br[0], br[1]], [bl[0], bl[1]]];
}

// PaddleOCR polygon -> editor box. 4-point quads keep their rotation; anything
// else collapses to its bounding rect.
function polyToBox(poly, id) {
    if (!poly || poly.length === 0) return null;
    let pts;
    if (poly.length === 4) {
        pts = orderQuad(poly);
    } else {
        const xs = poly.map(p => p[0]);
        const ys = poly.map(p => p[1]);
        const x1 = Math.min(...xs), y1 = Math.min(...ys);
        const x2 = Math.max(...xs), y2 = Math.max(...ys);
        pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
    }
    return { id: id ?? uid(), points: pts, selected: false };
}

function boxToPoly(b) { return b.points; }

// Axis-aligned quad from two corners (draw mode).
function makeRectBox(x1, y1, x2, y2) {
    return {
        id: uid(),
        points: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        selected: true,
    };
}

// Point-in-convex-polygon.
function pointInPolygon(px, py, pts) {
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
        const xi = pts[i][0], yi = pts[i][1];
        const xj = pts[j][0], yj = pts[j][1];
        if (((yi > py) !== (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) inside = !inside;
    }
    return inside;
}

// Which corner was hit (0-3), or -1.
function hitCorner(pts, px, py, r) {
    for (let i = 0; i < pts.length; i++) {
        if (Math.hypot(px - pts[i][0], py - pts[i][1]) <= r) return i;
    }
    return -1;
}

// Drag a corner, keeping the box axis-aligned. The opposite corner is the anchor.
function resizeRectCorner(pts, cornerIdx, nx, ny) {
    const opposites = [2, 3, 0, 1];
    const oIdx = opposites[cornerIdx];
    const ox = pts[oIdx][0], oy = pts[oIdx][1];
    const x1 = Math.min(nx, ox), y1 = Math.min(ny, oy);
    const x2 = Math.max(nx, ox), y2 = Math.max(ny, oy);
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
}

function rectSize(pts) {
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    return { w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) };
}

function rotatePoint(px, py, cx, cy, angle) {
    const cos = Math.cos(angle), sin = Math.sin(angle);
    return [cx + (px - cx) * cos - (py - cy) * sin, cy + (px - cx) * sin + (py - cy) * cos];
}

function getBoxCenter(pts) {
    return [pts.reduce((s, p) => s + p[0], 0) / 4, pts.reduce((s, p) => s + p[1], 0) / 4];
}

// Rotation handle sits off the TL-TR edge midpoint, along the outward normal.
function getRotationHandlePos(pts, offsetDist) {
    const mx = (pts[0][0] + pts[1][0]) / 2;
    const my = (pts[0][1] + pts[1][1]) / 2;
    const [cx, cy] = getBoxCenter(pts);
    const dx = mx - cx, dy = my - cy;
    const len = Math.hypot(dx, dy) || 1;
    return [mx + (dx / len) * offsetDist, my + (dy / len) * offsetDist];
}

function hitRotationHandle(pts, px, py, r, offsetDist) {
    const [hx, hy] = getRotationHandlePos(pts, offsetDist);
    return Math.hypot(px - hx, py - hy) <= r;
}

// Corner resize that survives rotation — the angle is preserved and the
// opposite corner stays put.
function resizeRotatedCorner(pts, cornerIdx, nx, ny) {
    const opposites = [2, 3, 0, 1];
    const oIdx = opposites[cornerIdx];
    const [opx, opy] = pts[oIdx];
    const ncx = (opx + nx) / 2, ncy = (opy + ny) / 2;
    const edgeDx = pts[1][0] - pts[0][0], edgeDy = pts[1][1] - pts[0][1];
    const angle = Math.atan2(edgeDy, edgeDx);
    const cosA = Math.cos(-angle), sinA = Math.sin(-angle);
    const lox = (opx - ncx) * cosA - (opy - ncy) * sinA;
    const loy = (opx - ncx) * sinA + (opy - ncy) * cosA;
    const lnx = (nx - ncx) * cosA - (ny - ncy) * sinA;
    const lny = (nx - ncx) * sinA + (ny - ncy) * cosA;
    const hw = Math.abs(lnx - lox) / 2, hh = Math.abs(lny - loy) / 2;
    if (hw * 2 < MIN_BOX_SIZE || hh * 2 < MIN_BOX_SIZE) return pts;
    const local = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
    const cosR = Math.cos(angle), sinR = Math.sin(angle);
    return local.map(([lx, ly]) => [ncx + lx * cosR - ly * sinR, ncy + lx * sinR + ly * cosR]);
}

// ── BBoxEditor — multi-page, supports angled bounding boxes ──
export default function BBoxEditor({ pages: pagesProp = [], onSave, onCancel, initialPageNumber = null }) {

    // ── Multi-page ─────────────────────────────────────────────────────────
    // Frozen at mount. A batch detection running behind this modal reorders the
    // parent's list, which would slide currentIdx onto a different page and save
    // edits under the wrong page number.
    const pagesRef = useRef(pagesProp);
    const pages = pagesRef.current;
    const [currentIdx, setCurrentIdx] = useState(() => {
        if (initialPageNumber == null) return 0;
        const idx = pagesRef.current.findIndex(p => String(p.pageNumber) === String(initialPageNumber));
        return idx === -1 ? 0 : idx;
    });
    const pageBoxesRef = useRef({});
    const [modifiedPages, setModifiedPages] = useState(new Set());
    const currentPage = pages[currentIdx];

    // ── Refs ───────────────────────────────────────────────────────────────
    const bgCanvasRef = useRef(null);
    const overlayCanvasRef = useRef(null);
    const containerRef = useRef(null);
    const outerRef = useRef(null);
    // Scrolled into view on nav so later pages stay reachable.
    const activePageBtnRef = useRef(null);

    // Focus on open so arrow keys work straight away.
    useEffect(() => { outerRef.current?.focus(); }, []);

    useEffect(() => {
        activePageBtnRef.current?.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'smooth' });
    }, [currentIdx]);
    // ── Box state ──────────────────────────────────────────────────────────
    const [boxes, setBoxes] = useState([]);
    const [undoStack, setUndoStack] = useState([]);
    const [redoStack, setRedoStack] = useState([]);
    const boxesRef = useRef([]);
    useEffect(() => { boxesRef.current = boxes; }, [boxes]);

    // ── View ────────────────────────────────────────────────────────────────
    const [zoom, setZoom] = useState(1);
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [imageSize, setImageSize] = useState({ w: 0, h: 0 });
    const [imageReady, setImageReady] = useState(false);

    const zoomRef = useRef(zoom);
    const panRef = useRef(pan);
    const imageSizeRef = useRef(imageSize);

    // In the render body on purpose: effects fire too late.
    zoomRef.current = zoom;
    panRef.current = pan;
    imageSizeRef.current = imageSize;

    // 'select' | 'draw'
    const [mode, setMode] = useState('select');

    // One of: move | corner | rotate | draw | pan, each with its own payload.
    const dragRef = useRef(null);
    const isPanRef = useRef(false);
    const panStartRef = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

    const selectedId = useMemo(() => boxes.find(b => b.selected)?.id ?? null, [boxes]);

    // ── Page management ──
    const saveCurrentPage = useCallback(() => {
        if (!currentPage) return;
        pageBoxesRef.current[currentPage.pageNumber] = boxesRef.current.map(b => ({ ...b, selected: false }));
    }, [currentPage]);

    const loadPage = useCallback((page) => {
        setImageReady(false);
        setUndoStack([]);
        setRedoStack([]);

        const saved = pageBoxesRef.current[page.pageNumber];
        const initialBoxes = saved ?? (page.polygons || []).map(poly => polyToBox(poly)).filter(Boolean);
        if (!saved) pageBoxesRef.current[page.pageNumber] = initialBoxes;
        setBoxes(initialBoxes);
        // Synced here, not in an effect — flipping pages fast would otherwise
        // save the previous page's boxes under the next page's number.
        boxesRef.current = initialBoxes;

        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            const w = img.naturalWidth, h = img.naturalHeight;
            setImageSize({ w, h });

            const bg = bgCanvasRef.current;
            if (bg) { bg.width = w; bg.height = h; bg.getContext('2d').drawImage(img, 0, 0); }

            if (containerRef.current) {
                const cw = containerRef.current.clientWidth, ch = containerRef.current.clientHeight;
                const fitZ = Math.min(cw / w, ch / h, 1) * 0.9;
                const fitPan = { x: (cw - w * fitZ) / 2, y: (ch - h * fitZ) / 2 };
                setZoom(fitZ); setPan(fitPan);
                zoomRef.current = fitZ; panRef.current = fitPan;
            }
            setImageReady(true);
        };
        img.src = page.imageSrc;
    }, []);

    useEffect(() => { if (pages.length > 0) loadPage(pages[currentIdx] || pages[0]); }, []); // eslint-disable-line

    const goToPage = useCallback((idx) => {
        if (idx < 0 || idx >= pages.length || idx === currentIdx) return;
        saveCurrentPage();
        setCurrentIdx(idx);
        loadPage(pages[idx]);
    }, [currentIdx, pages, saveCurrentPage, loadPage]);

    // ── Overlay drawing ──
    const drawOverlay = useCallback((boxList, drawState) => {
        const canvas = overlayCanvasRef.current;
        const { w, h } = imageSizeRef.current;
        if (!canvas || w === 0) return;

        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, w, h);

        const z = zoomRef.current;
        const lw = Math.max(1.5, 2 / z);
        const cr = Math.max(4, HANDLE_RADIUS / z);     // corner handle radius in canvas coords

        for (const b of boxList) {
            const sel = b.selected;
            const pts = b.points;

            ctx.strokeStyle = sel ? '#f97316' : '#22c55e';
            ctx.lineWidth = lw;
            ctx.fillStyle = sel ? 'rgba(249,115,22,0.12)' : 'rgba(34,197,94,0.08)';
            ctx.lineJoin = 'round';
            ctx.beginPath();
            ctx.moveTo(pts[0][0], pts[0][1]);
            for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();

            if (sel) {
                ctx.fillStyle = '#ffffff';
                ctx.strokeStyle = '#f97316';
                ctx.lineWidth = lw;
                for (const [px, py] of pts) {
                    ctx.beginPath();
                    ctx.rect(px - cr, py - cr, cr * 2, cr * 2);
                    ctx.fill(); ctx.stroke();
                }

                const rotOffset = Math.max(8, 30 / z);
                const [rhx, rhy] = getRotationHandlePos(pts, rotOffset);
                const tmx = (pts[0][0] + pts[1][0]) / 2;
                const tmy = (pts[0][1] + pts[1][1]) / 2;
                ctx.setLineDash([3 / z, 2 / z]);
                ctx.strokeStyle = '#f97316';
                ctx.lineWidth = lw;
                ctx.beginPath(); ctx.moveTo(tmx, tmy); ctx.lineTo(rhx, rhy); ctx.stroke();
                ctx.setLineDash([]);
                const rhr = cr * 0.95;
                ctx.beginPath(); ctx.arc(rhx, rhy, rhr, 0, Math.PI * 2);
                ctx.fillStyle = '#ffffff'; ctx.fill();
                ctx.strokeStyle = '#f97316'; ctx.lineWidth = lw; ctx.stroke();
                ctx.strokeStyle = '#f97316';
                ctx.lineWidth = Math.max(1, lw * 0.8);
                ctx.beginPath();
                ctx.arc(rhx, rhy, rhr * 0.5, -Math.PI * 0.8, Math.PI * 0.2);
                ctx.stroke();
            }
        }

        if (drawState) {
            const { startX, startY, currentX, currentY } = drawState;
            const x1 = Math.min(startX, currentX), y1 = Math.min(startY, currentY);
            const x2 = Math.max(startX, currentX), y2 = Math.max(startY, currentY);
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = lw;
            ctx.setLineDash([6 / z, 4 / z]);
            ctx.fillStyle = 'rgba(59,130,246,0.10)';
            ctx.beginPath(); ctx.rect(x1, y1, x2 - x1, y2 - y1);
            ctx.fill(); ctx.stroke();
            ctx.setLineDash([]);
        }
    }, []);

    // imageSize is in the deps so we redraw once the image loads.
    useEffect(() => {
        if (imageSize.w > 0) drawOverlay(boxes, null);
    }, [boxes, imageSize, drawOverlay]);

    // ── Undo / Redo ──
    const pushUndo = useCallback((prevBoxes) => {
        const snap = prevBoxes.map(b => ({ ...b, points: b.points.map(p => [...p]), selected: false }));
        setUndoStack(s => { const n = [...s, snap]; return n.length > MAX_UNDO ? n.slice(-MAX_UNDO) : n; });
        setRedoStack([]);
        setModifiedPages(prev => new Set([...prev, currentPage?.pageNumber]));
    }, [currentPage]);

    const handleUndo = useCallback(() => {
        setUndoStack(s => {
            if (!s.length) return s;
            const prev = s[s.length - 1];
            setRedoStack(r => [...r, boxes.map(b => ({ ...b, points: b.points.map(p => [...p]), selected: false }))]);
            setBoxes(prev); return s.slice(0, -1);
        });
    }, [boxes]);

    const handleRedo = useCallback(() => {
        setRedoStack(r => {
            if (!r.length) return r;
            const next = r[r.length - 1];
            setUndoStack(s => [...s, boxes.map(b => ({ ...b, points: b.points.map(p => [...p]), selected: false }))]);
            setBoxes(next); return r.slice(0, -1);
        });
    }, [boxes]);

    // ── Coordinates ──
    const screenToImage = useCallback((clientX, clientY) => {
        if (!containerRef.current) return { x: 0, y: 0 };
        const rect = containerRef.current.getBoundingClientRect();
        return {
            x: (clientX - rect.left - panRef.current.x) / zoomRef.current,
            y: (clientY - rect.top - panRef.current.y) / zoomRef.current,
        };
    }, []);

    // ── Pointer events ──
    const handlePointerDown = useCallback((e) => {
        // Middle-click or ctrl+left-click pans.
        if (e.button === 1 || (e.button === 0 && (e.ctrlKey || e.metaKey))) {
            isPanRef.current = true;
            panStartRef.current = { x: e.clientX, y: e.clientY, panX: panRef.current.x, panY: panRef.current.y };
            e.preventDefault(); return;
        }
        if (e.button !== 0) return;

        const { x: imgX, y: imgY } = screenToImage(e.clientX, e.clientY);
        const z = zoomRef.current;
        const cr = Math.max(4, HANDLE_RADIUS / z);
        const rr = Math.max(5, (HANDLE_RADIUS + 2) / z);

        // ── Draw mode ──
        if (mode === 'draw') {
            dragRef.current = { type: 'draw', startX: imgX, startY: imgY, currentX: imgX, currentY: imgY };
            return;
        }

        // ── Select mode: handles first, then bodies, then empty space ──
        const selBox = boxes.find(b => b.selected);

        if (selBox) {
            const rotOffset = Math.max(8, 30 / z);
            if (hitRotationHandle(selBox.points, imgX, imgY, rr * 1.8, rotOffset)) {
                const center = getBoxCenter(selBox.points);
                pushUndo(boxes);
                dragRef.current = {
                    type: 'rotate',
                    id: selBox.id,
                    origPts: selBox.points.map(p => [...p]),
                    center,
                    startAngle: Math.atan2(imgY - center[1], imgX - center[0]),
                };
                return;
            }
        }

        if (selBox) {
            const ci = hitCorner(selBox.points, imgX, imgY, cr * 1.5);
            if (ci !== -1) {
                pushUndo(boxes);
                dragRef.current = {
                    type: 'corner',
                    id: selBox.id,
                    cornerIdx: ci,
                    origPts: selBox.points.map(p => [...p]),
                };
                return;
            }
        }

        // Top-most box wins.
        const hit = [...boxes].reverse().find(b => pointInPolygon(imgX, imgY, b.points));
        if (hit) {
            pushUndo(boxes);
            dragRef.current = {
                type: 'move',
                id: hit.id,
                origPts: hit.points.map(p => [...p]),
                startX: imgX, startY: imgY,
            };
            setBoxes(prev => prev.map(b => ({ ...b, selected: b.id === hit.id })));
            return;
        }

        setBoxes(prev => prev.map(b => ({ ...b, selected: false })));
        dragRef.current = null;
    }, [mode, boxes, screenToImage, pushUndo]);

    const handlePointerMove = useCallback((e) => {
        if (isPanRef.current) {
            const dx = e.clientX - panStartRef.current.x;
            const dy = e.clientY - panStartRef.current.y;
            const np = { x: panStartRef.current.panX + dx, y: panStartRef.current.panY + dy };
            setPan(np); panRef.current = np; return;
        }

        const drag = dragRef.current;
        if (!drag) return;
        const { x: imgX, y: imgY } = screenToImage(e.clientX, e.clientY);

        if (drag.type === 'draw') {
            drag.currentX = imgX; drag.currentY = imgY;
            drawOverlay(boxesRef.current, drag); return;
        }

        if (drag.type === 'move') {
            const dx = imgX - drag.startX, dy = imgY - drag.startY;
            setBoxes(prev => prev.map(b => {
                if (b.id !== drag.id) return b;
                return { ...b, points: drag.origPts.map(([px, py]) => [px + dx, py + dy]) };
            }));
            return;
        }

        if (drag.type === 'rotate') {
            const [cx, cy] = drag.center;
            const currentAngle = Math.atan2(imgY - cy, imgX - cx);
            // 0.35x so the box turns gently, not wildly.
            const delta = (currentAngle - drag.startAngle) * 0.35;
            setBoxes(prev => prev.map(b => {
                if (b.id !== drag.id) return b;
                const newPts = drag.origPts.map(([px, py]) => rotatePoint(px, py, cx, cy, delta));
                return { ...b, points: newPts };
            }));
            return;
        }

        if (drag.type === 'corner') {
            setBoxes(prev => prev.map(b => {
                if (b.id !== drag.id) return b;
                const newPts = resizeRotatedCorner(drag.origPts, drag.cornerIdx, imgX, imgY);
                return { ...b, points: newPts };
            }));
            return;
        }
    }, [screenToImage, drawOverlay]);

    const handlePointerUp = useCallback(() => {
        isPanRef.current = false;
        const drag = dragRef.current;
        dragRef.current = null;
        if (!drag) return;

        if (drag.type === 'draw') {
            const x1 = Math.min(drag.startX, drag.currentX), y1 = Math.min(drag.startY, drag.currentY);
            const x2 = Math.max(drag.startX, drag.currentX), y2 = Math.max(drag.startY, drag.currentY);
            if (x2 - x1 >= MIN_BOX_SIZE && y2 - y1 >= MIN_BOX_SIZE) {
                pushUndo(boxesRef.current);
                const nb = makeRectBox(x1, y1, x2, y2);
                setBoxes(prev => [...prev.map(b => ({ ...b, selected: false })), nb]);
                setModifiedPages(prev => new Set([...prev, currentPage?.pageNumber]));
            } else {
                drawOverlay(boxesRef.current, null);
            }
        }

        if (drag.type === 'move' || drag.type === 'corner' || drag.type === 'rotate') {
            setModifiedPages(prev => new Set([...prev, currentPage?.pageNumber]));
        }
    }, [pushUndo, drawOverlay, currentPage]);

    useEffect(() => {
        const up = () => handlePointerUp();
        window.addEventListener('mouseup', up);
        return () => window.removeEventListener('mouseup', up);
    }, [handlePointerUp]);

    // ── Scroll zoom ──
    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;
        const onWheel = e => {
            e.preventDefault();
            // ~5% per tick — 10% overshoots constantly.
            const f = e.deltaY < 0 ? 1.05 : 1 / 1.05;
            setZoom(z => { const nz = Math.max(0.05, Math.min(8, z * f)); zoomRef.current = nz; return nz; });
        };
        el.addEventListener('wheel', onWheel, { passive: false });
        return () => el.removeEventListener('wheel', onWheel);
    }, []);

    // ── Keyboard ──
    const handleDelete = useCallback(() => {
        if (!selectedId) return;
        // Land the selection on the box above the deleted one.
        const sorted = [...boxes].sort((a, b) => {
            const topA = Math.min(...a.points.map(p => p[1]));
            const topB = Math.min(...b.points.map(p => p[1]));
            return topA - topB;
        });
        const delIdx = sorted.findIndex(b => b.id === selectedId);
        const prevId = delIdx > 0 ? sorted[delIdx - 1].id : null;
        pushUndo(boxes);
        setBoxes(prev =>
            prev
                .filter(b => b.id !== selectedId)
                .map(b => ({ ...b, selected: prevId !== null && b.id === prevId }))
        );
        setModifiedPages(prev => new Set([...prev, currentPage?.pageNumber]));
    }, [selectedId, boxes, pushUndo, currentPage]);

    const handleDuplicate = useCallback(() => {
        if (!selectedId) return;
        const src = boxes.find(b => b.id === selectedId);
        if (!src) return;
        pushUndo(boxes);
        const dup = {
            id: uid(),
            points: src.points.map(([px, py]) => [px + 12, py + 12]),
            selected: true,
        };
        setBoxes(prev => [...prev.map(b => ({ ...b, selected: false })), dup]);
        setModifiedPages(prev => new Set([...prev, currentPage?.pageNumber]));
    }, [selectedId, boxes, pushUndo, currentPage]);

    useEffect(() => {
        const onKey = e => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (e.key === 'n' || e.key === 'N') { setMode(m => m === 'draw' ? 'select' : 'draw'); return; }
            if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId) { e.preventDefault(); handleDelete(); return; }
            if ((e.key === 'd' || e.key === 'D') && selectedId) { e.preventDefault(); handleDuplicate(); return; }
            if (e.key === 'Escape') { setMode('select'); setBoxes(prev => prev.map(b => ({ ...b, selected: false }))); dragRef.current = null; return; }
            if (e.ctrlKey || e.metaKey) {
                if (e.key === 'z') { e.preventDefault(); handleUndo(); }
                else if (e.key === 'y' || (e.shiftKey && e.key === 'z')) { e.preventDefault(); handleRedo(); }
            }
            // Up/Down cycle boxes in top-to-bottom order.
            if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
                e.preventDefault();
                const sorted = [...boxesRef.current].sort((a, b) => {
                    const topA = Math.min(...a.points.map(p => p[1]));
                    const topB = Math.min(...b.points.map(p => p[1]));
                    return topA - topB;
                });
                if (sorted.length === 0) return;
                const selIdx = sorted.findIndex(b => b.selected);
                let nextIdx;
                if (selIdx === -1) {
                    nextIdx = e.key === 'ArrowDown' ? 0 : sorted.length - 1;
                } else {
                    nextIdx = e.key === 'ArrowDown' ? selIdx + 1 : selIdx - 1;
                    nextIdx = Math.max(0, Math.min(sorted.length - 1, nextIdx));
                }
                const targetId = sorted[nextIdx].id;
                setBoxes(prev => prev.map(b => ({ ...b, selected: b.id === targetId })));
                return;
            }
            if (e.key === 'ArrowLeft') goToPage(currentIdx - 1);
            if (e.key === 'ArrowRight') goToPage(currentIdx + 1);
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [selectedId, handleDelete, handleDuplicate, handleUndo, handleRedo, currentIdx, goToPage]);

    // ── Fit view ──
    const handleFit = useCallback(() => {
        const { w, h } = imageSizeRef.current;
        if (!containerRef.current || w === 0) return;
        const cw = containerRef.current.clientWidth, ch = containerRef.current.clientHeight;
        const fitZ = Math.min(cw / w, ch / h, 1) * 0.9;
        const fitPan = { x: (cw - w * fitZ) / 2, y: (ch - h * fitZ) / 2 };
        setZoom(fitZ); setPan(fitPan); zoomRef.current = fitZ; panRef.current = fitPan;
    }, []);

    // ── Cursor style ──
    const [cursor, setCursor] = useState('default');
    const handleMouseMoveForCursor = useCallback((e) => {
        if (mode === 'draw') { setCursor('crosshair'); return; }
        if (isPanRef.current) { setCursor('grabbing'); return; }
        if (dragRef.current) return; // already handling a drag

        const { x: imgX, y: imgY } = screenToImage(e.clientX, e.clientY);
        const z = zoomRef.current;
        const selBox = boxes.find(b => b.selected);
        const cr = Math.max(4, HANDLE_RADIUS / z);

        if (selBox) {
            const rr = Math.max(5, (HANDLE_RADIUS + 2) / z);
            const rotOffset = Math.max(8, 30 / z);
            if (hitRotationHandle(selBox.points, imgX, imgY, rr * 1.8, rotOffset)) { setCursor('grab'); return; }
            if (hitCorner(selBox.points, imgX, imgY, cr * 1.5) !== -1) { setCursor('nwse-resize'); return; }
        }
        const hit = [...boxes].reverse().find(b => pointInPolygon(imgX, imgY, b.points));
        setCursor(hit ? 'move' : 'default');
    }, [mode, boxes, screenToImage]);

    // ── Save all pages ──
    const handleSave = useCallback(() => {
        saveCurrentPage();
        const results = {};
        pages.forEach(page => {
            const bxs = pageBoxesRef.current[page.pageNumber];
            results[page.pageNumber] = bxs ? bxs.map(boxToPoly) : page.polygons || [];
        });
        onSave(results);
    }, [pages, onSave, saveCurrentPage]);

    const selBox = boxes.find(b => b.selected);

    // ── Render ──
    return (
        <div
            ref={outerRef}
            tabIndex={-1}
            className="fixed inset-0 z-50 flex flex-col bg-gray-950 outline-none"
            onContextMenu={e => e.preventDefault()}
        >

            {/* ── TOOLBAR ────────────────────────────────────────────── */}
            <div className="flex-shrink-0 flex items-center justify-between gap-3 px-4 py-2.5 bg-gray-900 border-b border-gray-800">

                {/* Left — title + mode toggle */}
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <div className="p-1.5 bg-gradient-to-br from-orange-500 to-amber-600 rounded-lg text-white">
                            <MousePointer2 size={16} />
                        </div>
                        <div>
                            <h2 className="text-white font-semibold text-sm leading-none">Box Editor</h2>
                            <p className="text-gray-500 text-[10px] mt-0.5">
                                {boxes.length} box{boxes.length !== 1 ? 'es' : ''}
                                {modifiedPages.size > 0 && <span className="ml-1 text-amber-500">· {modifiedPages.size} page{modifiedPages.size > 1 ? 's' : ''} modified</span>}
                            </p>
                        </div>
                    </div>

                    <div className="flex items-center gap-1 bg-gray-800 rounded-lg p-0.5">
                        <button onClick={() => setMode('select')} title="Select / Move / Resize / Rotate  (Esc)"
                            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${mode === 'select' ? 'bg-orange-500 text-white shadow-sm' : 'text-gray-400 hover:text-white'}`}>
                            <MousePointer2 size={13} /> Select
                        </button>
                        <button onClick={() => setMode(m => m === 'draw' ? 'select' : 'draw')} title="Draw New Box  (N)"
                            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${mode === 'draw' ? 'bg-blue-600 text-white shadow-sm' : 'text-gray-400 hover:text-white'}`}>
                            <Plus size={13} /> Add <kbd className="ml-1 px-1 text-[9px] bg-gray-700 text-gray-400 rounded">N</kbd>
                        </button>
                    </div>
                </div>

                {/* Center — info + undo/redo + zoom */}
                <div className="flex items-center gap-3">
                    {selBox ? (
                        <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 bg-gray-800 rounded-lg text-xs text-gray-300">
                            <span className="text-gray-500 text-[10px]">Drag corners to resize · Drag body to move · Drag ○ to rotate</span>
                        </div>
                    ) : (
                        <div className="hidden md:flex items-center gap-1.5 px-3 py-1 bg-gray-800 rounded-lg text-xs text-gray-500">
                            Click a box to select · drag to move or resize
                        </div>
                    )}

                    <div className="flex items-center gap-1 bg-gray-800 rounded-lg p-0.5">
                        <button onClick={handleUndo} disabled={!undoStack.length} title="Undo (Ctrl+Z)"
                            className={`p-1.5 rounded-md ${undoStack.length ? 'text-gray-300 hover:text-white hover:bg-gray-700' : 'text-gray-700 cursor-not-allowed'}`}><Undo2 size={15} /></button>
                        <button onClick={handleRedo} disabled={!redoStack.length} title="Redo (Ctrl+Y)"
                            className={`p-1.5 rounded-md ${redoStack.length ? 'text-gray-300 hover:text-white hover:bg-gray-700' : 'text-gray-700 cursor-not-allowed'}`}><Redo2 size={15} /></button>
                    </div>

                    <div className="flex items-center gap-1 bg-gray-800 rounded-lg p-0.5">
                        <button onClick={() => setZoom(z => Math.min(z * 1.25, 8))} className="p-1.5 text-gray-300 hover:text-white hover:bg-gray-700 rounded-md"><ZoomIn size={15} /></button>
                        <span className="text-xs text-gray-500 font-mono min-w-[42px] text-center">{Math.round(zoom * 100)}%</span>
                        <button onClick={() => setZoom(z => Math.max(z / 1.25, 0.05))} className="p-1.5 text-gray-300 hover:text-white hover:bg-gray-700 rounded-md"><ZoomOut size={15} /></button>
                        <button onClick={handleFit} title="Fit to view" className="p-1.5 text-gray-300 hover:text-white hover:bg-gray-700 rounded-md"><Maximize2 size={15} /></button>
                    </div>

                    {selectedId && (
                        <button onClick={handleDuplicate} title="Duplicate (D)"
                            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-blue-900/50 hover:bg-blue-800/60 text-blue-400 hover:text-blue-300 text-xs font-semibold transition-colors">
                            <Copy size={13} /> Dup
                        </button>
                    )}
                    {selectedId && (
                        <button onClick={handleDelete} title="Delete (Delete)"
                            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-red-900/50 hover:bg-red-800/60 text-red-400 hover:text-red-300 text-xs font-semibold transition-colors">
                            <Trash2 size={13} /> Delete
                        </button>
                    )}
                </div>

                {/* Right — cancel / save */}
                <div className="flex items-center gap-2">
                    <button onClick={onCancel} className="flex items-center gap-1.5 px-3 py-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg text-sm font-medium transition-colors">
                        <X size={15} /> Cancel
                    </button>
                    <button onClick={handleSave}
                        className="flex items-center gap-1.5 px-4 py-2 bg-gradient-to-r from-emerald-500 to-green-600 hover:from-emerald-600 hover:to-green-700 text-white font-semibold rounded-lg text-sm shadow-md hover:shadow-lg hover:-translate-y-0.5 transition-all">
                        <Check size={15} />
                        Save All
                        {modifiedPages.size > 0 && (
                            <span className="ml-1 px-1.5 py-0.5 bg-white/20 rounded text-[10px]">
                                {modifiedPages.size} page{modifiedPages.size > 1 ? 's' : ''}
                            </span>
                        )}
                    </button>
                </div>
            </div>

            {/* ── CANVAS AREA ──────────────────────────────────────────── */}
            <div
                ref={containerRef}
                className="flex-1 relative overflow-hidden select-none"
                style={{ background: 'repeating-conic-gradient(#1a1a1a 0% 25%, #222 0% 50%) 50% / 20px 20px', cursor }}
                onMouseDown={handlePointerDown}
                onMouseMove={(e) => { handlePointerMove(e); handleMouseMoveForCursor(e); }}
            >
                {/* Background image (drawn once) */}
                <canvas ref={bgCanvasRef} style={{ position: 'absolute', left: pan.x, top: pan.y, width: imageSize.w * zoom, height: imageSize.h * zoom, imageRendering: zoom > 2 ? 'pixelated' : 'auto', display: imageReady ? 'block' : 'none' }} />

                {/* Interactive overlay */}
                <canvas ref={overlayCanvasRef} style={{ position: 'absolute', left: pan.x, top: pan.y, width: imageSize.w * zoom, height: imageSize.h * zoom, imageRendering: zoom > 2 ? 'pixelated' : 'auto', display: imageReady ? 'block' : 'none', pointerEvents: 'none' }} />

                {!imageReady && (
                    <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-10 h-10 border-4 border-gray-700 border-t-orange-500 rounded-full animate-spin" />
                    </div>
                )}

                {mode === 'draw' && imageReady && (
                    <div className="absolute top-3 left-1/2 -translate-x-1/2 pointer-events-none">
                        <div className="px-3 py-1.5 bg-blue-600/90 text-white text-xs font-semibold rounded-full shadow-lg backdrop-blur-sm">
                            Drag to draw a rectangle · Press N or Esc to exit
                        </div>
                    </div>
                )}
            </div>

            {/* ── BOTTOM: page nav + shortcuts ─────────────────────────── */}
            <div className="flex-shrink-0 px-4 py-2 bg-gray-900 border-t border-gray-800 flex items-center justify-between">
                <div className="hidden lg:flex items-center gap-4 text-[11px] text-gray-500 shrink min-w-0 overflow-hidden">
                    <span><kbd className="px-1 py-0.5 bg-gray-800 text-gray-400 rounded font-mono text-[10px]">N</kbd> Add box</span>
                    <span><kbd className="px-1 py-0.5 bg-gray-800 text-gray-400 rounded font-mono text-[10px]">Del</kbd> Delete</span>
                    <span><kbd className="px-1 py-0.5 bg-gray-800 text-gray-400 rounded font-mono text-[10px]">D</kbd> Duplicate</span>
                    <span className="hidden md:inline"><kbd className="px-1 py-0.5 bg-gray-800 text-gray-400 rounded font-mono text-[10px]">○</kbd> Rotate</span>
                    <span><kbd className="px-1 py-0.5 bg-gray-800 text-gray-400 rounded font-mono text-[10px]">Ctrl+Z</kbd> Undo</span>
                    <span className="hidden sm:inline"><kbd className="px-1 py-0.5 bg-gray-800 text-gray-400 rounded font-mono text-[10px]">Ctrl+drag</kbd> Pan</span>
                    <span className="hidden sm:inline"><kbd className="px-1 py-0.5 bg-gray-800 text-gray-400 rounded font-mono text-[10px]">← →</kbd> Pages</span>
                    <span className="hidden sm:inline"><kbd className="px-1 py-0.5 bg-gray-800 text-gray-400 rounded font-mono text-[10px]">↑ ↓</kbd> Boxes</span>
                </div>

                {pages.length > 1 && (
                    <div className="flex items-center gap-2 flex-1 min-w-0 justify-center px-3">
                        <span className="shrink-0 text-[11px] text-gray-500 font-mono">{currentIdx + 1}/{pages.length}</span>
                        <button onClick={() => goToPage(currentIdx - 1)} disabled={currentIdx === 0}
                            className={`shrink-0 p-1.5 rounded-lg ${currentIdx === 0 ? 'text-gray-700' : 'text-gray-300 hover:text-white hover:bg-gray-800'}`}>
                            <ChevronLeft size={18} />
                        </button>
                        <div className="flex items-center gap-1 overflow-x-auto min-w-0 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
                            {pages.map((page, idx) => {
                                // Unvisited pages have no edited boxes yet; showing 0
                                // looked like detection had been reverted.
                                const pageBoxCount = idx === currentIdx
                                    ? boxes.length
                                    : (pageBoxesRef.current[page.pageNumber]?.length ?? page.polygons?.length ?? 0);
                                return (
                                <button key={page.pageNumber} ref={idx === currentIdx ? activePageBtnRef : null} onClick={() => goToPage(idx)} title={`Page ${page.pageNumber} — ${pageBoxCount} boxes`}
                                    className={`relative shrink-0 min-w-[2rem] px-1.5 py-1 rounded-lg transition-all flex flex-col items-center leading-none ${idx === currentIdx ? 'bg-orange-500 text-white shadow-sm' : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white'}`}>
                                    <span className="text-xs font-semibold">{shortPageLabel(page.pageNumber)}</span>
                                    <span className="text-[9px] opacity-70">{pageBoxCount}</span>
                                    {modifiedPages.has(page.pageNumber) && (
                                        <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-amber-400 rounded-full border border-gray-900" />
                                    )}
                                </button>
                                );
                            })}
                        </div>
                        <button onClick={() => goToPage(currentIdx + 1)} disabled={currentIdx === pages.length - 1}
                            className={`shrink-0 p-1.5 rounded-lg ${currentIdx === pages.length - 1 ? 'text-gray-700' : 'text-gray-300 hover:text-white hover:bg-gray-800'}`}>
                            <ChevronRight size={18} />
                        </button>
                    </div>
                )}

                <span className="hidden md:block shrink-0 text-[11px] text-gray-600">
                    {imageSize.w > 0 && `${imageSize.w} × ${imageSize.h} px`}
                </span>
            </div>
        </div>
    );
}
