import React, { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  XCircle,
  Eye,
  FileText,
  Image,
  Filter,
  AlertTriangle,
  Layers,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';

// Review which scanned page goes with which transcript page.
//
// Keys are normalised first ("3 - left", "3_left" and "3   left" all become
// "3_left"), then matched exact-first, falling back to split-image/whole-
// transcript and whole-image/split-transcript pairings.

// Lowercase, delimiters collapsed to '_'.
function normalizePageKey(raw) {
  return String(raw)
    .toLowerCase()
    .replace(/\s*[-–—]\s*/g, '_')   // "3 - left" → "3_left"
    .replace(/\s+/g, '_')           // spaces  → underscore
    .replace(/_+/g, '_')            // collapse multiple underscores
    .replace(/^_|_$/g, '')          // strip leading/trailing underscores
    .trim();
}

// -> { num, side: 'left' | 'right' | null }
function parseNormKey(normKey) {
  const m = normKey.match(/^(\d+)(?:_(left|right))?$/);
  if (!m) return null;
  return { num: m[1], side: m[2] || null };
}

export default function PageMatchReviewPage({ pages, parsedTranscript, onBack, onNext }) {
  const [filter, setFilter] = useState('all'); // 'all' | 'matched' | 'unmatched'
  const [selectedPageNum, setSelectedPageNum] = useState(null);
  const [detailTab, setDetailTab] = useState('both'); // 'image' | 'transcript' | 'both'

  // pageNumber -> string[]; undefined means not seeded yet.
  const [editedLines, setEditedLines] = useState({});
  const initializedRef = useRef(false);

  // ── Build the match table ──
  const matchData = useMemo(() => {
    if (!pages || !parsedTranscript) return [];

    // Two indexes: exact normalized key, and a per-number bucket holding the
    // plain/left/right variants.
    const exactByNorm = {};
    const byNumber = {};

    Object.keys(parsedTranscript).forEach((k) => {
      const norm = normalizePageKey(k);
      exactByNorm[norm] = k;

      const parsed = parseNormKey(norm);
      if (!parsed) return;
      const { num, side } = parsed;
      if (!byNumber[num]) byNumber[num] = { plain: null, left: null, right: null };
      if (side === 'left')  byNumber[num].left  = k;
      else if (side === 'right') byNumber[num].right = k;
      else                  byNumber[num].plain = k;
    });

    // -> { transcriptKeys, lines, matchType } or null.
    const resolve = (page) => {
      const norm = normalizePageKey(page.pageNumber);
      const parsed = parseNormKey(norm);

      if (exactByNorm[norm]) {
        const tk = exactByNorm[norm];
        return { transcriptKeys: [tk], lines: parsedTranscript[tk], matchType: 'exact' };
      }

      if (!parsed) return null;
      const { num, side } = parsed;
      const bucket = byNumber[num];
      if (!bucket) return null;

      if (side) {
        // Split image against a whole transcript page: "3_left" -> "3".
        if (bucket.plain) {
          return {
            transcriptKeys: [bucket.plain],
            lines: parsedTranscript[bucket.plain],
            matchType: 'split-image→unsplit-transcript',
          };
        }
        // Only the opposite side exists — that's a mismatch, so skip it.
      } else {
        // Whole image against a split transcript: combine left + right.
        if (bucket.left && bucket.right) {
          return {
            transcriptKeys: [bucket.left, bucket.right],
            lines: [
              ...parsedTranscript[bucket.left],
              ...parsedTranscript[bucket.right],
            ],
            matchType: 'unsplit-image→combined-transcript',
          };
        }
        if (bucket.left) {
          return { transcriptKeys: [bucket.left], lines: parsedTranscript[bucket.left], matchType: 'partial-left' };
        }
        if (bucket.right) {
          return { transcriptKeys: [bucket.right], lines: parsedTranscript[bucket.right], matchType: 'partial-right' };
        }
      }
      return null;
    };

    return pages.map((page) => {
      const result = resolve(page);
      return {
        pageNumber: page.pageNumber,
        thumbnail: page.thumbnail,
        isSplit: page.isSplit || false,
        splitSide: page.splitSide || null,
        transcriptKeys: result?.transcriptKeys || null,
        lines: result?.lines || null,
        matchType: result?.matchType || null,
        matched: !!result,
      };
    });
  }, [pages, parsedTranscript]);

  // Seed once, when matchData first lands.
  useEffect(() => {
    if (matchData.length > 0 && !initializedRef.current) {
      initializedRef.current = true;
      setEditedLines((prev) => {
        if (Object.keys(prev).length > 0) return prev; // already seeded
        const seed = {};
        matchData.forEach((d) => {
          seed[d.pageNumber] = d.matched && d.lines ? [...d.lines] : [];
        });
        return seed;
      });
    }
  }, [matchData]);

  const handleLinesChange = useCallback((pageNumber, newLines) => {
    setEditedLines((prev) => ({ ...prev, [pageNumber]: newLines }));
  }, []);

  const matchedCount = matchData.filter((d) => d.matched).length;
  const unmatchedCount = matchData.filter((d) => !d.matched).length;

  const usedTranscriptKeys = useMemo(() => {
    const used = new Set();
    matchData.forEach((d) => (d.transcriptKeys || []).forEach((k) => used.add(k)));
    return used;
  }, [matchData]);
  const transcriptOnlyCount = Object.keys(parsedTranscript || {}).filter(
    (k) => !usedTranscriptKeys.has(k)
  ).length;

  const canContinue = matchData.some(
    (d) => d.matched || (editedLines[d.pageNumber] || []).some((l) => l.trim())
  );

  // ── Filtered list ──
  const filteredData = useMemo(() => {
    if (filter === 'matched') return matchData.filter((d) => d.matched);
    if (filter === 'unmatched') return matchData.filter((d) => !d.matched);
    return matchData;
  }, [matchData, filter]);

  React.useEffect(() => {
    if (selectedPageNum === null && filteredData.length > 0) {
      setSelectedPageNum(filteredData[0].pageNumber);
    }
  }, [filteredData, selectedPageNum]);

  const selectedEntry = matchData.find((d) => d.pageNumber === selectedPageNum);
  const filteredIdx = filteredData.findIndex((d) => d.pageNumber === selectedPageNum);

  const navigatePage = useCallback((dir) => {
    const newIdx = filteredIdx + dir;
    if (newIdx >= 0 && newIdx < filteredData.length) {
      setSelectedPageNum(filteredData[newIdx].pageNumber);
    }
  }, [filteredIdx, filteredData]);

  const handleContinue = () => {
    // Unmatched pages count too if the user typed a transcript for them.
    const selectedPages = matchData
      .filter((d) => d.matched || (editedLines[d.pageNumber] || []).some((l) => l.trim()))
      .map((d) => d.pageNumber);

    // User edits win over the parsed lines.
    const mergedTranscript = {};
    matchData.forEach((d) => {
      const edited = editedLines[d.pageNumber];
      const nonEmpty = edited ? edited.filter((l) => l.trim()) : [];
      if (nonEmpty.length > 0) {
        mergedTranscript[String(d.pageNumber)] = nonEmpty;
      } else if (d.matched && d.lines) {
        mergedTranscript[String(d.pageNumber)] = d.lines;
      }
    });

    onNext(selectedPages, mergedTranscript);
  };

  return (
    <div className="flex flex-col h-full w-full overflow-hidden bg-white">
      {/* ── Header ──────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 bg-white/90 backdrop-blur-sm flex-shrink-0 shadow-sm">
        <div className="flex items-center gap-4">
          <button
            onClick={onBack}
            className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-500 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-all duration-150"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </button>
          <div className="w-px h-6 bg-gray-200" />
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-lg text-white shadow-md shadow-emerald-500/20">
              <Layers className="w-4 h-4" />
            </div>
            <div>
              <h1 className="text-base font-bold text-gray-800">Review Page Matches</h1>
              <p className="text-xs text-gray-500">Only matched pages will proceed to preprocessing</p>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-3">
          <StatPill count={matchData.length} label="Total Pages" color="gray" />
          <StatPill count={matchedCount} label="Matched" color="emerald" />
          {unmatchedCount > 0 && <StatPill count={unmatchedCount} label="Unmatched" color="amber" />}
          {transcriptOnlyCount > 0 && <StatPill count={transcriptOnlyCount} label="Transcript only" color="blue" />}
        </div>

        {/* Continue button */}
        <button
          onClick={handleContinue}
          disabled={!canContinue}
          className={`flex items-center gap-2 px-6 py-2.5 rounded-xl font-bold text-sm transition-all duration-200 ${
            canContinue
              ? 'bg-gradient-to-r from-emerald-500 to-teal-600 text-white shadow-lg shadow-emerald-500/25 hover:shadow-xl hover:-translate-y-0.5'
              : 'bg-gray-100 text-gray-400 cursor-not-allowed'
          }`}
        >
          Continue to Preprocessing
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* ── Mismatch warning ──────────────────────────────────────────── */}
      {unmatchedCount > 0 && (
        <div className="flex items-center gap-3 px-6 py-2.5 bg-amber-50 border-b border-amber-100 flex-shrink-0">
          <AlertTriangle className="w-4 h-4 text-amber-500 flex-shrink-0" />
          <p className="text-xs text-amber-700">
            <span className="font-semibold">{unmatchedCount} page{unmatchedCount !== 1 ? 's' : ''}</span> have no matching transcript entry and will be skipped.
            {transcriptOnlyCount > 0 && (
              <span className="ml-2 text-amber-600">
                {transcriptOnlyCount} transcript page{transcriptOnlyCount !== 1 ? 's' : ''} have no matching image.
              </span>
            )}
          </p>
        </div>
      )}

      {/* ── Body: thumbnail rail + detail panel ─────────────────────── */}
      <div className="flex-1 flex overflow-hidden">
        {/* LEFT: Thumbnail rail */}
        <div className="w-52 flex-shrink-0 border-r border-gray-100 flex flex-col overflow-hidden bg-gray-50/60">
          {/* Filter tabs */}
          <div className="flex gap-1 p-2 flex-shrink-0 border-b border-gray-100 bg-white">
            {[
              { key: 'all', label: 'All', count: matchData.length },
              { key: 'matched', label: 'Matched', count: matchedCount },
              { key: 'unmatched', label: 'Skip', count: unmatchedCount },
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => { setFilter(tab.key); setSelectedPageNum(null); }}
                className={`flex-1 py-1.5 px-1 text-[10px] font-semibold rounded-lg transition-all flex flex-col items-center gap-0.5 ${
                  filter === tab.key
                    ? tab.key === 'unmatched'
                      ? 'bg-amber-100 text-amber-700'
                      : 'bg-emerald-100 text-emerald-700'
                    : 'text-gray-500 hover:bg-gray-100'
                }`}
              >
                <span className="text-sm font-bold">{tab.count}</span>
                {tab.label}
              </button>
            ))}
          </div>

          {/* Page list */}
          <div className="flex-1 overflow-y-auto py-2 px-2 space-y-1.5">
            {filteredData.map((entry) => (
              <button
                key={entry.pageNumber}
                onClick={() => setSelectedPageNum(entry.pageNumber)}
                className={`w-full rounded-xl overflow-hidden border-2 transition-all duration-150 text-left group ${
                  selectedPageNum === entry.pageNumber
                    ? entry.matched
                      ? 'border-emerald-400 shadow-md shadow-emerald-500/20 bg-white'
                      : 'border-amber-400 shadow-md shadow-amber-500/20 bg-white'
                    : 'border-transparent hover:border-gray-200 bg-white/80 hover:shadow-sm'
                }`}
              >
                {/* Thumbnail */}
                <div className="relative w-full aspect-[3/4] bg-gray-100 overflow-hidden">
                  {entry.thumbnail ? (
                    <img
                      src={entry.thumbnail}
                      alt={`Page ${entry.pageNumber}`}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <Image className="w-6 h-6 text-gray-400" />
                    </div>
                  )}
                  {/* Match badge overlay */}
                  <div className={`absolute top-1.5 right-1.5 w-5 h-5 rounded-full flex items-center justify-center shadow ${
                    entry.matched ? 'bg-emerald-500' : 'bg-amber-400'
                  }`}>
                    {entry.matched
                      ? <CheckCircle2 className="w-3.5 h-3.5 text-white" />
                      : <XCircle className="w-3.5 h-3.5 text-white" />
                    }
                  </div>
                  {/* Split side badge */}
                  {entry.isSplit && (
                    <div className="absolute bottom-1 left-1 px-1.5 py-0.5 bg-black/50 text-white text-[9px] font-bold rounded uppercase tracking-wide">
                      {entry.splitSide}
                    </div>
                  )}
                </div>
                {/* Page number */}
                <div className="px-2 py-1.5 flex items-center justify-between">
                  <span className="text-xs font-bold text-gray-700 truncate">
                    {formatPageLabel(entry.pageNumber)}
                  </span>
                  {entry.matched
                    ? <span className="text-[10px] text-emerald-600 font-semibold flex-shrink-0">{(editedLines[entry.pageNumber] ?? entry.lines)?.length ?? 0}L</span>
                    : (editedLines[entry.pageNumber] || []).some((l) => l.trim())
                      ? <span className="text-[10px] text-indigo-600 font-semibold flex-shrink-0">{(editedLines[entry.pageNumber] || []).filter((l) => l.trim()).length}L</span>
                      : <span className="text-[10px] text-amber-500 font-semibold flex-shrink-0">skip</span>
                  }
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* RIGHT: Detail panel */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {selectedEntry ? (
            <>
              {/* Detail header */}
              <div className="flex items-center justify-between px-6 py-3 border-b border-gray-100 flex-shrink-0 bg-white/80">
                <div className="flex items-center gap-3 flex-wrap">
                  <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold ${
                    selectedEntry.matched
                      ? 'bg-emerald-100 text-emerald-700'
                      : 'bg-amber-100 text-amber-700'
                  }`}>
                    {selectedEntry.matched
                      ? <><CheckCircle2 className="w-3.5 h-3.5" /> Matched</>
                      : <><XCircle className="w-3.5 h-3.5" /> No transcript</>
                    }
                  </div>
                  <h2 className="font-bold text-gray-700 text-sm">
                    {formatPageLabel(selectedEntry.pageNumber)}
                  </h2>
                  {selectedEntry.matched && (
                    <span className="text-xs text-gray-400">
                      {(editedLines[selectedEntry.pageNumber] ?? selectedEntry.lines)?.length ?? 0} lines
                    </span>
                  )}
                  {selectedEntry.matched && selectedEntry.matchType && selectedEntry.matchType !== 'exact' && (
                    <MatchTypePill matchType={selectedEntry.matchType} transcriptKeys={selectedEntry.transcriptKeys} />
                  )}
                </div>

                <div className="flex items-center gap-2">
                  {/* Tab selector */}
                  <div className="flex bg-gray-100 rounded-lg p-0.5 gap-0.5">
                    {[
                      { key: 'both', label: 'Both', icon: Eye },
                      { key: 'image', label: 'Image', icon: Image },
                      { key: 'transcript', label: 'Text', icon: FileText },
                    ].map(({ key, label, icon: Icon }) => (
                      <button
                        key={key}
                        onClick={() => setDetailTab(key)}
                        className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-md transition-all ${
                          detailTab === key
                            ? 'bg-white text-gray-800 shadow-sm'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        <Icon className="w-3.5 h-3.5" />
                        {label}
                      </button>
                    ))}
                  </div>

                  {/* Navigate */}
                  <div className="flex items-center gap-1 ml-2">
                    <button
                      onClick={() => navigatePage(-1)}
                      disabled={filteredIdx === 0}
                      className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 transition-colors"
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                    <span className="text-xs text-gray-400 min-w-[48px] text-center">
                      {filteredIdx + 1} / {filteredData.length}
                    </span>
                    <button
                      onClick={() => navigatePage(1)}
                      disabled={filteredIdx >= filteredData.length - 1}
                      className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 transition-colors"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>

              {/* Detail content */}
              <div className="flex-1 overflow-hidden">
                {detailTab === 'both' ? (
                  <div className="h-full grid grid-cols-2 gap-0">
                    <ImagePanel entry={selectedEntry} />
                    <TranscriptPanel entry={selectedEntry} editedLines={editedLines} onLinesChange={handleLinesChange} />
                  </div>
                ) : detailTab === 'image' ? (
                  <div className="h-full">
                    <ImagePanel entry={selectedEntry} fullHeight />
                  </div>
                ) : (
                  <div className="h-full">
                    <TranscriptPanel entry={selectedEntry} fullHeight editedLines={editedLines} onLinesChange={handleLinesChange} />
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-gray-400 gap-3">
              <div className="w-16 h-16 bg-gray-100 rounded-2xl flex items-center justify-center">
                <Eye className="w-8 h-8 text-gray-300" />
              </div>
              <p className="text-sm font-medium">Select a page to preview</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ──

function ImagePanel({ entry, fullHeight }) {
  return (
    <div className={`flex flex-col bg-gray-50 border-r border-gray-100 overflow-hidden ${fullHeight ? 'h-full' : ''}`}>
      <div className="px-4 py-2.5 flex-shrink-0 border-b border-gray-100 bg-white/70 flex items-center gap-2">
        <Image className="w-3.5 h-3.5 text-gray-400" />
        <span className="text-xs font-semibold text-gray-600">Book Image</span>
        {entry.isSplit && (
          <span className={`ml-1 px-2 py-0.5 text-[10px] font-bold rounded-full uppercase tracking-wide ${
            entry.splitSide === 'left' ? 'bg-indigo-100 text-indigo-600' : 'bg-violet-100 text-violet-600'
          }`}>
            {entry.splitSide} half
          </span>
        )}
        <span className="ml-auto text-xs text-gray-400 font-mono">{formatPageLabel(entry.pageNumber)}</span>
      </div>
      <div className="flex-1 overflow-auto flex items-start justify-center p-4">
        {entry.thumbnail ? (
          <img
            src={entry.thumbnail}
            alt={`Page ${entry.pageNumber}`}
            className="max-w-full max-h-full object-contain rounded-lg shadow-md"
          />
        ) : (
          <div className="flex flex-col items-center justify-center gap-3 text-gray-400 h-full">
            <Image className="w-12 h-12" />
            <p className="text-sm">No image available</p>
          </div>
        )}
      </div>
    </div>
  );
}

function TranscriptPanel({ entry, fullHeight, editedLines = {}, onLinesChange }) {
  const lines = editedLines[entry.pageNumber] ?? (entry.lines ? [...entry.lines] : []);
  const lineCount = lines.filter((l) => l.trim()).length;
  const hasCustom = !entry.matched && lines.some((l) => l.trim());

  const handleTextChange = (e) => {
    const next = e.target.value.split('\n');
    onLinesChange?.(entry.pageNumber, next);
  };

  return (
    <div className={`flex flex-col overflow-hidden ${fullHeight ? 'h-full' : ''}`}>
      {/* Panel header */}
      <div className="px-4 py-2.5 flex-shrink-0 border-b border-gray-100 bg-white/70 flex items-center gap-2">
        <FileText className="w-3.5 h-3.5 text-gray-400" />
        <span className="text-xs font-semibold text-gray-600">Transcript</span>
        {entry.matched && entry.transcriptKeys && (
          <div className="flex items-center gap-1 ml-1">
            {entry.transcriptKeys.map((k) => (
              <span key={k} className="px-1.5 py-0.5 bg-emerald-50 text-emerald-600 text-[10px] font-mono font-bold rounded border border-emerald-100">
                {k}
              </span>
            ))}
          </div>
        )}
        {hasCustom && (
          <span className="px-1.5 py-0.5 bg-indigo-50 text-indigo-600 text-[10px] font-semibold rounded-full ml-1">
            Custom
          </span>
        )}
        <span className="ml-auto text-xs text-gray-400">{lineCount} lines</span>
      </div>

      {entry.matched && entry.matchType && entry.matchType !== 'exact' && (
        <div className="px-4 py-2 bg-blue-50/60 border-b border-blue-100 flex items-center gap-2 flex-shrink-0">
          <span className="text-[10px] text-blue-600 font-medium">
            {matchTypeHint(entry.matchType, entry.transcriptKeys)}
          </span>
        </div>
      )}

      {/* Notepad-style textarea */}
      <div className="flex-1 overflow-hidden p-3">
        <textarea
          value={lines.join('\n')}
          onChange={handleTextChange}
          spellCheck={false}
          placeholder={entry.matched
            ? 'Edit transcript lines here…'
            : 'Type transcript for this page to include it in the pipeline…'}
          className="w-full h-full resize-none text-sm text-gray-700 leading-relaxed font-mono bg-gray-50 border border-gray-200 rounded-xl p-4 focus:outline-none focus:ring-2 focus:ring-indigo-200 focus:border-indigo-300 placeholder-gray-300 transition-all"
        />
      </div>
    </div>
  );
}

// ── Helpers ──

// "3_left" -> "Page 3 · Left"
function formatPageLabel(pageNumber) {
  const s = String(pageNumber);
  const m = s.match(/^(\d+)(?:_(left|right))?$/i);
  if (!m) return `Page ${s}`;
  const num = m[1];
  const side = m[2];
  if (side) return `Page ${num} · ${side.charAt(0).toUpperCase() + side.slice(1)}`;
  return `Page ${num}`;
}


function matchTypeHint(matchType, transcriptKeys) {
  if (matchType === 'split-image→unsplit-transcript') {
    return `Split image half matched to unsplit transcript entry "${transcriptKeys?.[0]}"`;
  }
  if (matchType === 'unsplit-image→combined-transcript') {
    return `Unsplit image matched by combining transcript entries: ${(transcriptKeys || []).join(' + ')}`;
  }
  if (matchType === 'partial-left') {
    return `Only left-side transcript entry "${transcriptKeys?.[0]}" found`;
  }
  if (matchType === 'partial-right') {
    return `Only right-side transcript entry "${transcriptKeys?.[0]}" found`;
  }
  return matchType;
}

function MatchTypePill({ matchType, transcriptKeys }) {
  const isCombined = matchType === 'unsplit-image→combined-transcript';
  const isAdapted = matchType?.includes('→');
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold ${
      isCombined ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'
    }`}>
      {isCombined ? '⇔ Combined' : isAdapted ? '≈ Adapted' : matchType}
    </span>
  );
}

function StatPill({ count, label, color }) {
  const colors = {
    gray: 'bg-gray-100 text-gray-600',
    emerald: 'bg-emerald-100 text-emerald-700',
    amber: 'bg-amber-100 text-amber-700',
    blue: 'bg-blue-100 text-blue-700',
  };
  return (
    <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold ${colors[color]}`}>
      <span className="text-sm font-black">{count}</span>
      {label}
    </div>
  );
}

