import { useState, useEffect, useMemo, memo } from 'react';
import {
  FileText,
  Copy,
  Check,
  Download,
  Save,
  RotateCcw,
  Type,
  Loader2,
  File,
  FileType,
  BookOpen,
  Edit3,
  Eye,
  Sparkles,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { getLLMTemplates } from '../../services/llmApi';

// [illegible], [margin], [unclear], ...
const MARKER_REGEX = /(\[(?:illegible|margin|unclear|damaged|torn|faded|stained|missing|blank|unknown|(?:\?+))\])/gi;


function HighlightedTranscript({ text, monospace }) {
  const parts = useMemo(() => {
    if (!text) return [];
    
    const result = [];
    let lastIndex = 0;
    let match;
    
    // The regex is /g and module-level, so reset before each pass.
    MARKER_REGEX.lastIndex = 0;
    
    while ((match = MARKER_REGEX.exec(text)) !== null) {
      if (match.index > lastIndex) {
        result.push({
          type: 'text',
          content: text.slice(lastIndex, match.index),
          key: `text-${lastIndex}`,
        });
      }
      
      result.push({
        type: 'marker',
        content: match[0],
        key: `marker-${match.index}`,
      });
      
      lastIndex = match.index + match[0].length;
    }
    
    if (lastIndex < text.length) {
      result.push({
        type: 'text',
        content: text.slice(lastIndex),
        key: `text-${lastIndex}`,
      });
    }
    
    return result;
  }, [text]);

  return (
    <div 
      className={`w-full h-full p-3 text-sm leading-relaxed overflow-y-auto whitespace-pre-wrap ${
        monospace ? 'font-mono' : 'font-sans'
      }`}
    >
      {parts.map((part) => 
        part.type === 'marker' ? (
          <span 
            key={part.key} 
            className="text-red-600 font-semibold bg-red-50 px-0.5 rounded"
            title="OCR uncertainty marker"
          >
            {part.content}
          </span>
        ) : (
          <span key={part.key}>{part.content}</span>
        )
      )}
    </div>
  );
}

// Transcript viewer with copy, export and LLM cleanup.
function TranscriptPanel({
  pageNumber,
  transcript,
  originalTranscript,
  isProcessing,
  isProcessed,
  hasAnyTranscript,
  processedCount,
  onTranscriptChange,
  onReset,
  onExport,
  onSaveLocal,
  exporting,
  savingLocal,
  onLLMProcess,
  isLLMProcessing,
  llmEnabled,
}) {
  const [copied, setCopied] = useState(false);
  const [monospace, setMonospace] = useState(true);
  const [editMode, setEditMode] = useState(false);
  const [llmExpanded, setLlmExpanded] = useState(false);
  const [llmTemplates, setLlmTemplates] = useState([]);
  const [selectedTemplate, setSelectedTemplate] = useState('full_cleanup');

  useEffect(() => {
    if (!llmEnabled) return;
    let cancelled = false;
    getLLMTemplates()
      .then(data => {
        if (!cancelled && data.templates) {
          setLlmTemplates(data.templates);
        }
      })
      .catch(() => {}); // Silently fail - templates will just not show
    return () => { cancelled = true; };
  }, [llmEnabled]);

  const hasChanges = transcript !== originalTranscript;
  
  const markerCount = useMemo(() => {
    if (!transcript) return 0;
    MARKER_REGEX.lastIndex = 0;
    const matches = transcript.match(MARKER_REGEX);
    return matches ? matches.length : 0;
  }, [transcript]);

  const handleCopy = async () => {
    if (transcript) {
      await navigator.clipboard.writeText(transcript);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleDownloadTxt = () => {
    if (!transcript) return;
    // BOM keeps accents intact in Notepad/Excel.
    const utf8Bom = '\uFEFF';
    const blob = new Blob([utf8Bom + transcript], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `page_${pageNumber}_transcript.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white/95 backdrop-blur-sm rounded-xl border border-gray-200/80 shadow-sm overflow-hidden flex flex-col h-full">
      {/* Sticky Header */}
      <div className="px-3 py-2 bg-gradient-to-r from-blue-50/80 via-white to-indigo-50/50 border-b border-gray-100/80 flex items-center justify-between shrink-0 sticky top-0 z-10">
        <h3 className="font-bold text-gray-800 flex items-center gap-2 text-sm">
          <div className="p-1 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg text-white">
            <FileText size={12} />
          </div>
          <span>Transcript</span>
          {isProcessed && (
            <span className="text-xs text-gray-400 font-normal">
              #{pageNumber}
            </span>
          )}
        </h3>

        {/* Actions */}
        <div className="flex items-center gap-0.5">
          {isProcessed && transcript && (
            <>
              {/* Edit/View toggle */}
              <button
                onClick={() => setEditMode(!editMode)}
                className={`p-1.5 rounded-lg transition-all duration-200 ${editMode
                    ? 'bg-amber-100 text-amber-600'
                    : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
                  }`}
                title={editMode ? 'Switch to view mode' : 'Switch to edit mode'}
              >
                {editMode ? <Eye size={14} /> : <Edit3 size={14} />}
              </button>

              {/* Monospace toggle */}
              <button
                onClick={() => setMonospace(!monospace)}
                className={`p-1.5 rounded-lg transition-all duration-200 ${monospace
                    ? 'bg-blue-100 text-blue-600'
                    : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
                  }`}
                title={monospace ? 'Switch to sans-serif' : 'Switch to monospace'}
              >
                <Type size={14} />
              </button>

              {/* Copy button */}
              <button
                onClick={handleCopy}
                className={`p-1.5 rounded-lg transition-all duration-200 ${copied
                    ? 'bg-green-100 text-green-600'
                    : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
                  }`}
                title="Copy to clipboard"
              >
                {copied ? <Check size={14} /> : <Copy size={14} />}
              </button>

              {/* Download single page */}
              <button
                onClick={handleDownloadTxt}
                className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-all duration-200"
                title="Download this page as TXT"
              >
                <Download size={14} />
              </button>

              {/* Reset button */}
              {hasChanges && (
                <button
                  onClick={onReset}
                  className="p-1.5 rounded-lg text-amber-500 hover:text-amber-600 hover:bg-amber-50 transition-all duration-200"
                  title="Reset to original"
                >
                  <RotateCcw size={14} />
                </button>
              )}
            </>
          )}
        </div>
      </div>

      {/* Transcript Content - flex-1 takes remaining space */}
      <div className="flex-1 relative min-h-0 overflow-hidden">
        {isProcessing ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-gradient-to-br from-blue-50/50 to-indigo-50/50">
            <div className="relative">
              <Loader2 size={36} className="animate-spin text-blue-600" />
              <div className="absolute inset-0 blur-xl bg-blue-400/30 animate-pulse" />
            </div>
            <p className="text-blue-600 font-semibold mt-3 text-sm">Extracting text...</p>
            <p className="text-blue-500 text-xs">This may take a few seconds</p>
          </div>
        ) : !isProcessed ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400 bg-gradient-to-br from-gray-50/50 to-gray-100/30">
            <div className="p-3 bg-gradient-to-br from-gray-100 to-gray-50 rounded-xl">
              <FileText size={36} className="opacity-40" />
            </div>
            <p className="font-semibold mt-3 text-sm">No transcript yet</p>
            <p className="text-xs">Process this page to extract text</p>
          </div>
        ) : editMode ? (
          <textarea
            value={transcript || ''}
            onChange={(e) => onTranscriptChange(e.target.value)}
            className={`w-full h-full p-3 resize-none focus:outline-none text-sm leading-relaxed 
                       border-0 focus:ring-0 bg-transparent overflow-y-auto ${monospace ? 'font-mono' : 'font-sans'}`}
            placeholder="Transcript will appear here..."
          />
        ) : (
          <HighlightedTranscript text={transcript} monospace={monospace} />
        )}
      </div>

      {/* Footer with stats */}
      {isProcessed && transcript && (
        <div className="px-3 py-1.5 border-t border-gray-100 bg-gradient-to-r from-gray-50 to-white flex items-center justify-between text-xs text-gray-500 shrink-0">
          <div className="flex items-center gap-2">
            <span className="font-medium">{transcript.length.toLocaleString()} chars</span>
            {markerCount > 0 && (
              <span className="text-red-600 font-semibold flex items-center gap-1 px-1.5 py-0.5 bg-red-50 rounded-full">
                <span className="w-1 h-1 bg-red-500 rounded-full" />
                {markerCount} uncertain
              </span>
            )}
          </div>
          {hasChanges && (
            <span className="text-amber-600 font-semibold flex items-center gap-1 px-1.5 py-0.5 bg-amber-50 rounded-full">
              <span className="w-1 h-1 bg-amber-500 rounded-full animate-pulse" />
              Modified
            </span>
          )}
        </div>
      )}

      {/* AI Polish Section */}
      {llmEnabled && isProcessed && transcript && (
        <div className="border-t border-gray-100 shrink-0">
          <button
            onClick={() => setLlmExpanded(!llmExpanded)}
            className="w-full px-3 py-2 flex items-center justify-between text-xs font-bold text-gray-700 hover:bg-gray-50 transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <div className="p-1 bg-gradient-to-br from-purple-500 to-pink-500 rounded text-white">
                <Sparkles size={10} />
              </div>
              AI Polish
            </span>
            {llmExpanded ? <ChevronUp size={14} className="text-gray-400" /> : <ChevronDown size={14} className="text-gray-400" />}
          </button>
          {llmExpanded && (
            <div className="px-3 pb-3 space-y-2">
              <select
                value={selectedTemplate}
                onChange={(e) => setSelectedTemplate(e.target.value)}
                className="w-full text-xs border border-gray-200 rounded-lg px-2 py-1.5 bg-white focus:outline-none focus:ring-1 focus:ring-purple-300 focus:border-purple-300"
                disabled={isLLMProcessing}
              >
                {llmTemplates.length > 0 ? (
                  llmTemplates.map(t => (
                    <option key={t.id} value={t.id}>{t.name}</option>
                  ))
                ) : (
                  <>
                    <option value="full_cleanup">Full Cleanup</option>
                    <option value="spelling_correction">Spelling Correction</option>
                    <option value="formatting">Formatting</option>
                    <option value="historical_normalization">Historical Normalization</option>
                  </>
                )}
              </select>
              {llmTemplates.length > 0 && (
                <p className="text-[10px] text-gray-400 leading-tight">
                  {llmTemplates.find(t => t.id === selectedTemplate)?.description || ''}
                </p>
              )}
              <button
                onClick={() => onLLMProcess && onLLMProcess(selectedTemplate)}
                disabled={isLLMProcessing || !transcript}
                className="w-full flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-semibold
                         bg-gradient-to-r from-purple-500 to-pink-500 text-white
                         hover:from-purple-600 hover:to-pink-600 shadow-sm hover:shadow
                         transition-all duration-200
                         disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLLMProcessing ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    Polishing...
                  </>
                ) : (
                  <>
                    <Sparkles size={14} />
                    Polish with AI
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Export Panel - Compact */}
      {hasAnyTranscript && (
        <div className="px-3 py-3 border-t border-gray-200 bg-gradient-to-r from-white to-gray-50/50 shrink-0">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-bold text-gray-700 flex items-center gap-1.5">
              <div className="p-1 bg-gradient-to-br from-blue-500 to-indigo-600 rounded text-white">
                <Download size={10} />
              </div>
              Export All
            </h4>
            <span className="text-[10px] text-gray-500 font-medium bg-gray-100 px-1.5 py-0.5 rounded-full">
              {processedCount} pages
            </span>
          </div>

          <div className="grid grid-cols-4 gap-1.5">
            {[
              { format: 'txt', icon: FileText, label: 'TXT' },
              { format: 'csv', icon: File, label: 'CSV' },
              { format: 'json', icon: FileType, label: 'JSON' },
              { format: 'pdf', icon: BookOpen, label: 'PDF' },
            ].map(({ format, icon: Icon, label }) => (
              <button
                key={format}
                onClick={() => onExport(format)}
                disabled={exporting}
                className="flex flex-col items-center gap-1 py-2 rounded-lg border border-gray-200 
                         bg-white hover:border-blue-300 hover:bg-blue-50 hover:shadow-sm 
                         transition-all duration-200
                         disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {exporting === format ? (
                  <Loader2 size={16} className="animate-spin text-blue-600" />
                ) : (
                  <Icon size={16} className="text-blue-600" />
                )}
                <span className="text-[10px] font-semibold text-gray-700">{label}</span>
              </button>
            ))}
          </div>

          {onSaveLocal && (
            <button
              onClick={onSaveLocal}
              disabled={savingLocal}
              className="mt-2 w-full flex items-center justify-center gap-2 py-2 rounded-lg border border-cyan-200 bg-cyan-50 text-cyan-700 hover:bg-cyan-100 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {savingLocal ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
              Save to My Files
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// Typing is the hot path — memo keeps unrelated parent state (rate limit,
// zoom) from re-rendering the panel on every keystroke.
export default memo(TranscriptPanel);
