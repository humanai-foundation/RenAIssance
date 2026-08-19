import { useState } from 'react';
import {
  Key,
  Eye,
  EyeOff,
  Loader2,
  CheckCircle2,
  AlertCircle,
  ShieldCheck,
  ShieldX,
  RefreshCw,
  Zap,
  Minus,
  Plus,
  FileText,
  ChevronDown,
  ChevronUp,
  RotateCcw,
} from 'lucide-react';

const PROVIDER_LABELS = {
  gemini: 'Gemini',
  chatgpt: 'ChatGPT',
  deepseek: 'DeepSeek',
  qwen: 'Qwen',
};

const KEY_PLACEHOLDERS = {
  gemini: 'Enter Gemini API key...',
  chatgpt: 'Enter OpenAI API key...',
  deepseek: 'Enter DeepSeek API key...',
  qwen: 'Enter DashScope API key...',
};

// Provider + API key panel.
export default function SidebarConfig({
  apiKey,
  onApiKeyChange,
  selectedModel,
  onModelChange,
  models = [],
  isKeyValid,
  isValidating,
  onVerifyKey,
  backendOnline,
  batchSize,
  onBatchSizeChange,
  provider = 'gemini',
  customPrompt,
  onCustomPromptChange,
}) {
  const providerLabel = PROVIDER_LABELS[provider] || 'API';
  const [showKey, setShowKey] = useState(false);
  const [promptExpanded, setPromptExpanded] = useState(true);
  const isCustomPrompt = customPrompt !== null && customPrompt !== undefined && customPrompt !== '';

  const getKeyStatusBadge = () => {
    if (isValidating) {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-blue-50 text-blue-600 text-xs font-medium rounded-full">
          <Loader2 size={12} className="animate-spin" />
          Verifying...
        </span>
      );
    }
    if (isKeyValid === true) {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-green-50 text-green-600 text-xs font-medium rounded-full">
          <ShieldCheck size={12} />
          Valid
        </span>
      );
    }
    if (isKeyValid === false) {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-red-50 text-red-600 text-xs font-medium rounded-full">
          <ShieldX size={12} />
          Invalid
        </span>
      );
    }
    return null;
  };

  return (
    <div className="bg-white/95 backdrop-blur-sm rounded-xl border border-gray-200/80 shadow-sm overflow-hidden shrink-0">
      {/* Header */}
      <div className="px-3 py-2 bg-gradient-to-r from-blue-50/80 to-white border-b border-gray-100">
        <h3 className="font-semibold text-gray-800 flex items-center gap-2 text-sm">
          <Key size={16} className="text-blue-600" />
          {providerLabel} Config
        </h3>
      </div>

      <div className="p-3 space-y-3">
        {/* Backend Status */}
        {backendOnline === false && (
          <div className="px-2.5 py-1.5 bg-red-50 border border-red-100 rounded-lg">
            <p className="text-[11px] text-red-600 flex items-center gap-1.5">
              <AlertCircle size={12} />
              Backend offline
            </p>
          </div>
        )}

        {/* API Key Input */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className="text-xs font-medium text-gray-700">
              API Key
            </label>
            {getKeyStatusBadge()}
          </div>
          <div className="relative">
            <input
              type={showKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => onApiKeyChange(e.target.value)}
              placeholder={KEY_PLACEHOLDERS[provider] || 'Enter API key...'}
              className="w-full px-2.5 py-2 pr-16 text-sm border border-gray-200 rounded-lg 
                       focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-400
                       transition-colors bg-gray-50 focus:bg-white"
            />
            <div className="absolute right-1.5 top-1/2 -translate-y-1/2 flex items-center gap-0.5">
              <button
                onClick={() => setShowKey(!showKey)}
                className="p-1 text-gray-400 hover:text-gray-600 rounded transition-colors"
                title={showKey ? 'Hide key' : 'Show key'}
              >
                {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
          </div>

          {/* Verify Button */}
          <button
            onClick={onVerifyKey}
            disabled={!apiKey || apiKey.length < 10 || isValidating}
            className="mt-1.5 w-full flex items-center justify-center gap-1.5 px-2.5 py-1.5 
                     text-xs font-medium rounded-lg transition-colors
                     disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed
                     bg-blue-50 text-blue-600 hover:bg-blue-100"
          >
            {isValidating ? (
              <>
                <Loader2 size={12} className="animate-spin" />
                Verifying...
              </>
            ) : (
              <>
                <RefreshCw size={12} />
                Verify Key
              </>
            )}
          </button>
        </div>

        {/* Model Selection */}
        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1.5">
            Model
          </label>
          <select
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
            className="w-full px-2.5 py-2 text-sm border border-gray-200 rounded-lg 
                     focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-400
                     transition-colors bg-gray-50 focus:bg-white cursor-pointer"
          >
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
          <p className="mt-1 text-[10px] text-gray-500 leading-relaxed">
            {models.find((m) => m.id === selectedModel)?.description || ''}
          </p>
        </div>

        {/* Batch Size / Rate Limit Setting */}
        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1.5 flex items-center gap-1.5">
            <Zap size={12} className="text-amber-500" />
            Concurrent Requests
          </label>
          <div className="flex items-center gap-1">
            <button
              onClick={() => onBatchSizeChange(Math.max(1, batchSize - 1))}
              disabled={batchSize <= 1}
              className="p-1.5 rounded-lg border border-gray-200 text-gray-500 hover:bg-gray-100 
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <Minus size={14} />
            </button>
            <input
              type="number"
              min={1}
              value={batchSize}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                if (!isNaN(val) && val >= 1) {
                  onBatchSizeChange(val);
                }
              }}
              className="w-14 px-2 py-1.5 text-center text-sm font-bold border border-gray-200 rounded-lg 
                       focus:outline-none focus:ring-2 focus:ring-amber-100 focus:border-amber-400
                       bg-gray-50 focus:bg-white [appearance:textfield] 
                       [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
            />
            <button
              onClick={() => onBatchSizeChange(batchSize + 1)}
              className="p-1.5 rounded-lg border border-gray-200 text-gray-500 hover:bg-gray-100 transition-colors"
            >
              <Plus size={14} />
            </button>
          </div>
          <p className="mt-1 text-[10px] text-gray-500 leading-relaxed">
            Pages to process at once.{provider === 'gemini' ? ' Gemini free tier: 5 req/min.' : ''}
          </p>
        </div>

        {/* Custom Prompt Section */}
        <div>
          <button
            onClick={() => setPromptExpanded(!promptExpanded)}
            className="w-full flex items-center justify-between text-xs font-medium text-gray-700 mb-1.5 
                       hover:text-blue-600 transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <FileText size={12} className="text-purple-500" />
              OCR Prompt
              {isCustomPrompt && (
                <span className="inline-flex items-center px-1.5 py-0.5 bg-purple-50 text-purple-600 text-[9px] font-semibold rounded-full">
                  Custom
                </span>
              )}
              {!isCustomPrompt && (
                <span className="inline-flex items-center px-1.5 py-0.5 bg-gray-100 text-gray-500 text-[9px] font-semibold rounded-full">
                  Default
                </span>
              )}
            </span>
            {promptExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {promptExpanded && (
            <div className="space-y-2">
              {/* Toggle between default and custom */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => onCustomPromptChange('')}
                  className={`flex-1 px-2 py-1.5 text-[11px] font-medium rounded-lg border transition-colors ${
                    !isCustomPrompt
                      ? 'bg-blue-50 border-blue-200 text-blue-700'
                      : 'bg-white border-gray-200 text-gray-500 hover:bg-gray-50'
                  }`}
                >
                  System Default
                </button>
                <button
                  onClick={() => {
                    if (!isCustomPrompt) onCustomPromptChange(' ');
                  }}
                  className={`flex-1 px-2 py-1.5 text-[11px] font-medium rounded-lg border transition-colors ${
                    isCustomPrompt
                      ? 'bg-purple-50 border-purple-200 text-purple-700'
                      : 'bg-white border-gray-200 text-gray-500 hover:bg-gray-50'
                  }`}
                >
                  Custom Prompt
                </button>
              </div>

              {isCustomPrompt && (
                <>
                  <textarea
                    value={customPrompt}
                    onChange={(e) => onCustomPromptChange(e.target.value)}
                    placeholder="Enter your custom OCR prompt here..."
                    rows={6}
                    className="w-full px-2.5 py-2 text-xs border border-gray-200 rounded-lg 
                             focus:outline-none focus:ring-2 focus:ring-purple-100 focus:border-purple-400
                             transition-colors bg-gray-50 focus:bg-white resize-y
                             placeholder:text-gray-400 leading-relaxed"
                  />
                  <button
                    onClick={() => onCustomPromptChange('')}
                    className="flex items-center gap-1 text-[10px] text-gray-500 hover:text-purple-600 transition-colors"
                  >
                    <RotateCcw size={10} />
                    Reset to System Default
                  </button>
                </>
              )}

              {!isCustomPrompt && (
                <p className="text-[10px] text-gray-500 leading-relaxed">
                  Using the built-in prompt optimized for historical OCR transcription.
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
