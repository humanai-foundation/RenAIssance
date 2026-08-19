// Provider metadata, fallback model lists and UI config in one place.
// DeepSeek and Qwen are implemented server-side but hidden in the UI.

import {
    Sparkles,
    MessageSquare,
    Bot,
    Globe,
} from 'lucide-react';

export const PROVIDER_LABELS = {
    gemini: 'Gemini',
    chatgpt: 'ChatGPT',
    deepseek: 'DeepSeek',
    qwen: 'Qwen',
};

// Used when the backend can't be reached.
export const FALLBACK_MODELS = {
    gemini: [
        { id: 'gemini-3.1-flash-lite', name: 'Gemini 3.1 Flash Lite', description: 'Fast and light, best free-tier quota (recommended)' },
        { id: 'gemini-3-flash-preview', name: 'Gemini 3 Flash', description: 'Fast, high quality for most documents' },
        { id: 'gemini-3.5-flash', name: 'Gemini 3.5 Flash', description: 'Higher quality flash model' },
        { id: 'gemini-3.1-pro-preview', name: 'Gemini 3.1 Pro', description: 'Most capable (low daily quota)' },
        { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash', description: 'Stable fallback' },
    ],
    chatgpt: [
        { id: 'gpt-5.2', name: 'GPT-5.2', description: 'Latest and most capable multimodal model' },
        { id: 'gpt-5-mini', name: 'GPT-5 Mini', description: 'Smaller, faster, and affordable' },
    ],
    deepseek: [
        { id: 'deepseek-chat', name: 'DeepSeek Chat', description: 'General chat model' },
        { id: 'deepseek-reasoner', name: 'DeepSeek Reasoner', description: 'Reasoning model' },
    ],
    qwen: [
        { id: 'qwen-vl-max', name: 'Qwen VL Max', description: 'Most capable vision model' },
        { id: 'qwen-vl-ocr', name: 'Qwen VL OCR', description: 'Optimized for OCR tasks' },
        { id: 'qwen2.5-vl-72b-instruct', name: 'Qwen2.5 VL 72B', description: 'Large vision-language model' },
        { id: 'qwen2.5-vl-7b-instruct', name: 'Qwen2.5 VL 7B', description: 'Efficient vision model' },
    ],
};

export const DEFAULT_MODELS = {
    gemini: 'gemini-3.1-flash-lite',
    chatgpt: 'gpt-5.2',
    deepseek: 'deepseek-chat',
    qwen: 'qwen-vl-max',
};

export const METHOD_OPTIONS = [
    {
        id: 'api',
        name: 'Gemini',
        icon: Sparkles,
        tagline: 'Google\'s multimodal AI with state-of-the-art document understanding',
        gradient: 'from-blue-500 to-indigo-600',
        lightGradient: 'from-blue-50 to-indigo-50',
        borderColor: 'border-blue-500',
        accentColor: 'text-blue-600',
        shadowColor: 'shadow-blue-500/20',
        recommended: true,
    },
    {
        id: 'chatgpt',
        name: 'ChatGPT',
        icon: MessageSquare,
        tagline: 'OpenAI GPT-4 Vision with powerful multimodal OCR capabilities',
        gradient: 'from-green-500 to-emerald-600',
        lightGradient: 'from-green-50 to-emerald-50',
        borderColor: 'border-green-500',
        accentColor: 'text-green-600',
        shadowColor: 'shadow-green-500/20',
    },
    {
        id: 'deepseek',
        name: 'DeepSeek',
        icon: Bot,
        tagline: 'Cost-effective AI with strong reasoning and open-source foundation',
        gradient: 'from-purple-500 to-violet-600',
        lightGradient: 'from-purple-50 to-violet-50',
        borderColor: 'border-purple-500',
        accentColor: 'text-purple-600',
        shadowColor: 'shadow-purple-500/20',
        disabled: true,
        disabledReason: 'Does not provide API support for images',
    },
    {
        id: 'qwen',
        name: 'Qwen',
        icon: Globe,
        tagline: 'Alibaba\'s vision models with dedicated OCR and multilingual support',
        gradient: 'from-orange-500 to-amber-600',
        lightGradient: 'from-orange-50 to-amber-50',
        borderColor: 'border-orange-500',
        accentColor: 'text-orange-600',
        shadowColor: 'shadow-orange-500/20',
        disabled: true,
        disabledReason: 'Does not provide API support for images',
    },
];

// TextDetectionPage card id -> provider id.
export const PROVIDER_MAP = {
    'api': 'gemini',
    'chatgpt': 'chatgpt',
    'deepseek': 'deepseek',
    'qwen': 'qwen',
};
