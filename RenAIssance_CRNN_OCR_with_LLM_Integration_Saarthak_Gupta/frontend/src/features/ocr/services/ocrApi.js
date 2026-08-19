// All OCR provider calls in one place. Gemini uses X-Gemini-API-Key and is
// the only provider with server-side rate limiting; the rest use X-API-Key.

import { API_ORIGIN } from '../../../config';
import { toDataUrl } from '../../../utils/imageUrl';

const API_BASE = `${API_ORIGIN}/api`;

export async function getModels(provider) {
    const endpoints = {
        gemini: '/models',
        chatgpt: '/chatgpt-models',
        deepseek: '/deepseek-models',
        qwen: '/qwen-models',
    };

    const endpoint = endpoints[provider] || endpoints.gemini;
    const response = await fetch(`${API_BASE}${endpoint}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch ${provider} models`);
    }
    return response.json();
}

// Gemini only. Falls back to "ready" so a backend hiccup never blocks OCR.
export async function getRateLimitStatus() {
    try {
        const response = await fetch(`${API_BASE}/rate-limit-status`);
        if (!response.ok) {
            return { ready: true, wait_seconds: 0, available_slots: 5 };
        }
        const data = await response.json();
        return {
            ready: data.ready ?? true,
            wait_seconds: data.wait_seconds ?? 0,
            available_slots: data.available_slots ?? (data.ready ? 5 : 0),
            requests_in_window: data.requests_in_window ?? 0,
        };
    } catch (error) {
        console.error('Failed to fetch rate limit status:', error);
        return { ready: true, wait_seconds: 0, available_slots: 5 };
    }
}

// Format check only — the key is really validated on the first OCR call.
export async function verifyApiKey(apiKey) {
    if (!apiKey) {
        return { valid: false, error: 'API key is required' };
    }

    if (apiKey.length < 20) {
        return { valid: false, error: 'API key appears too short' };
    }

    try {
        const response = await fetch(`${API_BASE}/validate-key`, {
            method: 'POST',
            headers: {
                'X-Gemini-API-Key': apiKey,
            },
        });

        if (response.status === 401) {
            const data = await response.json().catch(() => ({}));
            return { valid: false, error: data.detail || 'Invalid API key format' };
        }

        if (response.ok) {
            const data = await response.json().catch(() => ({}));
            return {
                valid: true,
                message: data.message || 'API key format verified. Will be validated on first use.'
            };
        }

        return {
            valid: true,
            message: 'Could not verify with server, but format looks valid. Will be validated on first use.'
        };
    } catch (error) {
        if (apiKey.length >= 20) {
            return {
                valid: true,
                message: 'Server offline. Key format looks valid, will be verified on first use.'
            };
        }
        return { valid: false, error: 'Could not connect to server' };
    }
}

export async function processPageOCR(imageData, model, apiKey, provider = 'gemini', customPrompt = '') {
    const endpoints = {
        gemini: '/gemini-ocr-json',
        chatgpt: '/chatgpt-ocr-json',
        deepseek: '/deepseek-ocr-json',
        qwen: '/qwen-ocr-json',
    };

    const headerName = provider === 'gemini' ? 'X-Gemini-API-Key' : 'X-API-Key';

    if (provider === 'gemini') {
        const rateLimitStatus = await getRateLimitStatus();
        if (!rateLimitStatus.ready) {
            return {
                success: false,
                error: 'rate_limited',
                waitSeconds: rateLimitStatus.wait_seconds,
            };
        }
    }

    const endpoint = endpoints[provider] || endpoints.gemini;
    const requestBody = {
        // Page images are blob: object URLs in app state; backends want base64.
        image_data: await toDataUrl(imageData),
        model: model,
    };
    if (customPrompt && customPrompt.trim()) {
        requestBody.custom_prompt = customPrompt.trim();
    }
    const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            [headerName]: apiKey,
        },
        body: JSON.stringify(requestBody),
    });

    if (response.status === 429) {
        const errorData = await response.json().catch(() => ({}));
        const detail = errorData.detail;
        if (detail && detail.error === 'quota_exceeded') {
            return { success: false, error: 'quota_exceeded', message: detail.message };
        }
        return {
            success: false,
            error: 'rate_limited',
            waitSeconds: detail?.wait_seconds || 20,
        };
    }

    if (response.status === 401) {
        return {
            success: false,
            error: 'invalid_api_key',
        };
    }

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const detail = errorData.detail;
        const message = typeof detail === 'object' ? detail?.message : detail;
        return {
            success: false,
            error: message || `OCR failed (HTTP ${response.status})`,
        };
    }

    const data = await response.json();
    return data;
}

// Gemini only.
export async function processBatchOCR(items, model, apiKey, customPrompt = '') {
    if (!items || items.length === 0) {
        return { success: false, error: 'No items to process' };
    }

    try {
        const rateLimitStatus = await getRateLimitStatus();
        if (!rateLimitStatus.ready || rateLimitStatus.available_slots < items.length) {
            return {
                success: false,
                error: 'rate_limited',
                waitSeconds: rateLimitStatus.wait_seconds || 60,
                availableSlots: rateLimitStatus.available_slots || 0,
            };
        }

        const batchBody = {
            items: items.map(item => ({
                page_index: item.pageIndex,
                image_data: item.imageData,
            })),
            model: model,
        };
        if (customPrompt && customPrompt.trim()) {
            batchBody.custom_prompt = customPrompt.trim();
        }

        const response = await fetch(`${API_BASE}/gemini-ocr-batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Gemini-API-Key': apiKey,
            },
            body: JSON.stringify(batchBody),
        });

        if (response.status === 429) {
            const errorData = await response.json().catch(() => ({}));
            return {
                success: false,
                error: 'rate_limited',
                waitSeconds: errorData.detail?.wait_seconds || 60,
                availableSlots: errorData.detail?.available_slots || 0,
            };
        }

        if (response.status === 401) {
            return {
                success: false,
                error: 'invalid_api_key',
            };
        }

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            return {
                success: false,
                error: errorData.detail || `Batch OCR failed with status ${response.status}`,
            };
        }

        const data = await response.json();

        const quotaExceeded = data.results?.some(r =>
            !r.success && r.error && (
                r.error.toLowerCase().includes('quota') ||
                r.error.toLowerCase().includes('resource_exhausted') ||
                r.error.toLowerCase().includes('rate limit') ||
                r.error.includes('429')
            )
        );

        return {
            success: true,
            results: data.results,
            successfulCount: data.successful_count,
            failedCount: data.failed_count,
            totalProcessingTimeMs: data.total_processing_time_ms,
            quotaExceeded: quotaExceeded,
        };
    } catch (error) {
        console.error('Batch OCR error:', error);
        return {
            success: false,
            error: error.message || 'Batch OCR processing failed',
        };
    }
}

export function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ── Local recognition (CRNN / TrOCR, runs on the backend GPU) ──

export async function getLocalRecognitionModels() {
    const response = await fetch(`${API_BASE}/local-recognition-models`);
    if (!response.ok) {
        throw new Error('Failed to fetch local recognition models');
    }
    return response.json();
}

export async function runLocalRecognition(imageData, boxes, modelId) {
    const response = await fetch(`${API_BASE}/local-recognize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image_data: imageData,
            boxes: boxes,
            model_id: modelId,
        }),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        return {
            success: false,
            error: errorData.detail || `Local recognition failed (${response.status})`,
        };
    }

    const data = await response.json();
    return {
        success: true,
        results: data.results,
        processing_time_ms: data.processing_time_ms,
        model_used: data.model_used,
        model_type: data.model_type,
        device: data.device,
    };
}

// ── Export helpers (pure client-side, no backend round-trip) ──

export function exportCRNNResultsAsText(resultsByPage) {
    const lines = [];
    for (const [pageLabel, texts] of Object.entries(resultsByPage)) {
        lines.push(`── Page ${pageLabel} ──`);
        texts.forEach((t) => lines.push(t));
        lines.push('');
    }
    return new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
}

export function exportCRNNResultsAsJSON(resultsByPage) {
    return new Blob([JSON.stringify(resultsByPage, null, 2)], {
        type: 'application/json;charset=utf-8',
    });
}
