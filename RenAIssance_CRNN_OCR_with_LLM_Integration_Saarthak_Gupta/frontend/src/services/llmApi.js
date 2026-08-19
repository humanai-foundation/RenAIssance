// LLM post-processing client — cleans up raw OCR text.
// Provider (gemini/openai/deepseek/qwen) is chosen server-side from the id.

import { API_ORIGIN } from '../config';

const API_BASE = `${API_ORIGIN}/api`;

export async function getLLMTemplates() {
  const response = await fetch(`${API_BASE}/llm/templates`);
  if (!response.ok) {
    throw new Error('Failed to fetch LLM templates');
  }
  return response.json();
}

// -> { providers: [{ id, name, enabled, default_model, models, note? }] }
export async function getLLMProviders() {
  const response = await fetch(`${API_BASE}/llm/providers`);
  if (!response.ok) {
    throw new Error('Failed to fetch LLM providers');
  }
  return response.json();
}

// -> { success, processed_text?, error? }
export async function postProcessWithLLMProvider(
  provider,
  apiKey,
  text,
  model,
  template = 'full_cleanup',
) {
  const response = await fetch(`${API_BASE}/llm/post-process`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-LLM-API-Key': apiKey,
    },
    body: JSON.stringify({ provider, text, model, template }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'LLM processing failed' }));
    throw new Error(err.detail || 'LLM processing failed');
  }

  return response.json();
}
