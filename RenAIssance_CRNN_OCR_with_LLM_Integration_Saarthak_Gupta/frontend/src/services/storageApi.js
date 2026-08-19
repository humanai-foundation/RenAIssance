import { API_ORIGIN } from '../config';

const API_BASE = `${API_ORIGIN}/api/storage`;

export async function fetchStorageOverview() {
  const response = await fetch(`${API_BASE}/overview`);
  if (!response.ok) {
    throw new Error('Failed to load stored files');
  }
  return response.json();
}

export async function saveTranscriptSession(
  transcripts,
  source = 'ocr upload',
  mode = 'recognition',
  transcriptImages = {},
  bookName = 'transcript',
  modelInfo = {},
) {
  const response = await fetch(`${API_BASE}/transcripts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      transcripts,
      transcript_images: transcriptImages,
      source,
      mode,
      book_name: bookName,
      model_info: modelInfo,
    }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Failed to save transcript' }));
    throw new Error(err.detail || 'Failed to save transcript');
  }

  return response.json();
}

export async function saveDatasetToMyFiles(pages, { source = 'dataset generation', bookName = 'dataset', bboxFormat = 'txt', mode = 'recognition', modelInfo = {} } = {}) {
  const response = await fetch(`${API_BASE}/datasets`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      pages,
      source,
      book_name: bookName,
      bbox_format: bboxFormat,
      mode,
      model_info: modelInfo,
    }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Failed to save dataset' }));
    throw new Error(err.detail || 'Failed to save dataset');
  }

  return response.json();
}

export async function fetchTranscriptDetail(sessionId) {
  const response = await fetch(`${API_BASE}/transcripts/${encodeURIComponent(sessionId)}`);
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Failed to load transcript details' }));
    throw new Error(err.detail || 'Failed to load transcript details');
  }
  return response.json();
}

export async function fetchDatasetDetail(datasetId) {
  const response = await fetch(`${API_BASE}/datasets/${encodeURIComponent(datasetId)}`);
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Failed to load dataset details' }));
    throw new Error(err.detail || 'Failed to load dataset details');
  }
  return response.json();
}

export async function deleteStoredItem(kind, id) {
  const response = await fetch(`${API_BASE}/${encodeURIComponent(kind)}/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Delete failed' }));
    throw new Error(err.detail || 'Delete failed');
  }

  return response.json();
}

export async function downloadStoredItem(kind, id) {
  const response = await fetch(`${API_BASE}/download/${encodeURIComponent(kind)}/${encodeURIComponent(id)}`);
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Download failed' }));
    throw new Error(err.detail || 'Download failed');
  }
  return response.blob();
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
