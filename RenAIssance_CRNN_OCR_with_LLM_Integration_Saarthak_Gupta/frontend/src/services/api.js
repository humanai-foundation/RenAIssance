// Preprocessing calls into the Python/OpenCV backend.

import { API_ORIGIN } from '../config';
import { toDataUrl, dataUrlToObjectUrl } from '../utils/imageUrl';

const API_BASE = API_ORIGIN;

// Runs the pipeline server-side. Returns the original image URL on failure so
// the editor keeps showing something instead of blanking out.
export async function preprocessImage(imageUrl, pipeline) {
  try {
    const operations = pipeline.map(step => ({
      op: step.op,
      params: step.params || {},
      enabled: true,
    }));

    const response = await fetch(`${API_BASE}/api/preprocess`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        // Page images are blob: object URLs in app state; the backend wants base64.
        image_data: await toDataUrl(imageUrl),
        operations: operations,
        preview_mode: false,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail?.message || error.detail || 'Preprocessing failed');
    }

    const result = await response.json();

    if (!result.success) {
      console.warn('Preprocessing had errors:', result.errors);
    }

    // Re-wrap as an object URL — keeping base64 for every page of a large book
    // bloats the JS heap.
    return result.processed_image ? dataUrlToObjectUrl(result.processed_image) : imageUrl;

  } catch (error) {
    console.error('Preprocessing API error:', error);
    return imageUrl;
  }
}
