// Page images live in state as blob: object URLs, not data URLs. A data URL
// parks the whole encoded image on the JS heap for every page of the book,
// which is what made large books crawl. Convert to base64 only at request time.

// blob: / http(s): / data: -> base64 data URL.
export async function toDataUrl(url) {
    if (!url) return null;
    if (url.startsWith('data:')) return url;
    const blob = await (await fetch(url)).blob();
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let bin = '';
    const CHUNK = 0x8000; // avoid call-stack limits on large images
    for (let i = 0; i < bytes.length; i += CHUNK) {
        bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
    }
    return `data:${blob.type || 'image/png'};base64,${btoa(bin)}`;
}

// data URL -> object URL. Anything else passes through. The caller owns the
// result and must revoke it.
export function dataUrlToObjectUrl(dataUrl) {
    if (!dataUrl || typeof dataUrl !== 'string' || !dataUrl.startsWith('data:')) return dataUrl;
    const [header, b64] = dataUrl.split(',');
    const mime = header.match(/:(.*?);/)?.[1] || 'image/png';
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return URL.createObjectURL(new Blob([u8], { type: mime }));
}


export function revokeObjectUrls(urls) {
    for (const url of urls || []) {
        if (typeof url === 'string' && url.startsWith('blob:')) URL.revokeObjectURL(url);
    }
}
