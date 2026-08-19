// Empty means same-origin: nginx proxies /api/ in production, the Vite dev
// server proxies it locally. One code path for both. Only override for unusual
// deployments.
export const API_ORIGIN = import.meta.env.VITE_API_URL ?? '';
