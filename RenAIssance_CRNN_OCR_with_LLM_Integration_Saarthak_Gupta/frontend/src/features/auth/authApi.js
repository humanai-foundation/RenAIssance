// Every call sends credentials so the httpOnly session cookie goes back. The
// cookie is opaque to JS, so no secret ever lands in the bundle.

import { API_ORIGIN } from '../../config';

const BASE = `${API_ORIGIN}/api/auth`;

async function request(path, { method = 'GET', body } = {}) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    credentials: 'include',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (res.status === 204) return null;

  let data = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }

  if (!res.ok) {
    const message =
      (data && (data.detail?.message || data.detail || data.message)) ||
      `Request failed (${res.status})`;
    throw new Error(typeof message === 'string' ? message : 'Request failed');
  }
  return data;
}

// Current user, or null when not logged in.
export async function fetchMe() {
  try {
    return await request('/me');
  } catch {
    return null;
  }
}

export function signup({ username, password, name, email, institute }) {
  return request('/signup', {
    method: 'POST',
    body: { username, password, name, email, institute: institute || null },
  });
}

export function login({ username, password }) {
  return request('/login', { method: 'POST', body: { username, password } });
}

export function logout() {
  return request('/logout', { method: 'POST' });
}

export function updateProfile({ name, email, institute }) {
  return request('/me', { method: 'PATCH', body: { name, email, institute } });
}
