// Central API base URL.
//
// Configure via VITE_API_URL (see .env.example). Falls back to the local
// backend for development. In production this should be set to the deployed
// backend origin, e.g. https://tabula-api.onrender.com
export const API_BASE: string = (
  import.meta.env?.VITE_API_URL ?? 'http://localhost:8000'
).replace(/\/+$/, '');

/** Build an absolute URL for a backend path (leading slash optional). */
export function apiUrl(path: string): string {
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE}${p}`;
}

/** Read the auth token from localStorage (or null). */
export function getToken(): string | null {
  return localStorage.getItem('token');
}

/** Fired (on window) when the backend rejects the session with 401. */
export const SESSION_EXPIRED_EVENT = 'tabula:session-expired';

/**
 * Handle a 401 response: drop the stale credentials and notify AuthContext
 * (which logs out and returns the UI to the login screen).  Call this on any
 * response with status 401 before surfacing the error.
 */
export function handleSessionExpired(): void {
  localStorage.removeItem('token');
  localStorage.removeItem('user');
  window.dispatchEvent(new Event(SESSION_EXPIRED_EVENT));
}

/** Friendly message for a 401 so users know to log in again. */
export const SESSION_EXPIRED_MESSAGE =
  'Your session has expired. Please log in again.';
