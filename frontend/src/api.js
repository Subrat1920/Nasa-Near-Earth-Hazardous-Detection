/**
 * api.js — All fetch / SSE / WebSocket calls to the FastAPI backend.
 * This is the single source of truth for all network requests.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ── Health / wake-up ────────────────────────────────────────────────────────
export async function pingHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.ok;
}

// ── Global Stats ─────────────────────────────────────────────────────────────
export async function fetchStats() {
  const res = await fetch(`${API_BASE}/api/stats`);
  if (!res.ok) throw new Error('Failed to fetch stats');
  return res.json();
}

// ── Asteroid Full Load (for 3D scene) ───────────────────────────────────────
/**
 * Fetches all 32,001 unique asteroids at once.
 * Replaces the unstable SSE stream.
 * @returns {Promise<Array>}
 */
export async function fetchAllAsteroids() {
  const res = await fetch(`${API_BASE}/api/asteroids_all`);
  if (!res.ok) throw new Error('Failed to fetch all asteroids');
  return res.json();
}

// ── Single Asteroid Detail ───────────────────────────────────────────────────
export async function fetchAsteroid(id) {
  const res = await fetch(`${API_BASE}/api/asteroids/${id}`);
  if (!res.ok) throw new Error(`Asteroid ${id} not found`);
  return res.json();
}

// ── Leaderboard ──────────────────────────────────────────────────────────────
export async function fetchLeaderboard({ by = 'risk', top = 100, hazardous = null, sentry = null } = {}) {
  const params = new URLSearchParams({ by, top });
  if (hazardous !== null) params.set('hazardous', hazardous);
  if (sentry !== null)    params.set('sentry', sentry);
  const res = await fetch(`${API_BASE}/api/leaderboard?${params}`);
  if (!res.ok) throw new Error('Leaderboard fetch failed');
  return res.json();
}

// ── Compare ──────────────────────────────────────────────────────────────────
export async function fetchCompare(ids = []) {
  if (!ids.length) return [];
  const res = await fetch(`${API_BASE}/api/compare?ids=${ids.join(',')}`);
  if (!res.ok) throw new Error('Compare fetch failed');
  return res.json();
}

// ── Risk ─────────────────────────────────────────────────────────────────────
export async function fetchRisk({ page = 1, limit = 100, category = null } = {}) {
  const params = new URLSearchParams({ page, limit });
  if (category) params.set('category', category);
  const res = await fetch(`${API_BASE}/api/risk?${params}`);
  if (!res.ok) throw new Error('Risk fetch failed');
  return res.json();
}

// ── MLOps ─────────────────────────────────────────────────────────────────────
export async function fetchModelLogs() {
  const res = await fetch(`${API_BASE}/api/mlops/models`);
  return res.json();
}

export async function fetchLatestDrift() {
  const res = await fetch(`${API_BASE}/api/mlops/drift/latest`);
  return res.json();
}

// ── WebSocket live updates ────────────────────────────────────────────────────
/**
 * Opens a WebSocket connection to receive new-asteroid events.
 * @param {function} onNewAsteroid - called with asteroid data object
 * @returns {{close: function}} - object with close method
 */
export function connectLiveWS(onNewAsteroid) {
  const wsUrl = API_BASE.replace(/^http/, 'ws') + '/ws/live';
  let ws;
  let alive = true;

  function connect() {
    ws = new WebSocket(wsUrl);
    ws.onopen = () => {
      // Heartbeat every 25s to keep connection alive
      const hb = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send('ping');
      }, 25000);
      ws.onclose = () => { clearInterval(hb); if (alive) setTimeout(connect, 5000); };
    };
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.event === 'new_asteroid') onNewAsteroid(msg.data);
      } catch (_) {}
    };
    ws.onerror = () => ws.close();
  }

  connect();
  return { close: () => { alive = false; ws?.close(); } };
}
