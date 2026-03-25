/**
 * main.js — NASA NEO Interactive Universe
 * Entry point: initialises all views, wires navigation, starts data loading.
 */

import './style.css';
import { pingHealth, fetchStats, fetchAllAsteroids, connectLiveWS } from './api.js';
import { initOrrery, addAsteroidBatch, addSingleAsteroid, setFilter, focusAsteroid } from './orrery.js';
import { initExplorer, openExplorer, closeExplorer, getCompareQueue, syncCompareFromExplorer } from './explorer.js';
import { initLeaderboard, loadLeaderboard } from './leaderboard.js';
import { initComparator, addToCompare, renderComparator } from './comparator.js';
import { loadMLOps } from './mlops.js';

// ── Loading screen helpers ─────────────────────────────────────────────────
const loadingBar   = document.getElementById('loading-bar');
const loadingCount = document.getElementById('loading-count');

function setProgress(loaded, total) {
  const pct = Math.min(100, (loaded / total) * 100);
  loadingBar.style.width = pct + '%';
  loadingCount.textContent = `Loaded ${loaded.toLocaleString()} / ${total.toLocaleString()} asteroids…`;
}

function hideLoading() {
  const screen = document.getElementById('loading-screen');
  const app    = document.getElementById('app');
  screen.style.transition = 'opacity 0.6s';
  screen.style.opacity    = '0';
  setTimeout(() => { 
    screen.style.display = 'none'; 
    app.style.display = 'flex'; 
    window.dispatchEvent(new Event('resize')); 
  }, 600);
}

// ── Navigation ─────────────────────────────────────────────────────────────
let activeView   = 'orrery';
let leaderboardLoaded = false;
let mlopsLoaded       = false;

document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const view = btn.dataset.view;
    switchView(view);
  });
});

function switchView(view) {
  activeView = view;

  document.querySelectorAll('.nav-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.view === view));

  document.querySelectorAll('.view').forEach(v =>
    v.classList.toggle('active', v.id === `view-${view}`));

  // Lazy-load data for non-orrery views
  if (view === 'leaderboard' && !leaderboardLoaded) {
    leaderboardLoaded = true;
    loadLeaderboard();
  }
  if (view === 'mlops' && !mlopsLoaded) {
    mlopsLoaded = true;
    loadMLOps();
  }
  if (view === 'compare') {
    syncCompareFromExplorer(getCompareQueue());
    renderComparator();
  }
  if (view !== 'orrery') {
    closeExplorer();
  }
}

// ── Filter bar (Orrery) ────────────────────────────────────────────────────
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    setFilter(btn.dataset.filter);
  });
});

// ── Header Stats ───────────────────────────────────────────────────────────
function updateHeaderStats(stats) {
  document.querySelector('#chip-unique .chip-val').textContent  = (+stats.unique_asteroids).toLocaleString();
  document.querySelector('#chip-pho .chip-val').textContent     = (+stats.unique_pho).toLocaleString();
  document.querySelector('#chip-sentry .chip-val').textContent  = (+stats.unique_sentry).toLocaleString();
  document.querySelector('#chip-speed .chip-val').textContent   = (+stats.max_velocity_kps).toFixed(1);
}

// ── Bootstrap ─────────────────────────────────────────────────────────────
async function bootstrap() {
  loadingCount.textContent = 'Connecting to Mission Control…';
  loadingBar.style.width = '5%';

  // Wake up Render free-tier if needed (with retry)
  for (let attempt = 0; attempt < 3; attempt++) {
    try { if (await pingHealth()) break; } catch (_) {}
    await new Promise(r => setTimeout(r, 3000));
  }
  loadingBar.style.width = '10%';

  // Fetch global stats for header chips
  try {
    const stats = await fetchStats();
    updateHeaderStats(stats);
  } catch (_) {}

  loadingBar.style.width = '15%';

  // Init sub-modules
  initExplorer((queue) => {
    // Called when user clicks "Add to Compare" in Explorer
    if (queue.length > 0) {
      addToCompare(queue[queue.length - 1]);
    }
  });
  initLeaderboard();
  initComparator();

  // Init 3D Orrery (canvas)
  const canvas = document.getElementById('orrery-canvas');
  initOrrery(canvas, (asteroidBasic) => {
    // Called when an asteroid is clicked in the 3D scene
    openExplorer(asteroidBasic);
  });

  loadingBar.style.width = '20%';
  loadingCount.textContent = 'Streaming asteroid data…';

  // Fetch all 32,001 asteroids via standard JSON endpoint
  try {
    loadingCount.textContent = 'Downloading asteroid catalog… (3MB)';
    const allAsteroids = await fetchAllAsteroids();
    
    // Send to WebGL scene
    loadingCount.textContent = 'Rendering orbits…';
    addAsteroidBatch(allAsteroids.data ? allAsteroids.data : allAsteroids);
    
    // Complete
    loadingBar.style.width = '100%';
    setTimeout(hideLoading, 400);

    // Open WebSocket for live new-asteroid events
    connectLiveWS((newAsteroid) => {
      addSingleAsteroid(newAsteroid);
      console.log('[LIVE] New asteroid detected:', newAsteroid.name);
    });
  } catch (err) {
    console.error("Failed to load asteroids:", err);
    loadingCount.textContent = 'Failed to load data. Please refresh.';
  }
}

// ── Start ──────────────────────────────────────────────────────────────────
bootstrap().catch(err => {
  loadingCount.textContent = `Error: ${err.message}. Please refresh.`;
  loadingBar.style.background = 'var(--pho)';
  console.error('[BOOT ERROR]', err);
});
