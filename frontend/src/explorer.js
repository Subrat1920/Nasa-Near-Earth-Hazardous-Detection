/**
 * explorer.js — Asteroid detail side-panel
 * Opens when user clicks an asteroid in the Orrery or Leaderboard.
 */

import { fetchAsteroid } from './api.js';
import { gsap } from 'gsap';

let compareQueue = [];    // ids queued for comparison
let onCompareChange = null;
let onCloseChange = null;

export function initExplorer(onCompareCb, onCloseCb) {
  onCompareChange = onCompareCb;
  onCloseChange = onCloseCb;

  document.getElementById('explorer-close').addEventListener('click', () => {
    closeExplorer();
    onCloseChange?.();
  });
  document.getElementById('ex-compare-btn').addEventListener('click', () => {
    const idText = document.getElementById('explorer-id').textContent;
    const id = parseInt(idText.replace(/\D/g, ''));
    if (id && !compareQueue.includes(id) && compareQueue.length < 6) {
      compareQueue.push(id);
      onCompareChange?.(compareQueue);
    }
  });
}

export function openExplorer(asteroidBasic) {
  // Show panel immediately with basic SSE data, then enrich with full detail
  _renderBasic(asteroidBasic);
  _slideOpen();
  fetchAsteroid(asteroidBasic.id).then(_renderFull).catch(console.warn);
}

export function closeExplorer() {
  _slideClose();
}

function _slideOpen() {
  const panel = document.getElementById('explorer-panel');
  panel.classList.remove('panel-hidden');
  gsap.fromTo(panel, { opacity: 0, x: 30 }, { opacity: 1, x: 0, duration: 0.3, ease: 'power2.out' });
}

function _slideClose() {
  const panel = document.getElementById('explorer-panel');
  gsap.to(panel, {
    opacity: 0, x: 30, duration: 0.25, ease: 'power2.in',
    onComplete: () => panel.classList.add('panel-hidden'),
  });
}

function _badgeClass(d) {
  const isHazardous = d.is_potentially_hazardous || (d.risk_score_manual >= 0.5) || (d.probability_being_truely_hazardous >= 0.5);
  if (isHazardous && d.is_sentry_object) return 'both';
  if (isHazardous) return 'pho';
  if (d.is_sentry_object) return 'sentry';
  return 'safe';
}

function _badgeLabel(d) {
  const isHazardous = d.is_potentially_hazardous || (d.risk_score_manual >= 0.5) || (d.probability_being_truely_hazardous >= 0.5);
  if (isHazardous && d.is_sentry_object) return '💜 PHO + SENTRY';
  if (isHazardous) return '🔴 POTENTIALLY HAZARDOUS';
  if (d.is_sentry_object) return '🟠 SENTRY OBJECT';
  return '🔵 SAFE';
}

function _updateBadge(a) {
  const badge = document.getElementById('explorer-badge');
  badge.className = `explorer-badge ${_badgeClass(a)}`;
  badge.textContent = _badgeLabel(a);
}

function _renderBasic(a) {
  _updateBadge(a);

  document.getElementById('explorer-name').textContent = a.name;
  document.getElementById('explorer-id').textContent   = `NASA ID: ${a.id}`;
  document.getElementById('ex-dmin').textContent = `${a.min_diameter_km?.toFixed(3)} km`;
  document.getElementById('ex-dmax').textContent = `${a.max_diameter_km?.toFixed(3)} km`;
  document.getElementById('ex-vel').textContent  = `${a.relative_velocity_kps?.toFixed(2)} km/s`;
  document.getElementById('ex-dist').textContent = `${(a.miss_distance_km / 1000).toFixed(0).toLocaleString()} × 10³ km`;

  // Set JPL link
  document.getElementById('ex-jpl-link').href =
    a.nasa_jpl_url || `https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=${a.id}`;

  // Clear enriched sections
  document.getElementById('ex-ml-section').style.opacity  = '0.4';
  document.getElementById('ex-risk-section').style.opacity = '0.4';
  document.getElementById('ex-timeline').innerHTML = '<div style="color:var(--muted);font-size:0.75rem">Loading history...</div>';
}

function _renderFull(data) {
  // Upgrade the badge with the full data so ML predictive hazards apply successfully!
  _updateBadge(data);

  // ML section
  const mlSec = document.getElementById('ex-ml-section');
  if (data.probability_being_truely_hazardous != null) {
    const pct = (data.probability_being_truely_hazardous * 100).toFixed(1);
    document.getElementById('ex-prob-fill').style.width = `${pct}%`;
    document.getElementById('ex-prob-val').textContent  = `${pct}%`;
    mlSec.style.opacity = '1';
  }

  // Risk section
  const riskSec = document.getElementById('ex-risk-section');
  if (data.risk_score_manual != null) {
    document.getElementById('ex-risk').textContent   = data.risk_score_manual.toFixed(3);
    document.getElementById('ex-rcat').textContent   = data.risk_category_manual;
    document.getElementById('ex-mass').textContent   = _sciNotation(data.mass_kg) + ' kg';
    document.getElementById('ex-energy').textContent = _sciNotation(data.impact_energy_j) + ' J';
    riskSec.style.opacity = '1';
  }

  // Timeline
  const tl = document.getElementById('ex-timeline');
  if (data.approaches?.length) {
    tl.innerHTML = data.approaches.map(a => `
      <div class="timeline-row">
        <span class="timeline-date">${a.close_approach_date}</span>
        <span class="timeline-dist">${(a.miss_distance_km / 1_000_000).toFixed(2)} M km</span>
        <span style="font-size:0.7rem;color:var(--muted)">${a.relative_velocity_kps?.toFixed(1)} km/s</span>
      </div>
    `).join('');
  } else {
    tl.innerHTML = '<div style="color:var(--muted);font-size:0.75rem">No approach history found.</div>';
  }
}

function _sciNotation(val) {
  if (!val) return '—';
  const exp = Math.floor(Math.log10(Math.abs(val)));
  const coef = (val / Math.pow(10, exp)).toFixed(2);
  return `${coef} × 10^${exp}`;
}

export function getCompareQueue() { return [...compareQueue]; }
export function clearCompare()    { compareQueue = []; }
export function syncCompareFromExplorer(ids) {
  // Merge ids into compareQueue (used when switching to Compare view)
  ids.forEach(id => { if (!compareQueue.includes(id)) compareQueue.push(id); });
}

