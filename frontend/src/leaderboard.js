/**
 * leaderboard.js — Top-N asteroid table with sorting, filtering, and pagination
 */

import { fetchLeaderboard } from './api.js';
import { focusAsteroid }   from './orrery.js';
import { openExplorer }    from './explorer.js';

let currentSort   = 'risk';
let currentTop    = 100;
let currentFilter = 'all';  // 'all' | 'pho' | 'sentry'

export function initLeaderboard() {
  // Sort buttons
  document.querySelectorAll('.lb-sort').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.lb-sort').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentSort = btn.dataset.sort;
      loadLeaderboard();
    });
  });

  // Top-N buttons
  document.querySelectorAll('.lb-top').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.lb-top').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentTop = parseInt(btn.dataset.top);
      loadLeaderboard();
    });
  });

  // Filter buttons
  document.querySelectorAll('.lb-filter').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.lb-filter').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.dataset.filter;
      loadLeaderboard();
    });
  });
}

export async function loadLeaderboard() {
  const body = document.getElementById('leaderboard-body');
  body.innerHTML = '<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:32px">Loading...</td></tr>';

  const params = { by: currentSort, top: currentTop };
  if (currentFilter === 'pho')    params.hazardous = true;
  if (currentFilter === 'sentry') params.sentry    = true;

  try {
    const data = await fetchLeaderboard(params);
    body.innerHTML = data.map(a => _rowHTML(a)).join('');

    // Wire up row actions
    data.forEach(a => {
      document.getElementById(`lb-inspect-${a.id}`)?.addEventListener('click', () => {
        openExplorer(a);
        // Switch back to Orrery to show the highlighted asteroid
        document.querySelector('[data-view="orrery"]')?.click();
        setTimeout(() => focusAsteroid(a.id), 300);
      });
      document.getElementById(`lb-compare-${a.id}`)?.addEventListener('click', () => {
        document.querySelector('[data-view="compare"]')?.click();
        // The compare view will add this asteroid
        window._addToCompare?.(a.id);
      });
    });
  } catch (e) {
    body.innerHTML = `<tr><td colspan="9" style="color:var(--pho);text-align:center;padding:32px">Error: ${e.message}</td></tr>`;
  }
}

function _rowHTML(a) {
  const riskPct = a.risk_score_manual != null ? (a.risk_score_manual * 100).toFixed(1) : null;
  const riskCls = _riskClass(a.risk_category_manual);
  const hazard  = a.is_potentially_hazardous
    ? '<span class="badge-pho">🔴 Yes</span>'
    : '<span class="badge-safe">🔵 No</span>';

  return `
    <tr>
      <td class="td-rank">${a.rank}</td>
      <td class="td-name" title="${a.name}">${a.name}</td>
      <td>${a.min_diameter_km?.toFixed(3)} – ${a.max_diameter_km?.toFixed(3)}</td>
      <td>${a.relative_velocity_kps?.toFixed(2)}</td>
      <td>${(a.miss_distance_km / 1000).toFixed(0).toLocaleString()} k</td>
      <td>${riskPct != null ? `<div style="display:flex;align-items:center;gap:6px">
        <div style="width:60px;height:5px;background:var(--surface2);border-radius:99px">
          <div style="width:${riskPct}%;height:100%;background:var(--accent);border-radius:99px"></div>
        </div>${riskPct}%</div>` : '—'}</td>
      <td>${a.risk_category_manual ? `<span class="risk-pill ${riskCls}">${a.risk_category_manual}</span>` : '—'}</td>
      <td>${hazard}</td>
      <td>
        <button class="btn-xs" id="lb-inspect-${a.id}">🔭 View</button>
        <button class="btn-xs" id="lb-compare-${a.id}">⚖ Compare</button>
      </td>
    </tr>
  `;
}

function _riskClass(cat) {
  if (!cat) return '';
  if (cat.includes('High'))   return 'high';
  if (cat.includes('Medium')) return 'medium';
  if (cat.includes('Low'))    return 'low';
  return 'vlow';
}
