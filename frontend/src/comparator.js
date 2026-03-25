/**
 * comparator.js — Side-by-side multi-asteroid comparison with radar chart
 */

import { fetchCompare } from './api.js';
import { Chart, RadarController, RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend } from 'chart.js';

Chart.register(RadarController, RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

let compareIds = [];
let radarChart = null;

const CHART_COLORS = [
  'rgba(0,212,255,0.7)', 'rgba(255,59,59,0.7)', 'rgba(0,255,136,0.7)',
  'rgba(255,165,0,0.7)', 'rgba(155,89,182,0.7)', 'rgba(135,206,235,0.7)',
];

export function initComparator() {
  // Expose global for leaderboard to call
  window._addToCompare = addToCompare;
}

export function addToCompare(id) {
  if (!compareIds.includes(id) && compareIds.length < 6) {
    compareIds.push(id);
    renderComparator();
  }
}

export function syncCompareFromExplorer(ids) {
  compareIds = [...new Set([...compareIds, ...ids])].slice(0, 6);
}

export async function renderComparator() {
  const empty = document.getElementById('compare-empty');
  const slots = document.getElementById('compare-slots');
  const chartWrap = document.getElementById('compare-chart-wrap');

  if (!compareIds.length) {
    empty.style.display  = 'block';
    slots.innerHTML      = '';
    chartWrap.style.display = 'none';
    return;
  }

  empty.style.display     = 'none';
  chartWrap.style.display = 'block';
  slots.innerHTML         = '<div style="color:var(--muted);padding:16px 32px">Loading comparison...</div>';

  try {
    const data = await fetchCompare(compareIds);

    // ── Cards ─────────────────────────────────────────────────────────
    slots.innerHTML = data.map((a, i) => `
      <div class="compare-card" style="border-color:${CHART_COLORS[i].replace('0.7','0.4')}">
        <button class="ccard-remove" onclick="window._removeFromCompare(${a.id})">✕</button>
        <div class="ccard-name">${a.name}</div>
        <div class="ccard-info">
          <div>💨 ${a.relative_velocity_kps?.toFixed(2)} km/s</div>
          <div>📡 ${(a.miss_distance_km/1e6).toFixed(2)} M km</div>
          <div>📏 ${a.max_diameter_km?.toFixed(3)} km max</div>
          <div>⚡ ${a.is_potentially_hazardous ? '🔴 Hazardous' : '🔵 Safe'}</div>
          ${a.risk_score_manual != null ? `<div>⚠ Risk: ${(a.risk_score_manual*100).toFixed(1)}%</div>` : ''}
        </div>
      </div>
    `).join('');

    window._removeFromCompare = (id) => {
      compareIds = compareIds.filter(x => x !== id);
      renderComparator();
    };

    // ── Radar Chart ────────────────────────────────────────────────────
    const metrics = ['Velocity', 'Size', 'Miss Dist', 'Risk Score', 'Hazard Prob'];
    const maxValues = { vel: 66, size: 2, dist: 75e6, risk: 1, prob: 1 };

    const datasets = data.map((a, i) => ({
      label: a.name.length > 20 ? a.name.slice(0, 18) + '…' : a.name,
      data: [
        (a.relative_velocity_kps || 0) / maxValues.vel,
        (a.max_diameter_km || 0) / maxValues.size,
        1 - (a.miss_distance_km || 0) / maxValues.dist, // invert: closer = riskier
        a.risk_score_manual || 0,
        a.probability_being_truely_hazardous || (a.is_potentially_hazardous ? 0.8 : 0.1),
      ],
      fill: true,
      backgroundColor: CHART_COLORS[i].replace('0.7', '0.1'),
      borderColor:     CHART_COLORS[i],
      pointBackgroundColor: CHART_COLORS[i],
      borderWidth: 2,
    }));

    if (radarChart) { radarChart.destroy(); radarChart = null; }
    const ctx = document.getElementById('compare-radar').getContext('2d');
    radarChart = new Chart(ctx, {
      type: 'radar',
      data: { labels: metrics, datasets },
      options: {
        animation: { duration: 500 },
        scales: {
          r: {
            min: 0, max: 1,
            ticks: { display: false },
            grid:  { color: 'rgba(0,212,255,0.1)' },
            angleLines: { color: 'rgba(0,212,255,0.15)' },
            pointLabels: { color: '#e2e8f0', font: { family: 'Orbitron', size: 10 } },
          }
        },
        plugins: {
          legend: { labels: { color: '#e2e8f0', font: { size: 11 } } },
          tooltip: { callbacks: {
            label: (ctx) => {
              const raw = ctx.raw;
              const labels = ['km/s (norm)', 'km (norm)', 'Proximity (norm)', 'Risk Score', 'Hazard Prob'];
              return `${ctx.dataset.label}: ${(raw * 100).toFixed(1)}% ${labels[ctx.dataIndex]}`;
            }
          }}
        }
      }
    });

  } catch (e) {
    slots.innerHTML = `<div style="color:var(--pho);padding:32px">Error: ${e.message}</div>`;
  }
}
