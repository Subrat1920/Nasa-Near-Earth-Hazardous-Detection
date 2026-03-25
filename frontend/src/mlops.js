/**
 * mlops.js — MLOps dashboard: model history line chart + drift table + PSI bar chart
 */

import { fetchModelLogs, fetchLatestDrift } from './api.js';
import {
  Chart, LineController, BarController, CategoryScale, LinearScale,
  PointElement, LineElement, BarElement, Tooltip, Legend, Filler,
} from 'chart.js';

Chart.register(LineController, BarController, CategoryScale, LinearScale,
  PointElement, LineElement, BarElement, Tooltip, Legend, Filler);

let accuracyChart = null;
let psiChart      = null;

export async function loadMLOps() {
  try {
    const [models, drift] = await Promise.all([fetchModelLogs(), fetchLatestDrift()]);
    _renderModelCard(models[0]);
    _renderAccuracyChart(models);
    _renderDrift(drift);
  } catch (e) {
    document.getElementById('mlops-model-info').textContent = `Error: ${e.message}`;
  }
}

function _renderModelCard(m) {
  if (!m) return;
  const info = document.getElementById('mlops-model-info');
  info.innerHTML = `
    <div class="mi-row"><span class="mi-lbl">Model</span><span class="mi-val">${m.model_name}</span></div>
    <div class="mi-row"><span class="mi-lbl">Trained on</span><span class="mi-val">${m.training_date?.slice(0,10)}</span></div>
    <div class="mi-row"><span class="mi-lbl">Accuracy</span>
      <span class="mi-val green">${(m.accuracy*100).toFixed(1)}%</span></div>
    <div class="mi-row"><span class="mi-lbl">Recall</span>
      <span class="mi-val green">${(m.recall*100).toFixed(1)}%</span></div>
    <div class="mi-row"><span class="mi-lbl">Precision</span>
      <span class="mi-val ${m.precision < 0.4 ? 'red' : ''}">${(m.precision*100).toFixed(1)}%</span></div>
    <div class="mi-row"><span class="mi-lbl">F1 Score</span>
      <span class="mi-val">${(m.f1_score*100).toFixed(1)}%</span></div>
  `;
}

function _renderAccuracyChart(models) {
  const reversed = [...models].reverse();
  const labels   = reversed.map((m, i) => `v${i + 1}`);

  if (accuracyChart) { accuracyChart.destroy(); accuracyChart = null; }
  const ctx = document.getElementById('accuracy-chart').getContext('2d');
  accuracyChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Accuracy',
          data: reversed.map(m => +(m.accuracy * 100).toFixed(2)),
          borderColor: '#00d4ff', backgroundColor: 'rgba(0,212,255,0.08)',
          tension: 0.4, fill: true, pointBackgroundColor: '#00d4ff',
        },
        {
          label: 'Recall',
          data: reversed.map(m => +(m.recall * 100).toFixed(2)),
          borderColor: '#00ff88', backgroundColor: 'rgba(0,255,136,0.08)',
          tension: 0.4, fill: true, pointBackgroundColor: '#00ff88',
        },
        {
          label: 'F1 Score',
          data: reversed.map(m => +(m.f1_score * 100).toFixed(2)),
          borderColor: '#ffa500', backgroundColor: 'rgba(255,165,0,0.08)',
          tension: 0.4, fill: false, pointBackgroundColor: '#ffa500',
        },
      ],
    },
    options: {
      animation: { duration: 600 },
      scales: {
        x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: {
          min: 0, max: 100,
          ticks: { color: '#64748b', callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
      },
      plugins: { legend: { labels: { color: '#e2e8f0', font: { size: 11 } } } },
    },
  });
}

function _renderDrift(drift) {
  // Summary chips
  const s = drift.summary || {};
  document.getElementById('drift-summary').innerHTML = `
    <div class="drift-chip high"><span class="dc-val">${s.high_drift    || 0}</span><span class="dc-lbl">High Drift</span></div>
    <div class="drift-chip mod"> <span class="dc-val">${s.moderate_drift || 0}</span><span class="dc-lbl">Moderate</span></div>
    <div class="drift-chip no">  <span class="dc-val">${s.no_drift       || 0}</span><span class="dc-lbl">No Drift</span></div>
  `;

  // Drift table
  const tbody = document.getElementById('drift-body');
  if (!drift.features?.length) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--muted)">No drift data</td></tr>';
    return;
  }
  tbody.innerHTML = drift.features.map(f => {
    const cls = f.drift_status === 'High Drift' ? 'high' : f.drift_status === 'No Drift' ? 'no' : 'mod';
    return `
      <tr>
        <td style="font-family:var(--font-mono)">${f.feature}</td>
        <td>${f.feature_type}</td>
        <td>${f.method?.toUpperCase()}</td>
        <td style="font-family:var(--font-mono)">${f.psi != null ? f.psi.toExponential(2) : '—'}</td>
        <td class="drift-status-${cls}">${f.drift_status}</td>
      </tr>
    `;
  }).join('');

  // PSI bar chart
  const psiData = drift.features.filter(f => f.psi != null);
  if (!psiData.length) return;

  if (psiChart) { psiChart.destroy(); psiChart = null; }
  const ctx = document.getElementById('psi-chart').getContext('2d');
  psiChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: psiData.map(f => f.feature),
      datasets: [{
        label: 'PSI',
        data: psiData.map(f => f.psi),
        backgroundColor: psiData.map(f =>
          f.drift_status === 'High Drift' ? 'rgba(255,59,59,0.7)' :
          f.drift_status === 'Moderate Drift' ? 'rgba(255,165,0,0.7)' :
          'rgba(0,212,255,0.5)'
        ),
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      animation: { duration: 600 },
      scales: {
        x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#e2e8f0', font: { family: 'JetBrains Mono', size: 10 } }, grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });
}
