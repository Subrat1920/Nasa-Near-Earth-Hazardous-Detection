/**
 * comparator.js — Side-by-side multi-asteroid comparison with radar chart
 */

import { fetchCompare } from './api.js';
import { Chart, RadarController, RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend } from 'chart.js';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { gsap } from 'gsap';

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
  window._previewScale = (id) => {
    if (window._previewScaleData && window._previewScaleData[id]) {
      previewScale(window._previewScaleData[id]);
    }
  };
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

    window._previewScaleData = {};

    // ── Cards ─────────────────────────────────────────────────────────
    slots.innerHTML = data.map((a, i) => {
      window._previewScaleData[a.id] = a;
      return `
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
        <button class="btn-xs" style="margin-top:12px; width:100%; border-color:var(--accent); color:var(--text);" onclick="window._previewScale(${a.id})">🔭 Earth Scale</button>
      </div>
      `
    }).join('');

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

// ── Localized Earth Context ───────────────────────────────────────────────────
let ecScene, ecCamera, ecRenderer, ecControls, ecAnimId;
let ecCompareGroup;
let ecEarthMesh, ecEarthGlow;
let ecAsteroidMesh = null;
let ecReticleMesh = null;
let ecRenderScale = 1.0;
let ecAstX = 1.05;
let currentMag = 1.0;

function _createEarthTexture() {
  const size = 512;
  const cvs = document.createElement('canvas');
  cvs.width = size * 2; cvs.height = size;
  const ctx = cvs.getContext('2d');

  // Deep Ocean Gradient
  const grad = ctx.createLinearGradient(0, 0, 0, size);
  grad.addColorStop(0, '#1055b0');
  grad.addColorStop(0.5, '#1a73e8');
  grad.addColorStop(1, '#051c4a');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size * 2, size);

  // Continental Clusters
  ctx.fillStyle = '#2d6a4f';
  for (let i = 0; i < 20; i++) {
    const x = Math.random() * size * 2;
    const y = Math.random() * size;
    const w = 40 + Math.random() * 200;
    const h = 20 + Math.random() * 100;
    ctx.beginPath();
    ctx.ellipse(x, y, w / 2, h / 2, Math.random() * Math.PI, 0, Math.PI * 2);
    ctx.fill();
    // Wrap around for seamless texture
    if (x + w/2 > size * 2) {
      ctx.beginPath();
      ctx.ellipse(x - size * 2, y, w / 2, h / 2, Math.random() * Math.PI, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Specular map hint (Lighter areas for islands/shallows)
  ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
  for (let i = 0; i < 10; i++) {
    ctx.fillRect(Math.random() * size * 2, Math.random() * size, 50, 20);
  }

  return new THREE.CanvasTexture(cvs);
}

function _initEarthScene() {
  if (ecRenderer) return; 
  
  const canvas = document.getElementById('earth-compare-canvas');
  ecRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  ecRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  ecRenderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

  ecScene = new THREE.Scene();
  
  ecCamera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.01, 100);
  ecCamera.position.set(0, 0, 5);

  ecControls = new OrbitControls(ecCamera, ecRenderer.domElement);
  ecControls.enableDamping = true;
  ecControls.dampingFactor = 0.08;
  ecControls.minDistance = 1.05; 
  ecControls.maxDistance = 20;

  const amLight = new THREE.AmbientLight(0xffffff, 0.2);
  ecScene.add(amLight);
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
  dirLight.position.set(5, 3, 5);
  ecScene.add(dirLight);

  ecCompareGroup = new THREE.Group();
  ecScene.add(ecCompareGroup);

  const earthGeo = new THREE.SphereGeometry(1.0, 64, 64);
  const earthMat = new THREE.MeshPhongMaterial({
    map: _createEarthTexture(),
    shininess: 40,
    specular: 0x223344
  });
  ecEarthMesh = new THREE.Mesh(earthGeo, earthMat);
  ecCompareGroup.add(ecEarthMesh);
  
  const glowGeo = new THREE.SphereGeometry(1.05, 32, 32);
  const glowMat = new THREE.MeshBasicMaterial({
    color: 0x4a90e2, transparent: true, opacity: 0.1, blending: THREE.AdditiveBlending, side: THREE.BackSide
  });
  ecEarthGlow = new THREE.Mesh(glowGeo, glowMat);
  ecCompareGroup.add(ecEarthGlow);

  window.addEventListener('resize', () => {
    if (!ecRenderer || document.getElementById('earth-compare-wrap').style.display === 'none') return;
    ecCamera.aspect = canvas.clientWidth / canvas.clientHeight;
    ecCamera.updateProjectionMatrix();
    ecRenderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  });

  _animateEarthScene();
}

function _animateEarthScene() {
  ecAnimId = requestAnimationFrame(_animateEarthScene);
  
  if (ecControls) {
    // ── FIXED ANCHOR ZOOM LOGIC ───────────────────────────────────────
    // The camera target is fixed at the asteroid center (ecAstX, 0, 0)
    const fixedTarget = new THREE.Vector3(ecAstX, 0, 0);
    const dist = ecCamera.position.distanceTo(fixedTarget);
    
    // Zoom Magnitude based on proximity to the fixed asteroid target
    let targetMag = 1.0;
    if (dist < 5.0) {
      // Powerful magnification to show asteroid details alongside Earth
      targetMag = 1.0 + Math.pow(Math.max(0, 5.0 - dist), 2.5) * 2.0;
    }
    
    // Smoothly interpolate magnification
    currentMag += (targetMag - currentMag) * 0.12;

    // Apply scaling to the entire comparison group
    // The group origin is at the asteroid center, so Earth scales "away" from it.
    if (ecCompareGroup) {
      ecCompareGroup.scale.set(currentMag, currentMag, currentMag);
    }
    
    if (ecEarthMesh) ecEarthMesh.rotation.y += 0.001;
    
    if (ecAsteroidMesh) {
      ecAsteroidMesh.rotation.x += 0.003;
      ecAsteroidMesh.rotation.y += 0.005;
    }

    if (ecReticleMesh) {
      ecReticleMesh.lookAt(ecCamera.position);
      // Reticle pulse effect
      const pulse = 1.0 + Math.sin(Date.now() * 0.005) * 0.15;
      ecReticleMesh.scale.set(pulse, pulse, pulse);
    }

    // Stabilize the OrbitControls target to the asteroid center
    ecControls.target.copy(fixedTarget);
    ecControls.update();
    
    // Adjust min distance based on Earth's scaled geometry to prevent clipping
    // The closest point is Earth's surface which is at distance (ecAstX - 1.0) * currentMag
    const minSafeDist = Math.max(0.1, (ecAstX * currentMag) - (1.0 * currentMag));
    ecControls.minDistance = 0.05 + minSafeDist; 
  }
  
  if (document.getElementById('earth-compare-wrap').style.display !== 'none') {
    ecRenderer.render(ecScene, ecCamera);
  }
}

export function previewScale(a) {
  _initEarthScene();
  
  const wrap = document.getElementById('earth-compare-wrap');
  wrap.style.display = 'flex';
  
  const canvas = document.getElementById('earth-compare-canvas');
  ecCamera.aspect = canvas.clientWidth / canvas.clientHeight;
  ecCamera.updateProjectionMatrix();
  ecRenderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

  document.getElementById('ec-name').textContent = a.name;
  document.getElementById('ec-dmax').textContent = `${a.max_diameter_km?.toFixed(3)} km`;
  document.getElementById('ec-miss').textContent = `${(a.miss_distance_km/1e6).toFixed(2)} Million km`;

  const ratio = (a.max_diameter_km / 12742);
  document.getElementById('ec-ratio').textContent = `1 : ${(1/ratio).toLocaleString(undefined, {maximumFractionDigits:0})}`;

  // Preserve mathematically flawless scale down to microscopic limits so WebGL doesn't cull
  ecRenderScale = Math.max(0.0005, 2.0 * ratio);
  ecAstX = 1.05 + ecRenderScale * 0.5;
  currentMag = 1.0; // Reset magnification on new selection

  // RE-POSITION Group and Earth for asteroid-centric scaling
  // Group at ecAstX, Earth local at -ecAstX. World Earth = 0.
  if (ecCompareGroup) {
    ecCompareGroup.position.set(ecAstX, 0, 0);
    ecCompareGroup.scale.set(1, 1, 1);
  }
  if (ecEarthMesh) ecEarthMesh.position.set(-ecAstX, 0, 0);
  if (ecEarthGlow) ecEarthGlow.position.set(-ecAstX, 0, 0);

  if (ecAsteroidMesh) {
    ecCompareGroup.remove(ecAsteroidMesh);
    ecAsteroidMesh.geometry.dispose();
    ecAsteroidMesh.material.dispose();
    ecAsteroidMesh = null;
  }
  if (ecReticleMesh) {
    ecCompareGroup.remove(ecReticleMesh);
    ecReticleMesh.geometry.dispose();
    ecReticleMesh.material.dispose();
    ecReticleMesh = null;
  }

  // Red targeting reticle centered at (0,0,0) local to group (asteroid center)
  const retGeo = new THREE.RingGeometry(0.06, 0.065, 32);
  const retMat = new THREE.MeshBasicMaterial({ color: 0xff0044, transparent: true, opacity: 0.8, side: THREE.DoubleSide });
  ecReticleMesh = new THREE.Mesh(retGeo, retMat);
  ecReticleMesh.position.set(0, 0, 0);
  ecCompareGroup.add(ecReticleMesh);

  const geo = new THREE.IcosahedronGeometry(0.5, 2); 
  const pos = geo.attributes.position;
  for(let i=0; i<pos.count; i++) {
      const p = new THREE.Vector3().fromBufferAttribute(pos, i);
      const noise = (Math.random() - 0.5) * 0.2; 
      p.add(p.clone().normalize().multiplyScalar(noise));
      pos.setXYZ(i, p.x, p.y, p.z);
  }
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    color: (a.is_potentially_hazardous || a.risk_score_manual >= 0.5) ? 0xcc4444 : 0x88aabb,
    roughness: 0.9, metalness: 0.1, flatShading: true
  });

  ecAsteroidMesh = new THREE.Mesh(geo, mat);
  ecAsteroidMesh.scale.set(ecRenderScale * 1.2, ecRenderScale * 0.9, ecRenderScale * 1.1);
  ecAsteroidMesh.position.set(0, 0, 0); // Local origin of group is the asteroid
  ecCompareGroup.add(ecAsteroidMesh);
  
  // Update UI Camera Bounds dynamically so the macro zoom aligns perfectly with the size
  document.getElementById('cam-pov-front').onclick = () => { gsap.to(ecCamera.position, {x:0, y:0, z:4, duration:1.5, ease:'power3.inOut'}); ecControls.target.set(0.6,0,0); };
  document.getElementById('cam-pov-top').onclick   = () => { gsap.to(ecCamera.position, {x:0.6, y:3.5, z:0, duration:1.5, ease:'power3.inOut'}); ecControls.target.set(0.6,0,0); };
  document.getElementById('cam-pov-close').onclick = () => { 
    const pushBack = Math.max(0.02, ecRenderScale * 4.5);
    gsap.to(ecCamera.position, {x: (ecAstX + pushBack) * 2.5, y: pushBack, z: pushBack * 2, duration:1.5, ease:'power3.inOut'}); 
    ecControls.target.set(ecAstX, 0, 0); 
  };
  
  // Trigger default asteroid view
  const closeBack = Math.max(0.02, ecRenderScale * 2.5);
  gsap.to(ecCamera.position, {x: (ecAstX + closeBack) * 3, y: closeBack * 2, z: closeBack * 2, duration:1.5, ease:'power3.inOut'});
  ecControls.target.set(ecAstX, 0, 0);
}
