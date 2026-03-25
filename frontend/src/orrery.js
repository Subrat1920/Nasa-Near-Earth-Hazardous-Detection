/**
 * orrery.js — Three.js 3D Asteroid Orrery
 *
 * Performance strategy for i3 CPU / Intel UHD integrated graphics:
 *  • ONE InstancedMesh for all 32,001 asteroids (= 1 draw call total)
 *  • MeshBasicMaterial (no lighting calculations)
 *  • Low-poly sphere geometry (4×4 segments)
 *  • Group rotation for scene animation (O(1), not per-asteroid)
 *  • Orbit rings ONLY for hovered / selected asteroid
 *  • Stars as Points (1 draw call)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { gsap } from 'gsap';

// ── Constants ──────────────────────────────────────────────────────────────
const COLORS = {
  pho:    new THREE.Color(0xff3b3b),  // Hazardous
  sentry: new THREE.Color(0xffa500),  // Sentry
  both:   new THREE.Color(0x9b59b6),  // PHO + Sentry
  safe:   new THREE.Color(0x87ceeb),  // Safe
  earth:  new THREE.Color(0x1a73e8),
  orbit:  new THREE.Color(0x00d4ff),
};

const MISS_MIN_KM   = 6599;
const MISS_MAX_KM   = 74794677;
const ORBIT_MIN     = 2.5;   // Three.js scene units
const ORBIT_MAX     = 48;
const MAX_ASTEROIDS = 35000; // headroom for weekly growth

// ── Module State ───────────────────────────────────────────────────────────
let scene, camera, renderer, controls;
let instancedMesh;
let asteroidGroup;      // parent group — rotated as a unit each frame
let orbitRing = null;   // single reusable orbit ring
let hoveredId  = -1;
let selectedId = -1;

const asteroidData   = [];   // [{id, name, orbitR, angle, incl, color, ...}, ...]
const idToIndex      = {};   // {asteroid_id: instance_index}
const _matrix        = new THREE.Matrix4();
const _color         = new THREE.Color();
const _dummy         = new THREE.Object3D();

let onSelectCallback = null;   // set by main.js
let filterMode       = 'all';  // 'all' | 'pho' | 'sentry' | 'safe'
let canvas, animId;
let frameCount = 0;

// ── Init ───────────────────────────────────────────────────────────────────
export function initOrrery(canvasEl, onSelect) {
  canvas = canvasEl;
  onSelectCallback = onSelect;

  // Renderer
  renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'default' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // cap for integrated GPU
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x020817);

  // Camera
  camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 500);
  camera.position.set(0, 18, 35);

  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping  = true;
  controls.dampingFactor  = 0.08;
  controls.minDistance    = 3;
  controls.maxDistance    = 120;
  controls.autoRotate     = false;

  // Stars
  _addStars();

  // Earth
  _addEarth();

  // Asteroid InstancedMesh (empty, filled as SSE data arrives)
  const geo = new THREE.SphereGeometry(1, 4, 4);
  const mat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  instancedMesh = new THREE.InstancedMesh(geo, mat, MAX_ASTEROIDS);
  instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  instancedMesh.count = 0; // start with 0 visible

  asteroidGroup = new THREE.Group();
  asteroidGroup.add(instancedMesh);
  scene.add(asteroidGroup);

  // Raycaster for click & hover
  _setupInteraction();

  // Resize handler
  window.addEventListener('resize', _onResize);

  // Start render loop
  _animate();
}

// ── Earth ──────────────────────────────────────────────────────────────────
function _addEarth() {
  // Procedural Earth canvas texture
  const size = 512;
  const cvs  = document.createElement('canvas');
  cvs.width = size * 2; cvs.height = size;
  const ctx = cvs.getContext('2d');

  // Ocean
  const grad = ctx.createRadialGradient(size, size / 2, 50, size, size / 2, size);
  grad.addColorStop(0, '#1a73e8');
  grad.addColorStop(0.6, '#1055b0');
  grad.addColorStop(1, '#051c4a');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size * 2, size);

  // Continents (rough shapes for aesthetics)
  ctx.fillStyle = '#2d6a4f';
  const landMasses = [
    [100,80,220,180],[550,60,140,200],[750,160,90,130],
    [300,230,180,100],[480,220,100,80],[900,40,80,160],
    [120,280,80,60],[680,290,120,70],[200,310,60,40],
  ];
  landMasses.forEach(([x,y,w,h]) => {
    ctx.beginPath();
    ctx.ellipse(x, y, w/2, h/2, Math.random()*0.5, 0, Math.PI*2);
    ctx.fill();
  });

  const earthTex = new THREE.CanvasTexture(cvs);

  const earthGeo = new THREE.SphereGeometry(1, 48, 48);
  const earthMat = new THREE.MeshPhongMaterial({
    map: earthTex, shininess: 60, specular: new THREE.Color(0x334455),
  });
  const earthLight = new THREE.DirectionalLight(0xffffff, 1.2);
  earthLight.position.set(5, 3, 5);
  scene.add(earthLight);
  scene.add(new THREE.AmbientLight(0x112244, 0.5));
  scene.add(new THREE.Mesh(earthGeo, earthMat));

  // Atmosphere glow
  const glowGeo = new THREE.SphereGeometry(1.12, 32, 32);
  const glowMat = new THREE.MeshBasicMaterial({
    color: 0x1a73e8, transparent: true, opacity: 0.12,
    side: THREE.BackSide, blending: THREE.AdditiveBlending,
  });
  scene.add(new THREE.Mesh(glowGeo, glowMat));
}

// ── Stars ──────────────────────────────────────────────────────────────────
function _addStars() {
  const count  = 6000;
  const posArr = new Float32Array(count * 3);
  for (let i = 0; i < count * 3; i++) {
    posArr[i] = (Math.random() - 0.5) * 400;
  }
  const starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
  const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.2, sizeAttenuation: true });
  scene.add(new THREE.Points(starGeo, starMat));
}

// ── Orbit radius mapping (log scale) ─────────────────────────────────────
function _orbitRadius(missKm) {
  const t = Math.log(missKm - MISS_MIN_KM + 1) / Math.log(MISS_MAX_KM - MISS_MIN_KM + 1);
  return ORBIT_MIN + t * (ORBIT_MAX - ORBIT_MIN);
}

function _asteroidSize(minDiam, maxDiam) {
  const avg = (minDiam + maxDiam) / 2;
  // Map 0–10 km range to 0.03–0.18 scene units
  return Math.max(0.03, Math.min(0.18, avg * 0.025));
}

function _typeColor(pho, sentry) {
  if (pho && sentry) return COLORS.both;
  if (pho)           return COLORS.pho;
  if (sentry)        return COLORS.sentry;
  return COLORS.safe;
}

// ── Add Asteroid Batch ─────────────────────────────────────────────────────
export function addAsteroidBatch(batch) {
  const startIdx = instancedMesh.count;

  batch.forEach((a, i) => {
    const idx   = startIdx + i;
    if (idx >= MAX_ASTEROIDS) return;

    const r     = _orbitRadius(a.miss_distance_km);
    const size  = _asteroidSize(a.min_diameter_km, a.max_diameter_km);
    const angle = ((a.id % 1000) / 1000) * Math.PI * 2;
    const incl  = Math.sin(a.id * 0.0031) * 0.6; // inclination in radians (±~34°)
    const color = _typeColor(a.is_potentially_hazardous, a.is_sentry_object);

    // Position on orbit
    _dummy.position.set(
      r * Math.cos(angle) * Math.cos(incl),
      r * Math.sin(incl),
      r * Math.sin(angle) * Math.cos(incl),
    );
    _dummy.scale.setScalar(size);
    _dummy.updateMatrix();
    instancedMesh.setMatrixAt(idx, _dummy.matrix);
    instancedMesh.setColorAt(idx, color);

    asteroidData[idx] = { ...a, orbitR: r, angle, incl, size, color };
    idToIndex[a.id] = idx;
  });

  instancedMesh.count = Math.min(startIdx + batch.length, MAX_ASTEROIDS);
  instancedMesh.instanceMatrix.needsUpdate = true;
  instancedMesh.instanceColor.needsUpdate  = true;

  // Update visible count
  document.getElementById('visible-count').textContent =
    instancedMesh.count.toLocaleString();
}

// ── Add single new asteroid (from WebSocket) ──────────────────────────────
export function addSingleAsteroid(a) {
  addAsteroidBatch([a]);
  // Flash the count badge
  const badge = document.getElementById('chip-unique');
  if (badge) {
    const val = badge.querySelector('.chip-val');
    const cur = parseInt(val.textContent.replace(/,/g,'')) || 0;
    val.textContent = (cur + 1).toLocaleString();
    badge.style.borderColor = 'var(--green)';
    setTimeout(() => (badge.style.borderColor = ''), 2000);
  }
}

// ── Filter ─────────────────────────────────────────────────────────────────
export function setFilter(mode) {
  filterMode = mode;
  let visible = 0;
  for (let i = 0; i < instancedMesh.count; i++) {
    const a = asteroidData[i];
    if (!a) continue;
    let show = true;
    if (mode === 'pho')    show = a.is_potentially_hazardous;
    if (mode === 'sentry') show = a.is_sentry_object;
    if (mode === 'safe')   show = !a.is_potentially_hazardous && !a.is_sentry_object;

    instancedMesh.setColorAt(i, show ? a.color : new THREE.Color(0x000000));
    if (show) visible++;
  }
  instancedMesh.instanceColor.needsUpdate = true;
  document.getElementById('visible-count').textContent = visible.toLocaleString();
}

// ── Orbit Ring (shown on hover/select) ────────────────────────────────────
function _showOrbitRing(idx) {
  _removeOrbitRing();
  const a = asteroidData[idx];
  if (!a) return;

  const ringGeo = new THREE.TorusGeometry(a.orbitR, 0.012, 3, 80);
  const ringMat = new THREE.MeshBasicMaterial({
    color: a.color, transparent: true, opacity: 0.5,
  });
  orbitRing = new THREE.Mesh(ringGeo, ringMat);
  orbitRing.rotation.x = a.incl;
  asteroidGroup.add(orbitRing);
}

function _removeOrbitRing() {
  if (orbitRing) { asteroidGroup.remove(orbitRing); orbitRing.geometry.dispose(); orbitRing = null; }
}

// ── Highlight selected asteroid ────────────────────────────────────────────
function _highlightAsteroid(idx, on) {
  const a = asteroidData[idx];
  if (!a) return;
  const c = on ? new THREE.Color(0xffffff) : a.color;
  instancedMesh.setColorAt(idx, c);
  instancedMesh.instanceColor.needsUpdate = true;
}

// ── Interaction ────────────────────────────────────────────────────────────
function _setupInteraction() {
  const raycaster = new THREE.Raycaster();
  raycaster.params.Mesh.threshold = 0.05;
  const mouse = new THREE.Vector2();
  const tooltip = document.getElementById('asteroid-tooltip');
  const tooltipName = document.getElementById('tooltip-name');
  const tooltipType = document.getElementById('tooltip-type');

  renderer.domElement.addEventListener('pointermove', (e) => {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width)  * 2 - 1;
    mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(instancedMesh);
    if (hits.length) {
      const iid = hits[0].instanceId;
      if (iid !== hoveredId) {
        if (hoveredId >= 0 && hoveredId !== selectedId) _highlightAsteroid(hoveredId, false);
        hoveredId = iid;
        _highlightAsteroid(iid, true);
        if (selectedId < 0) _showOrbitRing(iid);
      }
      const a = asteroidData[iid];
      tooltipName.textContent = a.name;
      tooltipType.textContent = a.is_potentially_hazardous ? '🔴 HAZARDOUS' : a.is_sentry_object ? '🟠 Sentry' : '🔵 Safe';
      tooltip.style.left = (e.clientX - rect.left + 12) + 'px';
      tooltip.style.top  = (e.clientY - rect.top  + 12) + 'px';
      tooltip.classList.remove('hidden');
      renderer.domElement.style.cursor = 'pointer';
    } else {
      if (hoveredId >= 0 && hoveredId !== selectedId) _highlightAsteroid(hoveredId, false);
      if (hoveredId >= 0 && selectedId < 0) _removeOrbitRing();
      hoveredId = -1;
      tooltip.classList.add('hidden');
      renderer.domElement.style.cursor = 'grab';
    }
  });

  renderer.domElement.addEventListener('click', (e) => {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width)  * 2 - 1;
    mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(instancedMesh);
    if (hits.length) {
      if (selectedId >= 0) _highlightAsteroid(selectedId, false);
      selectedId = hits[0].instanceId;
      _highlightAsteroid(selectedId, true);
      _showOrbitRing(selectedId);
      onSelectCallback?.(asteroidData[selectedId]);
    } else {
      // Click empty space — deselect
      if (selectedId >= 0) { _highlightAsteroid(selectedId, false); selectedId = -1; }
      _removeOrbitRing();
    }
  });
}

// ── Focus on a specific asteroid (from leaderboard click) ─────────────────
export function focusAsteroid(asteroidId) {
  const idx = idToIndex[asteroidId];
  if (idx == null) return;

  if (selectedId >= 0) _highlightAsteroid(selectedId, false);
  selectedId = idx;
  _highlightAsteroid(idx, true);
  _showOrbitRing(idx);
  onSelectCallback?.(asteroidData[idx]);

  // Move camera to smoothly face the asteroid
  const a = asteroidData[idx];
  const target = new THREE.Vector3(
    a.orbitR * Math.cos(a.angle) * Math.cos(a.incl),
    a.orbitR * Math.sin(a.incl),
    a.orbitR * Math.sin(a.angle) * Math.cos(a.incl),
  );
  gsap.to(controls.target, {
    x: target.x, y: target.y, z: target.z,
    duration: 0.8, ease: 'power2.out'
  });
}

// ── Reset camera to Earth ──────────────────────────────────────────────────
export function resetCamera() {
  if (selectedId >= 0) {
    _highlightAsteroid(selectedId, false);
    selectedId = -1;
  }
  _removeOrbitRing();
  
  // Smoothly pan target back to origin (Earth)
  gsap.to(controls.target, {
    x: 0, y: 0, z: 0,
    duration: 1.0, ease: 'power2.inOut'
  });
}

// ── Render Loop ────────────────────────────────────────────────────────────
function _animate() {
  animId = requestAnimationFrame(_animate);
  frameCount++;

  // Slow scene rotation (the whole asteroid field turns as one unit)
  asteroidGroup.rotation.y += 0.00015;

  controls.update();
  renderer.render(scene, camera);
}

// ── Resize ─────────────────────────────────────────────────────────────────
function _onResize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}

// ── Cleanup ─────────────────────────────────────────────────────────────────
export function destroyOrrery() {
  cancelAnimationFrame(animId);
  window.removeEventListener('resize', _onResize);
  renderer.dispose();
}
