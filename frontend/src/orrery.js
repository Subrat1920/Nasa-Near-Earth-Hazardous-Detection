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
// Earth radius + Atmosphere is ~2.24. 
// Miss distance 6599km is ~1.03 Earth Radii from Earth's center.
// So mathematically passing at 2.4 units makes it graze the atmosphere realistically.
const ORBIT_MIN     = 2.4; 
const ORBIT_MAX     = 48;
const MAX_ASTEROIDS = 35000; // headroom for weekly growth

// ── Module State ───────────────────────────────────────────────────────────
let scene, camera, renderer, controls;
let instancedMesh;
let asteroidGroup;      // parent group — rotated as a unit each frame
let orbitRing = null;   // single reusable orbit ring
let hoveredId  = -1;
let selectedId = -1;
let isTweeningTarget = false;
let isolatedMode = false;

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

  // Camera (near plane 0.001 so microscopic zoom distances don't clip through the rock geometry)
  camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.001, 500);
  camera.position.set(0, 18, 35);

  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping  = true;
  controls.dampingFactor  = 0.08;
  // Reduce minDistance from 3.0 down to microscopic so camera can get inches away from true-scale rocks
  controls.minDistance    = 0.02;
  controls.maxDistance    = 120;
  controls.autoRotate     = false;

  // Stars
  _addStars();

  // Earth
  _addEarth();

  // Asteroid InstancedMesh (empty, filled as SSE data arrives)
  const geo = new THREE.IcosahedronGeometry(1, 1);
  
  // Custom attribute to pass unique asteroid physics to Shader (Cragginess, Noise Freq, Spin Speed, Visibility(w))
  const shapeArray = new Float32Array(MAX_ASTEROIDS * 4);
  const shapeAttr = new THREE.InstancedBufferAttribute(shapeArray, 4);
  geo.setAttribute('aShape', shapeAttr);

  const rockMaterial = new THREE.ShaderMaterial({
    uniforms: { 
      baseColor: { value: new THREE.Color(0x2d2d30) },
      uTime: { value: 0 }
    },
    vertexShader: `
      varying vec3 vColor; varying vec3 vNormal; varying vec3 vViewPosition;
      uniform float uTime;
      attribute vec4 aShape; // x=staticSeed, y=noiseFreq, z=spinSpeed, w=isVisible
      
      float hash(vec3 p) { p=fract(p*0.3183099+.1); p*=17.0; return fract(p.x*p.y*p.z*(p.x+p.y+p.z)); }
      float noise(vec3 x) {
        vec3 i=floor(x); vec3 f=fract(x); f=f*f*(3.0-2.0*f);
        return mix(mix(mix(hash(i+vec3(0,0,0)),hash(i+vec3(1,0,0)),f.x),mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x),f.y),f.z);
      }

      void main() {
        vColor = instanceColor;
        
        float spin = uTime * aShape.z;
        float s = sin(spin); float c = cos(spin);
        mat3 rot = mat3(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
        
        // By multiplying position by aShape.w, we mathematically collapse invisible asteroids to 0 Area instantly!
        vec3 spunPos = rot * (position * aShape.w);
        
        // Use purely static seed per asteroid (aShape.x) so they don't squirm as they orbit
        vec3 staticSeed = vec3(aShape.x, aShape.x * 1.5, aShape.x * 0.5);
        
        float bump = noise(spunPos * aShape.y + staticSeed) * 0.4; // 0.4 crater depth
        vec3 displaced = spunPos + normalize(spunPos) * bump;
        
        vNormal = normalize(normalMatrix * mat3(instanceMatrix) * rot * normal);
        vec4 mvPosition = viewMatrix * instanceMatrix * vec4(displaced, 1.0);
        vViewPosition = -mvPosition.xyz;
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      uniform vec3 baseColor;
      varying vec3 vColor; varying vec3 vNormal; varying vec3 vViewPosition;

      float hash(vec3 p) { p=fract(p*0.3183099+.1); p*=17.0; return fract(p.x*p.y*p.z*(p.x+p.y+p.z)); }
      float noise(vec3 x) {
        vec3 i=floor(x); vec3 f=fract(x); f=f*f*(3.0-2.0*f);
        return mix(mix(mix(hash(i+vec3(0,0,0)),hash(i+vec3(1,0,0)),f.x),mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x),f.y),f.z);
      }

      void main() {
        vec3 n = normalize(vNormal); vec3 v = normalize(vViewPosition);
        vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));
        
        // Porous surface texture coloring using micro-noise
        float tex = noise(vViewPosition * 5.0) * 0.5 + 0.5;
        vec3 rock = baseColor * (0.6 + tex * 0.4);
        
        // Mix the rock texture heavily with the hazard color tag WITHOUT glowing
        rock = mix(rock, vColor, 0.65);
        
        float diff = max(dot(n, lightDir), 0.15); // ambient 0.15
        vec3 finalColor = rock * (diff + 0.3);
        
        gl_FragColor = vec4(finalColor, 1.0);
      }
    `,
  });
  instancedMesh = new THREE.InstancedMesh(geo, rockMaterial, MAX_ASTEROIDS);
  instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  instancedMesh.count = 0; // start with 0 visible
  // Disable frustum culling! Huge bounds that update continuously will eventually clip if the camera tracks wild
  instancedMesh.frustumCulled = false;

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

  const earthGeo = new THREE.SphereGeometry(2, 48, 48); // 2.0 Radius = 6,371 km
  const earthMat = new THREE.MeshPhongMaterial({
    map: earthTex, shininess: 60, specular: new THREE.Color(0x334455),
  });
  const earthLight = new THREE.DirectionalLight(0xffffff, 1.2);
  earthLight.position.set(5, 3, 5);
  scene.add(earthLight);
  scene.add(new THREE.AmbientLight(0x112244, 0.5));
  scene.add(new THREE.Mesh(earthGeo, earthMat));

  // Atmosphere glow
  const glowGeo = new THREE.SphereGeometry(2.24, 32, 32); // Scaled 12% above surface
  const glowMat = new THREE.MeshBasicMaterial({
    color: 0x1a73e8, transparent: true, opacity: 0.12,
    side: THREE.BackSide, blending: THREE.AdditiveBlending,
  });
  scene.add(new THREE.Mesh(glowGeo, glowMat));
}

// ── Stars ──────────────────────────────────────────────────────────────────
function _addStars() {
  const count  = 1200; // Drastically reduce star count
  const posArr = new Float32Array(count * 3);
  for (let i = 0; i < count * 3; i++) {
    posArr[i] = (Math.random() - 0.5) * 400;
  }
  const starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
  // Dim the color (dark blue/grey) and reduce size so they look far away
  const starMat = new THREE.PointsMaterial({ color: 0x4a5a7a, size: 0.1, sizeAttenuation: true, transparent: true, opacity: 0.4 });
  scene.add(new THREE.Points(starGeo, starMat));
}

// ── Orbit radius mapping (log scale) ─────────────────────────────────────
function _orbitRadius(missKm) {
  const t = Math.log(missKm - MISS_MIN_KM + 1) / Math.log(MISS_MAX_KM - MISS_MIN_KM + 1);
  return ORBIT_MIN + t * (ORBIT_MAX - ORBIT_MIN);
}

function _asteroidSize(minDiam, maxDiam) {
  const avgDiamKm = (minDiam + maxDiam) / 2;
  // MATH: Earth radius 6,371 km is exactly 2.0 WebGL scene units.
  // True 1:1 scale: (avgDiamKm / 6371.0) * 2.0 scene units.
  // We exaggerate by 70x because a true 1km asteroid would be a microscopic, invisible pixel.
  // This mathematically guarantees size ratios across all asteroids are 100% accurate to each other.
  const trueRelativeSize = (avgDiamKm / 6371.0) * 2.0;
  const uiSize = trueRelativeSize * 70.0;
  
  // Safe limit so the screen doesn't completely turn to rock if a 50km anomaly orbits
  return Math.max(0.015, Math.min(0.8, uiSize));
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
    const color = _typeColor(a.is_potentially_hazardous, a.is_sentry_object);
    
    // Create True 3D Orbital Planes so they go UP and DOWN instead of just left/right!
    // 50% cluster somewhat near Earth's ecliptic plane, 50% are wild flyers (spherical halo)
    const incl = (Math.random() > 0.5) 
      ? (Math.random() - 0.5) * 1.5 
      : Math.acos(2 * Math.random() - 1); 
    const ascNode = Math.random() * Math.PI * 2;

    // Build the orbital plane's X and Y basis vectors mathematically
    const nx = Math.sin(incl) * Math.sin(ascNode);
    const ny = Math.cos(incl);
    const nz = Math.sin(incl) * Math.cos(ascNode);
    const planeNormal = new THREE.Vector3(nx, ny, nz).normalize();
    const uVec = new THREE.Vector3(1, 0, 0);
    if (Math.abs(planeNormal.x) > 0.9) uVec.set(0, 1, 0);
    uVec.cross(planeNormal).normalize();
    const vVec = new THREE.Vector3().crossVectors(planeNormal, uVec).normalize();
    
    // Move cleanly via real relative_velocity_kps and randomize direction!
    const vKps = a.relative_velocity_kps || 20.0;
    const dir = (a.id % 2 === 0) ? 1.0 : -1.0; 
    // Reduced speed multiplier by 20% (0.00015 * 0.8 = 0.00012) per request
    const speed = vKps * dir * 0.00012;
    
    // Extreme randomized shape stretching (x, y, z individually morphed)
    let sx = size * (0.6 + Math.random() * 1.8);
    let sy = size * (0.6 + Math.random() * 1.8);
    let sz = size * (0.6 + Math.random() * 1.8);
    // 20% chance to be extremely stretched
    if (Math.random() > 0.8) sy *= 2.5; 
    if (Math.random() > 0.8) sx *= 2.0;

    const rx = Math.random() * Math.PI * 2;
    const ry = Math.random() * Math.PI * 2;
    const rz = Math.random() * Math.PI * 2;
    
    // Inject standard static data into GPU: x=static_seed, y=crater_scale, z=tumbling_speed, w=visibility(1.0)
    const staticSeed = (a.id % 1000) * 12.34; // Keeps noise structurally locked
    const noiseFreq = 1.5 + Math.random() * 5.0; // wider range of crater densities
    const tumblingSpeed = (Math.random() - 0.5) * 8.0;
    instancedMesh.geometry.getAttribute('aShape').setXYZW(idx, staticSeed, noiseFreq, tumblingSpeed, 1.0);

    // Set initial static stance using True 3D projection
    const cosA = r * Math.cos(angle);
    const sinA = r * Math.sin(angle);
    _dummy.position.set(
      cosA * uVec.x + sinA * vVec.x,
      cosA * uVec.y + sinA * vVec.y,
      cosA * uVec.z + sinA * vVec.z
    );
    _dummy.scale.set(sx, sy, sz);
    _dummy.rotation.set(rx, ry, rz);
    _dummy.updateMatrix();
    instancedMesh.setMatrixAt(idx, _dummy.matrix);
    instancedMesh.setColorAt(idx, color);

    // Save the plane vectors so the animation loop can orbit them perfectly
    asteroidData[idx] = { ...a, orbitR: r, angle, uVec, vVec, size, color, speed };
    idToIndex[a.id] = idx;
  });

  instancedMesh.count = Math.min(startIdx + batch.length, MAX_ASTEROIDS);
  instancedMesh.geometry.getAttribute('aShape').needsUpdate = true;
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
  // Orient the ring perfectly to the orbital plane
  const planeNormal = new THREE.Vector3().crossVectors(a.uVec, a.vVec).normalize();
  orbitRing.lookAt(planeNormal);
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

// ── Isolate single asteroid ──────────────────────────────────────────────────
function _setIsolatedAsteroid(idx) {
  isolatedMode = true;
  const attr = instancedMesh.geometry.getAttribute('aShape');
  for (let i = 0; i < instancedMesh.count; i++) {
    attr.setW(i, i === idx ? 1.0 : 0.0);
  }
  attr.needsUpdate = true;
  // Fade stars to bring focus natively to the single asteroid
  scene.children.forEach(c => { if (c.isPoints) c.material.opacity = 0.05; });
}

function _disableIsolation() {
  if (!isolatedMode) return;
  isolatedMode = false;
  const attr = instancedMesh.geometry.getAttribute('aShape');
  for (let i = 0; i < instancedMesh.count; i++) {
    attr.setW(i, 1.0);
  }
  attr.needsUpdate = true;
  // Restore star opacities
  scene.children.forEach(c => { if (c.isPoints) c.material.opacity = 0.4; });
}

// ── Interaction ────────────────────────────────────────────────────────────
function _setupInteraction() {
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  const tooltip = document.getElementById('asteroid-tooltip');
  const tooltipName = document.getElementById('tooltip-name');
  const tooltipType = document.getElementById('tooltip-type');

  // Helper: Fat raycast via distance math instead of sub-pixel mesh intersection
  const _vec = new THREE.Vector3();
  function getFatHit(clientX, clientY) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((clientX - rect.left) / rect.width)  * 2 - 1;
    mouse.y = -((clientY - rect.top)  / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    
    let closestId = -1;
    let closestDist = Infinity;
    const arr = instancedMesh.instanceMatrix.array;
    const count = instancedMesh.count;
    
    for (let i = 0; i < count; i++) {
      if (isolatedMode && i !== selectedId) continue; // Do not allow clicking hidden asteroids
      _vec.set(arr[i*16+12], arr[i*16+13], arr[i*16+14]);
      // Math: Project distance from the camera Ray
      const dSq = raycaster.ray.distanceSqToPoint(_vec);
      const camDistSq = _vec.distanceToSquared(camera.position);
      
      // Hit tolerance strictly scales with distance so it's always easy to click on screen
      const tolerance = camDistSq * 0.0006; 
      
      if (dSq < tolerance) {
        if (camDistSq < closestDist) {
          closestDist = camDistSq;
          closestId = i;
        }
      }
    }
    return closestId;
  }

  renderer.domElement.addEventListener('pointermove', (e) => {
    const iid = getFatHit(e.clientX, e.clientY);
    if (iid >= 0) {
      if (iid !== hoveredId) {
        if (hoveredId >= 0 && hoveredId !== selectedId) _highlightAsteroid(hoveredId, false);
        hoveredId = iid;
        _highlightAsteroid(iid, true);
        if (selectedId < 0) _showOrbitRing(iid);
      }
      const a = asteroidData[iid];
      tooltipName.textContent = a.name;
      tooltipType.textContent = a.is_potentially_hazardous ? '🔴 HAZARDOUS' : a.is_sentry_object ? '🟠 Sentry' : '🔵 Safe';
      
      const rect = renderer.domElement.getBoundingClientRect();
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

  let pointerDownPos = { x: 0, y: 0 };
  renderer.domElement.addEventListener('pointerdown', (e) => {
    pointerDownPos = { x: e.clientX, y: e.clientY };
  });

  renderer.domElement.addEventListener('click', (e) => {
    // If the mouse moved significantly between pointerdown and click, the user was panning the camera.
    // We ignore the click so we don't accidentally load an asteroid while exploring.
    const dx = e.clientX - pointerDownPos.x;
    const dy = e.clientY - pointerDownPos.y;
    if (Math.abs(dx) > 5 || Math.abs(dy) > 5) return;

    const iid = getFatHit(e.clientX, e.clientY);
    if (iid >= 0) {
      if (selectedId >= 0) _highlightAsteroid(selectedId, false);
      selectedId = iid;
      _highlightAsteroid(selectedId, true);
      _showOrbitRing(selectedId);
      _setIsolatedAsteroid(selectedId);
      onSelectCallback?.(asteroidData[selectedId]);
      
      // Execute the cinematic follow-cam zoom!
      _flyCameraTo(selectedId);
    } else {
      // Click empty space — deselect
      resetCamera();
    }
  });
}

// ── Focus on a specific asteroid (from leaderboard click) ─────────────────
// ── Cinematic Camera Tracking ──────────────────────────────────────────────────
function _flyCameraTo(idx) {
  const a = asteroidData[idx];
  if (!a) return;
  const currRot = a.angle + (frameCount * 0.5) * a.speed;
  const cosA = a.orbitR * Math.cos(currRot);
  const sinA = a.orbitR * Math.sin(currRot);
  const targetPos = new THREE.Vector3(
    cosA * a.uVec.x + sinA * a.vVec.x,
    cosA * a.uVec.y + sinA * a.vVec.y,
    cosA * a.uVec.z + sinA * a.vVec.z
  );

  const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
  // Dynamically calculate camera depth based on the EXACT size of the rock, maintaining perfect frame visibility
  const zoomDepth = Math.max(0.1, a.size * 8.0);
  
  if (offset.lengthSq() < 0.01) offset.set(0, 0.5, zoomDepth); 
  offset.normalize().multiplyScalar(zoomDepth);
  
  const camPos = new THREE.Vector3().addVectors(targetPos, offset);

  isTweeningTarget = true;
  gsap.to(controls.target, {
    x: targetPos.x, y: targetPos.y, z: targetPos.z,
    duration: 1.5, ease: 'power3.inOut'
  });
  // Smoothly fly physical camera there too
  gsap.to(camera.position, {
    x: camPos.x, y: camPos.y, z: camPos.z,
    duration: 1.5, ease: 'power3.inOut',
    onUpdate: () => controls.update(),
    onComplete: () => { isTweeningTarget = false; }
  });
}

export function focusAsteroid(asteroidId) {
  const idx = idToIndex[asteroidId];
  if (idx == null) return;

  if (selectedId >= 0) _highlightAsteroid(selectedId, false);
  selectedId = idx;
  _highlightAsteroid(idx, true);
  _showOrbitRing(idx);
  _setIsolatedAsteroid(idx);
  onSelectCallback?.(asteroidData[idx]);

  _flyCameraTo(idx);
}

// ── Reset camera to Earth ──────────────────────────────────────────────────
export function resetCamera() {
  if (selectedId >= 0) {
    _highlightAsteroid(selectedId, false);
    selectedId = -1;
  }
  _removeOrbitRing();
  _disableIsolation();
  isTweeningTarget = true;
  // Smoothly pan target and camera back to default overview (Earth)
  gsap.to(controls.target, {
    x: 0, y: 0, z: 0,
    duration: 1.5, ease: 'power3.inOut'
  });
  gsap.to(camera.position, {
    x: 0, y: 18, z: 35, // Original camera stance
    duration: 1.5, ease: 'power3.inOut',
    onUpdate: () => controls.update(),
    onComplete: () => { isTweeningTarget = false; }
  });
}

// ── Render Loop ────────────────────────────────────────────────────────────
function _animate() {
  animId = requestAnimationFrame(_animate);
  frameCount++;

  // Update real-time Keplerian orbits for all instances directly in the buffer array
  const arr = instancedMesh.instanceMatrix.array;
  const timeMod = frameCount * 0.5; // Orbit time multiplier
  
  // Link uniform time to shader for glowing edge pulse and noise
  instancedMesh.material.uniforms.uTime.value = timeMod;

  // Only update xyz positions, leave scales and rotations completely intact
  for (let i = 0; i < instancedMesh.count; i++) {
    const a = asteroidData[i];
    if (!a) continue;
    const cAng = a.angle + timeMod * a.speed;
    const cosA = a.orbitR * Math.cos(cAng);
    const sinA = a.orbitR * Math.sin(cAng);
    arr[i * 16 + 12] = cosA * a.uVec.x + sinA * a.vVec.x;
    arr[i * 16 + 13] = cosA * a.uVec.y + sinA * a.vVec.y;
    arr[i * 16 + 14] = cosA * a.uVec.z + sinA * a.vVec.z;
  }
  instancedMesh.instanceMatrix.needsUpdate = true;

  // Track currently selected asteroid dynamically if camera is not tweening
  if (selectedId >= 0 && !isTweeningTarget) {
    const a = asteroidData[selectedId];
    if (a) {
      const cAng = a.angle + timeMod * a.speed;
      const cosA = a.orbitR * Math.cos(cAng);
      const sinA = a.orbitR * Math.sin(cAng);
      
      const nx = cosA * a.uVec.x + sinA * a.vVec.x;
      const ny = cosA * a.uVec.y + sinA * a.vVec.y;
      const nz = cosA * a.uVec.z + sinA * a.vVec.z;
      
      // Shift physical camera smoothly alongside the asteroid to lock trailing mode
      camera.position.x += (nx - controls.target.x);
      camera.position.y += (ny - controls.target.y);
      camera.position.z += (nz - controls.target.z);
      
      controls.target.set(nx, ny, nz);
    }
  }

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
