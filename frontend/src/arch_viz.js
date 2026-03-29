import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { gsap } from 'gsap';

let scene, camera, renderer, controls, canvas;
let mainNodes = [];
let subNodesGroup;
let connectorGroup;
let currentSubNodes = [];
let flowPulses = [];
let animId, isPaused = false;
let raycaster = new THREE.Raycaster();
let mouse = new THREE.Vector2();
let hoveredNode = null;
let onClickCallback = null;
let currentStageId = 'overview';
let currentStageIndex = -1;

// UI elements
let archLabelsContainer, archSubLabelsContainer, archBubble, archNavGroup;

const NODE_DATA = [
  { 
    id: 'ingestion', label: 'Data Ingestion', pos: [-6, 0, 0], color: 0x00d4ff,
    subs: [
      { id: 'api_extraction', label: 'API Extraction', work: 'Extracts batch orbital definitions from NASA NeoWs Remote API.', tech: 'NASA NeoWS API' },
      { id: 'data_storage', label: 'Data Storage', work: 'Maps raw NASA payloads into localized database structures.', tech: 'Python/SQL' },
      { id: 'ingestion_reg', label: 'Ingestion', work: 'Structures incoming JSON into flattened CSV schemas and registers transformations.', tech: 'Python/ SQL' },
      { id: 'transformation', label: 'Transformation', work: 'Cleanses data, normalizes distances, and applies feature engineering for ML.', tech: 'Pandas/NumPy' }
    ]
  },
  { 
    id: 'ml', label: 'ML Pipeline', pos: [-2, 0, 0], color: 0x9b59b6,
    subs: [
      { id: 'model_training', label: 'Model Training', work: 'Trains different independent and Stacked Classifier Model to identify hazards and logs each model performance in MLFlow model Registry with DagsHub for experiment Tracking.', tech: 'Scikit-Learn/XGBoost/LightGBM/Catboost/StackedModels and MLFow' },
      { id: 'batch_prediction', label: 'Batch Prediction', work: 'Executes unlabelled batch testing generating deterministic risk probabilities.', tech: 'Python' },
      { id: 'data_drift', label: 'Drift Detection', work: 'Uses PSI equations to monitor model stability against historical signatures.', tech: 'PSI (Population Stability Index and Chi-Square Test)' },
      { id: 'grafana', label: 'Grafana Monitor', work: 'Aggregates metrics and system loads onto remote dashboard nodes.', tech: 'Grafana' }
    ]
  },
  { 
    id: 'backend', label: 'FastAPI Backend', pos: [2, 0, 0], color: 0x00ff88,
    subs: [
      { id: 'serve_model', label: 'Serve Model', work: 'Boots the FastAPI environment and loads serialized .pkl models.', tech: 'FastAPI/Uvicorn' },
      { id: 'asgi_server', label: 'ASGI Server', work: 'Exposes high-concurrency event loops for 30,000+ data requests.', tech: 'Uvicorn' },
      { id: 'live_endpoint', label: 'Live API', work: 'Streams risk estimations and orbital parameters from NeonDB via JSON.', tech: 'PostgreSQL/NeonDB' }
    ]
  },
  { 
    id: 'frontend', label: 'WebGL Frontend', pos: [6, 0, 0], color: 0xffa500,
    subs: [
      { id: 'data_consumption', label: 'JSON Pipeline', work: 'Asynchronously ingests 32,001 data arrays from FastAPI endpoints.', tech: 'Vanilla JS/Fetch' },
      { id: 'geometry_mapping', label: 'Geometry Engine', work: 'Converts 2D miss-distance-km into 3D spherical orbits around the Sun.', tech: 'Trigonometry/Three.js' },
      { id: 'webgl_shaders', label: 'WebGL Shaders', work: 'GPU-bound InstancedMesh rendering using custom GLSL fractional routines.', tech: 'GLSL/Three.js' },
      { id: 'ui_isolation', label: 'UI Isolation', work: 'Cinematic Camera Tracking and GSAP-based visual projection.', tech: 'GSAP/Three.js' }
    ]
  }
];

export function initArchViz(canvasEl, onClick) {
  canvas = canvasEl;
  onClickCallback = onClick;
  if (!canvas) return;

  archLabelsContainer = document.getElementById('arch-labels');
  archSubLabelsContainer = document.getElementById('arch-sub-labels');
  archBubble = document.getElementById('arch-bubble');
  archNavGroup = document.getElementById('arch-nav-group');

  renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true, powerPreference: 'low-power' });
  renderer.setPixelRatio(1);
  renderer.setSize(canvas.clientWidth || 800, canvas.clientHeight || 450, false);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(35, (canvas.clientWidth || 800) / (canvas.clientHeight || 450), 0.1, 1000);
  camera.position.set(0, 4, 30);

  subNodesGroup = new THREE.Group();
  scene.add(subNodesGroup);
  connectorGroup = new THREE.Group();
  scene.add(connectorGroup);

  const geo = new THREE.IcosahedronGeometry(1.0, 1);
  const glowGeo = new THREE.IcosahedronGeometry(1.2, 1);
  
  NODE_DATA.forEach((data, i) => {
    const nodeGroup = new THREE.Group();
    nodeGroup.position.set(data.pos[0], data.pos[1], data.pos[2]);
    scene.add(nodeGroup);

    const mat = new THREE.MeshBasicMaterial({ color: data.color, transparent: true, opacity: 0.8 });
    const node = new THREE.Mesh(geo, mat);
    nodeGroup.add(node);

    const glowMat = new THREE.MeshBasicMaterial({ color: data.color, transparent: true, opacity: 0.15, blending: THREE.AdditiveBlending });
    const glowNode = new THREE.Mesh(glowGeo, glowMat);
    nodeGroup.add(glowNode);

    node.userData = { ...data, index: i, isMain: true, glow: glowNode, mat: mat };
    mainNodes.push(node);

    const ring = new THREE.Mesh(new THREE.TorusGeometry(1.6, 0.02, 8, 48), new THREE.MeshBasicMaterial({ color: data.color, transparent: true, opacity: 0.1 }));
    ring.rotation.x = Math.PI/2;
    nodeGroup.add(ring);
    node.userData.group = nodeGroup;

    const label = document.createElement('div');
    label.className = 'arch-label';
    label.textContent = data.label;
    label.style.borderLeftColor = new THREE.Color(data.color).getStyle();
    archLabelsContainer.appendChild(label);
    node.userData.labelEl = label;
  });

  _initPulseLogic();

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.enableZoom = false;
  controls.enablePan = false;

  renderer.domElement.addEventListener('pointermove', _onMouseMove);
  renderer.domElement.addEventListener('click', _onClick);
  window.addEventListener('resize', _onResize);

  // Nav Buttons
  document.getElementById('arch-prev-btn')?.addEventListener('click', () => _goToStage(currentStageIndex - 1));
  document.getElementById('arch-next-btn')?.addEventListener('click', () => _goToStage(currentStageIndex + 1));

  _animate();
}

let overviewCurves = [];
let subCurves = [];
let pulseTimer = 0;

function _initPulseLogic() {
  overviewCurves = [];
  for (let i = 0; i < NODE_DATA.length - 1; i++) {
    const start = NODE_DATA[i].pos;
    const end = NODE_DATA[i+1].pos;
    overviewCurves.push({
      curve: new THREE.CatmullRomCurve3([
        new THREE.Vector3(...start),
        new THREE.Vector3((start[0] + end[0]) / 2, 0.8, 0.5),
        new THREE.Vector3(...end)
      ]),
      color: NODE_DATA[i].color
    });
  }
}

function _goToStage(index) {
  if (index < 0 || index >= NODE_DATA.length) return;
  _drillDown(NODE_DATA[index]);
}

function _drillDown(stageData) {
  currentStageId = stageData.id;
  currentStageIndex = NODE_DATA.findIndex(s => s.id === stageData.id);
  _hideBubble();
  
  // Show/Hide Nav Buttons
  archNavGroup?.classList.remove('hidden');
  document.getElementById('arch-prev-btn')?.classList.toggle('disabled', currentStageIndex === 0);
  document.getElementById('arch-next-btn')?.classList.toggle('disabled', currentStageIndex === NODE_DATA.length - 1);

  mainNodes.forEach(node => {
    const isTarget = node.userData.id === stageData.id;
    const targetX = isTarget ? 0 : (node.userData.pos[0] < stageData.pos[0] ? -25 : 25);
    const targetY = isTarget ? 8 : 0;
    
    gsap.to(node.userData.group.position, { x: targetX, y: targetY, duration: 0.8, ease: 'power2.inOut' });
    gsap.to(node.userData.group.scale, { x: isTarget ? 0.6 : 0.3, y: isTarget ? 0.6 : 0.3, z: isTarget ? 0.6 : 0.3, duration: 0.8 });
    gsap.to(node.userData.labelEl, { opacity: isTarget ? 1 : 0, duration: 0.4 });
  });

  // Cleanup sub nodes & connectors
  _cleanupSubScene();

  const subGeo = new THREE.IcosahedronGeometry(0.5, 1);
  const spacing = 5.5;
  const startX = -((stageData.subs.length - 1) * spacing) / 2;

  stageData.subs.forEach((sub, i) => {
    const subGroup = new THREE.Group();
    subGroup.position.set(startX + i * spacing, -8, 0); // Drop in animation
    subNodesGroup.add(subGroup);

    const subMat = new THREE.MeshBasicMaterial({ color: stageData.color, transparent: true, opacity: 0 });
    const subNode = new THREE.Mesh(subGeo, subMat);
    subGroup.add(subNode);

    const glow = new THREE.Mesh(subGeo, new THREE.MeshBasicMaterial({ color: stageData.color, transparent: true, opacity: 0, blending: THREE.AdditiveBlending }));
    glow.scale.set(1.4, 1.4, 1.4);
    subGroup.add(glow);

    // Create Sub Label
    const slabel = document.createElement('div');
    slabel.className = 'arch-sub-label';
    slabel.textContent = sub.label;
    archSubLabelsContainer.appendChild(slabel);

    subNode.userData = { ...sub, isSub: true, mat: subMat, glow: glow, labelEl: slabel };
    currentSubNodes.push(subNode);

    gsap.to(subMat, { opacity: 0.8, duration: 0.5, delay: 0.2 + i * 0.1 });
    gsap.to(glow.material, { opacity: 0.2, duration: 0.5, delay: 0.2 + i * 0.1 });
    gsap.to(subGroup.position, { y: 0, duration: 0.8, ease: 'back.out(1.7)', delay: 0.2 + i * 0.1 });

    // Connectors
    if (i < stageData.subs.length - 1) {
      const p1 = new THREE.Vector3(startX + i * spacing, 0, 0);
      const p2 = new THREE.Vector3(startX + (i + 1) * spacing, 0, 0);
      const curve = new THREE.CatmullRomCurve3([p1, new THREE.Vector3((p1.x+p2.x)/2, 0.5, 0.2), p2]);
      
      // Draw actual path line
      const lineGeo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(20));
      const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: stageData.color, transparent: true, opacity: 0.15 }));
      connectorGroup.add(line);

      subCurves.push({ curve, color: stageData.color });
    }
  });

  gsap.to(camera.position, { z: 22, y: 3, duration: 1, ease: 'power2.inOut' });
  document.getElementById('arch-back-btn')?.classList.remove('hidden');
}

function _cleanupSubScene() {
  while(subNodesGroup.children.length) {
    const child = subNodesGroup.children[0];
    subNodesGroup.remove(child);
  }
  while(connectorGroup.children.length) {
    const child = connectorGroup.children[0];
    if(child.geometry) child.geometry.dispose();
    if(child.material) child.material.dispose();
    connectorGroup.remove(child);
  }
  archSubLabelsContainer.innerHTML = '';
  currentSubNodes = [];
  subCurves = [];
}

export function resetToOverview() {
  currentStageId = 'overview';
  currentStageIndex = -1;
  _hideBubble();
  archNavGroup?.classList.add('hidden');
  
  mainNodes.forEach(node => {
    gsap.to(node.userData.group.position, { x: node.userData.pos[0], y: node.userData.pos[1], z: node.userData.pos[2], duration: 1, ease: 'power3.inOut' });
    gsap.to(node.userData.group.scale, { x: 1, y: 1, z: 1, duration: 1 });
    gsap.to(node.userData.labelEl, { opacity: 1, duration: 0.5 });
  });

  currentSubNodes.forEach(node => {
    gsap.to(node.userData.mat, { opacity: 0, duration: 0.4 });
    gsap.to(node.userData.glow.material, { opacity: 0, duration: 0.4 });
    gsap.to(node.parent.position, { y: -8, duration: 0.5 });
    gsap.to(node.userData.labelEl, { opacity: 0, duration: 0.3 });
  });

  gsap.to(camera.position, { z: 30, y: 4, duration: 1, ease: 'power2.inOut' });
  document.getElementById('arch-back-btn')?.classList.add('hidden');
}

function _onMouseMove(e) {
  const rect = canvas.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const targets = [...mainNodes, ...currentSubNodes];
  const intersects = raycaster.intersectObjects(targets);

  if (intersects.length > 0) {
    const obj = intersects[0].object;
    if (hoveredNode !== obj) {
      if (hoveredNode) _onHover(hoveredNode, false);
      hoveredNode = obj;
      _onHover(hoveredNode, true);
      canvas.style.cursor = 'pointer';
    }
  } else {
    if (hoveredNode) {
       _onHover(hoveredNode, false);
       hoveredNode = null;
    }
    canvas.style.cursor = 'default';
  }
}

function _onHover(node, active) {
   _highlightNode(node, active);
   if (active && node.userData.isSub) {
     _showBubble(node);
   } else {
     _hideBubble();
   }
}

function _showBubble(node) {
  const data = node.userData;
  const title = document.getElementById('bubble-title');
  const tech = document.getElementById('bubble-tech');
  const work = document.getElementById('bubble-work');
  if (title) title.textContent = data.label;
  if (tech) tech.textContent = `TECH: ${data.tech}`;
  if (work) work.textContent = data.work;
  archBubble.classList.add('active');
  node.userData.isHoveredForBubble = true;
}

function _hideBubble() {
  if (archBubble) archBubble.classList.remove('active');
  currentSubNodes.forEach(n => n.userData.isHoveredForBubble = false);
}

function _onClick(e) {
  if (hoveredNode && hoveredNode.userData.isMain && currentStageId === 'overview') {
    _drillDown(hoveredNode.userData);
    onClickCallback?.(hoveredNode.userData.id);
  }
}

function _highlightNode(node, active) {
  if (currentStageId !== 'overview' && node.userData.isMain) return;
  const scale = active ? 1.3 : 1.0;
  const op = active ? 0.3 : 0.15;
  gsap.to(node.scale, { x: scale, y: scale, z: scale, duration: 0.3 });
  if (node.userData.glow) gsap.to(node.userData.glow.material, { opacity: op, duration: 0.3 });
}

function _onResize() {
  if (!canvas) return;
  const w = canvas.clientWidth || 800;
  const h = canvas.clientHeight || 450;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}

function _updateHTMLOverlays() {
  const widthHalf = canvas.clientWidth / 2;
  const heightHalf = canvas.clientHeight / 2;
  if (widthHalf === 0 || heightHalf === 0) return;

  const pos = new THREE.Vector3();

  // Update Main Labels
  mainNodes.forEach(node => {
     node.userData.group.getWorldPosition(pos);
     pos.y += 1.8;
     pos.project(camera);
     node.userData.labelEl.style.left = `${(pos.x * widthHalf) + widthHalf}px`;
     node.userData.labelEl.style.top = `${-(pos.y * heightHalf) + heightHalf}px`;
  });

  // Update Sub Labels
  currentSubNodes.forEach(node => {
     node.getWorldPosition(pos);
     pos.y += 1.0; // Margin
     pos.project(camera);
     node.userData.labelEl.style.left = `${(pos.x * widthHalf) + widthHalf}px`;
     node.userData.labelEl.style.top = `${-(pos.y * heightHalf) + heightHalf}px`;
     node.userData.labelEl.style.opacity = node.userData.mat.opacity > 0.5 ? 1 : 0;
  });

  // Update Bubble Position
  const activeNode = currentSubNodes.find(n => n.userData.isHoveredForBubble);
  if (activeNode) {
     activeNode.getWorldPosition(pos);
     pos.project(camera);
     archBubble.style.left = `${(pos.x * widthHalf) + widthHalf}px`;
     archBubble.style.top = `${-(pos.y * heightHalf) + heightHalf}px`;
  }
}

function _animate() {
  if (isPaused) return;
  animId = requestAnimationFrame(_animate);
  const now = Date.now();
  
  mainNodes.forEach((node, i) => {
    node.rotation.y += 0.01;
    if (currentStageId === 'overview') {
      node.userData.group.position.y = Math.sin(now * 0.001 + i) * 0.15;
    }
  });

  currentSubNodes.forEach((node, i) => {
    node.rotation.y += 0.015;
    node.parent.position.y = Math.sin(now * 0.002 + i) * 0.08;
  });

  pulseTimer++;
  if (pulseTimer % 180 === 0) {
    if (currentStageId === 'overview') {
      overviewCurves.forEach(c => _createPulse(c.curve, c.color));
    } else {
      subCurves.forEach(c => _createPulse(c.curve, c.color, true));
    }
  }

  for (let i = flowPulses.length - 1; i >= 0; i--) {
    const p = flowPulses[i];
    p.progress += p.speed;
    if (p.progress >= 1) {
      scene.remove(p.mesh);
      p.mesh.geometry.dispose();
      p.mesh.material.dispose();
      flowPulses.splice(i, 1);
    } else {
      p.mesh.position.copy(p.curve.getPointAt(p.progress));
      p.mesh.material.opacity = p.progress > 0.8 ? (1 - p.progress) * 5 : 0.6;
    }
  }

  _updateHTMLOverlays();
  controls.update();
  renderer.render(scene, camera);
}

function _createPulse(curve, color, isSmall = false) {
  const geo = new THREE.SphereGeometry(isSmall ? 0.1 : 0.15, 8, 8);
  const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6 });
  const mesh = new THREE.Mesh(geo, mat);
  scene.add(mesh);
  flowPulses.push({ curve, mesh, progress: 0, speed: isSmall ? 0.008 : 0.004 });
}

export function pauseArchViz() { isPaused = true; if (animId) cancelAnimationFrame(animId); }
export function resumeArchViz() { if (!isPaused) return; isPaused = false; _animate(); }
