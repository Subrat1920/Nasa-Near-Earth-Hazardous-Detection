# 🚀 NASA NEO Interactive Universe — Full Implementation Plan

## What We're Building

A **stunning, self-updating interactive web universe** where:
- 🌍 Earth sits at the center of a living 3D solar system
- ☄️ **32,001 unique asteroids** orbit Earth with real trajectories — auto-growing weekly
- Each asteroid = one permanent orbit object (not 71,085 rows — the same asteroid appears many times as it makes repeated close approaches over the years)
- 🔴 Color-coded by hazard (PHO=red, Sentry=orange, safe=blue/white)
- **32,001 unique orbit rings** — one per unique asteroid ID
- Click any asteroid → full detail panel + close approach history timeline
- Compare, rank, and explore every unique asteroid ever logged by NASA

---

## Real Data From Your NeonDB (as of 2026-03-25)

| Table | Rows | Unique Asteroids | Key Info |
|---|---|---|---|
| `train_neo` | **71,085** | **32,001** | One asteroid = multiple rows (avg 2.2 close approaches/asteroid) |
| `test_neo` | 3,082 | — | Holdout set |
| `prediction_table` | 2,964 | — | Model predictions with probabilities |
| [risk_analysis](file:///d:/Data%20Science/Data%20Science%20Projects/NASA_Near_Earth_Object_Detection/src/custom/asteroid_risk_analysis.py#264-284) | 2,896 | — | Composite risk scores (manual + data-driven) |
| `neo_data_drift` | 238 | — | Feature drift records across 34 runs |
| `model_training_logs` | 5 | — | All training history (AdaBoost best so far) |
| `alert_recipients` | 1 | — | Email alert config |

**Key unique asteroid stats:**
- Unique asteroids: **32,001** (32,001 orbiting objects in the 3D scene)
- Unique PHOs (Hazardous): **1,957** | Unique Sentry Objects: **1,732**
- Each asteroid has on average **~2.2 logged close approaches** (= 71,085 total rows)
- Miss distance range: **6,599 km** → **74.8M km**
- Velocity: up to **65.8 km/s**
- Date range logged: **2015 → 2026+**

> **Design implication:** In the 3D scene, each unique [id](file:///d:/Data%20Science/Data%20Science%20Projects/NASA_Near_Earth_Object_Detection/src/custom/asteroid_risk_analysis.py#48-311) = **one orbit ring**. When you click an asteroid, the Explorer shows **all its historical close approaches** as a timeline (from its repeated rows in `train_neo`).

---

## Architecture

```
NeonDB (PostgreSQL / NeonDB)
       │  weekly GitHub Action pushes new asteroids
       ▼
FastAPI Backend  (replaces Flask)
  ├── REST endpoints (paginated asteroid data, filters, leaderboards)
  ├── WebSocket endpoint (/ws/live) — broadcasts new asteroid events
  └── SSE endpoint (/stream/asteroids) — streams full catalog to 3D scene
       │
       ▼
Three.js Frontend (Vite + Vanilla JS)
  ├── 3D Orrery  ← Earth + asteroid orbits, live-updating
  ├── Explorer   ← click-to-inspect
  ├── Compare    ← side-by-side multi-asteroid
  ├── Leaderboard ← Top 10/50/100/1000 tables
  ├── MLOps Panel ← model metrics + drift
  └── API Docs   ← beautiful interactive docs
```

---

## Backend: Flask → FastAPI

### Why FastAPI
- **Native async** — non-blocking DB queries with `asyncpg`
- **Built-in WebSockets** — push new asteroid events to all open UIs instantly
- **Auto Swagger docs** at `/docs` (no manual work)
- **Pydantic models** — type-validated responses
- **3-5× faster** than Flask under load

### New File Structure
```
backend/
├── main.py              ← FastAPI app entry point
├── database.py          ← asyncpg connection pool to NeonDB
├── models.py            ← Pydantic response models
├── routers/
│   ├── asteroids.py     ← /api/asteroids/* routes
│   ├── predictions.py   ← /api/predictions/* routes
│   ├── risk.py          ← /api/risk/* routes
│   ├── mlops.py         ← /api/mlops/* routes
│   └── ws.py            ← WebSocket /ws/live
└── services/
    └── predictor.py     ← ML inference (ported from app.py)
```

### Key API Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/asteroids` | Paginated unique asteroids (`?page=1&limit=100&hazardous=true`) |
| `GET` | `/api/asteroids/{id}` | Single asteroid — latest approach + all historical approaches |
| `GET` | `/api/asteroids/stream` | SSE stream — all **32,001 unique** asteroids for 3D scene |
| `GET` | `/api/asteroids/new?since=epoch` | Only newly added asteroids (for live update) |
| `GET` | `/api/leaderboard?by=risk&top=1000` | Top N by risk/velocity/size/miss distance |
| `GET` | `/api/compare?ids=123,456,789` | Multi-asteroid comparison payload |
| `GET` | `/api/risk` | All risk analysis data |
| `GET` | `/api/mlops/models` | Model training history |
| `GET` | `/api/mlops/drift` | Drift detection history |
| `GET` | `/api/stats` | Global KPI stats (counts, averages) |
| `POST` | `/api/predict` | Hazard prediction (ML inference) |
| `WS` | `/ws/live` | WebSocket — push new asteroid events |

---

## Frontend: 6 Pages/Views

### 🌌 Page 1 — The Orrery (Main 3D Scene)
> The headline feature. Earth at center, asteroids orbiting.

**Visual Design:**
- Deep space background with star field (star shader)
- Earth with realistic texture + atmosphere glow (Three.js `TextureLoader`)
- Moon orbit ring for scale reference
- Each asteroid = a sphere, sized proportionally to [(min_diameter_km + max_diameter_km) / 2](file:///d:/Data%20Science/Data%20Science%20Projects/NASA_Near_Earth_Object_Detection/app.py#54-87)

**Color Coding:**
| Condition | Color |
|---|---|
| `is_potentially_hazardous = true` | 🔴 Crimson red |
| `is_sentry_object = true` | 🟠 Orange |
| PHO + Sentry | 💜 Deep purple (rarest, most dangerous) |
| Neither | 🔵 Ice blue / white |

**Orbital Mapping (how DB columns → 3D orbit):**
- `miss_distance_km` → orbit radius (log-scaled: 6,599–74.8M km → 2–50 scene units) — use **latest approach row** per unique [id](file:///d:/Data%20Science/Data%20Science%20Projects/NASA_Near_Earth_Object_Detection/src/custom/asteroid_risk_analysis.py#48-311)
- `relative_velocity_kps` → orbital speed (angular velocity) — use latest row
- `close_approach_date` → initial position on orbit (epoch seed)
- Orbits are elliptical rings (inclined slightly for 3D look)
- **SQL for unique asteroids:** `SELECT DISTINCT ON (id) * FROM train_neo ORDER BY id, epoch_date_close_approach DESC`

**Auto-Update (Zero Human Intervention):**
- On page load: streams all asteroids via SSE `/api/asteroids/stream`
- WebSocket `/ws/live` stays open — when GitHub Action adds new rows to `train_neo`, FastAPI detects via DB polling every 60s and broadcasts new asteroid data
- Frontend receives event → spawns new orbit + asteroid sphere instantly

**Controls:**
- Mouse drag → rotate scene
- Scroll wheel → zoom
- Left panel: filter by PHO / Sentry / date range / velocity
- Asteroid count badge top-right (live counter)

---

### 🔭 Page 2 — Asteroid Explorer (Click to Inspect)
> Click any asteroid in the Orrery → slide-in detail panel

**Panel Contents:**
```
┌─────────────────────────────────────┐
│  ☄️ 85990 (1999 JV6)               │
│  ID: 2085990                        │
│─────────────────────────────────────│
│  📏 Diameter: 0.23 – 0.52 km       │
│  💨 Velocity: 7.69 km/s            │
│  📡 Miss Distance: 12,463,291 km   │
│  🗓️ Close Approach: 2015-Jan-05    │
│─────────────────────────────────────│
│  🔴 POTENTIALLY HAZARDOUS          │
│  ⚡ Sentry Object: No              │
│─────────────────────────────────────│
│  ML Prediction                      │
│  Hazardous: True (55.3% confidence)│
│─────────────────────────────────────│
│  Risk Score: 0.511 (Medium Risk)   │
│  Impact Energy: 2.9 × 10¹⁹ J      │
│  Estimated Mass: 1.68 × 10¹¹ kg   │
│─────────────────────────────────────│
│  [🔗 NASA JPL Link]  [+ Compare]   │
└─────────────────────────────────────┘
```

**Data joins:**
- `train_neo` → all close approach rows for this asteroid (shown as timeline)
- `prediction_table` (join on [id](file:///d:/Data%20Science/Data%20Science%20Projects/NASA_Near_Earth_Object_Detection/src/custom/asteroid_risk_analysis.py#48-311)) → ML probability
- [risk_analysis](file:///d:/Data%20Science/Data%20Science%20Projects/NASA_Near_Earth_Object_Detection/src/custom/asteroid_risk_analysis.py#264-284) (join on [id](file:///d:/Data%20Science/Data%20Science%20Projects/NASA_Near_Earth_Object_Detection/src/custom/asteroid_risk_analysis.py#48-311)) → risk score, mass, energy

**Timeline section in Explorer:** All `close_approach_date` rows for this asteroid listed chronologically — shows how the same asteroid has passed Earth multiple times

---

### ⚖️ Page 3 — Multi-Asteroid Comparator

**How it works:**
- Click "Add to Compare" in explorer (up to 6 asteroids at once)
- Side-by-side comparison card grid

**Comparison Metrics:**
| Metric | Source Column |
|---|---|
| Diameter | `min_diameter_km` / `max_diameter_km` |
| Velocity | `relative_velocity_kps` |
| Miss Distance | `miss_distance_km` |
| Hazard Status | `is_potentially_hazardous` |
| Sentry Object | `is_sentry_object` |
| ML Hazard % | `probability_being_truely_hazardous` |
| Risk Score | `RiskScorenormManual` |
| Impact Energy | `impact_energy_j` |
| Mass | `mass_kg` |
| Risk Category | `RiskCategoryManual` |

**Visualization:** Radar chart (spider chart) overlaying all selected asteroids — each metric is a spoke. Instantly see which asteroid is most dangerous across all dimensions.

---

### 🏆 Page 4 — Leaderboard

**Tabs (switchable):**
1. **Top N by Risk Score** — `ORDER BY RiskScorenormManual DESC`
2. **Top N by Velocity** — `ORDER BY relative_velocity_kps DESC`
3. **Top N by Closest Approach** — `ORDER BY miss_distance_km ASC`
4. **Top N by Size** — `ORDER BY max_diameter_km DESC`
5. **PHO Only** — filter `is_potentially_hazardous = true`
6. **Sentry Only** — filter `is_sentry_object = true`

**N selector:** 10 / 50 / 100 / 1000 (toggleable)

**Table Columns:** Rank | Name | ID | Diameter | Velocity | Miss Dist | Hazardous | Risk Score | Risk Category | [Inspect] [Compare]

**Clicking a leaderboard row** → highlights that asteroid in the 3D Orrery and opens the Explorer panel

---

### 🧠 Page 5 — MLOps Dashboard

**Sections:**
1. **Model Version Timeline** — Line chart of accuracy/recall/F1 across all 5 training runs
2. **Current Model Card** — Best model name, training date, accuracy 86.8%, recall 98.7%, F1 46.7%
3. **Data Drift Status Panel** — Feature-by-feature drift table from `neo_data_drift`
   - `epoch_date_close_approach` → 🔴 HIGH DRIFT
   - Others → 🟢 NO DRIFT (with PSI values)
4. **Population Stability Index Chart** — Bar chart per feature
5. **GitHub Actions Status** — 6 workflow badges w/ last run date
6. **Embedded Grafana** — iframe embed of your existing Grafana dashboard

---

### 📖 Page 6 — API Documentation

- Beautiful custom Swagger-alternative
- Auto-generated from FastAPI's OpenAPI schema
- Embedded live request tester
- Code examples: Python / curl / PowerShell

---

## Auto-Update Flow (Zero Human Intervention Detail)

```
Every Monday (GitHub Action: data_pusher.yml)
    │
    ▼
NASA API fetches new week's close approaches
    │
    ▼
new rows INSERT into train_neo (NeonDB)
  (may be a repeat approach for existing asteroid OR a brand-new unique asteroid)
    │
    ▼
FastAPI background task polls every 60s:
    SELECT DISTINCT id FROM train_neo
    WHERE id NOT IN (:known_unique_ids_set)
    │
    ├── Only new close approach row for EXISTING asteroid
    │       → update its orbit data silently, no new sphere added
    │
    └── Brand new unique asteroid ID found
            → broadcast via WebSocket /ws/live:
              {
                "event": "new_asteroid",
                "data": { id, name, miss_distance_km,
                          relative_velocity_kps, is_potentially_hazardous,
                          is_sentry_object, ... }
              }
              ▼
    All connected 3D Orrery UIs receive WebSocket event
              ▼
    JavaScript spawns new orbit ring + asteroid sphere
    Live counter badge increments: "32,001 → 32,002 unique asteroids"
    Leaderboard auto-refreshes if new asteroid ranks in Top N
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | **FastAPI** (Python) | Async, WebSocket, auto-docs |
| DB Driver | **asyncpg** | Async PostgreSQL, fastest Python driver |
| 3D Engine | **Three.js + InstancedMesh** | 32,001 asteroids = **1 draw call** — works on Intel integrated graphics |
| Frontend | **Vite + Vanilla JS** | Fast build, no framework overhead |
| Charts | **Chart.js** | Lightweight, beautiful, radar + line |
| Styling | **Custom CSS** | Dark space theme, glassmorphism |
| Fonts | **Google Fonts: Orbitron + Inter** | Space aesthetic + readability |
| Animations | **GSAP** | Smooth panel transitions |

---

## 🖥️ Performance Strategy (i3 CPU, No Dedicated GPU)

> i3 processors have **Intel UHD integrated graphics** — WebGL works fine. The real enemy is draw call count, not the lack of a dedicated GPU.

### The Core Solution: InstancedMesh
Instead of 32,001 individual Three.js `Mesh` objects (= 32,001 draw calls → instant lag), we use **one** `InstancedMesh`:
```js
// ONE draw call for ALL 32,001 asteroids
const geometry = new THREE.SphereGeometry(1, 6, 6); // low-poly sphere
const material = new THREE.MeshBasicMaterial();      // no lighting calc
const asteroids = new THREE.InstancedMesh(geometry, material, 32001);
// Each asteroid gets its own matrix (position/scale) and color
```
This renders all 32,001 objects at the cost of a **single draw call** — comfortably runs at 60fps on Intel UHD graphics.

### Additional Optimisations
| Technique | What it does |
|---|---|
| **Low-poly spheres** | 6×6 segments instead of 32×32 — 97% fewer vertices |
| **`MeshBasicMaterial`** | No lighting calculations — cheapest possible material |
| **Level of Detail (LOD)** | Zoomed out → hide orbit rings; zoomed in → show detail |
| **Frustum culling** | Three.js auto-skips objects outside camera view |
| **Lazy SSE loading** | Stream asteroids in batches of 500 → scene populates gradually, no freeze |
| **Orbit rings on demand** | Only draw orbit ellipse for the hovered/selected asteroid |
| **WebWorker for data** | Parse 32K asteroid JSON off the main thread → UI never blocks |

### Orbit Rings Detail
Showing 32,001 orbit rings simultaneously would tank performance. Strategy:
- **Default view:** Only the asteroid dot is visible
- **Hover:** Orbit ring appears for that single asteroid (1 ring drawn)
- **Selected (Explorer open):** Orbit ring stays visible
- **Filter active** (e.g. Top 50 PHOs): Show rings only for filtered set

---

## UI Design System

**Color Palette:**
```
Background:   #020817  (deep space black)
Surface:      #0d1b2a  (dark navy)
Glass:        rgba(13, 27, 42, 0.7) + blur(12px)
Accent Blue:  #00d4ff  (electric cyan)
Hazard Red:   #ff3b3b  (PHO asteroids + alerts)
Sentry Amber: #ffa500  (sentry objects)
Safe Green:   #00ff88  (confirmed safe)
Purple:       #9b59b6  (PHO + Sentry both)
Text Primary: #e2e8f0
Text Muted:   #64748b
```

**Typography:**
- `Orbitron` — headings, stats, NASA-style feel
- `Inter` — body text, tables
- `JetBrains Mono` — IDs, coordinate values, code

---

## Files To Create (New)

### Backend
- `backend/main.py` — FastAPI app
- `backend/database.py` — asyncpg pool
- `backend/models.py` — Pydantic schemas
- `backend/routers/asteroids.py`
- `backend/routers/risk.py`
- `backend/routers/mlops.py`
- `backend/routers/ws.py`
- `backend/services/predictor.py`

### Frontend
- `frontend/index.html`
- `frontend/src/main.js` — app entry, routing
- `frontend/src/orrery.js` — Three.js 3D scene
- `frontend/src/explorer.js` — detail panel
- `frontend/src/comparator.js` — compare view
- `frontend/src/leaderboard.js` — top-N tables
- `frontend/src/mlops.js` — MLOps dashboard
- `frontend/src/api.js` — all fetch/WebSocket calls
- `frontend/src/styles/` — CSS modules per view

### Config
- `backend/requirements.txt` — fastapi, asyncpg, uvicorn, pydantic, mlflow, etc.
- `frontend/package.json` — vite, three, chart.js, gsap

---

## 🆓 Zero-Cost Deployment Stack

> Every service below has a permanent free tier — no credit card required for the base usage.

| Service | What it hosts | Free Tier Limits | Cost |
|---|---|---|---|
| **Vercel** | Frontend (Vite build) | Unlimited bandwidth, CDN, HTTPS auto | $0 |
| **Render** | FastAPI backend | 750 hrs/month, sleeps after 15min idle | $0 |
| **NeonDB** | PostgreSQL database | Already in use, 0.5 GB, always-on | $0 |
| **GitHub Actions** | CI/CD, data pipelines | 2,000 min/month | $0 |
| **Grafana Cloud** | Monitoring dashboard | Already set up | $0 |

### How it Works Together
```
Vercel (frontend CDN, global)
    │  fetches from
    ▼
Render (FastAPI, free instance)
    │  reads from
    ▼
NeonDB (PostgreSQL, Neon free tier)
```

### Render Sleep-Wake Issue & Fix
Render free tier sleeps after 15 min of inactivity. Fix:
- On frontend load, send a `/api/health` ping → wakes the backend (takes ~10s)
- Show a **"Connecting to mission control..."** loading animation during wake-up
- This is handled gracefully — user sees a space-themed loading screen instead of an error

### Deployment Steps (one-time setup)
1. Push `frontend/` to GitHub → connect to **Vercel** → auto-deploys on every push
2. Push `backend/` to GitHub → connect to **Render** → auto-deploys on every push
3. Set environment variables on Render: `DATABASE_URL`, `MLFLOW_TRACKING_URI`, etc.
4. Point Vercel env var `VITE_API_URL` to your Render backend URL

---

## Verification Plan

### Automated
```bash
# 1. Start FastAPI backend
cd backend && uvicorn main:app --reload --port 8000

# 2. Test all REST endpoints
curl http://localhost:8000/api/stats
curl http://localhost:8000/api/asteroids?limit=5
curl "http://localhost:8000/api/leaderboard?by=risk&top=10"
curl "http://localhost:8000/api/compare?ids=2085990,2523915"

# 3. Start Vite dev server
cd frontend && npm run dev  # opens at localhost:5173
```

### Manual Browser Verification
1. Open `http://localhost:5173` — 3D Orrery loads, Earth visible, asteroids start populating
2. Rotate/zoom the scene — smooth 60fps
3. Click a red asteroid — Explorer panel slides in with correct data
4. Go to Leaderboard → switch to Top 1000 → table loads
5. Click any leaderboard row → 3D scene highlights that asteroid
6. Add 2 asteroids to Compare → Comparator opens with radar chart
7. Open DevTools → Network → check WebSocket connection at `/ws/live`
