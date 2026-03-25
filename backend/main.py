"""
main.py — FastAPI entry point for NASA NEO Interactive Universe API v2
Replaces the original Flask app.py (which remains untouched).

Usage:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import create_pool, close_pool, background_new_asteroid_watcher
from routers import asteroids, risk, mlops, ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    await create_pool()
    # Launch background watcher (polls DB every 60s for new unique asteroids)
    asyncio.create_task(background_new_asteroid_watcher())
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────────
    await close_pool()


app = FastAPI(
    title="NASA NEO Interactive Universe API",
    description=(
        "Read-only API serving 32,001+ unique Near-Earth Asteroids from PHO Database. "
        "Powers the interactive 3D Orrery, Explorer, Leaderboard, and MLOps dashboard."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server and Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],           # read-only API
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(asteroids.router, prefix="/api")
app.include_router(risk.router,      prefix="/api")
app.include_router(mlops.router,     prefix="/api")
app.include_router(ws.router)                       # WebSocket at /ws/live


@app.get("/api/health", tags=["Health"])
async def health():
    """Health check — also used by frontend to wake Render free-tier instance."""
    return {"status": "ok", "service": "NASA NEO API", "version": "2.0.0"}
