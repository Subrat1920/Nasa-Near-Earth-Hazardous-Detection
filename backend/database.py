"""
database.py — asyncpg connection pool for NeonDB (READ-ONLY)
All queries in this project use SELECT only. No INSERT/UPDATE/DELETE.
"""
import asyncpg
import asyncio
import os
import ssl
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from dotenv import load_dotenv

# Load from project root .env (one level up from backend/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

_pool: asyncpg.Pool = None
_known_unique_ids: set = set()
_ws_clients: set = set()


def _clean_db_url(raw_url: str) -> str:
    """Strip unsupported params (channel_binding) from the DSN."""
    parsed = urlparse(raw_url)
    query = parse_qs(parsed.query)
    query.pop("channel_binding", None)
    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


async def create_pool() -> asyncpg.Pool:
    global _pool
    raw_url = os.getenv("DATABASE_URL", "")
    if not raw_url:
        raise RuntimeError("DATABASE_URL not found in environment")
    clean_url = _clean_db_url(raw_url)

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    _pool = await asyncpg.create_pool(
        dsn=clean_url,
        ssl=ssl_ctx,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        await create_pool()
    return _pool


def get_ws_clients() -> set:
    return _ws_clients


def get_known_ids() -> set:
    return _known_unique_ids


async def background_new_asteroid_watcher():
    """
    Polls NeonDB every 60s for brand-new unique asteroid IDs.
    On first run: populates the known set silently.
    On subsequent runs: broadcasts truly new unique IDs to all WS clients.
    """
    import json
    global _known_unique_ids

    pool = await get_pool()
    first_run = True

    while True:
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT DISTINCT id FROM train_neo")
                current_ids = {r["id"] for r in rows}

                if first_run:
                    _known_unique_ids = current_ids
                    first_run = False
                else:
                    new_ids = current_ids - _known_unique_ids
                    if new_ids:
                        for new_id in new_ids:
                            row = await conn.fetchrow(
                                """
                                SELECT DISTINCT ON (id)
                                    id, name, min_diameter_km, max_diameter_km,
                                    relative_velocity_kps, miss_distance_km,
                                    is_potentially_hazardous, is_sentry_object,
                                    close_approach_date, epoch_date_close_approach
                                FROM train_neo
                                WHERE id = $1
                                ORDER BY id, epoch_date_close_approach DESC
                                """,
                                new_id,
                            )
                            if row:
                                payload = json.dumps(
                                    {"event": "new_asteroid", "data": dict(row)},
                                    default=str,
                                )
                                dead = set()
                                for ws in _ws_clients.copy():
                                    try:
                                        await ws.send_text(payload)
                                    except Exception:
                                        dead.add(ws)
                                _ws_clients -= dead

                        _known_unique_ids = current_ids

        except Exception as e:
            print(f"[WATCHER ERROR] {e}")

        await asyncio.sleep(60)
