"""
routers/ws.py — WebSocket live updates (zero-intervention new asteroid broadcast)
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from database import get_ws_clients

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    Keep-alive WebSocket connection.
    Clients receive {"event": "new_asteroid", "data": {...}} whenever
    the background watcher detects a new unique asteroid ID in train_neo.
    """
    await websocket.accept()
    clients = get_ws_clients()
    clients.add(websocket)
    try:
        while True:
            # Keep connection alive; client can send heartbeat pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception:
        clients.discard(websocket)
