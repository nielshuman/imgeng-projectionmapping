"""
server.py — Raspberry Pi projector calibration + click-to-project
Replaces arcade with a FastAPI/WebSocket server + p5.js frontend.

Setup:
  - Run:  uvicorn server:app --host 0.0.0.0 --port 8000
  - Open projector browser at:  http://localhost:8000/
  - Debug monitor shows OpenCV imshow windows as usual

Phases:
  1. WHITE  — browser shows white fullscreen for camera to settle
  2. DETECT — cornerdetect() runs each frame until 4 corners found
  3. LIVE   — browser goes black; click in OpenCV window maps through H
              → red crosshair drawn by p5.js on projector

Keys (in OpenCV debug window):
  R — re-run calibration
  Q — quit server
"""

import asyncio
import json
import threading
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
# from picamera2 import Picamera2

from detection import cornerdetect

# ── Resolutions ──────────────────────────────────────────────────────────────
PROJ_W, PROJ_H = 1280, 720
CAM_W,  CAM_H  = 640,  480

WHITE_SETTLE_TIME = 2.0   # seconds
MODE="face" # marker | face

# ── Shared state ─────────────────────────────────────────────────────────────
shared = {
    "state":      "white",  # white | detect | live
    "H":          None,
    "corners":    [],
    "marker":     None,     # (proj_x, proj_y)
    "cam_marker": None,
    "reset":      False,
}
lock = threading.Lock()

# ── WebSocket connection manager ─────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)

manager = ConnectionManager()

# asyncio event loop reference — set once uvicorn starts
_loop = None

def send_state(data: dict):
    """Thread-safe fire-and-forget broadcast from the OpenCV thread."""
    if _loop and manager.active:
        asyncio.run_coroutine_threadsafe(manager.broadcast(data), _loop)

# ── Geometry helpers ──────────────────────────────────────────────────────────
def order_points(pts):
    pts  = np.array(pts, dtype="float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype="float32")

def build_homography(camera_corners):
    src = order_points(camera_corners)
    dst = np.array([
        [0,          0         ],
        [PROJ_W - 1, 0         ],
        [PROJ_W - 1, PROJ_H - 1],
        [0,          PROJ_H - 1],
    ], dtype="float32")
    H, _ = cv2.findHomography(src, dst)
    return H

def cam_to_proj(H, cx, cy):
    pt = np.array([[[cx, cy]]], dtype="float32")
    r  = cv2.perspectiveTransform(pt, H)
    return float(r[0][0][0]), float(r[0][0][1])

# ── OpenCV thread ─────────────────────────────────────────────────────────────
def opencv_thread():
    # cam = Picamera2()
    # cam.configure(cam.create_preview_configuration(
    #     main={"size": (CAM_W, CAM_H), "format": "RGB888"}
    # ))
    # cam.start()
    # cam.set_controls({
    #     "AwbEnable":    False,
    #     "AeEnable":     False,
    #     "ExposureTime": 13000,
    #     "AnalogueGain": 1.0,
    # })

    white_start = time.monotonic()

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        with lock:
            state = shared["state"]
            H     = shared["H"]
        if state != "live" or MODE != "marker" or shared["H"] is None:
            return
        proj_x, proj_y = cam_to_proj(H, x, y)
        proj_x = max(0.0, min(float(PROJ_W - 1), proj_x))
        proj_y = max(0.0, min(float(PROJ_H - 1), proj_y))
        with lock:
            shared["marker"]     = (proj_x, proj_y)
            shared["cam_marker"] = (x, y)
        print(f"[click] cam=({x},{y})  ->  proj=({proj_x:.0f},{proj_y:.0f})")
        send_state({"type": "marker", "x": proj_x, "y": proj_y})

    cv2.namedWindow("Debug - camera")
    cv2.setMouseCallback("Debug - camera", mouse_callback)

    while True:
        # frame = cam.capture_array()
        frame = cv2.imread("../test_material/normal/frame_360.jpg")
        
        with lock:
            state = shared["state"]
            reset = shared["reset"]

        if reset:
            white_start = time.monotonic()
            with lock:
                shared.update(state="white", H=None, corners=[],
                              marker=None, cam_marker=None, reset=False)
            state = "white"
            send_state({"type": "state", "state": "white"})

        if state == "white":
            elapsed = time.monotonic() - white_start
            vis = frame.copy()
            cv2.putText(vis, f"Projecting white... ({elapsed:.1f}s)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if elapsed >= WHITE_SETTLE_TIME:
                with lock:
                    shared["state"] = "detect"
                send_state({"type": "state", "state": "detect"})

        elif state == "detect":
            corners, vis, thresh = cornerdetect(frame)
            cv2.imshow("Debug - threshold", thresh)

            if len(corners) == 4:
                H = build_homography(corners)
                corner_list = corners.tolist() if hasattr(corners, "tolist") else list(corners)
                with lock:
                    shared["H"]       = H
                    shared["corners"] = corner_list
                    shared["state"]   = "live"
                print("[calibration] corners found, homography computed")
                send_state({"type": "state", "state": "live"})
            else:
                cv2.putText(vis, "Detecting corners...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        elif state == "live" and MODE == "marker":
            vis = frame.copy()
            with lock:
                marker     = shared["marker"]
                cam_marker = shared["cam_marker"]
                corners_d  = shared["corners"]

            for c in corners_d:
                cv2.drawMarker(vis, (int(c[0]), int(c[1])),
                               (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
            if cam_marker and marker:
                cx, cy = cam_marker
                cv2.drawMarker(vis, (cx, cy),
                               (0, 0, 255), cv2.MARKER_CROSS, 20, 3)
                cv2.putText(vis, f"proj ({marker[0]:.0f},{marker[1]:.0f})",
                            (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(vis, "Click to project marker  |  R=recalibrate  Q=quit",
                        (10, CAM_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
        elif state == "live" and MODE == "face":
            vis = frame.copy()
            cv2.putText(vis, "Face tracking mode (not implemented)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        cv2.imshow("Debug - camera", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            send_state({"type": "quit"})
            break
        elif key == ord("r"):
            with lock:
                shared["reset"] = True

    # cam.stop()
    cv2.destroyAllWindows()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()

@app.get("/")
async def serve_projector():
    return FileResponse("projector.html")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    # Send current state immediately on connect
    with lock:
        state  = shared["state"]
        marker = shared["marker"]
    await ws.send_text(json.dumps({"type": "state", "state": state}))
    if marker:
        await ws.send_text(json.dumps({"type": "marker", "x": marker[0], "y": marker[1]}))

    try:
        while True:
            await ws.receive_text()   # keep connection alive; server is push-only
    except WebSocketDisconnect:
        manager.disconnect(ws)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=opencv_thread, daemon=True)
    t.start()

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)

    # Grab the event loop after uvicorn sets it up
    async def _run():
        global _loop
        _loop = asyncio.get_running_loop()
        await server.serve()

    asyncio.run(_run())
