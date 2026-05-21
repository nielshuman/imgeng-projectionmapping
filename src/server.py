"""
server.py — Raspberry Pi projector calibration + click-to-project
Replaces arcade with a FastAPI/WebSocket server + p5.js frontend.

Keys (in OpenCV debug window):
    R — re-run calibration
    E — quantify calibration error
    Q — quit server
"""

MODE = "marker"  # marker | face | laser
DEBUG = True
WHITE_SETTLE_TIME = 2 # .08

import sys

DEV = sys.argv[1] == "dev" if len(sys.argv) > 1 else False

if DEV:
    print("[mode] running in DEV mode with test image")

import asyncio
import json
import threading
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

if not DEV:
    from picamera2 import Picamera2

from detection import cornerdetect, laserdetect_fast, laserdetect
from quantification import run_quantification

# ── Resolutions ──────────────────────────────────────────────────────────────
PROJ_W, PROJ_H = 1280, 720
CAM_W, CAM_H = 640, 480


# ── Shared state ─────────────────────────────────────────────────────────────
shared = {
    "state": "white",
    "H": None,
    "corners": [],
    "marker": None,
    "cam_marker": None,
    "detect_frame": None,
    "reset": False,
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

# asyncio event loop reference
_loop = None


def send_state(data: dict):
    if _loop and manager.active:
        asyncio.run_coroutine_threadsafe(
            manager.broadcast(data),
            _loop,
        )


# ── Geometry helpers ─────────────────────────────────────────────────────────
def order_points(pts):
    pts = np.array(pts, dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    return np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)],
        ],
        dtype="float32",
    )


def build_homography(camera_corners):
    src = order_points(camera_corners)

    dst = np.array(
        [
            [0, 0],
            [PROJ_W - 1, 0],
            [PROJ_W - 1, PROJ_H - 1],
            [0, PROJ_H - 1],
        ],
        dtype="float32",
    )

    H, _ = cv2.findHomography(src, dst)

    return H


def cam_to_proj(H, cx, cy):
    pt = np.array([[[cx, cy]]], dtype="float32")

    r = cv2.perspectiveTransform(pt, H)

    return float(r[0][0][0]), float(r[0][0][1])


# ── OpenCV thread ────────────────────────────────────────────────────────────
def opencv_thread():
    if not DEV:
        cam = Picamera2()

        cam.configure(
            cam.create_preview_configuration(
                main={"size": (CAM_W, CAM_H), "format": "RGB888"}
            )
        )

        cam.start()

        cam.set_controls(
            {
                "AwbEnable": False,
                "AeEnable": False,
                "ExposureTime": 13000,
                "AnalogueGain": 1.0,
            }
        )

    white_start = time.monotonic()

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        with lock:
            state = shared["state"]
            H = shared["H"]

        if state != "live" or MODE != "marker" or H is None:
            return

        proj_x, proj_y = cam_to_proj(H, x, y)

        proj_x = max(0.0, min(float(PROJ_W - 1), proj_x))
        proj_y = max(0.0, min(float(PROJ_H - 1), proj_y))

        with lock:
            shared["marker"] = (proj_x, proj_y)
            shared["cam_marker"] = (x, y)

        print(f"[click] cam=({x},{y}) -> proj=({proj_x:.0f},{proj_y:.0f})")

        send_state(
            {
                "type": "marker",
                "x": proj_x,
                "y": proj_y,
            }
        )

    if DEBUG:
        cv2.namedWindow("Debug - camera")
        cv2.setMouseCallback("Debug - camera", mouse_callback)

    while True:
        if DEV:
            frame = cv2.imread("../test_material/normal/frame_360.jpg")
        else:
            frame = cam.capture_array()

        with lock:
            state = shared["state"]
            reset = shared["reset"]

        if reset:
            white_start = time.monotonic()

            with lock:
                shared.update(
                    state="white",
                    H=None,
                    corners=[],
                    marker=None,
                    cam_marker=None,
                    detect_frame=None,
                    reset=False,
                )

            state = "white"

            send_state(
                {
                    "type": "state",
                    "state": "white",
                }
            )

        if state == "white":
            elapsed = time.monotonic() - white_start

            corners, vis, thresh = cornerdetect(frame)

            if DEBUG:
                cv2.imshow("Debug - threshold", thresh)

            cv2.putText(
                vis,
                f"Projecting white... ({elapsed:.1f}s)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if elapsed >= WHITE_SETTLE_TIME:
                with lock:
                    shared["state"] = "detect"

                send_state(
                    {
                        "type": "state",
                        "state": "detect",
                    }
                )

        elif state == "detect":
            corners, vis, thresh = cornerdetect(frame)

            if DEBUG:
                cv2.imshow("Debug - threshold", thresh)

            if len(corners) == 4:
                H = build_homography(corners)

                corner_list = (
                    corners.tolist()
                    if hasattr(corners, "tolist")
                    else list(corners)
                )

                with lock:
                    shared["H"] = H
                    shared["corners"] = corner_list
                    shared["detect_frame"] = frame.copy()
                    shared["state"] = "live"

                print("[calibration] corners found, homography computed")

                send_state(
                    {
                        "type": "state",
                        "state": "live",
                    }
                )

            else:
                cv2.putText(
                    vis,
                    "Detecting corners...",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 100, 255),
                    2,
                )

        elif state == "live" and MODE == "marker":
            vis = frame.copy()

            with lock:
                marker = shared["marker"]
                cam_marker = shared["cam_marker"]
                corners_d = shared["corners"]

            for c in corners_d:
                cv2.drawMarker(
                    vis,
                    (int(c[0]), int(c[1])),
                    (0, 255, 0),
                    cv2.MARKER_CROSS,
                    12,
                    2,
                )

            if cam_marker and marker:
                cx, cy = cam_marker

                cv2.drawMarker(
                    vis,
                    (cx, cy),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    20,
                    3,
                )

                cv2.putText(
                    vis,
                    f"proj ({marker[0]:.0f},{marker[1]:.0f})",
                    (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            cv2.putText(
                vis,
                "Click=marker  E=quantify  R=recalibrate  Q=quit",
                (10, CAM_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

        elif state == "live" and MODE == "face":
            vis = frame.copy()

            cv2.putText(
                vis,
                "Face tracking mode (not implemented)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 100, 255),
                2,
            )

        elif state == "live" and MODE == "laser":
            point, vis, thresh = laserdetect_fast(frame, draw=True)

            cv2.putText(
                vis,
                "Laser tracking mode",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 100, 255),
                2,
            )

            if DEBUG:
                cv2.imshow("Debug - threshold laser", thresh)

            if point is not None:
                with lock:
                    H = shared["H"]

                if H is not None:
                    cx, cy = point

                    proj_x, proj_y = cam_to_proj(H, cx, cy)

                    send_state(
                        {
                            "type": "laser",
                            "x": float(proj_x),
                            "y": float(proj_y),
                        }
                    )

        if DEBUG:
            cv2.imshow("Debug - camera", vis)

        key = cv2.waitKey(1) & 0xFF if DEBUG else -1

        if key == ord("q"):
            send_state({"type": "quit"})
            break

        elif key == ord("r"):
            with lock:
                shared["reset"] = True

        elif key == ord("e"):
            with lock:
                detect_frame = shared["detect_frame"]
                detection_points = shared["corners"]

            if detect_frame is None or len(detection_points) == 0:
                print("[quantify] no calibration frame available")
                continue

            run_quantification(
                detect_frame,
                detection_points,
            )

    if not DEV:
        cam.stop()

    if DEBUG:
        cv2.destroyAllWindows()


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI()


@app.get("/")
async def serve_projector():
    return FileResponse("projector.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)

    with lock:
        state = shared["state"]
        marker = shared["marker"]

    await ws.send_text(
        json.dumps(
            {
                "type": "state",
                "state": state,
            }
        )
    )

    if marker:
        await ws.send_text(
            json.dumps(
                {
                    "type": "marker",
                    "x": marker[0],
                    "y": marker[1],
                }
            )
        )

    try:
        while True:
            raw = await ws.receive_text()

            try:
                msg = json.loads(raw)
            except Exception:
                continue

            if msg.get("type") == "reset":
                print("[client] recalibration requested")

                with lock:
                    shared["reset"] = True

    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(
        target=opencv_thread,
        daemon=True,
    )

    t.start()

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )

    server = uvicorn.Server(config)

    async def _run():
        global _loop

        _loop = asyncio.get_running_loop()

        await server.serve()

    asyncio.run(_run())
