"""
projector_sketch.py — Raspberry Pi projector calibration + click-to-project

Setup:
  - Projector connected to Pi → runs the arcade fullscreen window
  - Debug monitor → shows OpenCV imshow windows
  - Camera → pointed at the projection surface

Phases:
  1. WHITE     — arcade projects white fullscreen for 2s to let camera settle
  2. DETECT    — keeps trying cornerdetect() each frame until 4 corners found
  3. LIVE      — arcade goes black; OpenCV shows camera feed.
                 Click in the OpenCV window → point is mapped through H
                 (camera space → projector space) → red crosshair drawn by arcade

Keys (in OpenCV window):
  R — re-run calibration
  Q — quit
"""

import threading
import cv2
import numpy as np
import arcade
from picamera2 import Picamera2

from detection import cornerdetect

# ── Resolutions ─────────────────────────────────────────────────────────────
PROJ_W, PROJ_H = 1280, 720
CAM_W,  CAM_H  = 640, 480

WHITE_SETTLE_TIME = 2.0   # seconds to show white before detecting

# ── Shared state between OpenCV thread and arcade window ────────────────────
shared = {
    "state":      "white",   # white | detect | live
    "H":          None,      # homography matrix, set once corners found
    "marker":     None,      # (proj_x, proj_y) in projector pixel space
    "cam_marker": None,      # (cam_x, cam_y) — the raw click point for debug display
    "reset":      False,     # set True from OpenCV thread to trigger recalibration
}
lock = threading.Lock()


# ── Geometry helpers ─────────────────────────────────────────────────────────

def order_points(pts):
    pts  = np.array(pts, dtype="float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],    # top-left
        pts[np.argmin(diff)], # top-right
        pts[np.argmax(s)],    # bottom-right
        pts[np.argmax(diff)], # bottom-left
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


# ── OpenCV thread — camera capture, detection, debug display ────────────────

def opencv_thread():
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (CAM_W, CAM_H), "format": "RGB888"}
    ))
    cam.start()
    cam.set_controls({
        "AwbEnable":    False,
        "AeEnable":     False,
        "ExposureTime": 13000,
        "AnalogueGain": 1.0,
    })

    white_start = cv2.getTickCount()
    freq        = cv2.getTickFrequency()

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        with lock:
            state = shared["state"]
            H     = shared["H"]
        if state != "live" or H is None:
            return
        proj_x, proj_y = cam_to_proj(H, x, y)
        proj_x = max(0.0, min(float(PROJ_W - 1), proj_x))
        proj_y = max(0.0, min(float(PROJ_H - 1), proj_y))
        with lock:
            shared["marker"]     = (proj_x, proj_y)
            shared["cam_marker"] = (x, y)
        print(f"[click] cam=({x},{y})  ->  proj=({proj_x:.0f},{proj_y:.0f})")

    cv2.namedWindow("Debug - camera")
    cv2.setMouseCallback("Debug - camera", mouse_callback)

    while True:
        rgb = cam.capture_array()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        with lock:
            state = shared["state"]
            reset = shared["reset"]

        # Handle reset from keypress
        if reset:
            white_start = cv2.getTickCount()
            with lock:
                shared["state"]      = "white"
                shared["H"]          = None
                shared["marker"]     = None
                shared["cam_marker"] = None
                shared["reset"]      = False
            state = "white"

        # ── State machine ────────────────────────────────────────────────
        if state == "white":
            elapsed = (cv2.getTickCount() - white_start) / freq
            vis = bgr.copy()
            cv2.putText(vis, f"Projecting white... ({elapsed:.1f}s)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if elapsed >= WHITE_SETTLE_TIME:
                with lock:
                    shared["state"] = "detect"

        elif state == "detect":
            corners, vis, thresh = cornerdetect(bgr)
            cv2.imshow("Debug - threshold", thresh)

            if len(corners) == 4:
                H = build_homography(corners)
                with lock:
                    shared["H"]     = H
                    shared["state"] = "live"
                print("[calibration] corners found, homography computed")
            else:
                cv2.putText(vis, "Detecting corners...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        elif state == "live":
            vis = bgr.copy()
            with lock:
                marker = shared["marker"]
                H      = shared["H"]

            # Draw the 4 calibration corners for reference
            corners_disp, _, _ = cornerdetect(bgr)
            for c in corners_disp:
                cv2.drawMarker(vis, (int(c[0]), int(c[1])),
                               (0, 255, 0), cv2.MARKER_CROSS, 12, 2)

            # Draw the click point directly — no need to back-project,
            # it's just the original mouse coordinates
            with lock:
                cam_marker = shared["cam_marker"]
            if cam_marker is not None and marker is not None:
                cx, cy = cam_marker
                cv2.drawMarker(vis, (cx, cy),
                               (0, 0, 255), cv2.MARKER_CROSS, 20, 3)
                cv2.putText(vis, f"proj ({marker[0]:.0f},{marker[1]:.0f})",
                            (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.putText(vis, "Click to project marker  |  R=recalibrate  Q=quit",
                        (10, CAM_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Debug - camera", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            arcade.exit()
            break
        elif key == ord("r"):
            with lock:
                shared["reset"] = True

    cam.stop()
    cv2.destroyAllWindows()


# ── Arcade window — runs on the projector ───────────────────────────────────

class ProjectorWindow(arcade.Window):

    def __init__(self):
        super().__init__(PROJ_W, PROJ_H, "Projector", fullscreen=True)

    def on_update(self, delta_time):
        pass  # all logic lives in the OpenCV thread

    def on_draw(self):
        self.clear()

        with lock:
            state  = shared["state"]
            marker = shared["marker"]

        if state in ("white", "detect"):
            # Project solid white for calibration
            arcade.draw_lrbt_rectangle_filled(0, PROJ_W, 0, PROJ_H,
                                              arcade.color.WHITE)
            return

        # state == "live" -> black background
        arcade.draw_lrbt_rectangle_filled(0, PROJ_W, 0, PROJ_H,
                                          arcade.color.BLACK)

        if marker:
            px, py = marker
            # Arcade y=0 is bottom; projector/image y=0 is top -> flip
            py_arcade = PROJ_H - py

            r = 18
            t = 4
            arcade.draw_line(px - r, py_arcade, px + r, py_arcade,
                             arcade.color.RED, t)
            arcade.draw_line(px, py_arcade - r, px, py_arcade + r,
                             arcade.color.RED, t)
            arcade.draw_circle_outline(px, py_arcade, r,
                                       arcade.color.RED, t)

    def on_key_press(self, key, modifiers):
        if key in (arcade.key.ESCAPE, arcade.key.Q):
            self.close()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Start OpenCV camera/detection loop in a background thread
    t = threading.Thread(target=opencv_thread, daemon=True)
    t.start()

    # Run arcade on the main thread (required on most platforms)
    window = ProjectorWindow()
    arcade.run()