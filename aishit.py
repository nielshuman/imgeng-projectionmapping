"""
tune_gui.py  —  DearPyGui parameter tuner for cornerdetect()
"""

import dearpygui.dearpygui as dpg
import cv2
import numpy as np
from detection import cornerdetect, MIN_DETECTION_AREA, THRESHOLD, BLUR_AMOUNT

# ── state ────────────────────────────────────────────────────────────────────
SOURCE_IMAGE = cv2.imread("test_material/normal/frame_360.jpg")
H, W = SOURCE_IMAGE.shape[:2]
PANEL_W, PANEL_H = 640, int(640 * H / W)   # display size for each panel

params = {
    "blur":     BLUR_AMOUNT,
    "threshold": THRESHOLD,
    "min_area": MIN_DETECTION_AREA,
}

# ── helpers ──────────────────────────────────────────────────────────────────
def bgr_to_rgba_flat(img_bgr: np.ndarray, w: int, h: int) -> list:
    """Resize, convert BGR→RGBA, return flat float list DPG expects."""
    resized = cv2.resize(img_bgr, (w, h))
    rgba    = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
    return (rgba / 255.0).flatten().tolist()

def gray_to_rgba_flat(img_gray: np.ndarray, w: int, h: int) -> list:
    resized = cv2.resize(img_gray, (w, h))
    rgba    = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGBA)
    return (rgba / 255.0).flatten().tolist()

def run_detection():
    blur  = params["blur"]
    blur  = max(1, blur | 1)          # force odd
    thresh = params["threshold"]
    area  = params["min_area"]

    result_img = SOURCE_IMAGE.copy()
    thresh_img = None
    pts        = None
    corner_info = []

    try:
        pts, result_img, thresh_img = cornerdetect(
            SOURCE_IMAGE,
            blur_amount=blur,
            threshold=thresh,
            min_detection_area=area,
        )
        if pts is not None:
            for i, (x, y) in enumerate(pts):
                corner_info.append(f"  [{i}]  ({x:>4}, {y:>4})")
    except Exception as e:
        corner_info.append(f"  no quad detected")

    return result_img, thresh_img, pts, corner_info

def refresh():
    result_img, thresh_img, pts, corner_info = run_detection()

    # detection panel
    dpg.set_value("tex_result",
                  bgr_to_rgba_flat(result_img, PANEL_W, PANEL_H))

    # threshold panel
    if thresh_img is not None:
        dpg.set_value("tex_thresh",
                      gray_to_rgba_flat(thresh_img, PANEL_W, PANEL_H))

    # histogram overlay on threshold image
    update_histogram(thresh_img)

    # corner info text
    if corner_info:
        dpg.set_value("txt_corners", "\n".join(corner_info))
        dpg.configure_item("txt_corners", color=[120, 230, 120])
    else:
        dpg.set_value("txt_corners", "  no quad detected")
        dpg.configure_item("txt_corners", color=[230, 100, 100])

    # status bar
    blur_safe = max(1, params["blur"] | 1)
    dpg.set_value("txt_status",
                  f"blur={blur_safe}   threshold={params['threshold']}   "
                  f"min_area={params['min_area']}")

def update_histogram(gray_img):
    if gray_img is None:
        return
    hist = cv2.calcHist([gray_img], [0], None, [64], [0, 256]).flatten()
    hist_norm = (hist / hist.max() * 80).tolist()   # normalise to 80 px tall
    thresh_x  = int(params["threshold"] / 256 * 64)

    # rebuild series data
    xs = [i * 4 for i in range(64)]                 # bin centres (0-252)
    dpg.set_value("hist_series", [xs, hist_norm])
    dpg.set_value("thresh_line", [[params["threshold"], params["threshold"]], [0, 80]])

# ── slider callbacks ─────────────────────────────────────────────────────────
def on_blur(s, v, u):
    params["blur"] = v
    refresh()

def on_threshold(s, v, u):
    params["threshold"] = v
    refresh()

def on_area(s, v, u):
    params["min_area"] = v * 100
    refresh()

def on_reset(s, v, u):
    dpg.set_value("sl_blur",      BLUR_AMOUNT)
    dpg.set_value("sl_threshold", THRESHOLD)
    dpg.set_value("sl_area",      MIN_DETECTION_AREA // 100)
    params["blur"]      = BLUR_AMOUNT
    params["threshold"] = THRESHOLD
    params["min_area"]  = MIN_DETECTION_AREA
    refresh()

def on_save(s, v, u):
    blur_safe = max(1, params["blur"] | 1)
    snippet = (
        f"MIN_DETECTION_AREA = {params['min_area']}\n"
        f"THRESHOLD          = {params['threshold']}\n"
        f"BLUR_AMOUNT        = {blur_safe}\n"
    )
    print("\n── copy into detection.py ──────────────────\n" + snippet)
    dpg.set_value("txt_status", "✓ values printed to console")

# ── build UI ─────────────────────────────────────────────────────────────────
dpg.create_context()

with dpg.font_registry():
    pass   # uses built-in font; swap path here for a custom .ttf

# texture registry — must exist before windows reference textures
with dpg.texture_registry():
    init_flat = [0.1] * (PANEL_W * PANEL_H * 4)
    dpg.add_dynamic_texture(PANEL_W, PANEL_H, init_flat, tag="tex_result")
    dpg.add_dynamic_texture(PANEL_W, PANEL_H, init_flat, tag="tex_thresh")

with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg,       (18,  18,  24))
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg,        (26,  26,  34))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        (38,  38,  52))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (52,  52,  72))
        dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,     (100, 140, 240))
        dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive,(130,170,255))
        dpg.add_theme_color(dpg.mvThemeCol_Button,         (55,  90, 180))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (75, 115, 210))
        dpg.add_theme_color(dpg.mvThemeCol_Header,         (55,  90, 180, 180))
        dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,  (30,  50, 120))
        dpg.add_theme_color(dpg.mvThemeCol_TitleBg,        (20,  35,  90))
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,  8)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,   6)
        dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,    4)
        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,    10, 8)
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,  14, 14)

dpg.bind_theme(global_theme)

# ── main window ──────────────────────────────────────────────────────────────
with dpg.window(label="Corner Detection Tuner",
                width=PANEL_W * 2 + 320, height=PANEL_H + 240,
                no_close=True, tag="main_win"):

    # ── top row: two image panels ────────────────────────────────────────
    with dpg.group(horizontal=True):
        with dpg.child_window(width=PANEL_W + 16, height=PANEL_H + 40,
                              border=True, label="Detection result"):
            dpg.add_text("Detection result",
                         color=[160, 180, 255])
            dpg.add_image("tex_result", width=PANEL_W, height=PANEL_H)

        dpg.add_spacer(width=6)

        with dpg.child_window(width=PANEL_W + 16, height=PANEL_H + 40,
                              border=True):
            dpg.add_text("Threshold view",
                         color=[160, 180, 255])
            dpg.add_image("tex_thresh", width=PANEL_W, height=PANEL_H)

        dpg.add_spacer(width=6)

        # ── right column: controls ───────────────────────────────────
        with dpg.child_window(width=280, height=PANEL_H + 40, border=True):
            dpg.add_text("Parameters", color=[200, 210, 255])
            dpg.add_separator()
            dpg.add_spacer(height=4)

            dpg.add_text("Blur kernel", color=[140, 150, 180])
            dpg.add_slider_int(tag="sl_blur", min_value=1, max_value=31,
                               default_value=BLUR_AMOUNT,
                               callback=on_blur, width=-1)
            dpg.add_spacer(height=8)

            dpg.add_text("Threshold", color=[140, 150, 180])
            dpg.add_slider_int(tag="sl_threshold", min_value=0, max_value=255,
                               default_value=THRESHOLD,
                               callback=on_threshold, width=-1)
            dpg.add_spacer(height=8)

            dpg.add_text("Min area  (×100 px²)", color=[140, 150, 180])
            dpg.add_slider_int(tag="sl_area",
                               min_value=1,
                               max_value=500,
                               default_value=MIN_DETECTION_AREA // 100,
                               callback=on_area, width=-1)

            dpg.add_spacer(height=16)
            dpg.add_separator()
            dpg.add_spacer(height=8)

            dpg.add_text("Detected corners", color=[200, 210, 255])
            dpg.add_text("  —", tag="txt_corners", color=[150, 150, 150])

            dpg.add_spacer(height=16)
            dpg.add_separator()
            dpg.add_spacer(height=8)

            dpg.add_text("Brightness histogram", color=[200, 210, 255])
            with dpg.plot(height=100, width=-1, no_title=True,
                          no_mouse_pos=True):
                dpg.add_plot_axis(dpg.mvXAxis, label="",
                                  no_tick_labels=True, tag="hist_x")
                dpg.set_axis_limits("hist_x", 0, 255)
                with dpg.plot_axis(dpg.mvYAxis, label="",
                                   no_tick_labels=True, tag="hist_y"):
                    dpg.set_axis_limits("hist_y", 0, 85)
                    dpg.add_bar_series([], [], tag="hist_series",
                                       weight=4)
                    dpg.add_line_series([], [], tag="thresh_line",
                                        label="threshold")

            dpg.add_spacer(height=16)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reset",  callback=on_reset, width=120)
                dpg.add_button(label="Save ↗", callback=on_save,  width=120)

    # ── status bar ───────────────────────────────────────────────────────
    dpg.add_spacer(height=6)
    dpg.add_separator()
    dpg.add_text("", tag="txt_status", color=[120, 130, 160])

# ── run ───────────────────────────────────────────────────────────────────────
dpg.create_viewport(title="Detection Tuner",
                    width=PANEL_W * 2 + 360, height=PANEL_H + 290)
dpg.setup_dearpygui()
dpg.show_viewport()

refresh()   # populate on first frame

dpg.set_primary_window("main_win", True)
dpg.start_dearpygui()
dpg.destroy_context()