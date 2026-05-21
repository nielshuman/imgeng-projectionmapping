"""
Detection script
"""

import cv2
import numpy as np

# Define parameters at the top of the file or in a config section
MIN_DETECTION_AREA = 5000
THRESHOLD = 210
BLUR_AMOUNT = 5

def cornerdetect(frame, blur_amount=BLUR_AMOUNT, threshold=THRESHOLD, min_detection_area=MIN_DETECTION_AREA):
    detection_frame = frame.copy() 
    gray = cv2.cvtColor(detection_frame, cv2.COLOR_RGB2GRAY)

    # Blur - to smooth out noise
    if blur_amount > 0:
        blur = cv2.GaussianBlur(gray, (blur_amount, blur_amount), 0)
    else:
        blur = gray.copy()

    # Threshold image
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL, # only outer boundaries (no holes / no inner contours)
        cv2.CHAIN_APPROX_SIMPLE # compress segments to only their corner points
    )

    pts = []
    
    if contours:
        largest = max(contours, key=cv2.contourArea)

        # Ignore tiny blobs
        if cv2.contourArea(largest) > min_detection_area:
            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
            print(approx)

            if len(approx) == 4: # only if we have a quadrilateral
                pts = approx.reshape(4, 2)

                # Draw corners
                for x, y in pts:
                    cv2.circle(detection_frame, (x, y), 8, (0, 255, 0), -1)

                cv2.polylines(detection_frame, [approx], True, (255, 0, 0), 2)
    return pts, detection_frame, thresh

def tune_parameters(frame):
    window = "tune parameters"
    cv2.namedWindow(window)

    # Trackbars
    cv2.createTrackbar("blur",       window, BLUR_AMOUNT,        31,      lambda v: None)
    cv2.createTrackbar("threshold",  window, THRESHOLD,          255,     lambda v: None)
    # cv2.createTrackbar("min area /10", window, MIN_DETECTION_AREA // 10, 5000, lambda v: None)

    while True:
        blur  = cv2.getTrackbarPos("blur",         window)
        threshold = cv2.getTrackbarPos("threshold",    window)
        # area  = cv2.getTrackbarPos("min area /10", window) * 10
        area = MIN_DETECTION_AREA

        # blur must be odd and > 0 for GaussianBlur
        blur_safe = max(1, blur | 1)   # bitwise OR with 1 forces odd
        pts, vis, thresh = cornerdetect(frame, blur_amount=blur_safe,
                                             threshold=threshold,
                                             min_detection_area=area)

        # Overlay current values on the preview
        info = f"blur={blur_safe}  thresh={threshold}  min_area={area}"
        cv2.putText(vis, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window, vis)
        cv2.imshow("threshold view", thresh)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 13):   # q or Enter to confirm 
            break

    cv2.destroyAllWindows()
    print(f"Final values → BLUR_AMOUNT={blur_safe}, THRESHOLD={threshold}, MIN_DETECTION_AREA={area}")
    return blur_safe, threshold, area

# HSV ranges for common laser colors
LASER_COLORS = {
    "red": [
        (np.array([0, 120, 200]),   np.array([10, 255, 255])),
        (np.array([170, 120, 200]), np.array([180, 255, 255])),
    ],
    "green": [
        (np.array([40, 100, 180]), np.array([80, 255, 255])),
    ],
    "blue": [
        (np.array([100, 120, 180]), np.array([130, 255, 255])),
    ],
}

def laserdetect(frame, color="red", min_area=5000, max_area=50000):
    """
    Detect a laser point in an RGB888 frame (from Picamera2).
    Returns (x, y) centroid or None if not found.
    Displays the annotated frame via cv2.imshow.
    """
    # Picamera2 RGB888 → OpenCV expects BGR for display, HSV conversion needs BGR
    bgr = frame.copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    ranges = LASER_COLORS[color]
    mask = cv2.inRange(hsv, ranges[0][0], ranges[0][1])
    for lo, hi in ranges[1:]:
        mask |= cv2.inRange(hsv, lo, hi)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    point = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if point is None or area < point[2]:
                    point = (cx, cy, area)

    if point:
        x, y = point[0], point[1]
        cv2.circle(bgr, (x, y), 10, (0, 255, 0), 2)
        cv2.circle(bgr, (x, y), 2,  (0, 255, 0), -1)
        cv2.putText(bgr, f"({x}, {y})", (x + 14, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Laser detection", bgr)
    cv2.waitKey(1)  # needed to actually render the imshow window

    return (point[0], point[1]) if point else None