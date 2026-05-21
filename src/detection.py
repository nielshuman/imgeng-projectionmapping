"""
Detection script
"""

import cv2

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


def laserdetect(frame):
    # Blur slightly to reduce noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Red wraps around HSV hue range, so use two masks
    lower_red1 = np.array([0, 120, 200])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 200])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Largest contour
        c = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(c)

        # Ignore tiny noise
        if area > 5:
            (x, y), radius = cv2.minEnclosingCircle(c)

            center = (int(x), int(y))

            # Draw detection
            cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
            cv2.circle(frame, center, 3, (255, 0, 0), -1)

            print("Laser detected at:", center)

    cv2.imshow("Laser Detection", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)