"""
Detection script
"""

import cv2

# Define parameters at the top of the file or in a config section
MIN_DETECTION_AREA = 5000
THRESHOLD = 210
BLUR_AMOUNT = 5

def cornerdetect(frame, show=True, blur_amount=BLUR_AMOUNT, threshold=THRESHOLD, min_detection_area=MIN_DETECTION_AREA):
    detection_frame = frame.copy() 
    gray = cv2.cvtColor(detection_frame, cv2.COLOR_RGB2GRAY)

    # Blur - to smooth out noise
    if BLUR_AMOUNT > 0:
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
                
                return pts

    if show:
        cv2.imshow("Projector Corner Detect", detection_frame)
        cv2.imshow("Blur", blur)
        cv2.imshow("Threshold", thresh)