import cv2
import numpy as np
from picamera2 import Picamera2

MIN_DETECTION_AREA = 5000
THRESHOLD = 180
BLUR_AMOUNT = 5

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
))
picam2.start()

# cv2.waitKey(1000) # let camera settle/inir

# Lock settings to reduce projector rainbow artifacts
picam2.set_controls({
    # "AwbEnable": False
    "AeEnable": False,    
    # "ExposureTime": 10000,   # try 10000–30000
    # "AnalogueGain": 1.0
})

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Blur - to smooth out shit
    blur = cv2.GaussianBlur(gray, (BLUR_AMOUNT, BLUR_AMOUNT), 0)
    
    # Threshold image
    _, thresh = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL, # only outer boundaries (no holes / no inner contours)
        cv2.CHAIN_APPROX_SIMPLE # compress segments to only their coner points
    ) # together, just detect big rectagle

    if contours:
        largest = max(contours, key=cv2.contourArea)

        # Ignore tiny blobs
        if cv2.contourArea(largest) > MIN_DETECTION_AREA:
            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

            if len(approx) == 4: #only if we have a rectangle (well, vierhoek)
                pts = approx.reshape(4, 2)

                # Draw corners
                for x, y in pts:
                    cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

                cv2.polylines(frame, [approx], True, (255, 0, 0), 2)

    cv2.imshow("Projector Corner Detect", frame)
    cv2.imshow("Blur", blur)
    cv2.imshow("Threshold", thresh)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()