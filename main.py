import cv2
import numpy as np
from picamera2 import Picamera2
import time

MIN_DETECTION_AREA = 5000
THRESHOLD = 150
BLUR_AMOUNT = 5
RUN_NAME="checkerboard"
CAPTURE_INTERVAL = 60 # save a frame every 60 frames (roughly every 2 seconds at 30fps)
CAPTURE=True

        #ideal projector angle?


picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
))
picam2.start()

# cv2.waitKey(1000) # let camera settle/inir

# Lock settings to reduce projector rainbow artifacts
picam2.set_controls({
    "AwbEnable": False,
    "AeEnable": False,    
    "ExposureTime": 13000,   # try 10000–30000
    "AnalogueGain": 1.0
})

framecount = 0
# if directory exists, delete it and remake it
import os
import shutil
if CAPTURE:
    if os.path.exists(f"test_material/{RUN_NAME}"):
        shutil.rmtree(f"test_material/{RUN_NAME}")
    os.makedirs(f"test_material/{RUN_NAME}")


while True: # means repeat forever
    frame = picam2.capture_array()
    detection_frame = frame.copy() 
    gray = cv2.cvtColor(detection_frame, cv2.COLOR_RGB2GRAY)

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

            if len(approx) == 4: #only if we have a rectangle (well, vierhoek) quadrilateral!!!!
                pts = approx.reshape(4, 2)

                # Draw corners
                for x, y in pts:
                    cv2.circle(detection_frame, (x, y), 8, (0, 255, 0), -1)

                cv2.polylines(detection_frame, [approx], True, (255, 0, 0), 2)

    cv2.imshow("Projector Corner Detect", detection_frame)
    cv2.imshow("Blur", blur)
    cv2.imshow("Threshold", thresh)

    framecount += 1
    if framecount % CAPTURE_INTERVAL == 0 and CAPTURE:
        cv2.imwrite(f"test_material/{RUN_NAME}/frame_{framecount}.jpg", frame)
        print('Saved frame', framecount)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()