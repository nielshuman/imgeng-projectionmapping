"""
Main script - to be run on raspberry Pi
"""

import cv2
from picamera2 import Picamera2
import os
import shutil

from detection import cornerdetect

RUN_NAME="checkerboard"
CAPTURE_INTERVAL = 60 # save a frame every 60 frames (roughly every 2 seconds at 30fps)
CAPTURE=True

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
))
picam2.start()

# Lock settings to reduce projector rainbow artifacts
picam2.set_controls({
    "AwbEnable": False,
    "AeEnable": False,
    "ExposureTime": 13000,   # try 10000–30000
    "AnalogueGain": 1.0
})

framecount = 0

if CAPTURE:
    if os.path.exists(f"test_material/{RUN_NAME}"):
        shutil.rmtree(f"test_material/{RUN_NAME}")
    os.makedirs(f"test_material/{RUN_NAME}")


while True: 
    frame = picam2.capture_array()
    framecount += 1

    r, detection_frame, thresh = cornerdetect(frame)
    cv2.imshow("Projector Corner Detect", detection_frame)
    cv2.imshow("Threshold", thresh)

    if CAPTURE and framecount % CAPTURE_INTERVAL == 0:
        cv2.imwrite(f"test_material/{RUN_NAME}/frame_{framecount}.jpg", frame)
        print('Saved frame', framecount)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()