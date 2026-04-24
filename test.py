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
    "AwbEnable": False
    "AeEnable": False,    
    "ExposureTime": 10000,   # try 10000–30000
    "AnalogueGain": 1.0
})

while True: # means repeat forever
    frame = picam2.capture_array()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    cv2.imshow("Projector Corner Detect", frame)
    cv2.imshow("gray", gray)
    
    cv2.waitKey(1)

cv2.destroyAllWindows()