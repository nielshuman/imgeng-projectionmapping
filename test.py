import cv2
import numpy as np
from picamera2 import Picamera2
from src.detection import laserdetect, laserdetect_fast

MIN_DETECTION_AREA = 5000
THRESHOLD = 180
BLUR_AMOUNT = 5

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
))
picam2.start()

# Lock settings to reduce projector rainbow artifacts
picam2.set_controls({
    "AwbEnable": False,
    "AeEnable": False,
    "ExposureTime": 15000,   # try 10000–30000
    "AnalogueGain": 0.7
})

while True:
    frame = picam2.capture_array()

    point, vis, thresh = laserdetect_fast(frame)
    
    if point:
        print("Laser:", point)
        
    cv2.imshow("vis", vis)
    cv2.imshow("threshold", thresh)
    cv2.waitKey(1)

cv2.destroyAllWindows()