from src.detection import cornerdetect, MIN_DETECTION_AREA, THRESHOLD, BLUR_AMOUNT
import cv2
import os
import numpy as np

RUN_NAME = 'normal'

# for frame_filename in os.listdir(f'test_material/{RUN_NAME}/'):
#     corners, vis, thresh = cornerdetect(cv2.imread(f"test_material/{RUN_NAME}/{frame_filename}"))
#     cv2.imshow("corners", vis)
#     cv2.imshow("threshold", thresh)
#     cv2.waitKey(0)
  
def pick_points(frame, line=True):
    points = []
    
    def click_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            points.append((x, y))
    
    cv2.namedWindow("quantification")
    cv2.setMouseCallback("quantification", click_callback)
    
    while True:
        canvas = frame.copy()
        for (x, y) in points:
            cv2.drawMarker(canvas, (x, y), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1, line_type=cv2.LINE_AA)
        if line:
            cv2.polylines(canvas, [np.array(points)], True, (255, 0, 0), 1)
        cv2.imshow("quantification", canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            points = []
            pass
        
        if key == ord("q"):
            break
        
        #enter
        if key == 13: 
            break
        
        if key == ord("z"):
            points.pop()
        
    cv2.destroyAllWindows()
    return points

def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def quantify_error(human_points, detection_points):
    error = []

    for point in human_points:
        closest_point = min(detection_points, key=lambda point2: distance(point, point2))
        error.append(distance(point, closest_point))
        print(f"Point from quantification: {point} corresponds to point from corner detection: {closest_point}, error: {distance(point, closest_point)}")
    
    return error

# test_frame = cv2.imread('test_material/normal/frame_360.jpg')
# human_points = pick_points(test_frame)
# detection_points = cornerdetect(test_frame)[0]

# error = quantify_error(human_points, detection_points)
# print("Average error:", np.mean(error))
# print("Max error:", np.max(error))
# print("Total error:", np.sum(error))

def order_points(pts):
    pts = np.array(pts, dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


frame = cv2.imread('test_material/checkerboard/frame_780.jpg')
# corners, vis, thresh = cornerdetect(frame)
corners = pick_points(frame)
ordered_corners = order_points(corners)
# cv2.imshow("corners", vis)

dst = np.array([[0, 0], [1023, 0], [1023, 767], [0, 767]], dtype=np.float32)

H, _ = cv2.findHomography(ordered_corners, dst)

pts_on_photo = pick_points(frame, line=False)
pts_on_original = cv2.perspectiveTransform(np.array([pts_on_photo], dtype=np.float32), H)[0]

# display a white 1024x768 canvas with the selected points
canvas = cv2.imread('checkboard.png')
for (x, y) in pts_on_original:
    cv2.drawMarker(canvas, (int(x), int(y)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3, line_type=cv2.LINE_AA)
for (x, y) in pts_on_photo:
    cv2.drawMarker(frame, (x, y), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3, line_type=cv2.LINE_AA)
cv2.imwrite("quantification_result.jpg", canvas)
cv2.imwrite("quantification_photo.jpg", frame)
# cv2.imshow("threshold", thresh)
while True:
        cv2.imshow("photo", frame)
        cv2.imshow("original", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cv2.destroyAllWindows()