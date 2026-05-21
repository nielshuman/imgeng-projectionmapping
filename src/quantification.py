import cv2
import numpy as np


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
            cv2.drawMarker(
                canvas,
                (x, y),
                (255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=1,
                line_type=cv2.LINE_AA,
            )

        if line and len(points) > 1:
            cv2.polylines(
                canvas,
                [np.array(points)],
                True,
                (255, 0, 0),
                1,
            )

        cv2.putText(
            canvas,
            "ENTER=finish  Z=undo  R=reset  Q=cancel",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("quantification", canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            points = []

        elif key == ord("z"):
            if len(points) > 0:
                points.pop()

        elif key == ord("q"):
            cv2.destroyWindow("quantification")
            return None

        elif key == 13:  # ENTER
            break

    cv2.destroyWindow("quantification")
    return points


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def quantify_error(human_points, detection_points):
    error = []

    for point in human_points:
        closest_point = min(
            detection_points,
            key=lambda point2: distance(point, point2),
        )

        err = distance(point, closest_point)
        error.append(err)

        print(
            f"[quantify] human={point} "
            f"closest={closest_point} "
            f"error={err:.2f}px"
        )

    return error


def run_quantification(frame, detection_points):
    print("[quantify] click points to compare against detected corners")
    print("[quantify] ENTER=finish  Q=cancel")

    human_points = pick_points(frame, line=False)

    if human_points is None or len(human_points) == 0:
        print("[quantify] cancelled")
        return

    error = quantify_error(human_points, detection_points)

    print("\n──── Quantification Results ────")
    print(f"Points tested : {len(error)}")
    print(f"Average error : {np.mean(error):.2f}px")
    print(f"Max error     : {np.max(error):.2f}px")
    print(f"Total error   : {np.sum(error):.2f}px")
    print("────────────────────────────────\n")

    vis = frame.copy()

    # Human-picked points (red)
    for p in human_points:
        cv2.drawMarker(
            vis,
            (int(p[0]), int(p[1])),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            20,
            2,
        )

    # Detection points (green)
    for p in detection_points:
        cv2.drawMarker(
            vis,
            (int(p[0]), int(p[1])),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            20,
            2,
        )

    cv2.imshow("quantification result", vis)