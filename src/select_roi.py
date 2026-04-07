import argparse
import json
import os

import cv2
import numpy as np

ROI_CONFIG_PATH = "../config/roi_config.json"

points = []
current_frame = None
display_frame = None


def redraw():
    global display_frame
    display_frame = current_frame.copy()

    if len(points) >= 2:
        cv2.polylines(display_frame, [np.array(points, dtype=np.int32)], len(points) == 4, (0, 255, 255), 1, cv2.LINE_AA)

    for i, p in enumerate(points):
        cv2.circle(display_frame, tuple(p), 2, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(display_frame, str(i + 1), (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)


def mouse_callback(event, x, y, flags, param):
    del flags, param
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        redraw()


def save_roi(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if os.path.exists(ROI_CONFIG_PATH):
        with open(ROI_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    data[video_name] = points

    with open(ROI_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved ROI for {video_name}: {points}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=str)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read first frame from video.")

    current_frame = frame.copy()
    display_frame = frame.copy()
    redraw()

    cv2.namedWindow("ROI Selector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Selector", mouse_callback)

    while True:
        temp = display_frame.copy()
        cv2.putText(
            temp,
            "Left click: 4 points | z: undo | s: save | q: quit",
            (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("ROI Selector", temp)
        key = cv2.waitKey(10) & 0xFF

        if key == ord("z") and points:
            points.pop()
            redraw()
        elif key == ord("s"):
            if len(points) != 4:
                print("ROI phải có đúng 4 điểm.")
            else:
                save_roi(args.video)
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
