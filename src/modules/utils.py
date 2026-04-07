import json
import os
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

ROI_CONFIG_PATH = "../config/roi_config.json"


def get_video_stem(video_filename: str) -> str:
    return os.path.splitext(os.path.basename(video_filename))[0]


def load_roi_configuration(video_filename: str) -> Optional[List[List[float]]]:
    video_name = get_video_stem(video_filename)
    if not os.path.exists(ROI_CONFIG_PATH):
        return None

    with open(ROI_CONFIG_PATH, "r", encoding="utf-8") as file:
        roi_config = json.load(file)

    roi_points = roi_config.get(video_name)
    if roi_points is None or len(roi_points) != 4:
        return None
    return roi_points


def parse_roi_points_string(roi_points_text: str) -> Optional[List[List[float]]]:
    text = (roi_points_text or "").strip()
    if not text:
        return None

    points = []
    for chunk in text.split(";"):
        x_str, y_str = chunk.strip().split(",")
        points.append([float(x_str), float(y_str)])

    if len(points) != 4:
        raise ValueError("ROI inline phải có đúng 4 điểm.")
    return points


def select_roi_points_interactive(video_filename: str) -> List[List[float]]:
    cap = cv2.VideoCapture(video_filename)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read first frame for ROI selection.")

    points = []
    window_name = "ROI Selector"

    def redraw(base):
        canvas = base.copy()
        for i, p in enumerate(points):
            pt = (int(p[0]), int(p[1]))
            cv2.circle(canvas, pt, 2, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, str(i + 1), (pt[0] + 4, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        if len(points) >= 2:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], len(points) == 4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Left click: 4 points | z: undo | s: save | q: quit", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        return canvas

    def mouse_callback(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, redraw(frame))
        key = cv2.waitKey(10) & 0xFF
        if key == ord("z") and points:
            points.pop()
        elif key == ord("s"):
            if len(points) != 4:
                print("ROI phải có đúng 4 điểm.")
            else:
                cv2.destroyWindow(window_name)
                return points
        elif key == ord("q"):
            cv2.destroyWindow(window_name)
            raise RuntimeError("User cancelled ROI prompt.")


def resolve_roi_points(video_filename: str, roi_points_text: str = "", roi_source: str = "config") -> List[List[float]]:
    inline_points = parse_roi_points_string(roi_points_text)
    if inline_points is not None:
        return inline_points

    if roi_source == "prompt":
        return select_roi_points_interactive(video_filename)

    roi_points = load_roi_configuration(video_filename)
    if roi_points is None:
        raise RuntimeError("Không tìm thấy ROI hợp lệ trong roi_config.json. Hãy dùng --roi-source prompt hoặc --roi-points.")
    return roi_points


def initialize_model(model_path: str):
    return YOLO(model_path, task="detect")


def initialize_video_capture(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0
    return cap, fps


def initialize_video_writer(output_path: str, fps: float, frame_size):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def get_model_name_from_path(model_path: str) -> str:
    base = os.path.basename(os.path.normpath(model_path))
    name, ext = os.path.splitext(base)
    return name if ext else base


def _get_unique_path(video_path: str, model_path: str, size, output_dir: str, extension: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    video_name = get_video_stem(video_path)
    model_name = get_model_name_from_path(model_path)
    width, height = size
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{video_name}_{model_name}_{width}x{height}_{timestamp}{extension}"
    return os.path.normpath(os.path.join(output_dir, filename))


def get_unique_output_path(video_path, model_path, size, output_dir="../result/video", extension=".avi"):
    return _get_unique_path(video_path, model_path, size, output_dir, extension)


def get_log_file_path(video_path, model_path, size, output_dir="../result/benchmark", extension=".txt"):
    return _get_unique_path(video_path, model_path, size, output_dir, extension)
