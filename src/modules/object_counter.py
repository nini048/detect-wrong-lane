from collections import Counter, defaultdict, deque
from time import time
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics.solutions.solutions import SolutionAnnotator
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import colors

FONT = cv2.FONT_HERSHEY_SIMPLEX


class ObjectCounter:
    def __init__(self):
        self.im0 = None
        self.tf = 2
        self.view_img = False
        self.names = {}
        self.annotator = None
        self.window_name = "DETECT THE VEHICLES IN WRONG LANE"

        self.track_history: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=20))
        self.lane_history: Dict[int, Deque[int]] = defaultdict(lambda: deque(maxlen=7))
        self.stable_lane_idx: Dict[int, int] = {}
        self.lane_switch_streak: Dict[int, int] = defaultdict(int)
        self.lane_change_min_frames = 3

        self.flow_vote_history: Deque[str] = deque(maxlen=15)
        self.flow_direction = "unknown"
        self.reverse_lane_order = False

        self.track_thickness = 1
        self.draw_tracks = False
        self.track_color = None

        self.lane_rules = []
        self.violation_min_frames = 5
        self.wrong_way_min_frames = 4
        self.direction_min_dy = 12.0

        self.violation_streak: Dict[int, int] = {}
        self.wrong_way_streak: Dict[int, int] = {}

        self.current_wrong_ids = set()
        self.current_wrong_way_ids = set()
        self.total_violation_count = 0
        self.total_wrong_way_count = 0
        self.seen_violation_ids = set()
        self.seen_wrong_way_ids = set()

        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        classes_names,
        lane_rules,
        violation_min_frames=5,
        wrong_way_min_frames=4,
        direction_min_dy=12.0,
        line_thickness=2,
        track_thickness=1,
        view_img=False,
        draw_tracks=False,
        track_color=None,
    ):
        self.tf = line_thickness
        self.view_img = view_img
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks
        self.names = classes_names
        self.track_color = track_color
        self.lane_rules = lane_rules or []
        self.violation_min_frames = violation_min_frames
        self.wrong_way_min_frames = wrong_way_min_frames
        self.direction_min_dy = float(direction_min_dy)

    def _extract_track_rows(self, tracks):
        if not tracks or tracks[0].boxes.id is None:
            return []

        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()

        rows = []
        for box, track_id, cls in zip(boxes, track_ids, clss):
            bottom_center = (float((box[0] + box[2]) / 2), float(box[3]))
            rows.append(
                {
                    "box": box,
                    "track_id": int(track_id),
                    "cls": int(cls),
                    "bottom_center": bottom_center,
                }
            )
        return rows

    def _cleanup_missing_tracks(self, active_track_ids):
        active_track_ids = set(active_track_ids)
        stale_ids = [tid for tid in list(self.track_history.keys()) if tid not in active_track_ids]

        for tid in stale_ids:
            self.track_history.pop(tid, None)
            self.lane_history.pop(tid, None)
            self.stable_lane_idx.pop(tid, None)
            self.lane_switch_streak.pop(tid, None)
            self.violation_streak.pop(tid, None)
            self.wrong_way_streak.pop(tid, None)

    def _estimate_track_direction(self, track_id: int) -> str:
        history = self.track_history.get(track_id)
        if history is None or len(history) < 5:
            return "unknown"

        start_y = float(history[0][1])
        end_y = float(history[-1][1])
        dy = end_y - start_y

        if abs(dy) < self.direction_min_dy:
            return "unknown"
        return "far_to_near" if dy > 0 else "near_to_far"

    def _update_flow_direction(self, active_track_ids: List[int]):
        votes = []
        for tid in active_track_ids:
            direction = self._estimate_track_direction(tid)
            if direction != "unknown":
                votes.append(direction)

        if len(votes) < 2:
            return

        frame_flow = Counter(votes).most_common(1)[0][0]
        self.flow_vote_history.append(frame_flow)

        if len(self.flow_vote_history) < 4:
            return

        stable_flow = Counter(self.flow_vote_history).most_common(1)[0][0]
        self.flow_direction = stable_flow
        self.reverse_lane_order = stable_flow == "near_to_far"

    def prime_tracks(self, tracks):
        rows = self._extract_track_rows(tracks)
        active_track_ids = [row["track_id"] for row in rows]
        self._cleanup_missing_tracks(active_track_ids)

        for row in rows:
            self.track_history[row["track_id"]].append(row["bottom_center"])

        self._update_flow_direction(active_track_ids)

    def draw_track_trail(self, track_line, color=(0, 255, 0), track_thickness=1):
        if len(track_line) < 2:
            return

        pts = [(int(p[0]), int(p[1])) for p in track_line]
        for i in range(1, len(pts)):
            cv2.line(self.im0, pts[i - 1], pts[i], color, track_thickness, cv2.LINE_AA)

    def _normalize_dynamic_regions(self, dynamic_regions):
        polygons = []
        if dynamic_regions is None:
            return polygons

        for region in dynamic_regions:
            if region is None:
                continue
            pts = region.tolist() if hasattr(region, "tolist") else region
            if len(pts) < 3:
                continue
            poly = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
            polygons.append(poly)

        return polygons

    def _get_lane_index(self, point_xy, polygons):
        px = float(point_xy[0])
        py = float(point_xy[1])
        for idx, poly in enumerate(polygons):
            if cv2.pointPolygonTest(poly, (px, py), False) >= 0:
                return idx
        return -1

    def _resolve_lane_index(self, bottom_center, polygons, lane_detector=None):
        if lane_detector is not None and getattr(lane_detector, "locked", False):
            cal_lane = lane_detector.get_vehicle_lane(bottom_center)
            return cal_lane - 1 if cal_lane > 0 else -1
        return self._get_lane_index(bottom_center, polygons)

    def _stabilize_lane_index(self, track_id: int, raw_lane_idx: int) -> int:
        if raw_lane_idx < 0:
            return self.stable_lane_idx.get(track_id, -1)

        history = self.lane_history[track_id]
        history.append(raw_lane_idx)

        majority_lane = Counter(history).most_common(1)[0][0]
        prev_lane = self.stable_lane_idx.get(track_id, majority_lane)

        if majority_lane != prev_lane:
            self.lane_switch_streak[track_id] += 1
            if self.lane_switch_streak[track_id] >= self.lane_change_min_frames:
                self.stable_lane_idx[track_id] = majority_lane
                self.lane_switch_streak[track_id] = 0
        else:
            self.stable_lane_idx[track_id] = prev_lane
            self.lane_switch_streak[track_id] = 0

        return self.stable_lane_idx[track_id]

    def _is_allowed_in_lane(self, cls_name, lane_idx):
        if lane_idx < 0 or lane_idx >= len(self.lane_rules):
            return True

        allowed = self.lane_rules[lane_idx].get("allowed_classes", [])
        if not allowed:
            return True
        return cls_name in allowed

    def _draw_lane_name(self, lane_idx, point_xy):
        cv2.putText(
            self.im0,
            f"L{lane_idx + 1}",
            (int(point_xy[0]) + 5, int(point_xy[1]) - 5),
            FONT,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_normal_track(self, box, track_id, color):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(self.im0, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            self.im0,
            f"#{track_id}",
            (x1, max(14, y1 - 4)),
            FONT,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )

    def _draw_alert_state(self, box, bottom_center, label, color):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(self.im0, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, FONT, 0.46, 1)
        tag_y1 = max(0, y1 - th - 10)
        cv2.rectangle(self.im0, (x1, tag_y1), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            self.im0,
            label,
            (x1 + 4, y1 - 4),
            FONT,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.circle(
            self.im0,
            (int(bottom_center[0]), int(bottom_center[1])),
            6,
            color,
            -1,
        )

    def _draw_status_banner(self):
        flow_map = {"far_to_near": "F->N", "near_to_far": "N->F", "unknown": "?"}
        text = (
            f"WL:{len(self.current_wrong_ids)} | WW:{len(self.current_wrong_way_ids)} | "
            f"TotalWL:{self.total_violation_count} | TotalWW:{self.total_wrong_way_count} | "
            f"Flow:{flow_map.get(self.flow_direction, '?')}"
        )

        (tw, th), _ = cv2.getTextSize(text, FONT, 0.52, 1)
        pad_x, pad_y = 10, 8
        x1 = self.im0.shape[1] - (tw + pad_x * 2) - 18
        y1 = 18
        x2 = x1 + tw + pad_x * 2
        y2 = y1 + th + pad_y * 2

        cv2.rectangle(self.im0, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.putText(
            self.im0,
            text,
            (x1 + pad_x, y2 - pad_y),
            FONT,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def extract_and_process_tracks(self, tracks, dynamic_regions, lane_detector=None):
        self.annotator = SolutionAnnotator(
            self.im0,
            line_width=self.tf,
            example=str(self.names),
        )
        self.current_wrong_ids = set()
        self.current_wrong_way_ids = set()

        rows = self._extract_track_rows(tracks)
        if not rows:
            return

        polygons = self._normalize_dynamic_regions(dynamic_regions)

        for row in rows:
            box = row["box"]
            track_id = row["track_id"]
            cls_name = self.names[row["cls"]]
            bottom_center = row["bottom_center"]
            base_color = colors(track_id, True)

            if self.draw_tracks:
                self.draw_track_trail(
                    self.track_history[track_id],
                    color=self.track_color if self.track_color else base_color,
                    track_thickness=self.track_thickness,
                )

            raw_lane_idx = self._resolve_lane_index(bottom_center, polygons, lane_detector=lane_detector)
            lane_idx = self._stabilize_lane_index(track_id, raw_lane_idx)

            if lane_idx >= 0:
                self._draw_lane_name(lane_idx, bottom_center)

            track_direction = self._estimate_track_direction(track_id)
            is_wrong_lane = lane_idx >= 0 and not self._is_allowed_in_lane(cls_name, lane_idx)
            is_wrong_way = (
                self.flow_direction != "unknown"
                and track_direction != "unknown"
                and track_direction != self.flow_direction
            )

            self.violation_streak[track_id] = self.violation_streak.get(track_id, 0) + 1 if is_wrong_lane else 0
            self.wrong_way_streak[track_id] = self.wrong_way_streak.get(track_id, 0) + 1 if is_wrong_way else 0

            wrong_way_confirmed = self.wrong_way_streak[track_id] >= self.wrong_way_min_frames
            wrong_lane_confirmed = self.violation_streak[track_id] >= self.violation_min_frames

            if wrong_way_confirmed:
                self.current_wrong_way_ids.add(track_id)
                if track_id not in self.seen_wrong_way_ids:
                    self.seen_wrong_way_ids.add(track_id)
                    self.total_wrong_way_count += 1
                self._draw_alert_state(box, bottom_center, f"WW #{track_id}", (255, 0, 255))
            elif wrong_lane_confirmed:
                self.current_wrong_ids.add(track_id)
                if track_id not in self.seen_violation_ids:
                    self.seen_violation_ids.add(track_id)
                    self.total_violation_count += 1
                self._draw_alert_state(box, bottom_center, f"WL #{track_id}", (0, 0, 255))
            else:
                self._draw_normal_track(box=box, track_id=track_id, color=base_color)

        self._draw_status_banner()

    def display_frames(self, frame_start_time):
        current_fps = 0.0
        if self.env_check:
            frame_processing_time = time() - frame_start_time
            current_fps = 1.0 / frame_processing_time if frame_processing_time > 0 else 0.0

            cv2.putText(
                self.im0,
                f"FPS: {current_fps:.2f}",
                (20, self.im0.shape[0] - 18),
                FONT,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.namedWindow(self.window_name)
            cv2.imshow(self.window_name, self.im0)

        return current_fps

    def start_counting(self, im0, tracks, frame_start_time, dynamic_regions=None, lane_detector=None):
        self.im0 = im0
        self.extract_and_process_tracks(tracks, dynamic_regions, lane_detector=lane_detector)

        current_fps = 0.0
        if self.view_img:
            current_fps = self.display_frames(frame_start_time)

        return self.im0, current_fps
