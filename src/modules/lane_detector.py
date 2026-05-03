from __future__ import annotations
import copy
from collections import Counter, deque
from typing import Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np
import scipy.signal


class LaneDetector:
    """
    Lane detector for fixed CCTV cameras - v3.16
    Changelog v3.16:
      - FIX 1: Bỏ hard-cap [:3] trên chosen_boundaries khi LOCKED — không còn xóa boundary của L1
      - FIX 2: outer_boundary dùng poly_degree nhất quán (3 coeffs) thay vì bậc 1 (2 coeffs)
      - FIX 3: edge_penalty trong _select_best_subset nới lỏng từ 0.12/0.88 → 0.06/0.94
               để boundary gần cạnh BEV (L1) không bị loại khi calibration
      - FIX 4: _vehicle_partition_boundaries bỏ hard-cap [:3] — nhất quán với detect()
      - FIX 5: get_vehicle_lane() sửa dead-code if/else trả về cùng idx+1
      - FIX 6: histogram EMA giảm alpha 0.82→0.72, tăng trọng current 0.18→0.28
               để peak L1 bị che khuất hồi phục nhanh hơn
    """

    def __init__(
        self,
        roi_points: List[List[float]],
        bev_width: int = 900,
        bev_height: int = 650,
        min_line_distance_ratio: float = 0.072,
        edge_exclusion_ratio: float = 0.012,
        calibration_frames: int = 120,
        calibration_stride: int = 2,
        min_calibration_frames: int = 18,
        lock_history: int = 8,
        max_missed_frames: int = 18,
        persistence_alpha: float = 0.82,
        infer_outer_edges: bool = True,
        locked_update_interval: int = 5,
        debug: bool = False,
        poly_degree: int = 2,
    ):
        if roi_points is None or len(roi_points) != 4:
            raise ValueError("roi_points phải chứa đúng 4 điểm ROI.")

        self.bev_w = int(bev_width)
        self.bev_h = int(bev_height)
        self.debug = bool(debug)
        self.edge_exclusion_ratio = float(edge_exclusion_ratio)
        self.min_line_gap_px = max(35, int(self.bev_w * float(min_line_distance_ratio)))

        self.calibration_frames = max(16, int(calibration_frames))
        self.calibration_stride = max(1, int(calibration_stride))
        self.min_calibration_frames = max(6, int(min_calibration_frames))
        self.lock_history = max(3, int(lock_history))
        self.max_missed_frames = max(3, int(max_missed_frames))
        self.persistence_alpha = float(np.clip(persistence_alpha, 0.82, 0.97))
        self.infer_outer_edges = bool(infer_outer_edges)
        self.locked_update_interval = max(1, int(locked_update_interval))
        self.poly_degree = int(poly_degree)

        self.src_quad = self._sort_roi_points(np.asarray(roi_points, dtype=np.float32))
        self.dst_quad = np.array(
            [[0, 0], [self.bev_w - 1, 0], [self.bev_w - 1, self.bev_h - 1], [0, self.bev_h - 1]],
            dtype=np.float32,
        )
        self.M = cv2.getPerspectiveTransform(self.src_quad, self.dst_quad)
        self.Minv = cv2.getPerspectiveTransform(self.dst_quad, self.src_quad)

        # FIX 2: khởi tạo outer boundary với poly_degree+1 coeffs (zero-padded)
        self.left_outer_boundary = np.zeros(self.poly_degree + 1, dtype=np.float64)
        self.right_outer_boundary = np.zeros(self.poly_degree + 1, dtype=np.float64)

        self.frame_index = 0
        self.boundary_tracks: List[Dict[str, object]] = []
        self.active_boundaries_bev: List[Dict[str, object]] = []
        self.visual_boundaries_bev: List[Dict[str, object]] = []
        self.locked_boundaries_bev: List[Dict[str, object]] = []
        self.locked = False
        self.lock_hits = 0
        self.last_signature: Optional[np.ndarray] = None
        self.last_result: Dict[str, object] = {}
        self.dominant_boundary_count: Optional[int] = None
        self.sampled_bev_frames: Deque[np.ndarray] = deque(maxlen=self.calibration_frames)
        self.calibration_background_bev: Optional[np.ndarray] = None
        self.persistence_map: Optional[np.ndarray] = None
        self.histogram_ema: Optional[np.ndarray] = None
        self.boundary_count_history: Deque[int] = deque(maxlen=max(12, self.lock_history * 4))

    def reset(self) -> None:
        self.frame_index = 0
        self.boundary_tracks = []
        self.active_boundaries_bev = []
        self.visual_boundaries_bev = []
        self.locked_boundaries_bev = []
        self.locked = False
        self.lock_hits = 0
        self.last_signature = None
        self.last_result = {}
        self.dominant_boundary_count = None
        self.sampled_bev_frames.clear()
        self.calibration_background_bev = None
        self.persistence_map = None
        self.histogram_ema = None
        self.boundary_count_history.clear()

    def reset_calibration(self) -> None:
        self.reset()

    @staticmethod
    def _sort_roi_points(points: np.ndarray) -> np.ndarray:
        y_sorted = points[np.argsort(points[:, 1])]
        top = y_sorted[:2]
        bottom = y_sorted[2:]
        tl, tr = top[np.argsort(top[:, 0])]
        bl, br = bottom[np.argsort(bottom[:, 0])]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _build_roi_mask(self, frame_shape: Tuple[int, int, int]) -> np.ndarray:
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.src_quad.astype(np.int32)], 255)
        return mask

    def _extract_track_boxes(self, tracks) -> List[Tuple[int, int, int, int]]:
        if tracks is None or len(tracks) == 0:
            return []
        track0 = tracks[0]
        boxes_obj = getattr(track0, "boxes", None)
        if boxes_obj is None or boxes_obj.xyxy is None:
            return []
        boxes = []
        xyxy = boxes_obj.xyxy.cpu().numpy()
        for box in xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2))
        return boxes

    def _build_vehicle_mask_bev(self, frame: np.ndarray, tracks) -> Optional[np.ndarray]:
        boxes = self._extract_track_boxes(tracks)
        if not boxes:
            return None
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            w = x2 - x1
            h = y2 - y1
            pad_x = max(10, int(w * 0.16))
            pad_y = max(10, int(h * 0.16))
            xx1 = max(0, x1 - pad_x)
            yy1 = max(0, y1 - pad_y)
            xx2 = min(frame.shape[1] - 1, x2 + pad_x)
            yy2 = min(frame.shape[0] - 1, y2 + pad_y)
            cv2.rectangle(mask, (xx1, yy1), (xx2, yy2), 255, -1)
        roi_mask = self._build_roi_mask(frame.shape)
        mask = cv2.bitwise_and(mask, roi_mask)
        mask_bev = cv2.warpPerspective(mask, self.M, (self.bev_w, self.bev_h))
        mask_bev = cv2.dilate(mask_bev, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=2)
        return mask_bev

    def _warp_to_bev(self, frame: np.ndarray, tracks=None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        roi_mask = self._build_roi_mask(frame.shape)
        roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
        bev = cv2.warpPerspective(roi_frame, self.M, (self.bev_w, self.bev_h))
        vehicle_mask_bev = self._build_vehicle_mask_bev(frame, tracks)
        return roi_frame, bev, vehicle_mask_bev

    def _update_calibration_background(self, bev: np.ndarray) -> np.ndarray:
        if (self.frame_index - 1) % self.calibration_stride == 0:
            self.sampled_bev_frames.append(bev.copy())
            stack = np.stack(list(self.sampled_bev_frames), axis=0)
            self.calibration_background_bev = np.median(stack, axis=0).astype(np.uint8)
        if self.calibration_background_bev is None:
            self.calibration_background_bev = bev.copy()
        return self.calibration_background_bev

    def _remove_edge_margins(self, binary: np.ndarray) -> np.ndarray:
        out = binary.copy()
        margin = int(self.bev_w * self.edge_exclusion_ratio)
        if margin > 0:
            out[:, :margin] = 0
            out[:, self.bev_w - margin:] = 0
        out[: int(self.bev_h * 0.02), :] = 0
        return out

    def _largest_component(self, binary: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return binary
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.zeros_like(binary)
        out[labels == largest_label] = 255
        return out

    def _filter_connected_components(self, binary: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        filtered = np.zeros_like(binary)
        min_area = max(15, int(self.bev_w * self.bev_h * 0.000022))
        min_height = max(14, int(self.bev_h * 0.028))
        max_width = max(18, int(self.bev_w * 0.042))
        max_area = int(self.bev_w * self.bev_h * 0.028)
        for label_idx in range(1, num_labels):
            x, y, w, h, area = stats[label_idx]
            if area < min_area or area > max_area:
                continue
            if x < int(self.bev_w * self.edge_exclusion_ratio * 1.0):
                continue
            if (x + w) > int(self.bev_w * (1.0 - self.edge_exclusion_ratio * 1.0)):
                continue
            aspect = h / float(max(1, w))
            slender = h >= min_height and aspect >= 1.75
            tall_line_like = h >= int(self.bev_h * 0.14) and w <= max_width
            if not (slender or tall_line_like):
                continue
            if w > max_width:
                continue
            filtered[labels == label_idx] = 255
        return filtered

    def _thin_vertical_features(self, binary: np.ndarray) -> np.ndarray:
        thinned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13)))
        thinned = cv2.morphologyEx(thinned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 45)))
        thinned = self._filter_connected_components(thinned)
        return thinned

    def _bridge_dashed_markings(self, binary: np.ndarray) -> np.ndarray:
        bridged = self._thin_vertical_features(binary)
        bridged = cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 35)))
        bridged = cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 68)))
        bridged = self._thin_vertical_features(bridged)
        return bridged

    def _update_temporal_binary(self, candidate_binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        current_float = (candidate_binary > 0).astype(np.float32)
        if self.persistence_map is None:
            self.persistence_map = current_float.copy()
        else:
            self.persistence_map = self.persistence_alpha * self.persistence_map + (1.0 - self.persistence_alpha) * current_float
        stable_thr = 0.26 if len(self.sampled_bev_frames) < self.min_calibration_frames else 0.38
        stable_binary = np.zeros_like(candidate_binary)
        stable_binary[self.persistence_map >= stable_thr] = 255
        stable_binary = self._filter_connected_components(stable_binary)
        fused = cv2.bitwise_or(candidate_binary, stable_binary)
        bridged = self._bridge_dashed_markings(fused)
        detection_binary = self._thin_vertical_features(bridged)
        return stable_binary, bridged, detection_binary

    def _preprocess(self, bev_bgr: np.ndarray, exclusion_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2LAB)
        lightness = hls[:, :, 1]
        saturation = hsv[:, :, 1]
        lab_a = lab[:, :, 1].astype(np.int16)
        lab_b = lab[:, :, 2].astype(np.int16)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        light_eq = clahe.apply(lightness)

        yellow_mask = cv2.inRange(hsv, (10, 30, 30), (50, 255, 255))
        light_mask = (light_eq > 45).astype(np.uint8) * 255
        yellow_mask = cv2.bitwise_and(yellow_mask, light_mask)

        lower_slice = slice(int(self.bev_h * 0.06), self.bev_h)
        light_roi = light_eq[lower_slice, :]
        sat_roi = saturation[lower_slice, :]

        bright_thr = int(np.clip(np.percentile(light_roi, 78), 110, 225))
        sat_thr = int(np.clip(np.percentile(sat_roi, 55) + 12, 28, 125))

        road_mask = np.zeros_like(gray_eq)
        road_mask[(np.abs(lab_a - 128) <= 22) & (np.abs(lab_b - 128) <= 26) & (light_eq >= 36)] = 255
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
        road_mask = self._largest_component(road_mask)

        white_mask = np.zeros_like(gray_eq)
        white_mask[(light_eq >= bright_thr) & (saturation <= sat_thr)] = 255
        white_mask = cv2.bitwise_and(white_mask, road_mask)

        adaptive = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -6)
        adaptive[(light_eq < int(np.percentile(light_roi, 58)))] = 0
        adaptive = cv2.bitwise_and(adaptive, road_mask)

        tophat = cv2.morphologyEx(gray_eq, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        tophat_thr = int(np.clip(np.percentile(tophat[lower_slice, :], 88), 10, 65))
        tophat_mask = cv2.threshold(tophat, tophat_thr, 255, cv2.THRESH_BINARY)[1]

        sobelx = cv2.Sobel(light_eq, cv2.CV_32F, 1, 0, ksize=3)
        abs_sobelx = np.abs(sobelx)
        grad_scaled = np.uint8(np.clip(abs_sobelx / (abs_sobelx.max() + 1e-6) * 255, 0, 255))
        grad_thr = int(np.clip(np.percentile(grad_scaled[lower_slice, :], 82), 14, 95))
        grad_mask = cv2.threshold(grad_scaled, grad_thr, 255, cv2.THRESH_BINARY)[1]

        edge_debug = cv2.Canny(gray_eq, 55, 150)

        candidate = cv2.bitwise_or(white_mask, cv2.bitwise_and(adaptive, grad_mask))
        candidate = cv2.bitwise_or(candidate, cv2.bitwise_and(tophat_mask, grad_mask))
        candidate = cv2.bitwise_and(candidate, road_mask)
        candidate = cv2.bitwise_or(candidate, yellow_mask)
        candidate = self._remove_edge_margins(candidate)

        if exclusion_mask is not None:
            exclusion_mask = cv2.erode(exclusion_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=1)
            candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(exclusion_mask))
            edge_debug = cv2.bitwise_and(edge_debug, cv2.bitwise_not(exclusion_mask))

        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 22)))
        candidate = self._filter_connected_components(candidate)

        stable_binary, bridged_binary, detection_binary = self._update_temporal_binary(candidate)
        return candidate, edge_debug, stable_binary, bridged_binary, detection_binary

    def _dominant_count(self) -> Optional[int]:
        valid = [count for count in self.boundary_count_history if count >= 2]
        if len(valid) < max(4, self.lock_history // 2):
            return None
        freq = Counter(valid)
        dominant_count, dominant_hits = max(freq.items(), key=lambda item: (item[1], item[0]))
        if dominant_hits < max(3, self.lock_history // 2):
            return None
        return int(dominant_count)

    def _select_best_subset(self, candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if len(candidates) <= 2:
            return candidates
        xs = [float(np.polyval(item["coeffs"], self.bev_h - 1)) for item in candidates]
        target_count = self._dominant_count()
        candidate_lengths: List[int]
        if target_count is None:
            candidate_lengths = list(range(2, len(candidates) + 1))
        else:
            candidate_lengths = sorted({max(2, target_count - 1), target_count, min(len(candidates), target_count + 1)})
        best_score = -1e9
        best_subset = candidates
        for length in candidate_lengths:
            if length > len(candidates):
                continue
            for start in range(0, len(candidates) - length + 1):
                subset = candidates[start : start + length]
                subset_xs = xs[start : start + length]
                gaps = np.diff(subset_xs)
                if np.any(gaps < self.min_line_gap_px * 0.58):
                    continue
                mean_gap = float(np.mean(gaps)) if len(gaps) > 0 else float(self.min_line_gap_px)
                gap_uniformity = 1.0
                if len(gaps) > 1:
                    gap_uniformity = max(0.0, 1.0 - float(np.std(gaps) / (mean_gap + 1e-6)))
                score_sum = float(np.sum([float(item["score"]) for item in subset]))
                coverage = float(np.mean([float(item["coverage_ratio"]) for item in subset]))
                continuity = float(np.mean([float(item["continuity_ratio"]) for item in subset]))

                # FIX 3: nới lỏng edge_penalty từ 0.12/0.88 → 0.06/0.94
                # Boundary L1 (lane ngoài cùng) thường nằm gần cạnh BEV,
                # ngưỡng cũ 0.12/0.88 quá chặt dẫn đến loại nhầm L1.
                edge_penalty = 0.0
                if subset_xs[0] < self.bev_w * 0.06:
                    edge_penalty += 0.35
                if subset_xs[-1] > self.bev_w * 0.94:
                    edge_penalty += 0.35

                target_bonus = 0.0
                if target_count is not None:
                    if length == target_count:
                        target_bonus += 0.35
                    else:
                        target_bonus -= 0.18 * abs(length - target_count)
                subset_score = (score_sum + 0.45 * gap_uniformity + 0.20 * coverage + 0.15 * continuity + 0.10 * length + target_bonus - edge_penalty)
                if subset_score > best_score:
                    best_score = subset_score
                    best_subset = subset
        return best_subset

    def _find_boundary_candidates(self, binary: np.ndarray) -> Tuple[List[Dict[str, object]], np.ndarray]:
        search_start = int(self.bev_h * 0.12)
        histogram_current = np.sum(binary[search_start:, :] > 0, axis=0).astype(np.float32)
        if self.histogram_ema is None:
            self.histogram_ema = histogram_current.copy()
        else:
            # FIX 6: giảm EMA alpha 0.82→0.72, tăng trọng current 0.18→0.28
            # Để peak của L1 khi bị xe che khuất nhiều frame hồi phục nhanh hơn
            self.histogram_ema = 0.72 * self.histogram_ema + 0.28 * histogram_current
        histogram = 0.35 * histogram_current + 0.65 * self.histogram_ema
        if histogram.max() <= 0:
            return [], histogram
        kernel = np.ones(31, dtype=np.float32) / 31.0
        histogram_smooth = np.convolve(histogram, kernel, mode="same")
        inner_margin = max(16, int(self.bev_w * max(self.edge_exclusion_ratio, 0.04)))
        histogram_smooth[:inner_margin] = 0
        histogram_smooth[self.bev_w - inner_margin:] = 0
        peak_height = max(float(np.mean(histogram_smooth) + 0.38 * np.std(histogram_smooth)), float(0.15 * np.max(histogram_smooth)), 4.0)
        prominence = max(1.8, float(0.05 * np.max(histogram_smooth)))
        peaks, _ = scipy.signal.find_peaks(histogram_smooth, distance=self.min_line_gap_px, height=peak_height, prominence=prominence)
        nonzero_y, nonzero_x = binary.nonzero()
        nonzero_y = np.asarray(nonzero_y)
        nonzero_x = np.asarray(nonzero_x)
        nwindows = 20
        window_height = max(12, self.bev_h // nwindows)
        margin = max(14, int(self.bev_w * 0.026))
        minpix = 4
        max_window_gap = 6
        candidates: List[Dict[str, object]] = []
        for peak_x in peaks:
            x_current = float(peak_x)
            lane_indices = []
            window_centers = []
            windows_hit = 0
            gap_run = 0
            y_min = self.bev_h - 1
            y_max = 0
            for window_idx in range(nwindows):
                y_low = self.bev_h - (window_idx + 1) * window_height
                y_high = self.bev_h - window_idx * window_height
                y_center = 0.5 * (y_low + y_high)
                x_low = int(max(0, x_current - margin))
                x_high = int(min(self.bev_w, x_current + margin))
                good = ((nonzero_y >= y_low) & (nonzero_y < y_high) & (nonzero_x >= x_low) & (nonzero_x < x_high)).nonzero()[0]
                if len(good) > 0:
                    lane_indices.append(good)
                    y_min = min(y_min, int(nonzero_y[good].min()))
                    y_max = max(y_max, int(nonzero_y[good].max()))
                if len(good) >= minpix:
                    x_current = float(np.median(nonzero_x[good]))
                    windows_hit += 1
                    gap_run = 0
                    window_centers.append((float(x_current), float(y_center), True))
                else:
                    gap_run += 1
                    if gap_run <= max_window_gap:
                        window_centers.append((float(x_current), float(y_center), False))
                    else:
                        break
            if not lane_indices or len(window_centers) < 5:
                continue
            lane_indices = np.concatenate(lane_indices)
            point_count = int(len(lane_indices))
            ys = np.array([item[1] for item in window_centers], dtype=np.float32)
            xs = np.array([item[0] for item in window_centers], dtype=np.float32)
            if point_count < 18:
                continue
            coeffs = np.polyfit(ys, xs, self.poly_degree)
            slope = float(coeffs[1] if self.poly_degree == 2 else coeffs[0])
            if abs(slope) > 0.55:
                continue
            coverage_ratio = float((y_max - y_min + 1) / max(1, self.bev_h))
            continuity_ratio = float(len(window_centers) / float(nwindows))
            if max(coverage_ratio, continuity_ratio) < 0.12:
                continue
            bottom_x = float(np.polyval(coeffs, self.bev_h - 1))
            if bottom_x < inner_margin or bottom_x > (self.bev_w - inner_margin):
                continue
            score = (0.33 * min(1.0, point_count / 240.0) + 0.26 * min(1.0, windows_hit / float(nwindows)) + 0.21 * min(1.0, continuity_ratio / 0.70) + 0.15 * min(1.0, coverage_ratio / 0.50) + 0.05 * max(0.0, 1.0 - abs(slope) / 0.50))
            candidates.append({"coeffs": coeffs, "peak_x": float(peak_x), "point_count": point_count, "coverage_ratio": coverage_ratio, "continuity_ratio": continuity_ratio, "windows_hit": int(windows_hit), "score": float(score), "y_min": 0, "y_max": self.bev_h - 1})
        candidates.sort(key=lambda item: np.polyval(item["coeffs"], self.bev_h - 1))
        merged: List[Dict[str, object]] = []
        for candidate in candidates:
            if not merged:
                merged.append(candidate)
                continue
            prev = merged[-1]
            prev_x = float(np.polyval(prev["coeffs"], self.bev_h - 1))
            curr_x = float(np.polyval(candidate["coeffs"], self.bev_h - 1))
            if abs(curr_x - prev_x) < self.min_line_gap_px * 0.62:
                if float(candidate["score"]) > float(prev["score"]):
                    merged[-1] = candidate
            else:
                merged.append(candidate)
        selected = self._select_best_subset(merged)
        return selected, histogram_smooth

    def _update_boundary_tracks(self, current_boundaries: List[Dict[str, object]]) -> List[Dict[str, object]]:
        y_eval = self.bev_h - 1
        matched_indices = set()
        for track in self.boundary_tracks:
            track_x = float(np.polyval(track["coeffs"], y_eval))
            best_idx = -1
            best_diff = float("inf")
            for idx, boundary in enumerate(current_boundaries):
                if idx in matched_indices:
                    continue
                boundary_x = float(np.polyval(boundary["coeffs"], y_eval))
                diff = abs(boundary_x - track_x)
                if diff < self.min_line_gap_px * 0.72 and diff < best_diff:
                    best_diff = diff
                    best_idx = idx
            if best_idx >= 0:
                boundary = current_boundaries[best_idx]
                alpha = 0.24 if self.locked else 0.34
                track["coeffs"] = alpha * boundary["coeffs"] + (1.0 - alpha) * track["coeffs"]
                track["score"] = 0.65 * float(track.get("score", 0.0)) + 0.35 * float(boundary["score"])
                track["coverage_ratio"] = float(boundary["coverage_ratio"])
                track["continuity_ratio"] = float(boundary.get("continuity_ratio", 0.0))
                track["y_min"] = 0
                track["y_max"] = self.bev_h - 1
                track["age"] = int(track.get("age", 0)) + 1
                track["misses"] = 0
                matched_indices.add(best_idx)
            else:
                track["misses"] = int(track.get("misses", 0)) + 1
                track["score"] = max(0.0, float(track.get("score", 0.0)) * 0.92)
        for idx, boundary in enumerate(current_boundaries):
            if idx in matched_indices:
                continue
            self.boundary_tracks.append({"coeffs": boundary["coeffs"].copy(), "score": float(boundary["score"]), "coverage_ratio": float(boundary["coverage_ratio"]), "continuity_ratio": float(boundary.get("continuity_ratio", 0.0)), "y_min": 0, "y_max": self.bev_h - 1, "age": 1, "misses": 0})
        self.boundary_tracks = [track for track in self.boundary_tracks if int(track.get("misses", 0)) <= self.max_missed_frames]
        self.boundary_tracks.sort(key=lambda item: np.polyval(item["coeffs"], y_eval))
        active: List[Dict[str, object]] = []
        for track in self.boundary_tracks:
            if float(track.get("score", 0.0)) < 0.18:
                continue
            if int(track.get("age", 0)) >= 2 or int(track.get("misses", 0)) == 0:
                active.append(track)
        active = self._select_best_subset(active) if len(active) > 2 else active
        self.active_boundaries_bev = active
        return active

    def _estimate_virtual_outer_boundaries(self, active_boundaries: List[Dict[str, object]]) -> List[Dict[str, object]]:
        return []

    def _update_lock_state(self, active_boundaries: List[Dict[str, object]]) -> None:
        if self.locked:
            return
        if len(self.sampled_bev_frames) < self.min_calibration_frames:
            self.lock_hits = 0
            self.last_signature = None
            return
        if len(active_boundaries) < 2:
            self.lock_hits = max(0, self.lock_hits - 1)
            self.last_signature = None
            return
        self.boundary_count_history.append(len(active_boundaries))
        dominant_count = self._dominant_count()
        self.dominant_boundary_count = dominant_count
        y_eval = self.bev_h - 1
        signature = np.array([float(np.polyval(item["coeffs"], y_eval)) for item in active_boundaries], dtype=np.float32)
        mean_score = float(np.mean([float(item.get("score", 0.0)) for item in active_boundaries]))
        score_ok = mean_score >= 0.26
        track_ok = all(int(item.get("age", 0)) >= 2 for item in active_boundaries)
        count_ok = dominant_count is None or len(active_boundaries) == dominant_count
        if count_ok and score_ok and track_ok:
            if self.last_signature is not None and len(signature) == len(self.last_signature):
                diff = np.max(np.abs(signature - self.last_signature)) if len(signature) > 0 else 0.0
                if diff <= 10.0:
                    self.lock_hits += 1
                else:
                    self.lock_hits = 1
            else:
                self.lock_hits = 1
            self.last_signature = signature
        else:
            self.lock_hits = max(0, self.lock_hits - 1)
            self.last_signature = signature if count_ok else None
        if self.lock_hits >= self.lock_history:
            self.locked_boundaries_bev = [copy.deepcopy(item) for item in active_boundaries]
            self.locked = True
            self.dominant_boundary_count = len(active_boundaries)

    def _build_output_boundaries(self, boundaries_bev: List[Dict[str, object]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        lines_orig: List[np.ndarray] = []
        lane_polygons_orig: List[np.ndarray] = []
        for boundary in boundaries_bev:
            ys = np.linspace(0, self.bev_h - 1, 320, dtype=np.float32)
            xs = np.polyval(boundary["coeffs"], ys)
            xs = np.clip(xs, 0, self.bev_w - 1)
            pts_bev = np.stack([xs, ys], axis=1).astype(np.float32)
            pts_orig = cv2.perspectiveTransform(pts_bev.reshape(-1, 1, 2), self.Minv).reshape(-1, 2)
            lines_orig.append(pts_orig)
        for idx in range(len(lines_orig) - 1):
            left_line = lines_orig[idx]
            right_line = lines_orig[idx + 1]
            if len(left_line) < 2 or len(right_line) < 2:
                continue
            poly = np.vstack([left_line, right_line[::-1]])
            lane_polygons_orig.append(poly)
        return lines_orig, lane_polygons_orig

    def _make_outer_boundary_dict(self, coeffs: np.ndarray) -> Dict[str, object]:
        """Helper tạo dict cho outer boundary với đầy đủ fields."""
        return {
            "coeffs": coeffs,
            "score": 1.0,
            "coverage_ratio": 1.0,
            "continuity_ratio": 1.0,
            "y_min": 0,
            "y_max": self.bev_h - 1,
            "age": 999,
            "misses": 0,
        }

    def _build_frozen_result(self, frame: np.ndarray, roi_frame: np.ndarray, bev: np.ndarray) -> Dict[str, object]:
        # FIX 1 (frozen path): bỏ hard-cap [:3] — giữ toàn bộ locked boundaries
        chosen_boundaries = list(self.locked_boundaries_bev)

        visual_boundaries = list(chosen_boundaries)

        if len(visual_boundaries) >= 2:
            visual_boundaries = (
                [self._make_outer_boundary_dict(self.left_outer_boundary)]
                + visual_boundaries
                + [self._make_outer_boundary_dict(self.right_outer_boundary)]
            )

        visual_boundaries.sort(key=lambda item: np.polyval(item["coeffs"], self.bev_h - 1))
        boundaries_orig, lane_polygons_orig = self._build_output_boundaries(visual_boundaries)
        detected_boundaries_orig, _ = self._build_output_boundaries(chosen_boundaries)

        n_inner = len(chosen_boundaries)
        total_boundaries = n_inner + 2  # +2 outer
        lane_count = max(0, total_boundaries - 1)

        result = {
            "mode": "LOCKED",
            "frame_index": self.frame_index,
            "roi_frame": roi_frame,
            "bev": bev,
            "calibration_bev": self.calibration_background_bev if self.calibration_background_bev is not None else bev,
            "binary_bev": self.last_result.get("binary_bev"),
            "stable_binary_bev": self.last_result.get("stable_binary_bev"),
            "bridged_binary_bev": self.last_result.get("bridged_binary_bev"),
            "detection_binary_bev": self.last_result.get("detection_binary_bev"),
            "edge_bev": self.last_result.get("edge_bev"),
            "histogram": self.last_result.get("histogram"),
            "boundaries": boundaries_orig,
            "detected_boundaries": detected_boundaries_orig,
            "lane_polygons": lane_polygons_orig,
            "boundary_count": total_boundaries,
            "visual_boundary_count": len(visual_boundaries),
            "estimated_outer_edges": 2,
            "lane_count": lane_count,
            "confidence": float(max(self.last_result.get("confidence", 0.0), 0.90)),
            "locked": True,
            "lock_hits": int(self.lock_hits),
            "calibration_progress": 1.0,
            "sample_count": len(self.sampled_bev_frames),
            "dominant_boundary_count": self.dominant_boundary_count,
        }
        self.last_result = result
        return result

    def detect(self, frame: np.ndarray, tracks=None) -> Dict[str, object]:
        self.frame_index += 1
        roi_frame, bev, vehicle_mask_bev = self._warp_to_bev(frame, tracks=tracks)

        if self.frame_index == 1:
            # FIX 2: tính outer boundary với poly_degree nhất quán
            # Lấy nhiều điểm dọc theo cạnh BEV rồi polyfit bậc poly_degree
            # để coeffs luôn có poly_degree+1 phần tử, tránh mismatch với polyval
            bev_ys = np.linspace(0, self.bev_h - 1, 20, dtype=np.float32)

            left_bev_pts = np.stack([np.zeros_like(bev_ys), bev_ys], axis=1).astype(np.float32)
            left_orig_pts = cv2.perspectiveTransform(left_bev_pts.reshape(-1, 1, 2), self.Minv).reshape(-1, 2)
            self.left_outer_boundary = np.polyfit(bev_ys, left_orig_pts[:, 0], self.poly_degree)

            right_bev_pts = np.stack([np.full_like(bev_ys, self.bev_w - 1), bev_ys], axis=1).astype(np.float32)
            right_orig_pts = cv2.perspectiveTransform(right_bev_pts.reshape(-1, 1, 2), self.Minv).reshape(-1, 2)
            self.right_outer_boundary = np.polyfit(bev_ys, right_orig_pts[:, 0], self.poly_degree)

        if self.locked and self.last_result and self.locked_update_interval > 1:
            if (self.frame_index % self.locked_update_interval) != 0:
                return self._build_frozen_result(frame, roi_frame, bev)

        calibration_bev = self._update_calibration_background(bev)
        current_binary, edge_bev, stable_binary, bridged_binary, detection_binary = self._preprocess(calibration_bev, exclusion_mask=vehicle_mask_bev)
        current_boundaries, histogram = self._find_boundary_candidates(detection_binary)
        detected_boundaries = self._update_boundary_tracks(current_boundaries)
        self._update_lock_state(detected_boundaries)

        chosen_boundaries = self.locked_boundaries_bev if self.locked and self.locked_boundaries_bev else detected_boundaries

        # FIX 1 (detect path): bỏ hoàn toàn hard-cap [:3] trên chosen_boundaries.
        # Trước đây code cắt bỏ boundary thứ 4 trở đi TRƯỚC khi thêm outer edges,
        # khiến L1 (boundary ngoài cùng) bị xóa và lane L1 không có biên phân cách.
        # Bây giờ giữ toàn bộ chosen_boundaries, outer edges được thêm vào sau.

        visual_boundaries = list(chosen_boundaries)

        if self.locked and len(visual_boundaries) >= 2:
            visual_boundaries = (
                [self._make_outer_boundary_dict(self.left_outer_boundary)]
                + visual_boundaries
                + [self._make_outer_boundary_dict(self.right_outer_boundary)]
            )

        visual_boundaries.sort(key=lambda item: np.polyval(item["coeffs"], self.bev_h - 1))
        self.visual_boundaries_bev = visual_boundaries

        boundaries_orig, lane_polygons_orig = self._build_output_boundaries(visual_boundaries)
        detected_boundaries_orig, _ = self._build_output_boundaries(chosen_boundaries)

        detected_count = len(chosen_boundaries)
        n_inner = detected_count
        total_boundaries = (n_inner + 2) if self.locked else detected_count
        lane_count = max(0, total_boundaries - 1) if self.locked else len(lane_polygons_orig)

        track_scores = [float(item.get("score", 0.0)) for item in chosen_boundaries]
        confidence = float(np.mean(track_scores)) if track_scores else 0.0
        if self.locked:
            confidence = max(confidence, 0.90 if detected_count >= 2 else confidence)
        calibration_progress = min(1.0, len(self.sampled_bev_frames) / float(self.calibration_frames))

        if self.locked:
            mode = "LOCKED"
        elif len(self.sampled_bev_frames) < self.min_calibration_frames:
            mode = "CALIBRATING"
        elif detected_count >= 2:
            mode = "TRACKING"
        elif detected_count == 1:
            mode = "PARTIAL"
        else:
            mode = "SEARCHING"

        result = {
            "mode": mode,
            "frame_index": self.frame_index,
            "roi_frame": roi_frame,
            "bev": bev,
            "calibration_bev": calibration_bev,
            "binary_bev": current_binary,
            "stable_binary_bev": stable_binary,
            "bridged_binary_bev": bridged_binary,
            "detection_binary_bev": detection_binary,
            "edge_bev": edge_bev,
            "histogram": histogram,
            "boundaries": boundaries_orig,
            "detected_boundaries": detected_boundaries_orig,
            "lane_polygons": lane_polygons_orig,
            "boundary_count": total_boundaries,
            "visual_boundary_count": len(visual_boundaries),
            "estimated_outer_edges": 2 if self.locked else 0,
            "lane_count": lane_count,
            "confidence": float(confidence),
            "locked": self.locked,
            "lock_hits": int(self.lock_hits),
            "calibration_progress": float(calibration_progress),
            "sample_count": len(self.sampled_bev_frames),
            "dominant_boundary_count": self.dominant_boundary_count,
        }
        self.last_result = result
        return result

    def _vehicle_partition_boundaries(self) -> List[Dict[str, object]]:
        if self.locked_boundaries_bev:
            boundaries = list(self.locked_boundaries_bev)
        elif self.visual_boundaries_bev:
            boundaries = list(self.visual_boundaries_bev)
        else:
            boundaries = list(self.active_boundaries_bev)

        if self.locked and len(boundaries) >= 2:
            # FIX 4: bỏ hard-cap [:3] — nhất quán với detect() và _build_frozen_result()
            # Giữ toàn bộ locked boundaries trước khi thêm outer edges
            left_outer = self._make_outer_boundary_dict(self.left_outer_boundary)
            right_outer = self._make_outer_boundary_dict(self.right_outer_boundary)
            boundaries = [left_outer] + boundaries + [right_outer]
            boundaries.sort(key=lambda item: np.polyval(item["coeffs"], self.bev_h - 1))
        return boundaries

    def get_vehicle_lane(self, bottom_center: Tuple[float, float]) -> int:
        try:
            if cv2.pointPolygonTest(self.src_quad.astype(np.int32), (int(bottom_center[0]), int(bottom_center[1])), False) < 0:
                return -1
        except Exception:
            return -1

        boundaries = self._vehicle_partition_boundaries()
        if len(boundaries) < 2:
            return -1

        point = np.array([[[float(bottom_center[0]), float(bottom_center[1])]]], dtype=np.float32)
        try:
            mapped = cv2.perspectiveTransform(point, self.M)
        except Exception:
            return -1

        bev_x, bev_y = mapped[0][0]
        bev_y = float(np.clip(bev_y, 0, self.bev_h - 1))

        for idx in range(len(boundaries) - 1):
            left_x = float(np.polyval(boundaries[idx]["coeffs"], bev_y))
            right_x = float(np.polyval(boundaries[idx + 1]["coeffs"], bev_y))

            if idx == 0:
                buffer = 20
            elif idx == len(boundaries) - 2:
                buffer = 15
            else:
                buffer = 8

            # FIX 5: sửa dead-code — cả 2 nhánh if/else đều trả về idx+1 (vô nghĩa).
            # Bây giờ trả về lane_index đúng dựa trên vị trí bev_x trong khoảng [left_x, right_x].
            # Lane index = idx + 1 (1-based), không thay đổi logic, nhưng loại bỏ dead code.
            if left_x - buffer <= bev_x <= right_x + buffer:
                return idx + 1

        # Xe nằm ngoài tất cả các lane đã biết nhưng gần boundary đầu tiên → gán lane 1
        if bev_x < float(np.polyval(boundaries[0]["coeffs"], bev_y)) + 25:
            return 1

        return -1

    def draw_overlay(self, frame: np.ndarray, result: Dict[str, object]) -> np.ndarray:
        output = frame.copy()
        for polygon in result.get("lane_polygons", []):
            polygon_int = np.round(polygon).astype(np.int32)
            cv2.fillPoly(output, [polygon_int], (45, 110, 45))
        output = cv2.addWeighted(output, 0.24, frame, 0.76, 0)
        line_color = (0, 255, 180) if result.get("locked", False) else (0, 255, 255)
        for line in result.get("detected_boundaries", []):
            pts = np.round(line).astype(np.int32)
            cv2.polylines(output, [pts], False, line_color, 2, cv2.LINE_AA)
        cv2.polylines(output, [self.src_quad.astype(np.int32)], True, (255, 255, 0), 1, cv2.LINE_AA)
        hud_lines = [
            f"{result.get('mode', 'UNKNOWN')} | B={result.get('boundary_count', 0)} | L={result.get('lane_count', 0)} | C={result.get('confidence', 0.0):.2f}",
            f"locked={result.get('locked', False)} | samples={result.get('sample_count', 0)} | hits={result.get('lock_hits', 0)}/{self.lock_history}",
        ]
        if self.debug and result.get("dominant_boundary_count") is not None:
            hud_lines.append(f"target_B={result.get('dominant_boundary_count')}")
        y = 24
        for text in hud_lines:
            cv2.putText(output, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(output, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
            y += 18
        return output