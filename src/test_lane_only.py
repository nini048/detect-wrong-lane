import cv2

from modules.arguments import parse_arguments
from modules.lane_detector import LaneDetector
from modules.utils import resolve_roi_points, initialize_video_capture


def _to_bgr(view):
    if view is None:
        return None
    if len(view.shape) == 2:
        return cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    return view


def main():
    args = parse_arguments()
    roi_points = resolve_roi_points(args.video, args.roi_points, args.roi_source)
    lane_detector = LaneDetector(
        roi_points=roi_points,
        bev_width=args.bev_width,
        bev_height=args.bev_height,
        min_line_distance_ratio=args.min_line_distance_ratio,
        edge_exclusion_ratio=args.edge_exclusion_ratio,
        calibration_frames=args.calibration_frames,
        calibration_stride=args.calibration_stride,
        lock_history=args.lock_history,
        max_missed_frames=args.max_missed_frames,
        locked_update_interval=args.locked_update_interval,
        debug=args.debug_lane,
    )

    cap, _ = initialize_video_capture(args.video)
    frame_count = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        if args.max_frames > 0 and frame_count > args.max_frames:
            break

        result = lane_detector.detect(frame, tracks=None)
        overlay = lane_detector.draw_overlay(frame, result)
        cv2.imshow("Lane Overlay", overlay)

        if args.show_debug_views:
            for win_name, view in (
                ("BEV", result.get("bev")),
                ("Binary", result.get("binary_bev")),
                ("Stable Binary", result.get("stable_binary_bev")),
                ("Detection Binary", result.get("detection_binary_bev")),
            ):
                if view is not None:
                    cv2.imshow(win_name, _to_bgr(view))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
