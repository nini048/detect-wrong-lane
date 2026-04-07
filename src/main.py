import cv2

from modules.arguments import parse_arguments
from modules.lane_detector import LaneDetector
from modules.lane_semantics import LaneSemanticsInferer
from modules.processor import process_video_frames, setup_object_counter
from modules.utils import (
    get_log_file_path,
    get_unique_output_path,
    initialize_model,
    initialize_video_capture,
    initialize_video_writer,
    resolve_roi_points,
)


def main():
    args = parse_arguments()

    roi_points = resolve_roi_points(
        args.video,
        roi_points_text=args.roi_points,
        roi_source=args.roi_source,
    )

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
    lane_semantics = LaneSemanticsInferer()

    model = initialize_model(args.model)
    cap, fps = initialize_video_capture(args.video)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame from video.")
    frame_height, frame_width = frame.shape[:2]
    frame_size = (frame_width, frame_height)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_video_path = get_unique_output_path(args.video, args.model, args.size)
    log_path = get_log_file_path(args.video, args.model, args.size)
    video_writer = initialize_video_writer(output_video_path, fps, frame_size)

    object_count = setup_object_counter(
        model.names,
        lane_rules=[],
        violation_min_frames=args.violation_min_frames,
        wrong_way_min_frames=args.wrong_way_min_frames,
        direction_min_dy=args.direction_min_dy,
    )

    average_fps, min_fps, max_fps, avg_infer_time = process_video_frames(
        cap,
        model,
        video_writer,
        object_count,
        5,
        args.size,
        lane_detector=lane_detector,
        lane_semantics_inferer=lane_semantics,
        detection_conf=args.det_conf,
        max_frames=args.max_frames,
        show_debug_views=args.show_debug_views,
    )

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Average FPS: {average_fps:.2f}\n")
        log_file.write(f"Min FPS: {min_fps:.2f}\n")
        log_file.write(f"Max FPS: {max_fps:.2f}\n")
        log_file.write(f"Average Inference Time: {avg_infer_time:.2f} ms\n")
        log_file.write(f"Lane locked: {lane_detector.locked}\n")
        log_file.write(f"Flow direction: {object_count.flow_direction}\n")
        log_file.write(f"Reverse lane order: {object_count.reverse_lane_order}\n")

    print(f"Video output saved to: {output_video_path}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
