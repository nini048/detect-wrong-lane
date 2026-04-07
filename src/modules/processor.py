import cv2
from time import time

from . import object_counter

FONT = cv2.FONT_HERSHEY_SIMPLEX


def setup_object_counter(model_names, lane_rules, violation_min_frames, wrong_way_min_frames=4, direction_min_dy=12.0):
    counter = object_counter.ObjectCounter()
    counter.set_args(
        classes_names=dict(model_names),
        lane_rules=lane_rules,
        violation_min_frames=violation_min_frames,
        wrong_way_min_frames=wrong_way_min_frames,
        direction_min_dy=direction_min_dy,
        view_img=True,
        draw_tracks=False,
        line_thickness=2,
        track_thickness=1,
    )
    return counter


def _draw_semantics_hud(frame, semantics, flow_direction="unknown", reverse_order=False):
    if not semantics:
        return frame

    output = frame.copy()
    panel_w = 300
    line_h = 22
    panel_h = 14 + line_h * (len(semantics) + 2)

    x2 = output.shape[1] - 18
    x1 = x2 - panel_w
    y1 = 58
    y2 = y1 + panel_h

    overlay = output.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (25, 25, 25), -1)
    output = cv2.addWeighted(overlay, 0.42, output, 0.58, 0)

    flow_text = {
        "far_to_near": "far->near",
        "near_to_far": "near->far",
        "unknown": "unknown",
    }.get(flow_direction, "unknown")

    y = y1 + 20
    cv2.putText(output, "Lane rules", (x1 + 12, y), FONT, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
    y += line_h
    cv2.putText(
        output,
        f"Flow: {flow_text} | reverse={reverse_order}",
        (x1 + 12, y),
        FONT,
        0.46,
        (200, 255, 255),
        1,
        cv2.LINE_AA,
    )
    y += line_h

    for item in semantics[:6]:
        text = f"L{item['lane_index'] + 1}: {item['label']}"
        cv2.putText(output, text, (x1 + 12, y), FONT, 0.50, (255, 255, 0), 1, cv2.LINE_AA)
        y += line_h

    return output


def process_video_frames(
    cap,
    model,
    video_writer,
    object_count,
    fps_warmup_frames,
    img_size,
    lane_detector=None,
    lane_semantics_inferer=None,
    detection_conf=0.10,
    max_frames=0,
    show_debug_views=False,
):
    total_fps = 0.0
    frame_count = 0
    min_fps = float("inf")
    max_fps_value = 0.0
    inference_time_ms = 0.0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video processing completed.")
            break

        if max_frames > 0 and frame_count >= max_frames:
            print("Reached max_frames.")
            break

        raw_frame = frame.copy()
        frame_start_time = time()

        tracks = model.track(
            frame,
            persist=True,
            show=False,
            imgsz=img_size,
            conf=detection_conf,
            verbose=False,
        )

        object_count.prime_tracks(tracks)

        display_frame = frame.copy()
        lane_result = None
        lane_ready = False

        if lane_detector is not None:
            lane_result = lane_detector.detect(raw_frame, tracks=tracks)
            display_frame = lane_detector.draw_overlay(display_frame, lane_result)
            lane_ready = bool(lane_detector.locked and lane_result.get("lane_count", 0) > 0)

        if lane_ready and lane_semantics_inferer is not None:
            semantics = lane_semantics_inferer.infer(
                lane_result["lane_count"],
                reverse_order=object_count.reverse_lane_order,
            )
            object_count.lane_rules = semantics
            display_frame = _draw_semantics_hud(
                display_frame,
                semantics,
                flow_direction=object_count.flow_direction,
                reverse_order=object_count.reverse_lane_order,
            )
            dynamic_regions = lane_result["lane_polygons"]
            lane_detector_for_count = lane_detector
        else:
            object_count.lane_rules = []
            dynamic_regions = None
            lane_detector_for_count = None

        display_frame, current_fps = object_count.start_counting(
            display_frame,
            tracks,
            frame_start_time,
            dynamic_regions=dynamic_regions,
            lane_detector=lane_detector_for_count,
        )

        if show_debug_views and lane_result is not None:
            for win_name, view in (
                ("Lane BEV", lane_result.get("bev")),
                ("Lane Binary", lane_result.get("binary_bev")),
                ("Lane Stable Binary", lane_result.get("stable_binary_bev")),
                ("Lane Detection Binary", lane_result.get("detection_binary_bev")),
            ):
                if view is None:
                    continue
                if len(view.shape) == 2:
                    view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
                cv2.imshow(win_name, view)

        speed = tracks[0].speed if len(tracks) > 0 else {"inference": 0.0}
        inference_time_ms += float(speed.get("inference", 0.0))

        if frame_count >= fps_warmup_frames:
            total_fps += current_fps
            min_fps = min(min_fps, current_fps)
            max_fps_value = max(max_fps_value, current_fps)

        print(f"FPS: {current_fps:.2f}")
        video_writer.write(display_frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    average_fps = total_fps / (frame_count - fps_warmup_frames) if frame_count > fps_warmup_frames else 0.0
    average_inference_time = inference_time_ms / frame_count if frame_count > 0 else 0.0

    print(f"Average inference time: {average_inference_time:.2f} ms")
    print(f"Average FPS: {average_fps:.2f}")
    print(f"Min FPS: {min_fps:.2f}")
    print(f"Max FPS: {max_fps_value:.2f}")

    return average_fps, min_fps, max_fps_value, average_inference_time
