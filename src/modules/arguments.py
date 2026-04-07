import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Wrong-lane and wrong-way detection for fixed CCTV cameras.")

    parser.add_argument("--video", type=str, default="../video/Video/bentre.mp4", help="Path to input video")
    parser.add_argument("--model", type=str, default="../train/imgsz_224/best.pt", help="Path to YOLO model")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("WIDTH", "HEIGHT"),
        help="YOLO inference image size",
    )

    parser.add_argument("--roi-points", type=str, default="", help='Inline ROI points: "x1,y1;x2,y2;x3,y3;x4,y4"')
    parser.add_argument(
        "--roi-source",
        type=str,
        default="prompt",
        choices=["config", "prompt", "inline"],
        help="Where to obtain ROI",
    )

    parser.add_argument("--bev-width", type=int, default=900)
    parser.add_argument("--bev-height", type=int, default=650)
    parser.add_argument("--min-line-distance-ratio", type=float, default=0.10)
    parser.add_argument("--edge-exclusion-ratio", type=float, default=0.035)

    parser.add_argument("--calibration-frames", type=int, default=120)
    parser.add_argument("--calibration-stride", type=int, default=2)
    parser.add_argument("--lock-history", type=int, default=8)
    parser.add_argument("--max-missed-frames", type=int, default=18)
    parser.add_argument("--locked-update-interval", type=int, default=8)

    parser.add_argument("--det-conf", type=float, default=0.10)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--debug-lane", action="store_true")
    parser.add_argument("--show-debug-views", action="store_true")

    parser.add_argument("--violation-min-frames", type=int, default=5)
    parser.add_argument("--wrong-way-min-frames", type=int, default=4)
    parser.add_argument("--direction-min-dy", type=float, default=12.0)

    return parser.parse_args()
