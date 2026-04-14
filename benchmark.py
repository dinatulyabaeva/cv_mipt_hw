import argparse
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from app import VideoBackgroundRemoval, build_background, composite, parse_color


def get_cpu_name() -> str:
    return platform.processor() or platform.platform()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="замер latency/FPS по обработанным кадрам"
    )
    parser.add_argument("--video", required=True, help="Путь к входному видео")
    parser.add_argument(
        "--mode",
        choices=["color", "image", "blur"],
        default="blur",
    )
    parser.add_argument("--bg-color", default="0,255,0")
    parser.add_argument("--bg-image", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--model", choices=["general", "landscape"], default="landscape")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--smooth-alpha", type=float, default=0.65)
    parser.add_argument("--mask-blur", type=int, default=7)
    parser.add_argument("--blur-kernel", type=int, default=35)
    parser.add_argument("--infer-every", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()

    if not Path(args.video).exists():
        raise FileNotFoundError(f"видео не найдено: {args.video}")

    bg_color = parse_color(args.bg_color)
    bg_image = None
    if args.bg_image is not None:
        bg_image = cv2.imread(args.bg_image)
        if bg_image is None:
            raise FileNotFoundError(f"не удалось прочитать фон: {args.bg_image}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"не удалось открыть видео: {args.video}")

    segmentor = VideoBackgroundRemoval(
        model_selection=1 if args.model == "landscape" else 0,
        threshold=args.threshold,
        smooth_alpha=args.smooth_alpha,
        mask_blur=args.mask_blur,
        infer_every=args.infer_every,
    )

    latencies_ms = []
    frame_count = 0

    try:
        while frame_count < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (args.width, args.height))
            start = time.perf_counter()
            mask = segmentor.get_mask(frame)
            background = build_background(
                frame=frame,
                mode=args.mode,
                bg_color=bg_color,
                bg_image=bg_image,
                blur_ksize=args.blur_kernel,
            )
            _ = composite(frame, mask, background)
            end = time.perf_counter()

            if frame_count >= args.warmup:
                latencies_ms.append((end - start) * 1000.0)

            frame_count += 1
    finally:
        segmentor.close()
        cap.release()

    if not latencies_ms:
        raise RuntimeError("недостаточно кадров для замера. Уменьши --warmup или проверь видео.")

    mean_latency = statistics.mean(latencies_ms)
    median_latency = statistics.median(latencies_ms)
    mean_fps = 1000.0 / mean_latency

    print("=== Benchmark results ===")
    print(f"CPU: {get_cpu_name()}")
    print(f"Video: {args.video}")
    print(f"Mode: {args.mode}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames measured: {len(latencies_ms)}")
    print(f"Mean latency (ms): {mean_latency:.2f}")
    print(f"Median latency (ms): {median_latency:.2f}")
    print(f"Mean FPS (processed): {mean_fps:.2f}")


if __name__ == "__main__":
    main()
