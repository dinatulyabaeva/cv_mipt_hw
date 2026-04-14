import argparse
import platform
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np


class FPSMeter:
    def __init__(self, window_size: int = 30) -> None:
        self.timestamps: deque[float] = deque(maxlen=window_size)

    def update(self) -> float:
        now = time.perf_counter()
        self.timestamps.append(now)
        if len(self.timestamps) < 2:
            return 0.0
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / duration


class MaskSmoother:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.prev_mask: Optional[np.ndarray] = None

    def apply(self, mask: np.ndarray) -> np.ndarray:
        if self.prev_mask is None:
            self.prev_mask = mask.astype(np.float32)
        else:
            self.prev_mask = (
                self.alpha * self.prev_mask + (1.0 - self.alpha) * mask.astype(np.float32)
            )
        return self.prev_mask


class VideoBackgroundRemoval:
    def __init__(
        self,
        model_selection: int,
        threshold: float,
        smooth_alpha: float,
        mask_blur: int,
        infer_every: int,
    ) -> None:
        self.segmentor = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )
        self.threshold = threshold
        self.smoother = MaskSmoother(alpha=smooth_alpha)
        self.mask_blur = max(0, int(mask_blur))
        self.infer_every = max(1, int(infer_every))
        self.cached_mask: Optional[np.ndarray] = None
        self.frame_idx = 0

    def close(self) -> None:
        self.segmentor.close()

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = np.clip(mask, 0.0, 1.0)
        mask = self.smoother.apply(mask)

        if self.mask_blur > 0:
            blur_size = self.mask_blur if self.mask_blur % 2 == 1 else self.mask_blur + 1
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        binary_mask = (mask > self.threshold).astype(np.float32)
        binary_mask = cv2.medianBlur((binary_mask * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
        return binary_mask

    def get_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        self.frame_idx += 1
        if self.cached_mask is not None and self.frame_idx % self.infer_every != 1:
            return self.cached_mask

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.segmentor.process(frame_rgb)
        raw_mask = results.segmentation_mask
        self.cached_mask = self._postprocess_mask(raw_mask)
        return self.cached_mask


def parse_color(color_str: str) -> Tuple[int, int, int]:
    parts = [part.strip() for part in color_str.split(",")]
    if len(parts) != 3:
        raise ValueError("цвет должен быть в формате R, G, B")
    values = tuple(int(part) for part in parts)
    if any(not 0 <= value <= 255 for value in values):
        raise ValueError("каждый канал цвета должен быть в диапазоне 0..255")
    return values


def build_background(
    frame: np.ndarray,
    mode: str,
    bg_color: Tuple[int, int, int],
    bg_image: Optional[np.ndarray],
    blur_ksize: int,
) -> np.ndarray:
    if mode == "color":
        return np.full_like(frame, bg_color, dtype=np.uint8)

    if mode == "image":
        if bg_image is None:
            raise ValueError("Для режима image требуется --bg-image")
        return cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))

    if mode == "blur":
        kernel = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        kernel = max(3, kernel)
        return cv2.GaussianBlur(frame, (kernel, kernel), 0)

    raise ValueError(f"Неизвестный режим фона: {mode}")


def composite(frame: np.ndarray, mask: np.ndarray, background: np.ndarray) -> np.ndarray:
    mask_3 = np.repeat(mask[:, :, None], 3, axis=2)
    foreground = frame.astype(np.float32) * mask_3
    bg = background.astype(np.float32) * (1.0 - mask_3)
    output = np.clip(foreground + bg, 0, 255).astype(np.uint8)
    return output


def add_overlay(
    frame: np.ndarray,
    fps: float,
    source_label: str,
    mode: str,
    resolution: Tuple[int, int],
    infer_every: int,
) -> np.ndarray:
    overlay = frame.copy()
    lines = [
        f"FPS (processed): {fps:.2f}",
        f"Source: {source_label}",
        f"Mode: {mode}",
        f"Resolution: {resolution[0]}x{resolution[1]}",
        f"Infer every N frames: {infer_every}",
        "Keys: q/ESC quit, m switch mode, s save frame",
    ]

    y = 28
    for line in lines:
        cv2.putText(
            overlay,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28
    return overlay


def create_capture(source: Union[int, str], width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def parse_source(source_arg: str) -> Union[int, str]:
    return int(source_arg) if source_arg.isdigit() else source_arg


def get_cpu_name() -> str:
    return platform.processor() or platform.platform()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Удаление или замена фона в реальном времени на CPU"
    )
    parser.add_argument("--source", default="0", help="Индекс камеры или путь к видео")
    parser.add_argument(
        "--mode",
        choices=["color", "image", "blur"],
        default="blur",
        help="Режим замены фона",
    )
    parser.add_argument(
        "--bg-color",
        default="0,255,0",
        help="Цвет фона в формате B,G,R для режима color",
    )
    parser.add_argument("--bg-image", default=None, help="Путь к изображению фона")
    parser.add_argument("--width", type=int, default=640, help="Ширина входного кадра")
    parser.add_argument("--height", type=int, default=480, help="Высота входного кадра")
    parser.add_argument(
        "--model",
        choices=["general", "landscape"],
        default="landscape",
        help="Вариант модели MediaPipe",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Порог бинаризации маски сегментации",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.65,
        help="Сила сглаживания маски во времени: ближе к 1 сильнее сглаживание",
    )
    parser.add_argument(
        "--mask-blur",
        type=int,
        default=7,
        help="Размер Gaussian blur для маски",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=35,
        help="Размер ядра размытия для режима blur",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=1,
        help="Выполнять инференс 1 раз в N кадров",
    )
    parser.add_argument(
        "--save-dir",
        default="outputs",
        help="Папка для сохранения кадров по клавише s",
    )
    args = parser.parse_args()

    bg_color = parse_color(args.bg_color)
    source = parse_source(args.source)
    cap = create_capture(source, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть источник видео: {args.source}")

    bg_image = None
    if args.bg_image is not None:
        bg_image = cv2.imread(args.bg_image)
        if bg_image is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение фона: {args.bg_image}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode
    modes = ["color", "image", "blur"] if bg_image is not None else ["color", "blur"]

    segmentor = VideoBackgroundRemoval(
        model_selection=1 if args.model == "landscape" else 0,
        threshold=args.threshold,
        smooth_alpha=args.smooth_alpha,
        mask_blur=args.mask_blur,
        infer_every=args.infer_every,
    )
    fps_meter = FPSMeter(window_size=30)

    print("Background Removal started")
    print(f"CPU: {get_cpu_name()}")
    print(f"Source: {args.source}")
    print(f"Initial mode: {mode}")
    print("Нажмите q или ESC для выхода, m для переключения режима, s для сохранения кадра.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (args.width, args.height))
            mask = segmentor.get_mask(frame)
            background = build_background(
                frame=frame,
                mode=mode,
                bg_color=bg_color,
                bg_image=bg_image,
                blur_ksize=args.blur_kernel,
            )
            output = composite(frame, mask, background)

            fps = fps_meter.update()
            output = add_overlay(
                output,
                fps=fps,
                source_label=str(args.source),
                mode=mode,
                resolution=(args.width, args.height),
                infer_every=args.infer_every,
            )
            cv2.imshow("Background Removal", output)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                idx = modes.index(mode)
                mode = modes[(idx + 1) % len(modes)]
                print(f"Режим переключен на: {mode}")
            if key == ord("s"):
                filename = save_dir / f"frame_{int(time.time())}.png"
                cv2.imwrite(str(filename), output)
                print(f"Кадр сохранен: {filename}")
    finally:
        segmentor.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
