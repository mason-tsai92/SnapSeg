from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interactive.sam_service import SamEmbeddingCacheService


def overlay(base_bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    canvas = base_bgr.copy()
    if mask is not None:
        color = np.zeros_like(canvas)
        color[:, :, 2] = 255
        mm = mask.astype(bool)
        canvas[mm] = (0.58 * canvas[mm] + 0.42 * color[mm]).astype(np.uint8)
    return canvas


def annotate_text(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (14, 14), (out.shape[1] - 14, 64), (22, 30, 40), thickness=-1)
    cv2.putText(out, text, (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (236, 242, 248), 2)
    return out


def main() -> None:
    root = ROOT
    image_path = root / "examples" / "sample.jpg"
    out_dir = root / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo.gif"

    svc = SamEmbeddingCacheService()
    svc.set_image(image_path)
    base_rgb = svc.image_rgb
    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)

    # Simulated interactive points (positive / negative)
    sequence = [
        ([[90.0, 100.0]], [1], "1) Positive click"),
        ([[90.0, 100.0], [150.0, 115.0]], [1, 1], "2) Add another positive click"),
        ([[90.0, 100.0], [150.0, 115.0], [115.0, 160.0]], [1, 1, 0], "3) Negative click to trim background"),
        ([[90.0, 100.0], [150.0, 115.0], [115.0, 160.0], [135.0, 140.0]], [1, 1, 0, 1], "4) Final refinement"),
    ]

    frames: list[np.ndarray] = []
    durations: list[float] = []

    # Intro frame
    intro = annotate_text(base_bgr, "SnapSeg: interactive point segmentation (~25ms P95 decode)")
    frames.append(cv2.cvtColor(intro, cv2.COLOR_BGR2RGB))
    durations.append(1.2)

    for coords, labels, text in sequence:
        pred = svc.predict(coords, labels, multimask_output=False)
        frame = overlay(base_bgr, pred.mask)
        for (x, y), lb in zip(coords, labels):
            c = (0, 255, 0) if lb == 1 else (0, 140, 255)
            cv2.circle(frame, (int(round(x)), int(round(y))), 4, c, -1)
        frame = annotate_text(frame, f"{text} | decoder {pred.latency_ms:.1f}ms")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        durations.append(0.7)

    pil_frames = [Image.fromarray(f) for f in frames]
    first = pil_frames[0]
    rest = pil_frames[1:]
    duration_ms = [int(d * 1000) for d in durations]
    first.save(
        out_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
