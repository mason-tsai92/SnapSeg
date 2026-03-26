from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from src.interactive import SamEmbeddingCacheService


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile SAM interactive decoder latency")
    p.add_argument("--image", type=Path, required=True, help="Input image path")
    p.add_argument("--runs", type=int, default=30, help="Number of predict runs")
    p.add_argument("--points", type=int, default=2, help="Points per run")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    return p.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * q))
    return float(arr[max(0, min(idx, len(arr) - 1))])


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    svc = SamEmbeddingCacheService(device=device)
    t0 = perf_counter()
    svc.set_image(args.image)
    embedding_ms = (perf_counter() - t0) * 1000.0

    h, w = svc.image_rgb.shape[:2]
    rng = np.random.default_rng(42)
    latencies: list[float] = []

    if device == "cuda":
        torch.cuda.synchronize()

    for _ in range(args.runs):
        coords: list[list[float]] = []
        labels: list[int] = []
        for i in range(max(1, args.points)):
            x = float(rng.integers(0, w))
            y = float(rng.integers(0, h))
            coords.append([x, y])
            labels.append(1 if i % 2 == 0 else 0)

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = perf_counter()
        _ = svc.predict(point_coords=coords, point_labels=labels, multimask_output=False)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((perf_counter() - t1) * 1000.0)

    print(f"device={device}")
    print(f"embedding_ms={embedding_ms:.2f}")
    print(f"runs={len(latencies)}")
    print(f"decoder_mean_ms={mean(latencies):.2f}")
    print(f"decoder_p50_ms={percentile(latencies, 0.50):.2f}")
    print(f"decoder_p95_ms={percentile(latencies, 0.95):.2f}")
    print(f"threshold_500ms_pass={percentile(latencies, 0.95) < 500.0}")


if __name__ == "__main__":
    main()

