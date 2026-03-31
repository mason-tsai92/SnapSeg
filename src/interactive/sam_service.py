from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep, time
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image


@dataclass
class SamPrediction:
    mask: np.ndarray
    score: float
    latency_ms: float


@dataclass
class SamImageCache:
    image_path: Path
    image_rgb: np.ndarray
    image_embeddings: torch.Tensor
    orig_h: int
    orig_w: int
    reshape_h: int
    reshape_w: int


_service_registry: dict[tuple[str, str, str], "SamEmbeddingCacheService"] = {}
_registry_lock = threading.Lock()


def get_global_service(
    backend: Literal["sam", "mobile_sam"] = "sam",
    model_id: str | None = None,
    device: str | None = None,
) -> "SamEmbeddingCacheService":
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_id = model_id or SamEmbeddingCacheService._default_model_id(backend)
    key = (backend, resolved_model_id, resolved_device)
    svc = _service_registry.get(key)
    if svc is not None:
        return svc
    with _registry_lock:
        svc = _service_registry.get(key)
        if svc is None:
            svc = SamEmbeddingCacheService(backend=backend, model_id=resolved_model_id, device=resolved_device)
            _service_registry[key] = svc
        return svc


class SamEmbeddingCacheService:
    def __init__(
        self,
        backend: Literal["sam", "mobile_sam"] = "sam",
        model_id: str | None = None,
        device: str | None = None,
    ) -> None:
        self.requested_backend = backend
        self.backend = backend
        self.model_id = model_id or self._default_model_id(backend)
        self.last_load_warning: str = ""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            # Keep desktop responsive when running SAM fully on CPU.
            max_threads = max(1, (os.cpu_count() or 2) // 2)
            try:
                torch.set_num_threads(max_threads)
            except Exception:
                pass
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
        self._processor: Any | None = None
        self._model: Any | None = None
        self._model_load_lock = threading.Lock()
        self.model_status: str = "idle"  # idle | loading | ready | error
        self.model_error: str = ""
        self.model_loading_started_at: float | None = None
        self.last_model_load_ms: float | None = None

        self._image_path: Path | None = None
        self._image_rgb: np.ndarray | None = None
        self._image_embeddings: torch.Tensor | None = None
        self._orig_h: int = 0
        self._orig_w: int = 0
        self._reshape_h: int = 0
        self._reshape_w: int = 0

    @staticmethod
    def _default_model_id(backend: Literal["sam", "mobile_sam"]) -> str:
        # Use a lightweight Transformers-compatible SAM checkpoint for the mobile_sam mode.
        if backend == "mobile_sam":
            return "nielsr/slimsam-50-uniform"
        return "facebook/sam-vit-base"

    @staticmethod
    def _is_cache_complete(model_id: str) -> bool:
        try:
            from huggingface_hub import try_to_load_from_cache
        except Exception:
            return False
        required_cfg = ("config.json", "preprocessor_config.json")
        for fname in required_cfg:
            if try_to_load_from_cache(model_id, fname) is None:
                return False
        weight_candidates = ("pytorch_model.bin", "model.safetensors", "model.safetensors.index.json")
        return any(try_to_load_from_cache(model_id, fname) is not None for fname in weight_candidates)

    def _ensure_model_unlocked(self) -> None:
        from transformers import SamModel, SamProcessor

        def _load(model_id: str, local_only: bool) -> tuple[Any, Any]:
            processor = SamProcessor.from_pretrained(model_id, local_files_only=local_only)
            model = SamModel.from_pretrained(model_id, local_files_only=local_only).to(self.device)
            return processor, model

        def _load_with_retries(model_id: str) -> tuple[Any, Any]:
            use_local = self._is_cache_complete(model_id)
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    return _load(model_id, local_only=use_local)
                except Exception as exc:  # pragma: no cover - dependent on runtime env/network
                    last_exc = exc
                    if use_local:
                        # Cache seems present but failed to load: fallback to online fetch.
                        use_local = False
                        continue
                    if attempt < 2:
                        sleep(2**attempt)
            if last_exc is None:
                raise RuntimeError("Unknown model load error")
            raise last_exc

        try:
            self._processor, self._model = _load_with_retries(self.model_id)
            self.last_load_warning = ""
        except Exception as exc:
            # MobileSAM checkpoints on HF are not always Transformers-SAM compatible.
            # Fallback to base SAM so the app still works instead of returning HTTP 400 on /api/config.
            if self.backend == "mobile_sam":
                fallback_id = "facebook/sam-vit-base"
                self._processor, self._model = _load_with_retries(fallback_id)
                self.backend = "sam"
                self.model_id = fallback_id
                self.last_load_warning = (
                    "Requested mobile_sam backend could not be loaded. "
                    "Fell back to sam/facebook-sam-vit-base."
                )
            else:
                raise RuntimeError(
                    f"Failed to load SAM model '{self.model_id}'. "
                    "Use --model-id to specify a valid Transformers SAM checkpoint."
                ) from exc
        self._model.eval()

    def ensure_model(self) -> None:
        if self._processor is not None and self._model is not None:
            if self.model_status != "ready":
                self.model_status = "ready"
            return
        with self._model_load_lock:
            if self._processor is not None and self._model is not None:
                if self.model_status != "ready":
                    self.model_status = "ready"
                return
            self.model_status = "loading"
            self.model_error = ""
            self.model_loading_started_at = time()
            t0 = perf_counter()
            try:
                self._ensure_model_unlocked()
                self.model_status = "ready"
                self.last_model_load_ms = round((perf_counter() - t0) * 1000.0, 2)
            except Exception as exc:
                self.model_status = "error"
                self.model_error = str(exc)
                self.last_model_load_ms = None
                raise

    def set_image(self, image_path: Path) -> None:
        self.ensure_model()
        self._image_path = image_path

        image = Image.open(image_path).convert("RGB")
        self._image_rgb = np.array(image, dtype=np.uint8)

        inputs = self._processor(images=image, return_tensors="pt")
        orig = inputs["original_sizes"][0].tolist()
        reshaped = inputs["reshaped_input_sizes"][0].tolist()
        self._orig_h, self._orig_w = int(orig[0]), int(orig[1])
        self._reshape_h, self._reshape_w = int(reshaped[0]), int(reshaped[1])
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            self._image_embeddings = self._model.get_image_embeddings(pixel_values)

    def snapshot_cache(self, to_cpu: bool = True) -> SamImageCache:
        if self._image_path is None or self._image_rgb is None or self._image_embeddings is None:
            raise RuntimeError("No image cache available. Call set_image first.")
        emb = self._image_embeddings.detach()
        if to_cpu:
            emb = emb.cpu()
        return SamImageCache(
            image_path=self._image_path,
            image_rgb=self._image_rgb.copy(),
            image_embeddings=emb,
            orig_h=self._orig_h,
            orig_w=self._orig_w,
            reshape_h=self._reshape_h,
            reshape_w=self._reshape_w,
        )

    def load_cache(self, cache: SamImageCache) -> None:
        self.ensure_model()
        self._image_path = cache.image_path
        self._image_rgb = cache.image_rgb.copy()
        self._orig_h = int(cache.orig_h)
        self._orig_w = int(cache.orig_w)
        self._reshape_h = int(cache.reshape_h)
        self._reshape_w = int(cache.reshape_w)
        self._image_embeddings = cache.image_embeddings.to(self.device)

    @property
    def image_rgb(self) -> np.ndarray:
        if self._image_rgb is None:
            raise RuntimeError("Image is not set. Call set_image first.")
        return self._image_rgb

    def predict(
        self,
        point_coords: list[list[float]] | None = None,
        point_labels: list[int] | None = None,
        box_xyxy: list[float] | None = None,
        multimask_output: bool = False,
    ) -> SamPrediction:
        if self._image_embeddings is None:
            raise RuntimeError("Image embeddings are not cached. Call set_image first.")
        has_points = bool(point_coords)
        has_box = bool(box_xyxy)
        if not has_points and not has_box:
            raise ValueError("At least one prompt is required (point or box).")
        if has_points:
            if point_labels is None:
                raise ValueError("point_labels is required when point_coords is provided.")
            if len(point_coords or []) != len(point_labels):
                raise ValueError("point_coords and point_labels must have same length.")

        input_points: torch.Tensor | None = None
        input_labels: torch.Tensor | None = None
        if has_points:
            coords_np = np.asarray(point_coords, dtype=np.float32)
            coords_np[:, 0] = (coords_np[:, 0] + 0.5) * (float(self._reshape_w) / max(1.0, float(self._orig_w)))
            coords_np[:, 1] = (coords_np[:, 1] + 0.5) * (float(self._reshape_h) / max(1.0, float(self._orig_h)))
            input_points = torch.from_numpy(coords_np).to(self.device).unsqueeze(0).unsqueeze(0)
            input_labels = torch.tensor(point_labels, dtype=torch.int64, device=self.device).unsqueeze(0).unsqueeze(0)

        input_boxes: torch.Tensor | None = None
        if has_box:
            if box_xyxy is None or len(box_xyxy) != 4:
                raise ValueError("box_xyxy must be [x1, y1, x2, y2].")
            bx = np.asarray(box_xyxy, dtype=np.float32).copy()
            bx[0] = (bx[0] + 0.5) * (float(self._reshape_w) / max(1.0, float(self._orig_w)))
            bx[1] = (bx[1] + 0.5) * (float(self._reshape_h) / max(1.0, float(self._orig_h)))
            bx[2] = (bx[2] + 0.5) * (float(self._reshape_w) / max(1.0, float(self._orig_w)))
            bx[3] = (bx[3] + 0.5) * (float(self._reshape_h) / max(1.0, float(self._orig_h)))
            input_boxes = torch.from_numpy(bx).to(self.device).unsqueeze(0).unsqueeze(0)

        t0 = perf_counter()
        with torch.no_grad():
            model_kwargs: dict[str, Any] = {
                "image_embeddings": self._image_embeddings,
                "multimask_output": multimask_output,
            }
            if input_points is not None and input_labels is not None:
                model_kwargs["input_points"] = input_points
                model_kwargs["input_labels"] = input_labels
            if input_boxes is not None:
                model_kwargs["input_boxes"] = input_boxes
            outputs = self._model(**model_kwargs)

            iou_scores = outputs.iou_scores[0, 0]  # [K]
            best_idx = int(torch.argmax(iou_scores).item())

            # Use SAM's official mask post-processing to map masks back to original image geometry.
            original_sizes = torch.tensor(
                [[self._orig_h, self._orig_w]],
                dtype=torch.int64,
                device=outputs.pred_masks.device,
            )
            reshaped_input_sizes = torch.tensor(
                [[self._reshape_h, self._reshape_w]],
                dtype=torch.int64,
                device=outputs.pred_masks.device,
            )
            post_masks = self._processor.image_processor.post_process_masks(
                outputs.pred_masks,
                original_sizes=original_sizes,
                reshaped_input_sizes=reshaped_input_sizes,
            )
            best_post = post_masks[0][0, best_idx]  # [orig_h, orig_w]
            binary = (best_post > 0.0).to(torch.uint8)
            score = float(iou_scores[best_idx].item())
        latency_ms = (perf_counter() - t0) * 1000.0

        mask_np = binary.detach().cpu().numpy().astype(np.uint8)
        return SamPrediction(mask=mask_np, score=score, latency_ms=latency_ms)
