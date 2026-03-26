from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class MaskAnnotation:
    image_path: Path
    category_name: str
    mask: np.ndarray
    score: float = 1.0


class AnnotationExporter:
    def __init__(self, polygon_epsilon_ratio: float = 0.005) -> None:
        self.polygon_epsilon_ratio = max(0.0, float(polygon_epsilon_ratio))

    @staticmethod
    def _to_binary(mask: np.ndarray) -> np.ndarray:
        return (mask > 0).astype(np.uint8) * 255

    def _mask_to_polygons(self, mask: np.ndarray) -> list[list[float]]:
        binary = self._to_binary(mask)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[list[float]] = []
        for c in contours:
            if c.shape[0] < 3:
                continue
            if self.polygon_epsilon_ratio > 0.0:
                perimeter = cv2.arcLength(c, True)
                epsilon = perimeter * self.polygon_epsilon_ratio
                c = cv2.approxPolyDP(c, epsilon, True)
            c2 = c.squeeze(axis=1)
            if c2.ndim != 2 or c2.shape[0] < 3:
                continue
            poly = c2.astype(float).reshape(-1).tolist()
            if len(poly) >= 6:
                polygons.append(poly)
        return polygons

    @staticmethod
    def _bbox_xywh(mask: np.ndarray) -> list[float]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max()), float(ys.max())
        return [x1, y1, x2 - x1 + 1.0, y2 - y1 + 1.0]

    def export_coco(self, annotations: list[MaskAnnotation], output_json: Path) -> None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        categories: dict[str, int] = {}
        images: list[dict] = []
        anns: list[dict] = []
        image_id_by_path: dict[Path, int] = {}
        ann_id = 1

        for ann in annotations:
            if ann.image_path not in image_id_by_path:
                img = cv2.imread(str(ann.image_path), cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
                iid = len(image_id_by_path) + 1
                image_id_by_path[ann.image_path] = iid
                images.append({"id": iid, "file_name": ann.image_path.name, "width": w, "height": h})

            if ann.category_name not in categories:
                categories[ann.category_name] = len(categories) + 1
            cat_id = categories[ann.category_name]

            bbox = self._bbox_xywh(ann.mask)
            area = float((ann.mask > 0).sum())
            polygons = self._mask_to_polygons(ann.mask)
            anns.append(
                {
                    "id": ann_id,
                    "image_id": image_id_by_path[ann.image_path],
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": polygons,
                    "score": float(ann.score),
                }
            )
            ann_id += 1

        coco = {
            "info": {"description": "Interactive SAM annotation"},
            "images": images,
            "annotations": anns,
            "categories": [{"id": cid, "name": name, "supercategory": "auto"} for name, cid in categories.items()],
        }
        output_json.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

    def export_yolo_seg(self, annotations: list[MaskAnnotation], labels_dir: Path, class_names_path: Path) -> None:
        labels_dir.mkdir(parents=True, exist_ok=True)
        class_map: dict[str, int] = {}
        by_image: dict[Path, list[MaskAnnotation]] = {}
        for ann in annotations:
            by_image.setdefault(ann.image_path, []).append(ann)
            if ann.category_name not in class_map:
                class_map[ann.category_name] = len(class_map)

        for img_path, items in by_image.items():
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            lines: list[str] = []
            for ann in items:
                cls_id = class_map[ann.category_name]
                polygons = self._mask_to_polygons(ann.mask)
                for poly in polygons:
                    coords: list[str] = []
                    for i in range(0, len(poly), 2):
                        coords.extend([f"{poly[i] / w:.6f}", f"{poly[i + 1] / h:.6f}"])
                    if len(coords) >= 6:
                        lines.append(f"{cls_id} " + " ".join(coords))
            (labels_dir / f"{img_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        classes = sorted(class_map.items(), key=lambda kv: kv[1])
        class_names_path.parent.mkdir(parents=True, exist_ok=True)
        class_names_path.write_text("\n".join([name for name, _ in classes]), encoding="utf-8")
