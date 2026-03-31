from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


class DatasetPackager:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root
        self.train_images_dir = dataset_root / "train" / "images"
        self.train_labels_dir = dataset_root / "train" / "labels"
        self.val_images_dir = dataset_root / "val" / "images"
        self.val_labels_dir = dataset_root / "val" / "labels"
        self.test_images_dir = dataset_root / "test" / "images"
        self.test_labels_dir = dataset_root / "test" / "labels"
        self.classes_path = dataset_root / "classes.txt"
        self.yaml_path = dataset_root / "dataset.yaml"

    @staticmethod
    def _stable_key(image_path: Path) -> str:
        raw = str(image_path.resolve()).encode("utf-8", errors="ignore")
        return hashlib.sha1(raw).hexdigest()[:10]

    def _dataset_stem(self, image_path: Path) -> str:
        return f"{image_path.stem}_{self._stable_key(image_path)}"

    def _load_global_classes(self) -> list[str]:
        if not self.classes_path.exists():
            return []
        lines = self.classes_path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]

    def _save_global_classes(self, classes: list[str]) -> None:
        self.classes_path.parent.mkdir(parents=True, exist_ok=True)
        self.classes_path.write_text("\n".join(classes), encoding="utf-8")

    @staticmethod
    def _normalize_classes(classes: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for c in classes:
            name = str(c).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    @staticmethod
    def _load_local_classes(path: Path) -> list[str]:
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]

    @staticmethod
    def _remap_label_lines(lines: list[str], local_classes: list[str], global_classes: list[str]) -> list[str]:
        global_index = {name: idx for idx, name in enumerate(global_classes)}
        remapped: list[str] = []
        for line in lines:
            text = line.strip()
            if not text:
                continue
            parts = text.split()
            if not parts:
                continue
            try:
                local_cls_id = int(parts[0])
            except ValueError:
                continue
            if local_cls_id < 0 or local_cls_id >= len(local_classes):
                continue

            cls_name = local_classes[local_cls_id]
            if cls_name not in global_index:
                global_index[cls_name] = len(global_classes)
                global_classes.append(cls_name)
            parts[0] = str(global_index[cls_name])
            remapped.append(" ".join(parts))
        return remapped

    def _write_yaml(self, classes: list[str]) -> None:
        self.yaml_path.parent.mkdir(parents=True, exist_ok=True)
        train_path = (self.dataset_root / "train" / "images").resolve().as_posix()
        val_path = (self.dataset_root / "val" / "images").resolve().as_posix()
        test_path = (self.dataset_root / "test" / "images").resolve().as_posix()
        names_text = ", ".join(classes)
        lines = [
            f"train: {train_path}",
            f"val:   {val_path}",
            f"test:  {test_path}",
            "",
            f"nc: {len(classes)}",
            f"names: [{names_text}]",
        ]
        self.yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def update_class_metadata(self, class_list: list[str] | None = None) -> list[str]:
        if class_list:
            classes = self._normalize_classes(class_list)
        else:
            classes = self._load_global_classes()
        if not classes:
            classes = ["object"]
        self._save_global_classes(classes)
        self._write_yaml(classes)
        return classes

    def package_yolo_seg(self, image_path: Path, image_out: Path, class_list: list[str] | None = None) -> None:
        local_labels_path = image_out / "labels_yolo_seg" / f"{image_path.stem}.txt"
        local_classes_path = image_out / "classes_yolo_seg.txt"

        self.train_images_dir.mkdir(parents=True, exist_ok=True)
        self.train_labels_dir.mkdir(parents=True, exist_ok=True)
        self.val_images_dir.mkdir(parents=True, exist_ok=True)
        self.val_labels_dir.mkdir(parents=True, exist_ok=True)
        self.test_images_dir.mkdir(parents=True, exist_ok=True)
        self.test_labels_dir.mkdir(parents=True, exist_ok=True)

        # Always refresh class metadata so YAML tracks current UI class settings.
        global_classes = self.update_class_metadata(class_list=class_list)

        if not local_labels_path.exists() or not local_classes_path.exists():
            return

        local_classes = self._load_local_classes(local_classes_path)
        lines = local_labels_path.read_text(encoding="utf-8").splitlines()
        remapped = self._remap_label_lines(lines, local_classes, global_classes)

        dataset_stem = self._dataset_stem(image_path)
        out_image = self.train_images_dir / f"{dataset_stem}{image_path.suffix}"
        out_label = self.train_labels_dir / f"{dataset_stem}.txt"

        shutil.copy2(image_path, out_image)
        out_label.write_text("\n".join(remapped), encoding="utf-8")
        self._save_global_classes(global_classes)
        self._write_yaml(global_classes)
