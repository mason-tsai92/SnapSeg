from __future__ import annotations

from typing import Any

from .exporter import AnnotationExporter, MaskAnnotation

__all__ = [
    "AnnotationExporter",
    "MaskAnnotation",
    "AsyncAutosaveManager",
    "AsyncSaveManager",
    "PrefetchQueue",
    "SaveTask",
    "SamEmbeddingCacheService",
    "get_global_service",
    "DEFAULT_CHECKPOINT_DIR",
    "SamImageCache",
    "SamPrediction",
]


_LAZY_IMPORT_MAP = {
    "AsyncAutosaveManager": (".runtime", "AsyncAutosaveManager"),
    "AsyncSaveManager": (".runtime", "AsyncSaveManager"),
    "PrefetchQueue": (".runtime", "PrefetchQueue"),
    "SaveTask": (".runtime", "SaveTask"),
    "DEFAULT_CHECKPOINT_DIR": (".sam_service", "DEFAULT_CHECKPOINT_DIR"),
    "SamEmbeddingCacheService": (".sam_service", "SamEmbeddingCacheService"),
    "SamImageCache": (".sam_service", "SamImageCache"),
    "SamPrediction": (".sam_service", "SamPrediction"),
    "get_global_service": (".sam_service", "get_global_service"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_IMPORT_MAP.get(name)
    if target is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_name, attr_name = target
    module = __import__(f"{__name__}{module_name}", fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
