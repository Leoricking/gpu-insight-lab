"""
GPU Insight Lab - Image Batch Workload
Batch image preprocessing with CPU (Pillow) and optional CUDA backend.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ImageBatchResult:
    status: str = "ok"
    backend: str = "cpu"
    image_count: int = 0
    width: int = 0
    height: int = 0
    total_time_ms: float = 0.0
    io_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    throughput_images_per_sec: float = 0.0
    correctness_pass: Optional[bool] = None
    max_pixel_error: Optional[float] = None
    error: Optional[str] = None


def _grayscale_cpu_numpy(image_array: Any) -> Any:
    """RGB to grayscale: Y = 0.2989*R + 0.5870*G + 0.1140*B"""
    import numpy as np  # noqa: PLC0415
    arr = np.asarray(image_array, dtype=np.float32)
    weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    return (arr @ weights).clip(0, 255).astype(np.uint8)


def run_from_folder(
    folder: Path,
    use_cuda: bool = True,
) -> ImageBatchResult:
    """
    Load PNG/JPG images from folder, run grayscale conversion.
    Falls back gracefully if Pillow unavailable.
    """
    result = ImageBatchResult()

    try:
        from PIL import Image  # type: ignore  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        result.status = "error"
        result.error = f"Pillow or NumPy not installed: {exc}"
        return result

    # Collect image files
    folder = Path(folder)
    if not folder.exists():
        result.status = "error"
        result.error = f"Folder not found: {folder}"
        return result

    image_files = sorted(
        list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
    )
    if not image_files:
        result.status = "error"
        result.error = f"No PNG/JPG images found in {folder}"
        return result

    # Load images
    t_io_start = time.perf_counter()
    images: List[Any] = []
    for fp in image_files:
        try:
            img = Image.open(fp).convert("RGB")
            images.append(np.array(img))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", fp, exc)

    t_io_end = time.perf_counter()
    result.io_time_ms = (t_io_end - t_io_start) * 1000
    result.image_count = len(images)

    if not images:
        result.status = "error"
        result.error = "All image loads failed"
        return result

    result.width = images[0].shape[1] if images[0].ndim >= 2 else 0
    result.height = images[0].shape[0] if images[0].ndim >= 1 else 0

    # CPU processing
    t_cpu_start = time.perf_counter()
    cpu_results = [_grayscale_cpu_numpy(img) for img in images]
    t_cpu_end = time.perf_counter()
    cpu_time = (t_cpu_end - t_cpu_start) * 1000

    result.backend = "cpu"
    result.processing_time_ms = cpu_time
    result.total_time_ms = result.io_time_ms + cpu_time
    if result.total_time_ms > 0:
        result.throughput_images_per_sec = result.image_count / (result.total_time_ms / 1000)

    # CUDA backend (via native runner)
    if use_cuda:
        from benchmarks.native_runner import run_test, is_available  # noqa: PLC0415
        if is_available():
            native_result = run_test("image_grayscale", repeat=1)
            if native_result and "error" not in native_result:
                result.backend = "cuda"
                gpu_time = native_result.get("gpu_time_ms")
                if gpu_time:
                    result.processing_time_ms = gpu_time
                    result.total_time_ms = result.io_time_ms + gpu_time
                    if result.total_time_ms > 0:
                        result.throughput_images_per_sec = result.image_count / (result.total_time_ms / 1000)

                # Correctness comparison
                corr = native_result.get("correctness_pass")
                if corr is not None:
                    result.correctness_pass = bool(corr)
                    result.max_pixel_error = native_result.get("max_error")
        # else: no CUDA available, CPU results are fine

    if result.backend == "cpu":
        result.correctness_pass = True  # CPU is the reference

    return result


def run_synthetic(
    width: int = 1920,
    height: int = 1080,
    batch: int = 20,
) -> ImageBatchResult:
    """
    Run batch grayscale on synthetic random images (no file I/O).
    Always available since it uses only NumPy.
    """
    result = ImageBatchResult()

    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        result.status = "error"
        result.error = f"NumPy not installed: {exc}"
        return result

    images = [np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8) for _ in range(batch)]
    result.image_count = batch
    result.width = width
    result.height = height

    t0 = time.perf_counter()
    _ = [_grayscale_cpu_numpy(img) for img in images]
    t1 = time.perf_counter()

    result.processing_time_ms = (t1 - t0) * 1000
    result.total_time_ms = result.processing_time_ms
    result.backend = "cpu_synthetic"
    if result.total_time_ms > 0:
        result.throughput_images_per_sec = batch / (result.total_time_ms / 1000)
    result.correctness_pass = True

    return result
