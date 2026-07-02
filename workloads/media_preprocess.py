"""
GPU Insight Lab - Media Preprocessing Workload
Orchestrates image preprocessing pipeline for benchmarking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from workloads.image_batch import ImageBatchResult, run_synthetic

logger = logging.getLogger(__name__)


@dataclass
class MediaPipelineResult:
    status: str = "ok"
    total_images_processed: int = 0
    avg_throughput_images_per_sec: float = 0.0
    total_pipeline_ms: float = 0.0
    error: Optional[str] = None


def run_pipeline(
    input_folder: Optional[Path] = None,
    width: int = 1920,
    height: int = 1080,
    batch: int = 20,
    use_cuda: bool = True,
) -> MediaPipelineResult:
    """
    Run media preprocessing pipeline.
    Uses folder if provided; falls back to synthetic data.
    """
    pipeline = MediaPipelineResult()

    if input_folder and input_folder.exists():
        from workloads.image_batch import run_from_folder  # noqa: PLC0415
        batch_result = run_from_folder(input_folder, use_cuda=use_cuda)
    else:
        batch_result = run_synthetic(width=width, height=height, batch=batch)

    if batch_result.status != "ok":
        pipeline.status = "error"
        pipeline.error = batch_result.error
        return pipeline

    pipeline.total_images_processed = batch_result.image_count
    pipeline.avg_throughput_images_per_sec = batch_result.throughput_images_per_sec
    pipeline.total_pipeline_ms = batch_result.total_time_ms
    return pipeline
