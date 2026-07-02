"""
GPU Insight Lab - Storage Models
Python dataclasses representing database rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SystemRecord:
    id: Optional[int] = None
    hostname: str = ""
    os: str = ""
    cpu: str = ""
    ram_gb: float = 0.0
    gpu_name: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    collected_at: str = ""


@dataclass
class BenchmarkSessionRecord:
    id: Optional[int] = None
    system_id: Optional[int] = None
    session_name: str = ""
    started_at: str = ""
    completed_at: str = ""
    health_score: Optional[float] = None
    confidence: Optional[float] = None
    status: str = "pending"


@dataclass
class BenchmarkResultRecord:
    id: Optional[int] = None
    session_id: Optional[int] = None
    test_name: str = ""
    schema_version: str = "1.0"
    result_json: str = "{}"
    created_at: str = ""


@dataclass
class DiagnosisRecord:
    id: Optional[int] = None
    session_id: Optional[int] = None
    rule_id: str = ""
    severity: str = "INFO"
    category: str = ""
    title: str = ""
    summary: str = ""
    evidence: str = ""
    confidence: float = 0.0
    created_at: str = ""


@dataclass
class ReportRecord:
    id: Optional[int] = None
    session_id: Optional[int] = None
    format: str = ""
    file_path: str = ""
    created_at: str = ""


@dataclass
class ApplicationSetting:
    key: str = ""
    value: str = ""
    updated_at: str = ""
