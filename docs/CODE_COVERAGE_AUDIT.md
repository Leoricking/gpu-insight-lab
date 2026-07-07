# GPU Insight Lab — Code Coverage Audit

**Generated:** 2026-07-07  
**Version:** v0.1.0  
**Auditor:** Automated audit pass (Phase 1)

---

## Feature Coverage Table

| Claimed feature | Existing implementation files | CLI support | GUI support | Native support | Report support | Test support | Current status | Missing code | Interview demo readiness | Priority |
|---|---|---|---|---|---|---|---|---|---|---|
| System Inspector | collectors/system_collector.py, collectors/nvidia_collector.py, collectors/cuda_collector.py, collectors/pcie_collector.py, collectors/tool_collector.py, collectors/amd_collector.py | `system-info` command | Partial (GUI page exists) | device_info.cu | All 5 formats include system section | test_collectors.py, test_smoke.py | IMPLEMENTED | None | HIGH — works without GPU | P1 |
| PCIe Analyzer | collectors/pcie_collector.py, diagnosis/rules.py (transfer_overhead) | `system-info` shows PCIe | Partial | device_info.cu reports compute cap | JSON/MD/HTML/XLSX include PCIe section | test_collectors.py | PARTIAL | Active PCIe bandwidth measurement via native binary not yet dispatched to Python CLI as separate command | MEDIUM — data collected, no standalone PCIe bench command | P1 |
| Memory Benchmark | native/cuda/memory_bandwidth.cu, benchmarks/runner.py, benchmarks/native_runner.py | `benchmark --test memory_bandwidth` | Partial | memory_bandwidth.cu (H2D/D2H/D2D) | Included in all formats | test_smoke.py (indirect) | IMPLEMENTED | MemoryBenchmarkResult schema not yet added to schemas.py | HIGH — runs CPU-only fallback if no GPU | P1 |
| Kernel Lab | native/cuda/vector_add.cu, reduction.cu, transpose.cu, gemm_naive.cu, gemm_tiled.cu, stream_pipeline.cu | `benchmark --test <name>` | Partial | All kernels compiled into gpu_insight_benchmark | benchmark section in all formats | test_smoke.py | IMPLEMENTED | prefix_sum.cu, convolution_2d.cu skeletons missing; KernelBenchmarkResult schema missing | HIGH — CPU fallback works | P1 |
| Workload Profiler | profilers/nsight_systems.py, nsight_compute.py, nvidia_smi_monitor.py, rocm_profiler.py | No direct CLI command | Partial | N/A | N/A | test_smoke.py (import only) | PARTIAL | No CLI integration; profilers are read-only wrappers, not invoked by benchmark pipeline | LOW — requires Nsight tools installed | P2 |
| Diagnosis Engine | diagnosis/engine.py, rules.py, scoring.py, evidence.py, recommendations.py | `diagnose --session` (session-id only) | Partial | N/A | Diagnosis section in all 5 formats | test_diagnosis.py | IMPLEMENTED | `--latest` flag missing from `diagnose` command; DiagnosisResult missing: gpu_insight_score, health_score, bottleneck_classification fields | HIGH — runs without GPU | P1 |
| Validation Center | benchmarks/cpu_baselines.py, diagnosis/scoring.py (correctness checks) | Indirect via quick-test/full-test | Partial | correctness_pass field in native kernels | correctness_pass shown in all formats | test_diagnosis.py | PARTIAL | No standalone `validate` CLI command; pass/fail policies not configurable | MEDIUM | P2 |
| History and Comparison | storage/database.py, storage/migrations.py | `history`, `compare` commands | Partial | N/A | Comparison sheet in Excel | test_storage.py | IMPLEMENTED | `export --latest` flag missing (only `--session` supported) | HIGH | P1 |
| Report Studio | reports/json_report.py, markdown_report.py, html_report.py, excel_report.py, csv_report.py | `export --session --format` | Partial | N/A | All 5 formats implemented | test_reports.py | IMPLEMENTED | `--latest` flag missing; demo-report command missing; interview demo summary line missing from reports | HIGH | P1 |
| CUDA Performance Lab | native/cuda/ (8 kernels), benchmarks/native_runner.py | `benchmark --test <name>` | Partial | 8 .cu files compiled | Benchmark section in all formats | test_smoke.py | IMPLEMENTED | prefix_sum.cu, convolution_2d.cu skeletons missing; softmax/layer_norm/GELU/Flash Attention are ROADMAP | HIGH | P1 |
| GPU PCIe Bandwidth Benchmark Tool | collectors/pcie_collector.py, native/cuda/memory_bandwidth.cu | `benchmark --test memory` or `--test pcie` | Partial | memory_bandwidth.cu measures H2D/D2H | PCIe section in reports | test_collectors.py | PARTIAL | `--test pcie` not dispatched separately from `--test memory_bandwidth`; native pcie_bandwidth sub-test not exposed to Python runner | MEDIUM | P1 |
| CUDA to HIP Portability Demo | native/hip/vector_add_hip.cpp, docs/CUDA_VS_HIP.md, collectors/amd_collector.py | No CLI flag | No GUI page | vector_add_hip.cpp only | NOT_VALIDATED label in reports | test_smoke.py (import only) | PARTIAL | reduction_hip.cpp, gemm_naive_hip.cpp missing; hipcc compile not automated | MEDIUM — good for interview discussion | P2 |
| GPU Benchmark Dashboard | app/gui/ (PySide6 MainWindow + sidebar) | N/A | MainWindow with pages | N/A | N/A | TestGUIImport in test_smoke.py | PARTIAL | GUI pages exist as stubs; benchmark progress display not fully wired | MEDIUM | P2 |
| AI Inference Kernel Roadmap | docs/12_WEEK_CUDA_JOB_ROADMAP.md | No CLI | No GUI | No native kernels | Not in reports | None | ROADMAP | softmax, layer_norm, GELU, Flash Attention, INT8 quant, PyTorch ext, TensorRT plugin all ROADMAP | LOW — document only | P3 |
| CLI Automation | app/cli.py (argparse, 8 commands) | system-info, quick-test, full-test, benchmark, history, compare, export, diagnose | N/A | N/A | N/A | test_smoke.py TestCLISmoke | IMPLEMENTED | `diagnose --latest` missing; `export --latest` missing; `demo-report` command missing | HIGH | P1 |
| Regression Baseline | storage/database.py (compare_sessions), schemas.py (BenchmarkSession) | `compare --session-a --session-b` | Partial | N/A | Comparison sheet in Excel | test_storage.py | PARTIAL | No automated regression threshold detection; no pass/fail policy file | LOW | P2 |
| Pass/Fail Policies | benchmarks/workload_profiles.py (timeout_seconds field) | None | None | None | None | None | ROADMAP | No policy YAML/JSON; no configurable thresholds for pass/fail | LOW | P3 |
| Multi-Machine Import | storage/database.py (import not implemented) | None | None | N/A | N/A | None | ROADMAP | No import-from-json or merge-sessions feature | LOW | P3 |
| Batch Execution | app/cli.py (no batch command) | None | None | N/A | N/A | None | ROADMAP | No batch runner or session manifest format | LOW | P3 |
| Company Report Templates | reports/templates/report.html.j2 (1 template) | None | None | N/A | HTML uses single template | test_reports.py | ROADMAP | No company-branded templates; no logo injection; no template selector | LOW | P3 |

---

## Status Summary

| Status | Count | Features |
|--------|-------|----------|
| IMPLEMENTED | 6 | System Inspector, Memory Benchmark (partial schema), Kernel Lab, Diagnosis Engine, History & Comparison, CLI Automation, Report Studio |
| PARTIAL | 7 | PCIe Analyzer, Workload Profiler, Validation Center, CUDA Performance Lab, GPU PCIe Bandwidth Tool, CUDA→HIP Portability Demo, GPU Benchmark Dashboard |
| ROADMAP | 5 | AI Inference Kernel Roadmap, Pass/Fail Policies, Multi-Machine Import, Batch Execution, Company Report Templates |
| DOC_ONLY | 1 | Regression Baseline (compare command exists but no policy thresholds) |
| MISSING | 1 | Regression Baseline policy engine |
| NOT_VALIDATED | 1 | AMD HIP real benchmark (hardware unavailable) |

---

## Key Gaps (Priority 1)

1. `diagnose --latest` flag — only `--session` works today
2. `export --latest` flag — only `--session` works today
3. `demo-report` CLI command — not implemented
4. `MemoryBenchmarkResult` and `KernelBenchmarkResult` schemas missing from benchmarks/schemas.py
5. `--test pcie` alias in CLI dispatch not wired up (maps to memory_bandwidth)
6. `--test vector-add` (hyphen) not handled by native benchmark_main.cu (only `vector_add` underscore)
7. `prefix_sum.cu` and `convolution_2d.cu` skeleton files missing from native/cuda/
8. `reduction_hip.cpp` and `gemm_naive_hip.cpp` missing from native/hip/
9. Sample example JSON files missing (examples/ directory doesn't exist)
10. Structured test files: test_cli_commands.py, test_native_manifest.py, test_interview_demo_readiness.py, test_code_coverage_audit.py — all missing

---

## Interview Demo Readiness Summary

**Works today (no GPU needed):**
- `python -m app.cli system-info` — full system info JSON
- `python -m app.cli quick-test --json --no-save` — CPU baseline + diagnosis
- `python -m app.cli history --json` — session list
- All 5 report formats generate without GPU

**Requires GPU / CUDA:**
- `benchmark --test vector-add/gemm/etc.` — native binary must be compiled
- Native binary is `gpu_insight_benchmark.exe`, target in CMakeLists.txt is confirmed correct

**ROADMAP / NOT_VALIDATED (do NOT claim as implemented):**
- AMD HIP real GPU benchmarks
- softmax, layer_norm, GELU, Flash Attention, INT8 quantization
- PyTorch extension, TensorRT plugin
- cuFFT, cuBLAS full benchmark suite
- Streamlit dashboard
- Parquet storage
- Multi-machine import
- Company report templates
- Batch execution
