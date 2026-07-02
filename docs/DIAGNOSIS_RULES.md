# GPU Insight Lab — Diagnosis Rules Reference

## Overview

The GPU Insight Lab diagnosis engine applies a set of evidence-based rules to each
`BenchmarkSession`. Every finding must include a non-empty `evidence` string — no
rule may produce a diagnosis without supporting data. Rules that lack sufficient data
return `None` (not a low-severity finding; `None` means "cannot evaluate").

This document describes all 9 rules, their required evidence, what they detect, what
they do NOT detect, and the explicit limitations of each rule.

---

## Diagnostic Result Structure

Every finding produced by the engine is a `DiagnosisResult`:

| Field | Type | Description |
|-------|------|-------------|
| `rule_id` | str | Unique identifier, e.g. `"PCIE_BOTTLENECK"` |
| `title` | str | Short human-readable title |
| `severity` | str | `"CRITICAL"`, `"WARNING"`, `"INFO"`, or `"OK"` |
| `category` | str | One of 15 defined categories |
| `description` | str | Explanation of the finding |
| `evidence` | str | Raw data that supports this finding (never empty) |
| `recommendation` | str | Actionable next step |
| `confidence` | str | `"HIGH"`, `"MEDIUM"`, or `"LOW"` |

### Severity Levels

| Severity | Meaning |
|----------|---------|
| CRITICAL | A condition that likely prevents correct or useful GPU operation |
| WARNING | A condition that degrades performance or reliability |
| INFO | A noteworthy observation that requires no immediate action |
| OK | An explicit "no problem detected" confirmation |

---

## Rule Catalogue

### Rule 1: DRIVER_MISSING

**Category**: Environment  
**Trigger**: `nvidia_info` is None or `nvidia_info.driver_version` is empty  
**Severity if triggered**: CRITICAL

**What it detects**: No NVIDIA driver is installed or the driver cannot be queried via
pynvml or nvidia-smi.

**Evidence required**: The rule checks `nvidia_info.error` (non-empty means collection
failed) and the driver version string.

**What it does NOT detect**:
- Whether the driver is compatible with the installed CUDA toolkit
- Whether the driver is the correct version for a specific workload
- Driver issues on AMD GPUs (handled by `AMD_NOT_VALIDATED` rule)

**Limitations**:
- If the system has both an NVIDIA and AMD GPU and only the AMD driver is installed,
  this rule fires for the NVIDIA side even if the AMD GPU is functional.
- On Windows, driver detection via pynvml requires the NVIDIA driver service to be
  running. A freshly installed driver that requires a reboot will appear as missing.

---

### Rule 2: PCIE_BOTTLENECK

**Category**: PCIe / Memory  
**Trigger**: `pcie_info.current_width` < `pcie_info.max_width` by more than 1 lane, OR
             `pcie_info.current_gen` < `pcie_info.max_gen`  
**Severity if triggered**: WARNING

**What it detects**: The GPU is operating at a lower PCIe link width or generation than
its maximum capability. This reduces host-device transfer bandwidth.

**Evidence required**: Both `current_width` and `max_width` must be non-zero. The rule
does not fire if either value is 0 or unknown.

**Quantitative threshold**: PCIe bandwidth loss > 25% compared to maximum triggers WARNING.

**What it does NOT detect**:
- Whether the PCIe slot is electrically damaged
- Whether the bottleneck is on the CPU side (PCIe lanes from CPU vs. chipset)
- Whether the lower link width actually limits the current workload (compute-bound
  workloads are not PCIe-limited)

**Limitations**:
- A GPU running at PCIe 4.0 x8 in a PCIe 4.0 x16 slot will trigger this rule.
  PCIe 4.0 x8 provides 16 GB/s, which is sufficient for most inference workloads.
  Low link width does not always cause measurable performance loss.
- Some motherboards route secondary PCIe slots through the chipset with limited lane
  counts by design. This is not a defect.
- nvidia-smi reports link width from the GPU perspective. The slot's physical wiring
  (x16 physical, x4 electrical) is not always distinguishable.

---

### Rule 3: THERMAL_THROTTLE

**Category**: Thermal / Power  
**Trigger**: `nvidia_info.temperature_c >= 83` (WARNING) or `>= 90` (CRITICAL), OR
             `nvidia_info.throttle_reasons` contains non-idle reason codes  
**Severity if triggered**: WARNING or CRITICAL

**What it detects**: GPU core temperature is at or above the thermal throttle threshold,
or nvidia-smi reports active throttle reason bitmasks.

**Evidence required**: `nvidia_info.temperature_c` must be > 0. A temperature of 0 is
treated as "unknown" (sensor not available), not "cool."

**What it does NOT detect**:
- Whether the throttle is transient (single peak) or sustained
- Whether cooling is inadequate vs. workload being intentionally intense
- Memory junction temperature (GDDR6X Tjmax) — only core temperature is checked

**Limitations**:
- Temperature thresholds are approximate. Different GPU models have different Tjmax
  values. An RTX 3090 Ti has Tjmax of 93°C; an A100 SXM has different thermal
  characteristics entirely.
- A GPU at 82°C is not throttling but is one degree from this rule's threshold.
  The rule does not predict future throttling.
- **Low GPU utilization does NOT mean the GPU is thermally healthy.** A GPU at 30%
  utilization could still be thermally throttled if the workload is memory-intensive
  and the GPU is in a poorly ventilated case.

---

### Rule 4: LOW_MEMORY_BANDWIDTH

**Category**: Performance  
**Trigger**: `memory_bandwidth_gbs < expected_min_bandwidth_gbs` for the detected GPU  
**Severity if triggered**: WARNING

**What it detects**: Measured memory bandwidth is significantly below the theoretical
peak for the GPU model.

**Evidence required**: A `memory_bandwidth` benchmark result with `correctness_verified`
or at least a non-None measurement. Expected bandwidth is looked up from a built-in table
keyed by GPU model string.

**Expected bandwidth table (approximate)**:

| GPU Family | Expected Min (GB/s) |
|------------|---------------------|
| RTX 30xx (consumer) | 400 |
| RTX 40xx (consumer) | 700 |
| A100 SXM | 1800 |
| H100 SXM | 3200 |
| Unknown | Rule does not fire |

**What it does NOT detect**:
- Whether low bandwidth is caused by ECC being enabled (ECC reduces effective bandwidth ~10%)
- Whether the GPU has XMP/overclocked GDDR6 vs. stock
- Memory controller errors (would require more specialized tools)

**Limitations**:
- If the GPU model is not in the lookup table, this rule returns `None` (no finding).
  Unknown GPU models are not penalized with a false WARNING.
- Bandwidth tests measure streaming access patterns. Real workloads with irregular
  access patterns (hash tables, graph algorithms) will see much lower effective
  bandwidth regardless of this rule's outcome.

---

### Rule 5: LOW_COMPUTE_THROUGHPUT

**Category**: Performance  
**Trigger**: GEMM tiled benchmark shows throughput < expected minimum GFLOP/s for the GPU  
**Severity if triggered**: WARNING

**What it detects**: FP32 compute throughput is below expected for the GPU model, suggesting
SM utilization issues, clock throttling, or incorrect kernel configuration.

**Evidence required**: `gemm_tiled` benchmark result with `gflops` field populated.

**What it does NOT detect**:
- Tensor Core utilization (FP16/BF16 throughput — not benchmarked in v0.1.0)
- Whether the GEMM problem size is large enough to saturate the GPU
- INT8 inference throughput

**Limitations**:
- **Low occupancy does NOT necessarily mean low performance.** A kernel with low
  theoretical occupancy may still achieve high throughput if it is limited by
  instruction-level parallelism rather than warp count. This rule should not be
  interpreted as "you need higher occupancy."
- The hand-written tiled GEMM kernel is not as optimized as cuBLAS. Comparing its
  performance to cuBLAS benchmarks in other tools will show lower numbers. This is
  expected and does not indicate a GPU problem.
- Small matrix sizes (< 512×512) under-saturate the GPU and will show lower GFLOP/s
  regardless of GPU quality.

---

### Rule 6: CORRECTNESS_FAILURE

**Category**: Kernel Correctness  
**Trigger**: Any `BenchmarkResult.correctness_verified == False`  
**Severity if triggered**: CRITICAL

**What it detects**: A benchmark kernel produced output that differs from the CPU reference
beyond the defined tolerance.

**Evidence required**: `correctness_error` field from the failing BenchmarkResult, including
the benchmark name and the maximum observed error.

**What it does NOT detect**:
- The root cause of the correctness error (driver bug, memory error, race condition,
  wrong kernel logic)
- Whether the error is deterministic or intermittent
- ECC-correctable errors (these are transparent to the application)

**Limitations**:
- Floating-point non-associativity means parallel reductions may legitimately differ
  from the sequential CPU result by small amounts. The tolerance thresholds are set
  to allow for this.
- A correctness failure on one benchmark does not imply all benchmarks are incorrect.
  Each benchmark is checked independently.
- GPU memory errors that cause bit flips in unchecked memory regions will not be
  caught by this rule unless they affect the benchmark output arrays.

---

### Rule 7: HIGH_VARIANCE

**Category**: Performance Consistency  
**Trigger**: Any benchmark has coefficient of variation (CV) > 10%  
**Severity if triggered**: WARNING

**What it detects**: Benchmark results show high run-to-run variability, which indicates
the GPU is not in a stable operating state.

**Evidence required**: `raw_measurements` list from the BenchmarkResult with at least
5 elements to compute meaningful CV.

**Common causes**:
- Active thermal throttling during the benchmark run
- Power limit reached intermittently
- Background system processes interfering (other GPU workloads, PCIe DMA transfers)
- GPU boost clock fluctuating between measurements

**What it does NOT detect**:
- Whether the variability will affect user-facing application latency
- Whether the system is experiencing hardware instability (vs. OS scheduling noise)

**Limitations**:
- Host-device transfer benchmarks naturally have higher variance (5–8%) due to
  OS page locking and PCIe scheduling. The 10% threshold is intentionally conservative
  to avoid false positives on PCIe benchmarks.
- Short benchmarks (< 1 ms per iteration) are more susceptible to timing resolution
  noise and may show higher CV without a real GPU performance issue.

---

### Rule 8: CUDA_TOOLKIT_MISSING

**Category**: Environment  
**Trigger**: `cuda_info.nvcc_version` is empty AND `cuda_info.runtime_version` is None  
**Severity if triggered**: INFO (toolkit present but driver mismatch) or WARNING (no toolkit at all)

**What it detects**: The CUDA toolkit (nvcc compiler) is not installed or not on PATH.
Without the toolkit, native CUDA kernels cannot be compiled, and some profiling tools
may not function.

**Evidence required**: The `cuda_info.error` field and the nvcc detection result.

**What it does NOT detect**:
- Whether the driver-toolkit version mismatch will cause runtime failures (driver
  forward compatibility is version-dependent)
- Whether CUDA runtime is available via a precompiled binary (which does not need nvcc)

**Limitations**:
- The native benchmark binary (`gpu_insight_benchmark.exe`) ships precompiled and
  does not require nvcc to run. This rule fires when nvcc is absent, but the
  native benchmarks may still run if the precompiled binary is present.
- CUDA Home detection uses environment variables (`CUDA_HOME`, `CUDA_PATH`) and
  common installation paths. Non-standard installations may not be detected.

---

### Rule 9: AMD_NOT_VALIDATED

**Category**: Compatibility  
**Trigger**: `amd_info` is not None (AMD GPU detected)  
**Severity if triggered**: INFO

**What it detects**: An AMD GPU has been detected. GPU Insight Lab's kernel correctness
and performance baselines have not been validated on AMD/ROCm hardware.

**Evidence required**: `amd_info.device_name` and `amd_info.validation_status == "NOT_VALIDATED"`.

**This rule always fires when an AMD GPU is present.** It is not a performance finding;
it is an explicit transparency notice.

**What it does NOT detect**:
- Whether the AMD GPU is functional
- Whether ROCm is installed correctly
- AMD-specific performance issues

**Limitations**:
- This is not a diagnosis of a problem. AMD GPUs may be perfectly functional; GPU Insight Lab
  simply cannot validate that claim in v0.1.0.
- Future versions (v0.2.0 roadmap) plan to add HIP-validated kernel equivalents with
  proper AMD reference baselines.

---

## Rules That Are Intentionally NOT Implemented

The following diagnoses are commonly requested but deliberately excluded from v0.1.0:

### "GPU utilization is too low"

**Reason not implemented**: Low GPU utilization is NOT a GPU problem. It means the workload
is not GPU-bound. Possible causes include:
- CPU-GPU data transfer bottleneck (PCIe-limited)
- CPU preprocessing being too slow to feed the GPU
- Small batch sizes in ML inference
- The application is not designed to saturate the GPU

Flagging low GPU utilization as a GPU diagnosis would be misleading and generate
false alarms for correctly-behaving systems under light load.

### "Occupancy is too low"

**Reason not implemented**: Occupancy (active warps / maximum warps per SM) is a resource
utilization metric, not a performance metric. Many high-performance kernels intentionally
use large register footprints that reduce occupancy because memory-latency hiding
requires fewer warps than theoretical maximum. Roofline analysis or achieved vs. peak
bandwidth comparison is a more meaningful performance indicator.

### "Memory usage is too high"

**Reason not implemented**: Memory usage depends entirely on the workload. GPU Insight Lab
cannot know what "too high" means without knowledge of the application's requirements.

---

## Evidence Policy

Every `DiagnosisResult` in GPU Insight Lab must satisfy:

1. `evidence` is a non-empty string
2. `evidence` contains at least one specific measured value (e.g., a temperature reading,
   a bandwidth number, a version string)
3. Rules that cannot obtain sufficient data return `None`, not a low-confidence finding
4. No rule fires based on absence of data alone (exception: DRIVER_MISSING and
   CUDA_TOOLKIT_MISSING, where absence of a required component is itself the finding)

This policy prevents the diagnosis engine from generating spurious warnings on systems
where data collection failed for unrelated reasons (e.g., permissions, missing tools).
