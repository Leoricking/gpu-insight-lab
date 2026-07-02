# GPU Insight Lab — Commercial Roadmap

## Current Status: v0.1.0 — Free / Open Source

GPU Insight Lab v0.1.0 is fully open source under the MIT License. There is **no payment,
no license key, no account creation, and no feature gating** in v0.1.0. Every feature in
the `app/features.py` registry is enabled for all users.

This document describes the planned edition structure for future releases. Nothing in this
document represents a binding commitment. Roadmap items are subject to change based on
user feedback, hardware availability, and engineering priorities.

---

## Edition Structure (Planned for v1.0.0)

### Free Edition

**Target users**: Individual developers, students, hobbyists, GPU enthusiasts.

| Feature | Free |
|---------|------|
| System info collection | Yes |
| GPU info (pynvml / nvidia-smi) | Yes |
| CUDA / ROCm toolchain detection | Yes |
| PCIe info | Yes |
| CPU baselines (3 benchmarks) | Yes |
| Native CUDA benchmarks (quick mode) | Yes |
| Native CUDA benchmarks (full mode) | No |
| Diagnosis engine (5 rules) | Yes |
| Diagnosis engine (all 9 rules) | No |
| Session history (last 5 sessions) | Yes |
| Session history (unlimited) | No |
| JSON report | Yes |
| CSV report | Yes |
| Markdown report | Yes |
| HTML report | Yes |
| Excel report | No |
| GUI (PySide6) | Yes |
| CLI | Yes |
| Multi-GPU support | No |
| CI/CD integration mode | No |

### Pro Edition (Planned)

**Target users**: Professional ML engineers, performance engineers, QA teams.  
**Pricing model**: Annual subscription per user (pricing TBD).

All Free features, plus:

| Feature | Pro |
|---------|-----|
| Native CUDA benchmarks (full mode) | Yes |
| All 9 diagnosis rules | Yes |
| Unlimited session history | Yes |
| Session comparison (delta analysis) | Yes |
| Excel report (7 sheets) | Yes |
| PDF report | Yes |
| CI/CD integration mode (`--ci` flag, machine-readable exit codes) | Yes |
| Multi-GPU support (up to 4 GPUs) | Yes |
| Scheduled benchmark runs (cron) | Yes |
| Slack / Teams / webhook notifications | Yes |
| Custom diagnosis rule configuration | Yes |

### Lab Edition (Planned)

**Target users**: GPU validation teams, data center operators, hardware QA labs.  
**Pricing model**: Per-seat or site license (pricing TBD).

All Pro features, plus:

| Feature | Lab |
|---------|-----|
| Unlimited multi-GPU | Yes |
| Long-duration stress testing (hours) | Yes |
| Thermal cycling test sequences | Yes |
| ECC error monitoring and alerting | Yes |
| PCIe error rate monitoring | Yes |
| Custom workload plugin API | Yes |
| Central results server (self-hosted) | Yes |
| REST API for dashboard integration | Yes |
| Audit log for all benchmark runs | Yes |
| FIPS-compliant result signing | Yes |
| Priority support SLA | Yes |

---

## Versioning Plan

| Version | Target | Key Deliverables |
|---------|--------|-----------------|
| v0.1.0 | Current | Full NVIDIA CUDA MVP, Free/MIT, all features enabled |
| v0.2.0 | Q3 2026 | AMD ROCm validation, HIP kernel correctness, wavefront support |
| v0.3.0 | Q4 2026 | Nsight integration improvements, Roofline model visualization |
| v0.4.0 | Q1 2027 | Multi-GPU support, topology-aware benchmarks (NVLink, PCIe topology) |
| v1.0.0 | Q2 2027 | Edition gating, Pro/Lab commercial launch, stability guarantee |

---

## Feature Gating Design (for v1.0.0)

When edition gating is introduced, it will be implemented as:

1. A license key file stored in `~/.gpu_insight_lab/license.key`
2. The `app/features.py` `is_enabled()` function will check the edition against the
   feature's required edition
3. All existing v0.1.0 functionality will remain free
4. No telemetry, no call-home, no internet connection required for Free edition
5. License validation for Pro/Lab will use offline HMAC verification (no server required
   for normal operation; only needed for initial activation)

The feature registry (`app/features.py`) already contains all feature IDs. In v0.1.0
every feature returns `enabled=True`. The structure is in place for edition gating without
requiring architectural changes.

---

## What Will Never Be Paywalled

Regardless of edition structure, the following will remain free permanently:

- System information collection (CPU, RAM, OS)
- GPU detection and basic info (name, VRAM, driver version)
- CUDA/ROCm toolchain detection
- JSON and CSV report formats
- CLI (`gpu-insight` command)
- SQLite session storage
- The native benchmark binary (basic quick mode)
- All open-source code on GitHub

---

## Distribution Model

### v0.1.0 (Current)
- Source code on GitHub (MIT License)
- `pip install gpu-insight-lab[all]`
- No binary distribution

### v1.0.0 and Later (Planned)
- Free edition: `pip install gpu-insight-lab` (same as today)
- Pro/Lab edition: Separate `pip install gpu-insight-lab-pro` with license activation
- Windows installer (.exe) for users who prefer not to use pip
- Docker image for CI/CD integration

---

## Support Policy

| Edition | Support Channel | Response SLA |
|---------|----------------|-------------|
| Free | GitHub Issues | Best-effort, no SLA |
| Pro | Email support | 2 business days |
| Lab | Dedicated Slack + Email | 4 business hours |

---

## Notes

- This roadmap reflects planning as of v0.1.0 (July 2026). It is subject to change.
- No payment processing exists in the codebase. No payment will be collected before v1.0.0.
- "GPU Insight Lab" is an unregistered trademark. See `app/branding.py` for the full
  trademark notice policy.
