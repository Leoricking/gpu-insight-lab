"""Tests that the interview demo package is complete and consistent."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_demo_guide_exists():
    assert (ROOT / "docs" / "DEMO_GUIDE.md").exists()


def test_demo_script_exists():
    assert (ROOT / "scripts" / "run_interview_demo.ps1").exists()


def test_demo_manifest_exists():
    assert (ROOT / "examples" / "demo_manifest.json").exists()


def test_readme_contains_quick_interview_demo():
    content = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Quick Interview Demo" in content


def test_readme_zh_contains_interview_demo():
    content = (ROOT / "README_zh-TW.md").read_text(encoding="utf-8")
    assert "面試快速展示" in content


def test_demo_manifest_has_implemented_benchmarks():
    data = json.loads((ROOT / "examples" / "demo_manifest.json").read_text(encoding="utf-8"))
    implemented = data["implemented_benchmarks"]
    for name in ["vector-add", "reduction", "transpose", "gemm", "memory", "streams", "pcie"]:
        assert name in implemented, f"{name} missing from implemented_benchmarks"


def test_demo_manifest_has_roadmap_benchmarks():
    data = json.loads((ROOT / "examples" / "demo_manifest.json").read_text(encoding="utf-8"))
    roadmap_names = [b["name"] for b in data["roadmap_benchmarks"]]
    for name in ["softmax", "layer_norm", "gelu", "flash_attention"]:
        assert name in roadmap_names, f"{name} missing from roadmap_benchmarks"


def test_demo_manifest_has_interview_target_roles():
    data = json.loads((ROOT / "examples" / "demo_manifest.json").read_text(encoding="utf-8"))
    assert len(data["interview_target_roles"]) >= 3


def test_roadmap_benchmarks_not_in_implemented():
    data = json.loads((ROOT / "examples" / "demo_manifest.json").read_text(encoding="utf-8"))
    implemented = set(data["implemented_benchmarks"])
    roadmap_names = {b["name"] for b in data["roadmap_benchmarks"]}
    overlap = implemented & roadmap_names
    assert not overlap, f"Roadmap benchmarks incorrectly listed as implemented: {overlap}"


def test_cli_demo_report_command():
    result = subprocess.run(
        [sys.executable, "-m", "app.cli", "demo-report"],
        capture_output=True, text=True,
        cwd=str(ROOT)
    )
    assert result.returncode == 0, f"demo-report failed:\n{result.stderr}"
    assert "GPU Insight Lab" in result.stdout or "GPU Insight Lab" in result.stderr


def test_demo_guide_contains_core_positioning():
    content = (ROOT / "docs" / "DEMO_GUIDE.md").read_text(encoding="utf-8")
    assert "reproducible GPU engineering workflow" in content


def test_demo_guide_contains_nvidia_talking_points():
    content = (ROOT / "docs" / "DEMO_GUIDE.md").read_text(encoding="utf-8")
    assert "NVIDIA" in content


def test_demo_guide_contains_amd_talking_points():
    content = (ROOT / "docs" / "DEMO_GUIDE.md").read_text(encoding="utf-8")
    assert "AMD" in content
