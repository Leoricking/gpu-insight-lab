# GPU Insight Lab — Interview Demo Script
# Runs all 8 demo steps, records pass/fail, prints summary.
# No interactive prompts. CUDA/HIP/Nsight unavailable is not fatal.

$ErrorActionPreference = "Continue"

# Ensure output directory exists
$OutputDir = Join-Path $PSScriptRoot "..\output"
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$StepNames = @(
    "python -m app.cli --help",
    "python -m app.cli system-info",
    "python -m app.cli quick-test",
    "python -m app.cli benchmark --test memory",
    "python -m app.cli benchmark --test pcie",
    "python -m app.cli benchmark --test gemm",
    "python -m app.cli diagnose --latest",
    "python -m app.cli demo-report"
)

$StepResults = @{}

$TotalSteps = $StepNames.Count

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GPU Insight Lab Interview Demo" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

for ($i = 0; $i -lt $TotalSteps; $i++) {
    $StepNum = $i + 1
    $Cmd = $StepNames[$i]
    Write-Host "[GPU Insight Lab Demo] Step $StepNum/$TotalSteps - $Cmd" -ForegroundColor Yellow
    Write-Host ""

    # Split command into executable + args
    $Parts = $Cmd -split " "
    $Exe = $Parts[0]
    $CmdArgs = $Parts[1..($Parts.Length - 1)]

    & $Exe @CmdArgs
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -eq 0 -or $ExitCode -eq 2) {
        $StepResults[$StepNum] = "PASS"
        Write-Host ""
        Write-Host "  => Step $StepNum PASS (exit code $ExitCode)" -ForegroundColor Green
    } else {
        $StepResults[$StepNum] = "FAIL"
        Write-Host ""
        Write-Host "  => Step $StepNum FAIL (exit code $ExitCode)" -ForegroundColor Red
    }

    Write-Host ""
}

# ── Environment availability check ─────────────────────────────────────────

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Environment Availability Check" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$EnvTools = @{
    "nvidia-smi"   = "nvidia-smi"
    "nvcc"         = "nvcc --version"
    "nsys"         = "nsys --version"
    "ncu"          = "ncu --version"
    "rocminfo"     = "rocminfo"
    "hipcc"        = "hipcc --version"
}

foreach ($Tool in $EnvTools.Keys) {
    $ToolCmd = $EnvTools[$Tool] -split " "
    $ToolExe = $ToolCmd[0]
    $ToolArgs = if ($ToolCmd.Length -gt 1) { $ToolCmd[1..($ToolCmd.Length - 1)] } else { @() }

    $ToolPath = Get-Command $ToolExe -ErrorAction SilentlyContinue
    if ($ToolPath) {
        if ($ToolArgs.Count -gt 0) {
            $VersionOut = & $ToolExe @ToolArgs 2>&1 | Select-Object -First 1
        } else {
            $VersionOut = ""
        }
        Write-Host "  $($Tool.PadRight(16)) AVAILABLE   $VersionOut" -ForegroundColor Green
    } else {
        Write-Host "  $($Tool.PadRight(16)) unavailable" -ForegroundColor DarkGray
    }
}

Write-Host ""

# ── Generated files ─────────────────────────────────────────────────────────

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Generated Report Files" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ReportPatterns = @("*.json", "*.md", "*.html", "*.xlsx")
$AnyReport = $false
foreach ($Pattern in $ReportPatterns) {
    $Files = Get-ChildItem -Path $OutputDir -Filter $Pattern -ErrorAction SilentlyContinue
    foreach ($File in $Files) {
        $SizeKB = [math]::Round($File.Length / 1KB, 1)
        Write-Host "  $($File.Name.PadRight(50)) $SizeKB KB" -ForegroundColor White
        $AnyReport = $true
    }
}
if (-not $AnyReport) {
    Write-Host "  (no report files found in output/)" -ForegroundColor DarkGray
}

Write-Host ""

# ── Summary table ──────────────────────────────────────────────────────────

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Demo Step Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Step  Command                                    Result" -ForegroundColor White
Write-Host "  ----  -----------------------------------------  ------" -ForegroundColor DarkGray

$PassCount = 0
$FailCount = 0

for ($i = 0; $i -lt $TotalSteps; $i++) {
    $StepNum = $i + 1
    $Cmd = $StepNames[$i]
    $Result = $StepResults[$StepNum]
    $CmdDisplay = $Cmd.PadRight(45)
    if ($Result -eq "PASS") {
        Write-Host "  $($StepNum.ToString().PadLeft(4))  $CmdDisplay  PASS" -ForegroundColor Green
        $PassCount++
    } else {
        Write-Host "  $($StepNum.ToString().PadLeft(4))  $CmdDisplay  FAIL" -ForegroundColor Red
        $FailCount++
    }
}

Write-Host ""
Write-Host "  Results: $PassCount/$TotalSteps passed, $FailCount/$TotalSteps failed" -ForegroundColor Cyan
Write-Host ""
Write-Host "GPU Insight Lab Interview Demo Complete" -ForegroundColor Cyan
Write-Host ""
