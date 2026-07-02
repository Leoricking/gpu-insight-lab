# GPU Insight Lab - Run test suite

Write-Host "[GPU Insight Lab] Running tests..." -ForegroundColor Cyan
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

Write-Host "  Step 1: Compile-check all Python modules..." -ForegroundColor Yellow
python -m compileall app collectors benchmarks diagnosis storage reports profilers workloads tests -q
if ($LASTEXITCODE -ne 0) {
    Write-Error "Compilation check failed"
    Pop-Location
    exit 1
}
Write-Host "  All modules compiled OK" -ForegroundColor Green

Write-Host "  Step 2: Running pytest..." -ForegroundColor Yellow
python -m pytest tests/ -v --tb=short
$ExitCode = $LASTEXITCODE
Pop-Location
exit $ExitCode
