# GPU Insight Lab - Run CLI quick-test

Write-Host "[GPU Insight Lab] Running CLI quick-test..." -ForegroundColor Cyan
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
python -m app.cli quick-test
$ExitCode = $LASTEXITCODE
Pop-Location
exit $ExitCode
