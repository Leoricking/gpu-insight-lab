# GPU Insight Lab - Package for distribution

Write-Host "[GPU Insight Lab] Packaging..." -ForegroundColor Cyan
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputZip = Join-Path $ProjectRoot "output\gpu_insight_lab_$Timestamp.zip"

New-Item -ItemType Directory -Path (Join-Path $ProjectRoot "output") -Force | Out-Null

$Exclude = @("__pycache__", "*.pyc", ".venv", "venv", "build", "*.sqlite", ".pytest_cache", "*.egg-info")
Compress-Archive -Path $ProjectRoot -DestinationPath $OutputZip -Force

Write-Host "Package created: $OutputZip" -ForegroundColor Green
