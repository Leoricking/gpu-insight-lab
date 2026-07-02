# GPU Insight Lab - Clean build artifacts

Write-Host "[GPU Insight Lab] Cleaning build artifacts..." -ForegroundColor Cyan
$ProjectRoot = Split-Path -Parent $PSScriptRoot

$ToRemove = @("build", "__pycache__", ".pytest_cache", "*.egg-info")
foreach ($pattern in $ToRemove) {
    Get-ChildItem -Path $ProjectRoot -Include $pattern -Recurse -Force -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

Get-ChildItem -Path $ProjectRoot -Include "*.pyc" -Recurse -Force -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

Write-Host "Clean complete." -ForegroundColor Green
