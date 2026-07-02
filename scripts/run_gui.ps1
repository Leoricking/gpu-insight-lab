# GPU Insight Lab - Launch GUI

Write-Host "[GPU Insight Lab] Running GUI..." -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Install Python 3.11+ from https://python.org"
    exit 1
}

$PythonVersion = python --version 2>&1
Write-Host "  Python: $PythonVersion" -ForegroundColor Green

# Check PySide6
python -c "import PySide6" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "PySide6 not installed. Installing..."
    pip install PySide6>=6.6.0
}

Push-Location $ProjectRoot
python -m app.main
$ExitCode = $LASTEXITCODE
Pop-Location
exit $ExitCode
