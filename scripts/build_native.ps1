# GPU Insight Lab - Build Native Benchmark
# PowerShell script to build gpu_insight_benchmark.exe

Write-Host "[GPU Insight Lab] Building native benchmark..." -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$NativeDir = Join-Path $ProjectRoot "native"
$BuildDir = Join-Path $ProjectRoot "build"
$BinDir = Join-Path $ProjectRoot "bin"

# Check cmake
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Error "cmake not found. Install CMake from https://cmake.org"
    exit 1
}

# Check nvcc
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    Write-Error "nvcc not found. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
    exit 1
}

Write-Host "  cmake: $(cmake --version | Select-Object -First 1)" -ForegroundColor Green
Write-Host "  nvcc: $(nvcc --version | Select-Object -Last 1)" -ForegroundColor Green

# Create build directory
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Run cmake configure
Write-Host "[GPU Insight Lab] Configuring with cmake..." -ForegroundColor Cyan
Push-Location $BuildDir
cmake $NativeDir -G "Visual Studio 17 2022" -A x64 2>&1
if ($LASTEXITCODE -ne 0) {
    # Try Ninja as fallback
    Write-Host "  Visual Studio generator failed; trying Ninja..." -ForegroundColor Yellow
    cmake $NativeDir -G Ninja 2>&1
}

if ($LASTEXITCODE -ne 0) {
    Write-Error "cmake configure failed"
    Pop-Location
    exit 1
}

# Build
Write-Host "[GPU Insight Lab] Building (Release)..." -ForegroundColor Cyan
cmake --build . --config Release 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed"
    Pop-Location
    exit 1
}
Pop-Location

# Copy to bin/
if (-not (Test-Path $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir | Out-Null
}

$ExePaths = @(
    (Join-Path $BuildDir "bin\gpu_insight_benchmark.exe"),
    (Join-Path $BuildDir "Release\gpu_insight_benchmark.exe"),
    (Join-Path $BuildDir "gpu_insight_benchmark.exe")
)
$Copied = $false
foreach ($ExePath in $ExePaths) {
    if (Test-Path $ExePath) {
        Copy-Item $ExePath (Join-Path $BinDir "gpu_insight_benchmark.exe") -Force
        Write-Host "[GPU Insight Lab] Executable copied to bin/" -ForegroundColor Green
        $Copied = $true
        break
    }
}

if (-not $Copied) {
    Write-Warning "Executable not found after build. Check $BuildDir for output."
    exit 2
}

Write-Host "[GPU Insight Lab] Native build complete: $BinDir\gpu_insight_benchmark.exe" -ForegroundColor Green
