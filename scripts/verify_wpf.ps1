<#
.SYNOPSIS
  对已组装的机台包或当前开发目录执行无界面验收。
#>
param(
    [string]$PackageDir = "",
    [string]$PythonHome = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "DiamondDetect.sln"))) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

if (-not $PackageDir) {
    $pkg = Join-Path $Root "dist\缺陷分类系统"
    if (Test-Path (Join-Path $pkg "DiamondDetect.exe")) {
        $PackageDir = $pkg
    } else {
        $PackageDir = $Root
    }
}

$exe = Join-Path $PackageDir "DiamondDetect.exe"
if (-not (Test-Path $exe)) {
    # 开发模式：dotnet run
    Write-Host "使用开发工程验收..." -ForegroundColor Cyan
    if ($PythonHome) { $env:DIAMOND_PYTHON_HOME = $PythonHome }
    elseif (Test-Path "D:\Software\MiniAnaconda\envs\cv-yolo") {
        $env:DIAMOND_PYTHON_HOME = "D:\Software\MiniAnaconda\envs\cv-yolo"
    }
    $env:DEFECTS_VERIFY = "1"
    $env:DEFECTS_DEPLOY = "1"
    Push-Location $Root
    try {
        dotnet run --project (Join-Path $Root "src\DiamondDetect.Wpf\DiamondDetect.Wpf.csproj") -c Release --no-build 2>$null
        if ($LASTEXITCODE -ne 0) {
            dotnet run --project (Join-Path $Root "src\DiamondDetect.Wpf\DiamondDetect.Wpf.csproj") -c Release -- --verify
        }
        exit $LASTEXITCODE
    } finally {
        Pop-Location
        Remove-Item Env:DEFECTS_VERIFY -ErrorAction SilentlyContinue
    }
}

Push-Location $PackageDir
try {
    $env:DEFECTS_VERIFY = "1"
    $env:DEFECTS_DEPLOY = "1"
    $rt = Join-Path $PackageDir "python_runtime"
    if (Test-Path (Join-Path $rt "python.exe")) {
        $env:DIAMOND_PYTHON_HOME = $rt
    } elseif ($PythonHome) {
        $env:DIAMOND_PYTHON_HOME = $PythonHome
    }
    Write-Host "Verify: $exe" -ForegroundColor Cyan
    & $exe --verify
    Write-Host "exit=$LASTEXITCODE"
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
