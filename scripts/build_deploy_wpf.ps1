<#
.SYNOPSIS
  组装机台交付包：WPF 宿主 + python_core +（可选）Python 运行时 + checkpoints + YOLO。

.DESCRIPTION
  输出目录: dist/缺陷分类系统/
  推荐在已能运行开发版的机器上执行。

.PARAMETER PythonHome
  Conda 环境根目录（如 D:\Software\MiniAnaconda\envs\cv-yolo）。
  若省略，则使用环境变量 DIAMOND_PYTHON_HOME / CONDA_PREFIX。

.PARAMETER SkipPublish
  跳过 dotnet publish（复用已有 dist\_wpf_publish）。

.PARAMETER SkipPythonRuntime
  不复制 Python 运行时（机台需自备并设置 DIAMOND_PYTHON_HOME）。

.PARAMETER YoloPath
  YOLO 权重源路径；默认读取仓库 app_config.json 的 yolo_path，或 detect_weights/best.pt。

.PARAMETER OutDir
  组装输出目录。默认 dist/缺陷分类系统/

.EXAMPLE
  .\scripts\build_deploy_wpf.ps1 -PythonHome "D:\Software\MiniAnaconda\envs\cv-yolo"
#>
param(
    [string]$PythonHome = "",
    [switch]$SkipPublish,
    [switch]$SkipPythonRuntime,
    [string]$YoloPath = "",
    [string]$OutDir = "",
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "DiamondDetect.sln"))) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Resolve-PythonHome([string]$hint) {
    if ($hint -and (Test-Path $hint)) { return (Resolve-Path $hint).Path }
    foreach ($envName in @("DIAMOND_PYTHON_HOME", "CONDA_PREFIX")) {
        $v = [Environment]::GetEnvironmentVariable($envName)
        if ($v -and (Test-Path $v)) { return (Resolve-Path $v).Path }
    }
    $fallback = "D:\Software\MiniAnaconda\envs\cv-yolo"
    if (Test-Path $fallback) { return $fallback }
    return $null
}

function Copy-DirContents([string]$src, [string]$dst) {
    New-Item -ItemType Directory -Path $dst -Force | Out-Null
    Copy-Item -Path (Join-Path $src "*") -Destination $dst -Recurse -Force
}

$PublishDir = Join-Path $Root "dist\_wpf_publish"
$DistRoot = if ($OutDir) { $OutDir } else { Join-Path $Root "dist\缺陷分类系统" }
$CkptSrc = Join-Path $Root "checkpoints"
$PythonCoreSrc = Join-Path $Root "python_core"

if (-not $SkipPublish) {
    & (Join-Path $PSScriptRoot "publish_wpf.ps1") -Configuration $Configuration
}

if (-not (Test-Path (Join-Path $PublishDir "DiamondDetect.exe"))) {
    throw "未找到发布产物 DiamondDetect.exe，请先运行 publish_wpf.ps1"
}

$onnx = Join-Path $CkptSrc "model.onnx"
if (-not (Test-Path $onnx)) {
    throw "缺少 checkpoints/model.onnx，请先准备机台分类模型"
}

# YOLO 源
$cfgPath = Join-Path $Root "app_config.json"
if (-not $YoloPath -and (Test-Path $cfgPath)) {
    try {
        $cfg = Get-Content $cfgPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($cfg.yolo_path) { $YoloPath = [string]$cfg.yolo_path }
    } catch { }
}
if (-not $YoloPath) {
    $cand = Join-Path $Root "detect_weights\best.pt"
    if (Test-Path $cand) { $YoloPath = $cand }
}
if (-not $YoloPath -or -not (Test-Path $YoloPath)) {
    Write-Warning "未找到 YOLO 权重，将写入配置 detect_weights/best.pt，请稍后手动放入。"
    $YoloPath = ""
}

Write-Host "==> 组装 $DistRoot" -ForegroundColor Cyan
if (Test-Path $DistRoot) { Remove-Item $DistRoot -Recurse -Force }
New-Item -ItemType Directory -Path $DistRoot -Force | Out-Null

# 1) WPF 宿主
Copy-DirContents $PublishDir $DistRoot

# 2) python_core
$pcDst = Join-Path $DistRoot "python_core"
Copy-Item $PythonCoreSrc $pcDst -Recurse -Force
Get-ChildItem $pcDst -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# 3) checkpoints
$ckptDst = Join-Path $DistRoot "checkpoints"
New-Item -ItemType Directory -Path $ckptDst -Force | Out-Null
foreach ($name in @(
    "model.onnx", "model.onnx.data", "class_map.json",
    "train_config.json"
)) {
    $f = Join-Path $CkptSrc $name
    if (Test-Path $f) { Copy-Item $f $ckptDst -Force }
}

# 4) YOLO
$detDst = Join-Path $DistRoot "detect_weights"
New-Item -ItemType Directory -Path $detDst -Force | Out-Null
if ($YoloPath) {
    Copy-Item $YoloPath (Join-Path $detDst "best.pt") -Force
    Write-Host "YOLO -> detect_weights/best.pt"
}

# 5) 机台 app_config.json（相对路径）
$deployCfg = [ordered]@{
    pt_path                 = ""
    onnx_path               = "checkpoints/model.onnx"
    data_dir                = "data"
    corrections_dir         = "corrections"
    use_gpu                 = $true
    enable_local_diagnostics = $false
    yolo_path               = "detect_weights/best.pt"
    sahi_device             = "auto"
    sahi_slice_size         = 1280
    sahi_overlap            = 0.2
    sahi_det_conf           = 0.35
    sahi_batch_size         = 16
    sahi_crop_padding       = 15
    sahi_output_dir         = "sahi_output"
    sahi_ios_thresh         = 0.6
    sahi_min_area_ratio     = 0.45
    sahi_max_aspect_ratio   = 1.5
    sahi_edge_filter        = $true
    sahi_edge_margin_px     = 20
}
$deployCfg | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $DistRoot "app_config.json") -Encoding UTF8

# 6) Python 运行时（可选复制）
$pyHome = Resolve-PythonHome $PythonHome
$runtimeNote = ""
if (-not $SkipPythonRuntime) {
    if (-not $pyHome) {
        throw "未指定 PythonHome。请传 -PythonHome 或设置 DIAMOND_PYTHON_HOME，或加 -SkipPythonRuntime"
    }
    Write-Host "==> 复制 Python 运行时（可能较大）: $pyHome" -ForegroundColor Yellow
    $rtDst = Join-Path $DistRoot "python_runtime"
    # 使用 robocopy 排除部分缓存，加快并可续传
    $excludeDirs = @("__pycache__", ".git", "conda-meta")
    New-Item -ItemType Directory -Path $rtDst -Force | Out-Null
    $xd = @()
    foreach ($d in $excludeDirs) { $xd += @("/XD", $d) }
    & robocopy $pyHome $rtDst /E /MT:8 /NFL /NDL /NJH /NJS /nc /ns /np /XD __pycache__ .git @xd | Out-Null
    # robocopy exit codes 0-7 are success-ish
    if ($LASTEXITCODE -ge 8) { throw "robocopy python_runtime failed: $LASTEXITCODE" }
    $runtimeNote = "包内 python_runtime：分类=ONNX（有 NVIDIA 则 CUDA 快速模式，无独显回退 CPU）；检测=YOLO/torch CUDA 13.x。机台 GPU 需已装 NVIDIA 驱动（CUDA 13.2 兼容）。"
    Write-Host "Python runtime copied." -ForegroundColor Green

    $siteCustom = Join-Path $rtDst "Lib\site-packages\sitecustomize.py"
    $siteBody = @'
# Injected by build_deploy_wpf.ps1 — relocatable CUDA/torch DLL search.
import os
from pathlib import Path

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
if os.environ.get("DEFECTS_DEPLOY") == "1":
    os.environ.pop("CUDA_PATH", None)
    os.environ.pop("CUDA_HOME", None)

home = Path(os.environ.get("PYTHONHOME") or Path(__file__).resolve().parents[2])
dirs = []
torch_lib = home / "Lib" / "site-packages" / "torch" / "lib"
if torch_lib.is_dir():
    dirs.append(torch_lib)
nvidia = home / "Lib" / "site-packages" / "nvidia"
if nvidia.is_dir():
    for bin_dir in nvidia.glob("*/bin"):
        if bin_dir.parent.name.lower() == "cudnn":
            continue
        dirs.append(bin_dir)
libbin = home / "Library" / "bin"
if libbin.is_dir():
    dirs.append(libbin)
for d in dirs:
    p = str(d)
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(p)
        except OSError:
            pass
if dirs:
    os.environ["PATH"] = os.pathsep.join(str(d) for d in dirs) + os.pathsep + os.environ.get("PATH", "")
'@
    Set-Content -Path $siteCustom -Value $siteBody -Encoding UTF8
    Write-Host "Injected sitecustomize.py for CUDA DLL paths."
} else {
    $runtimeNote = "未打包运行时，机台需自备 Conda 并设置 DIAMOND_PYTHON_HOME"
    Write-Host $runtimeNote -ForegroundColor Yellow
}

# 7) 启动脚本：指向包内 python_runtime，并前置 CUDA/torch DLL 目录
$bat = @"
@echo off
cd /d "%~dp0"
set DEFECTS_DEPLOY=1
set CUDA_MODULE_LOADING=LAZY
set CUDA_PATH=
set CUDA_HOME=
if exist "%~dp0python_runtime\python.exe" (
  set DIAMOND_PYTHON_HOME=%~dp0python_runtime
  set PYTHONHOME=%~dp0python_runtime
  set PATH=%~dp0python_runtime;%~dp0python_runtime\Scripts;%~dp0python_runtime\Library\bin;%~dp0python_runtime\Lib\site-packages\torch\lib;%PATH%
)
if exist "%~dp0python_runtime\python312.dll" (
  set DIAMOND_PYTHON_DLL=%~dp0python_runtime\python312.dll
)
start "" /D "%~dp0" "%~dp0DiamondDetect.exe"
"@
Set-Content -Path (Join-Path $DistRoot "启动缺陷分类系统.bat") -Value $bat -Encoding ASCII

$verifyBat = @"
@echo off
cd /d "%~dp0"
set DEFECTS_DEPLOY=1
set DEFECTS_VERIFY=1
set CUDA_MODULE_LOADING=LAZY
set CUDA_PATH=
set CUDA_HOME=
if exist "%~dp0python_runtime\python.exe" (
  set DIAMOND_PYTHON_HOME=%~dp0python_runtime
  set PYTHONHOME=%~dp0python_runtime
  set PATH=%~dp0python_runtime;%~dp0python_runtime\Scripts;%~dp0python_runtime\Library\bin;%~dp0python_runtime\Lib\site-packages\torch\lib;%PATH%
)
if exist "%~dp0python_runtime\python312.dll" set DIAMOND_PYTHON_DLL=%~dp0python_runtime\python312.dll
DiamondDetect.exe --verify
echo exit=%ERRORLEVEL%
pause
"@
Set-Content -Path (Join-Path $DistRoot "验收_verify.bat") -Value $verifyBat -Encoding ASCII

$readme = @"
钻石缺陷图像分类系统（WPF 机台包）
================================
$runtimeNote

目录要点:
  DiamondDetect.exe     WPF 宿主
  启动缺陷分类系统.bat  机台模式启动（指向 python_runtime）
  验收_verify.bat       无界面验收模型加载
  python_core/          算法模块（勿删）
  python_runtime/       Python + torch/CUDA + ultralytics（完整包）
  checkpoints/          分类 ONNX（可热更新）
  detect_weights/       YOLO 权重 best.pt
  app_config.json       机台相对路径配置

机台要求: 已安装 NVIDIA 显卡驱动（CUDA 运行时已打进 python_runtime，无需再装 CUDA Toolkit）。
必须整包复制到机台，勿只拷 exe。
"@
Set-Content -Path (Join-Path $DistRoot "README_机台.txt") -Value $readme -Encoding UTF8

Write-Host ""
Write-Host "组装完成: $DistRoot" -ForegroundColor Green
Write-Host "请运行 验收_verify.bat 或: `$env:DEFECTS_VERIFY=1; .\DiamondDetect.exe --verify"
