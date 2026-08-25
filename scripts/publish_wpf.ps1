<#
.SYNOPSIS
  发布 WPF 自包含宿主到 dist/_wpf_publish/

.EXAMPLE
  .\scripts\publish_wpf.ps1
  .\scripts\publish_wpf.ps1 -Configuration Release
#>
param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [string]$Runtime = "win-x64"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "DiamondDetect.sln"))) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$Project = Join-Path $Root "src\DiamondDetect.Wpf\DiamondDetect.Wpf.csproj"
$OutDir = Join-Path $Root "dist\_wpf_publish"

Write-Host "==> dotnet publish ($Configuration, $Runtime)" -ForegroundColor Cyan
if (Test-Path $OutDir) { Remove-Item $OutDir -Recurse -Force }
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

dotnet publish $Project `
    -c $Configuration `
    -r $Runtime `
    --self-contained true `
    -p:PublishSingleFile=false `
    -p:IncludeNativeLibrariesForSelfExtract=true `
    -o $OutDir

if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed" }
Write-Host "Published: $OutDir" -ForegroundColor Green
