<#
.SYNOPSIS
  为 corrections-2 指定类别目录下的文件添加 "rename-" 前缀。

.DESCRIPTION
  目标目录：
    D:\迅雷下载\corrections\corrections-2\面朝上
    D:\迅雷下载\corrections\corrections-2\棱边朝上
  已带 rename- 前缀的文件会跳过，避免重复执行时二次改名。
#>
$ErrorActionPreference = "Stop"

$dirs = @(
    "D:\迅雷下载\corrections\corrections-2\面朝上",
    "D:\迅雷下载\corrections\corrections-2\棱边朝上"
)

$totalRenamed = 0
$totalSkipped = 0
$totalFailed = 0

foreach ($dir in $dirs) {
    if (-not (Test-Path -LiteralPath $dir)) {
        Write-Warning "目录不存在，跳过: $dir"
        continue
    }

    Write-Host ""
    Write-Host "处理: $dir"
    $files = Get-ChildItem -LiteralPath $dir -File
    $renamed = 0
    $skipped = 0

    foreach ($f in $files) {
        if ($f.Name.StartsWith("rename-")) {
            $skipped++
            continue
        }

        $newName = "rename-" + $f.Name
        $dest = Join-Path $f.DirectoryName $newName
        if (Test-Path -LiteralPath $dest) {
            Write-Warning "目标已存在，跳过: $($f.FullName) -> $newName"
            $skipped++
            $totalFailed++
            continue
        }

        try {
            Rename-Item -LiteralPath $f.FullName -NewName $newName
            $renamed++
        }
        catch {
            Write-Warning "改名失败: $($f.Name)  $($_.Exception.Message)"
            $totalFailed++
        }
    }

    Write-Host "  改名 $renamed  跳过 $skipped  共 $($files.Count)"
    $totalRenamed += $renamed
    $totalSkipped += $skipped
}

Write-Host ""
Write-Host "合计: 改名 $totalRenamed  跳过 $totalSkipped  失败/冲突 $totalFailed"
