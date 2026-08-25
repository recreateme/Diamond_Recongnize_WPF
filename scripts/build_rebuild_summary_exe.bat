@echo off
REM 打包 rebuild_summary_from_stats.py 为单文件 exe，并复制到 D:\迅雷下载\ECOA
setlocal
cd /d "%~dp0\.."

set PY=D:\Software\MiniAnaconda\envs\cv-yolo\python.exe
if not exist "%PY%" set PY=python

"%PY%" -m pip install -q pyinstaller
if errorlevel 1 exit /b 1

set OUT=dist_tools
if exist "%OUT%\rebuild_summary_from_stats" rmdir /s /q "%OUT%\rebuild_summary_from_stats"
if exist "%OUT%\rebuild_summary_from_stats.exe" del /f /q "%OUT%\rebuild_summary_from_stats.exe"

"%PY%" -m PyInstaller --noconfirm --clean --onefile --console ^
  --name rebuild_summary_from_stats ^
  --distpath "%OUT%" ^
  --workpath "%OUT%\build" ^
  --specpath "%OUT%" ^
  scripts\rebuild_summary_from_stats.py
if errorlevel 1 exit /b 1

set DEST=D:\迅雷下载\ECOA
if not exist "%DEST%" mkdir "%DEST%"
copy /Y "%OUT%\rebuild_summary_from_stats.exe" "%DEST%\rebuild_summary_from_stats.exe"
echo.
echo EXE: %DEST%\rebuild_summary_from_stats.exe
endlocal
