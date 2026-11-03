@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist "%~dp0venv-ssntagger\Scripts\python.exe" (
    echo Run setup_ssntagger.bat first. 1>&2
    exit /b 1
)
"%~dp0venv-ssntagger\Scripts\python.exe" "%~dp0ssntagger.py" %*
exit /b %errorlevel%
