@echo off
setlocal
if not exist "%~dp0venv-ssntagger\Scripts\python.exe" (
    echo Run setup_ssntagger.bat first. 1>&2
    exit /b 1
)
"%~dp0venv-ssntagger\Scripts\python.exe" "%~dp0ssntagger.py" %*
exit /b %errorlevel%
