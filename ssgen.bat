@echo off
setlocal
if not exist "%~dp0venv\Scripts\python.exe" (
    echo Run setup.bat first to create the project venv. 1>&2
    exit /b 1
)
"%~dp0venv\Scripts\python.exe" "%~dp0ssgen.py" %*
exit /b %errorlevel%
