@echo off
setlocal DisableDelayedExpansion
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%~dp0sstxt2tags.py" %*
) else (
    python "%~dp0sstxt2tags.py" %*
)
exit /b %errorlevel%
