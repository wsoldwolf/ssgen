@echo off
setlocal DisableDelayedExpansion
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%~dp0ssreorder.py" %*
) else (
    py -3 "%~dp0ssreorder.py" %*
)
set "SSREORDER_EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %SSREORDER_EXIT_CODE%
