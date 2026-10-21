@echo off
setlocal
pushd "%~dp0" || exit /b 1
if not exist "venv\Scripts\python.exe" (
    py -m venv venv
    if errorlevel 1 goto :failed
)
"venv\Scripts\python.exe" -m pip install -e ".[plot]"
if errorlevel 1 goto :failed
echo Setup complete. Run ssgen.bat or ssplot.bat.
popd
exit /b 0

:failed
popd
exit /b 1
