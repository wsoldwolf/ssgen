@echo off
setlocal
pushd "%~dp0" || exit /b 1
if not defined VSCMD_VER if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if errorlevel 1 goto :failed
)
if not defined SSNTAGGER_PYTHON set "SSNTAGGER_PYTHON=python"
if not exist "venv-ssntagger\Scripts\python.exe" (
    "%SSNTAGGER_PYTHON%" -m venv venv-ssntagger
    if errorlevel 1 goto :failed
)
if not defined CMAKE_ARGS set "CMAKE_ARGS=-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native"
if not defined CMAKE_GENERATOR set "CMAKE_GENERATOR=Ninja"
if not defined CMAKE_BUILD_PARALLEL_LEVEL set "CMAKE_BUILD_PARALLEL_LEVEL=8"
if not defined NVCC_PREPEND_FLAGS set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"
"venv-ssntagger\Scripts\python.exe" -m pip install --upgrade pip cmake ninja
if errorlevel 1 goto :failed
set "PATH=%~dp0venv-ssntagger\Scripts;%PATH%"
"venv-ssntagger\Scripts\python.exe" -m pip install --verbose --no-cache-dir --no-binary=llama-cpp-python "llama-cpp-python==0.3.36"
if errorlevel 1 goto :failed
"venv-ssntagger\Scripts\python.exe" -m pip install Pillow -e .
if errorlevel 1 goto :failed
"venv-ssntagger\Scripts\python.exe" -c "from llama_cpp.llama_chat_format import Gemma4ChatHandler; print('Gemma 4 support OK')"
if errorlevel 1 goto :failed
echo Setup complete. Run ssntagger.bat --help
popd
exit /b 0
:failed
echo Setup failed. Use an x64 Native Tools Command Prompt with CUDA Toolkit installed. 1>&2
popd
exit /b 1
