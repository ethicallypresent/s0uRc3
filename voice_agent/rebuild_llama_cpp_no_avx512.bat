@echo off
rem Rebuilds llama-cpp-python from source with AVX512 disabled (AVX2/FMA
rem still on). Fixes "OSError: [WinError -1073741795] 0xc000001d" on
rem CPUs without AVX512 (most laptop/mobile CPUs) -- see README.md's
rem "Troubleshooting" note under Stage 3 for why.
rem
rem Needs CMake and MSVC Build Tools:
rem   winget install Kitware.CMake
rem   winget install Microsoft.VisualStudio.2022.BuildTools
cd /d "%~dp0"

set "VCVARSALL="
for %%P in (
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat"
) do (
    if exist %%P set "VCVARSALL=%%~P"
)
if "%VCVARSALL%"=="" (
    echo Could not find vcvarsall.bat - install MSVC Build Tools first:
    echo   winget install Microsoft.VisualStudio.2022.BuildTools
    exit /b 1
)

where cmake >nul 2>nul
if errorlevel 1 (
    echo cmake not found on PATH - install it first:
    echo   winget install Kitware.CMake
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo .venv not found - create it first: python -m venv .venv
    exit /b 1
)

call "%VCVARSALL%" x64
if errorlevel 1 exit /b 1

set CMAKE_ARGS=-DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_AVX=ON -DGGML_F16C=ON
.venv\Scripts\python.exe -m pip install llama-cpp-python==0.3.30 --no-binary llama-cpp-python --force-reinstall --no-cache-dir
if errorlevel 1 exit /b 1

echo.
echo Done. Verifying build:
.venv\Scripts\python.exe -c "import llama_cpp; print(llama_cpp.llama_cpp.llama_print_system_info().decode())"
