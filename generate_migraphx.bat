@echo off
REM generate_migraphx.bat — Configure + build MIGraphX with DxGML frontend
REM
REM NINJA MODE (default, uses CMake presets):
REM   generate_migraphx.bat                                 Full configure + build (WinRelWithDebInfo)
REM   generate_migraphx.bat WinDebug                        Debug build
REM   generate_migraphx.bat WinRelWithDebInfo --build-only  Skip configure, rebuild only
REM   generate_migraphx.bat WinRelWithDebInfo --no-build    Configure only
REM   generate_migraphx.bat WinRelWithDebInfo --run-tests   Build + run DxGML tests
REM
REM VISUAL STUDIO MODE (--vs flag):
REM   generate_migraphx.bat --vs                            VS configure + build (RelWithDebInfo)
REM   generate_migraphx.bat --vs Debug                      VS Debug build
REM   generate_migraphx.bat --vs RelWithDebInfo --run-tests Build + run tests
REM   generate_migraphx.bat --vs --build-only               Rebuild without reconfigure
REM
REM Available Ninja presets : WinRelWithDebInfo (default), WinDebug, WinRelease, WinMinSizeRel
REM Available VS configs    : RelWithDebInfo (default), Debug, Release, MinSizeRel
REM
REM Options (both modes):
REM   --vs                  Use Visual Studio 2022 generator instead of Ninja
REM   --build-only          Skip CMake configure, rebuild only (faster)
REM   --skip-configure      Alias for --build-only
REM   --no-build            Configure only, do not build
REM   --skip-build          Alias for --no-build
REM   --run-tests           After build, run DxGML parse tests + driver tests
REM   --help, -h, /?        Show this help

setlocal enabledelayedexpansion

echo ==========================================
echo MIGraphX + DxGML Build
echo ==========================================
echo.

if "%1"=="--help" goto ShowHelp
if "%1"=="-h"     goto ShowHelp
if "%1"=="/?"     goto ShowHelp

REM ---------------------------------------------------------------
REM Parse all arguments (order-independent flags)
REM ---------------------------------------------------------------
set "USE_VS=false"
set "SKIP_CONFIGURE=false"
set "SKIP_BUILD=false"
set "RUN_TESTS=false"
set "ARG_NAME="

:ArgLoop
if "%~1"=="" goto ArgDone
if /i "%~1"=="--vs"              ( set "USE_VS=true"          & shift & goto ArgLoop )
if /i "%~1"=="--build-only"      ( set "SKIP_CONFIGURE=true"  & shift & goto ArgLoop )
if /i "%~1"=="--skip-configure"  ( set "SKIP_CONFIGURE=true"  & shift & goto ArgLoop )
if /i "%~1"=="--no-build"        ( set "SKIP_BUILD=true"      & shift & goto ArgLoop )
if /i "%~1"=="--skip-build"      ( set "SKIP_BUILD=true"      & shift & goto ArgLoop )
if /i "%~1"=="--run-tests"       ( set "RUN_TESTS=true"       & shift & goto ArgLoop )
REM First non-flag arg is the preset/config name
if "!ARG_NAME!"=="" set ARG_NAME=%~1
shift
goto ArgLoop
:ArgDone

REM ---------------------------------------------------------------
REM Environment defaults
REM ---------------------------------------------------------------
if "%GPU_TARGETS%"==""      set GPU_TARGETS=gfx1100;gfx1150;gfx1151;gfx1201;gfx1200
if "%ROCM_DIR%"==""         set ROCM_DIR=C:\opt\rocm
if "%ROCM_INSTALL_DIR%"=="" set ROCM_INSTALL_DIR=C:\opt
if "%DXGML_DROP_DIR%"==""   set DXGML_DROP_DIR=C:\Users\hubertgu\Documents\DXGML-Drop4.0-x64

set SCRIPT_DIR=%~dp0

REM ---------------------------------------------------------------
REM Mode: Ninja (preset) vs Visual Studio
REM ---------------------------------------------------------------
if "%USE_VS%"=="true" goto SetupVS

REM --- Ninja / preset mode ---
set PRESET=!ARG_NAME!
if "!PRESET!"=="" set PRESET=WinRelWithDebInfo
set BUILD_DIR=%SCRIPT_DIR%build\%PRESET%
set MODE_DESC=Ninja  (preset: %PRESET%)
goto PrintInfo

:SetupVS
REM --- Visual Studio mode ---
REM Accept either a Ninja preset name or a raw VS config name
set VS_CONFIG=!ARG_NAME!
if /i "!VS_CONFIG!"=="WinRelWithDebInfo" set VS_CONFIG=RelWithDebInfo
if /i "!VS_CONFIG!"=="WinDebug"          set VS_CONFIG=Debug
if /i "!VS_CONFIG!"=="WinRelease"        set VS_CONFIG=Release
if /i "!VS_CONFIG!"=="WinMinSizeRel"     set VS_CONFIG=MinSizeRel
if "!VS_CONFIG!"==""                     set VS_CONFIG=RelWithDebInfo
set BUILD_DIR=%SCRIPT_DIR%build_vs
set MODE_DESC=Visual Studio 2022  (config: %VS_CONFIG%)

:PrintInfo
echo Mode         : %MODE_DESC%
echo Build dir    : %BUILD_DIR%
echo GPU_TARGETS  : %GPU_TARGETS%
echo ROCM_DIR     : %ROCM_DIR%
echo ROCM_INSTALL : %ROCM_INSTALL_DIR%
echo DXGML_DROP   : %DXGML_DROP_DIR%
echo Skip config  : %SKIP_CONFIGURE%
echo Skip build   : %SKIP_BUILD%
echo Run tests    : %RUN_TESTS%
echo.

REM ---------------------------------------------------------------
REM Configure
REM ---------------------------------------------------------------
if "%SKIP_CONFIGURE%"=="true" goto SkipConfigure

if "%USE_VS%"=="true" goto ConfigureVS

REM --- Ninja configure via cmake --preset ---
echo ==========================================
echo Configuring CMake (preset: %PRESET%)
echo ==========================================
echo.

cd /d "%SCRIPT_DIR%"
cmake --preset %PRESET% ^
      -DBUILD_TESTING=ON ^
      -DGPU_TARGETS="%GPU_TARGETS%"

if errorlevel 1 (
    echo.
    echo ==========================================
    echo Configure Failed!
    echo ==========================================
    echo.
    echo Tip: to skip reconfigure and just rebuild:
    echo   generate_migraphx.bat %PRESET% --build-only
    echo.
    exit /b 1
)
goto ConfigureDone

:ConfigureVS
REM --- Visual Studio configure (direct cmake, no preset) ---
echo ==========================================
echo Configuring CMake (Visual Studio 2022)
echo ==========================================
echo.

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

cmake -G "Visual Studio 17 2022" ^
      -A x64 ^
      -T ClangCL ^
      -DCMAKE_PREFIX_PATH="%ROCM_INSTALL_DIR%/rocMLIR/WinRelWithDebInfo;%SCRIPT_DIR%depend/WinRelWithDebInfo" ^
      -DGPU_TARGETS="%GPU_TARGETS%" ^
      -DMIGRAPHX_ENABLE_DXGML=ON ^
      -DDXGML_DROP_DIR="%DXGML_DROP_DIR%" ^
      -DMIGRAPHX_ENABLE_ONNX=OFF ^
      -DMIGRAPHX_ENABLE_PYTHON=OFF ^
      -DMIGRAPHX_ENABLE_TENSORFLOW=OFF ^
      -DMIGRAPHX_USE_ROCBLAS=OFF ^
      -DMIGRAPHX_USE_HIPBLASLT=OFF ^
      -DMIGRAPHX_USE_MIOPEN=OFF ^
      -DMIGRAPHX_USE_HIPRTC=ON ^
      -DMIGRAPHX_DISABLE_LARGE_BUFFER_TESTS=ON ^
      -DMIGRAPHX_USE_BINSKIM_COMPLIANT_COMPILE_FLAGS=ON ^
      -DBUILD_TESTING=ON ^
      "%SCRIPT_DIR%"

if errorlevel 1 (
    echo.
    echo ==========================================
    echo Configure Failed!
    echo ==========================================
    cd /d "%SCRIPT_DIR%"
    exit /b 1
)

echo.
echo Visual Studio solution generated:
for %%f in ("%BUILD_DIR%\*.sln") do echo   %%f
echo.
cd /d "%SCRIPT_DIR%"

:ConfigureDone
echo.
echo ==========================================
echo Configure Done
echo ==========================================
echo.

:SkipConfigure

if "%SKIP_BUILD%"=="true" (
    echo Skipping build. CMake cache ready at:
    echo   %BUILD_DIR%\CMakeCache.txt
    echo.
    if "%USE_VS%"=="true" (
        for %%f in ("%BUILD_DIR%\*.sln") do echo Open in Visual Studio: %%f
        echo.
        echo To build from command line:
        echo   generate_migraphx.bat --vs %VS_CONFIG% --build-only
    ) else (
        echo To build from command line:
        echo   generate_migraphx.bat %PRESET% --build-only
    )
    echo.
    exit /b 0
)

REM ---------------------------------------------------------------
REM Build
REM ---------------------------------------------------------------
echo ==========================================
echo Building amdxgml + driver + tests
echo ==========================================
echo.

if "%USE_VS%"=="true" (
    cmake --build "%BUILD_DIR%" --config %VS_CONFIG% --parallel --target ^
        amdxgml ^
        driver ^
        test_dxgml_conv_relu_test ^
        test_dxgml_gelu_test ^
        test_dxgml_relu_erf_test ^
        test_dxgml_standalone_cluster_test
) else (
    cmake --build "%BUILD_DIR%" --parallel --target ^
        amdxgml ^
        driver ^
        test_dxgml_conv_relu_test ^
        test_dxgml_gelu_test ^
        test_dxgml_relu_erf_test ^
        test_dxgml_standalone_cluster_test
)

if errorlevel 1 (
    echo.
    echo ==========================================
    echo Build FAILED - check output above.
    echo ==========================================
    exit /b 1
)

echo.
echo ==========================================
echo Build Successful!
echo ==========================================
echo.
if "%USE_VS%"=="true" (
    echo Binaries  ^(VS multi-config layout^):
    echo   Library : %BUILD_DIR%\bin\%VS_CONFIG%\amdxgml.dll
    echo   Driver  : %BUILD_DIR%\bin\%VS_CONFIG%\migraphx-driver.exe
    echo   Tests   : %BUILD_DIR%\bin\%VS_CONFIG%\test_dxgml_*.exe
) else (
    echo Binaries:
    echo   Library : %BUILD_DIR%\bin\amdxgml.dll
    echo   Driver  : %BUILD_DIR%\bin\migraphx-driver.exe
    echo   Tests   : %BUILD_DIR%\bin\test_dxgml_*.exe
)
echo.

REM ---------------------------------------------------------------
REM Optional: run tests
REM ---------------------------------------------------------------
if "%RUN_TESTS%"=="false" goto ShowSummary

echo ==========================================
echo Running DxGML Parse Tests (ctest)
echo ==========================================
echo.
if "%USE_VS%"=="true" (
    ctest --test-dir "%BUILD_DIR%" -C %VS_CONFIG% -R test_dxgml -V
) else (
    ctest --test-dir "%BUILD_DIR%" -R test_dxgml -V
)
if errorlevel 1 (
    echo.
    echo Some parse tests FAILED!
) else (
    echo.
    echo Parse tests PASSED!
)

echo.
echo ==========================================
echo Running DxGML Driver Tests (.mlir files)
echo ==========================================
echo.
call "%SCRIPT_DIR%test\dxgml\run_dxgml_tests.bat"

:ShowSummary
echo.
echo ==========================================
echo Done.
echo ==========================================
echo.
echo Build dir    : !BUILD_DIR!
echo To run DxGML driver tests:
echo   test\dxgml\run_dxgml_tests.bat
echo.
echo To run parse unit tests:
echo   ctest --test-dir "!BUILD_DIR!" -R test_dxgml -V
echo.

exit /b 0

REM ---------------------------------------------------------------
:ShowHelp
REM ---------------------------------------------------------------
echo.
echo ==========================================
echo HELP — generate_migraphx.bat
echo ==========================================
echo.
echo USAGE:
echo   generate_migraphx.bat [PRESET^|CONFIG] [OPTIONS]
echo.
echo NINJA MODE (default, uses CMake presets + Ninja + Clang):
echo   generate_migraphx.bat                        Full configure + build (WinRelWithDebInfo)
echo   generate_migraphx.bat WinDebug               Debug build
echo   generate_migraphx.bat WinRelWithDebInfo --build-only   Rebuild (skip configure)
echo   generate_migraphx.bat WinRelWithDebInfo --no-build     Configure only
echo   generate_migraphx.bat WinRelWithDebInfo --run-tests    Build + run all DxGML tests
echo.
echo   Available presets: WinRelWithDebInfo (default), WinDebug, WinRelease, WinMinSizeRel
echo.
echo VISUAL STUDIO MODE (--vs flag, uses VS 2022 + ClangCL):
echo   generate_migraphx.bat --vs                   VS configure + build (RelWithDebInfo)
echo   generate_migraphx.bat --vs Debug             VS Debug build
echo   generate_migraphx.bat --vs --build-only      Rebuild without reconfigure
echo   generate_migraphx.bat --vs --run-tests       Build + run all DxGML tests
echo.
echo   Available configs: RelWithDebInfo (default), Debug, Release, MinSizeRel
echo   Also accepts Ninja preset names: WinRelWithDebInfo, WinDebug, etc.
echo   Solution generated at: build_vs\
echo.
echo OPTIONS (both modes):
echo   --vs                  Visual Studio 2022 generator (default: Ninja)
echo   --build-only          Skip configure, rebuild changed files only
echo   --skip-configure      Alias for --build-only
echo   --no-build            Configure/generate only, do not build
echo   --skip-build          Alias for --no-build
echo   --run-tests           Run parse tests + all 35 DxGML driver tests after build
echo   --help, -h, /?        Show this help
echo.
echo ENVIRONMENT VARIABLES (all have defaults):
echo   GPU_TARGETS       Semicolon-separated targets  (default: gfx1100;gfx1150;...;gfx1201)
echo   ROCM_DIR          ROCm install path            (default: C:\opt\rocm)
echo   ROCM_INSTALL_DIR  rocMLIR install root         (default: C:\opt)
echo   DXGML_DROP_DIR    DxGML drop directory         (default: Documents\shared_drive\...)
echo.
echo EXAMPLES:
echo   Full Ninja build + test:
echo     generate_migraphx.bat WinRelWithDebInfo --run-tests
echo.
echo   Ninja rebuild only (after code change):
echo     generate_migraphx.bat WinRelWithDebInfo --build-only
echo.
echo   VS configure only (open .sln in Visual Studio):
echo     generate_migraphx.bat --vs --no-build
echo.
echo   VS full build + test:
echo     generate_migraphx.bat --vs --run-tests
echo.
echo   VS debug build:
echo     generate_migraphx.bat --vs Debug
echo.
exit /b 0
