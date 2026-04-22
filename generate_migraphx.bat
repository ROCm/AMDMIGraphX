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
REM   --local-rocmlir [DIR] Use local rocMLIR build instead of C:\opt\rocMLIR.
REM                         DIR defaults to C:\Develop\rocMLIR\build\<config>.
REM                         Can also be set via ROCMLIR_LOCAL_DIR env var.
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
set "LOCAL_ROCMLIR_FLAG=false"
set "LOCAL_ROCMLIR_DIR_ARG="

:ArgLoop
if "%~1"=="" goto ArgDone
if /i "%~1"=="--vs"              ( set "USE_VS=true"         & shift & goto ArgLoop )
if /i "%~1"=="--build-only"      ( set "SKIP_CONFIGURE=true" & shift & goto ArgLoop )
if /i "%~1"=="--skip-configure"  ( set "SKIP_CONFIGURE=true" & shift & goto ArgLoop )
if /i "%~1"=="--no-build"        ( set "SKIP_BUILD=true"     & shift & goto ArgLoop )
if /i "%~1"=="--skip-build"      ( set "SKIP_BUILD=true"     & shift & goto ArgLoop )
if /i "%~1"=="--run-tests"       ( set "RUN_TESTS=true"      & shift & goto ArgLoop )
if /i "%~1"=="--local-rocmlir" (
    set "LOCAL_ROCMLIR_FLAG=true"
    shift
    REM Optional next arg: explicit path (must not start with --)
    if not "%~1"=="" (
        set _NEXT=%~1
        if "!_NEXT:~0,2!"=="--" goto ArgLoop
        set "LOCAL_ROCMLIR_DIR_ARG=%~1"
        shift
    )
    goto ArgLoop
)
REM First non-flag arg is the preset/config name
if "!ARG_NAME!"=="" set "ARG_NAME=%~1"
shift
goto ArgLoop
:ArgDone

REM Resolve local rocMLIR: flag > env var > off
set "USE_LOCAL_ROCMLIR=false"
set "LOCAL_ROCMLIR_ROOT="
if "!LOCAL_ROCMLIR_FLAG!"=="true" set "USE_LOCAL_ROCMLIR=true"
if not "!ROCMLIR_LOCAL_DIR!"=="" set "USE_LOCAL_ROCMLIR=true"
if not "!LOCAL_ROCMLIR_DIR_ARG!"=="" set "ROCMLIR_LOCAL_DIR=!LOCAL_ROCMLIR_DIR_ARG!"

REM ---------------------------------------------------------------
REM Environment defaults
REM ---------------------------------------------------------------
if "%GPU_TARGETS%"==""      set GPU_TARGETS=gfx1100;gfx1150;gfx1151;gfx1201;gfx1200
if "%ROCM_DIR%"==""         set ROCM_DIR=C:\opt\rocm
if "%ROCM_INSTALL_DIR%"=="" set ROCM_INSTALL_DIR=C:\opt
if "%DXGML_DROP_DIR%"==""   set DXGML_DROP_DIR=C:\Users\hubertgu\Documents\DXGML-Drop4.0-x64
if "%JOM_INSTALL_DIR%"==""   set JOM_INSTALL_DIR=C:\tools\jom
set ROCM_DIR_CMAKE=%ROCM_DIR:\=/%

set SCRIPT_DIR=%~dp0

REM ---------------------------------------------------------------
REM Mode: Ninja (preset) vs Visual Studio
REM ---------------------------------------------------------------
if /i "!USE_VS!"=="true" goto SetupVS

REM --- Ninja / preset mode ---
set PRESET=!ARG_NAME!
if "!PRESET!"=="" set PRESET=WinRelWithDebInfo
set PRESET_BUILD_DIR=!PRESET!
if /i "!PRESET_BUILD_DIR!"=="WinDebug"          set PRESET_BUILD_DIR=Debug
if /i "!PRESET_BUILD_DIR!"=="WinRelWithDebInfo" set PRESET_BUILD_DIR=RelWithDebInfo
if /i "!PRESET_BUILD_DIR!"=="WinRelease"        set PRESET_BUILD_DIR=Release
if /i "!PRESET_BUILD_DIR!"=="WinMinSizeRel"     set PRESET_BUILD_DIR=MinSizeRel
set BUILD_DIR=%SCRIPT_DIR%build\!PRESET_BUILD_DIR!
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
echo Local rocMLIR: %USE_LOCAL_ROCMLIR%
if "!USE_LOCAL_ROCMLIR!"=="true" echo rocMLIR dir  : !ROCMLIR_LOCAL_DIR!
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
set "CONFIGURE_FAILED=false"

REM Build the local-rocMLIR prefix override for Ninja mode.
REM When active we must pass CMAKE_PREFIX_PATH explicitly because -D overrides
REM the preset's cacheVariable value entirely — so we reconstruct the full list.
set "NINJA_PREFIX_OVERRIDE="
if "!USE_LOCAL_ROCMLIR!"=="true" (
    if "!ROCMLIR_LOCAL_DIR!"=="" (
        set "ROCMLIR_LOCAL_DIR=C:\Develop\rocMLIR\build\!PRESET_BUILD_DIR!"
    )
    echo Using local rocMLIR: !ROCMLIR_LOCAL_DIR!
    set "NINJA_PREFIX_OVERRIDE=-DCMAKE_PREFIX_PATH=!ROCMLIR_LOCAL_DIR!;%SCRIPT_DIR%depend\%PRESET_BUILD_DIR%;C:/opt/rocm"
)

set "_RC_FLAG="
if exist "%ROCM_DIR%\bin\llvm-rc.exe" set "_RC_FLAG=-DCMAKE_RC_COMPILER=%ROCM_DIR_CMAKE%/bin/llvm-rc.exe"

cmake --preset %PRESET% ^
    -DBUILD_TESTING=ON ^
    -DGPU_TARGETS="%GPU_TARGETS%" ^
    !NINJA_PREFIX_OVERRIDE! ^
    !_RC_FLAG!

if errorlevel 1 set "CONFIGURE_FAILED=true"

if "%CONFIGURE_FAILED%"=="true" (
    if exist "%BUILD_DIR%\CMakeCache.txt" (
        echo.
        echo Configure failed; clearing stale cache and retrying once...
        rd /s /q "%BUILD_DIR%"
        echo.
        cmake --preset %PRESET% ^
            -DBUILD_TESTING=ON ^
            -DGPU_TARGETS="%GPU_TARGETS%" ^
            !NINJA_PREFIX_OVERRIDE! ^
            !_RC_FLAG!
    )
)

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

REM Resolve the rocMLIR prefix for VS mode.
if "!USE_LOCAL_ROCMLIR!"=="true" (
    if "!ROCMLIR_LOCAL_DIR!"=="" (
        set "ROCMLIR_LOCAL_DIR=C:\Develop\rocMLIR\build\!VS_CONFIG!"
    )
    echo Using local rocMLIR: !ROCMLIR_LOCAL_DIR!
    set "VS_ROCMLIR_PREFIX=!ROCMLIR_LOCAL_DIR!"
) else (
    set "VS_ROCMLIR_PREFIX=%ROCM_INSTALL_DIR%/rocMLIR/%VS_CONFIG%"
)

cmake -G "Visual Studio 17 2022" ^
      -A x64 ^
      -T ClangCL ^
    -DCMAKE_PREFIX_PATH="!VS_ROCMLIR_PREFIX!;%SCRIPT_DIR%/depend/%VS_CONFIG%" ^
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
    echo Build FAILED! Check output above.
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
    ctest --test-dir "%BUILD_DIR%" -C %VS_CONFIG% -R "test_dxgml" -V
) else (
    ctest --test-dir "%BUILD_DIR%" -R "test_dxgml" -V
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
echo Done!
echo ==========================================
echo.
if "%USE_VS%"=="true" (
    for %%f in ("%BUILD_DIR%\*.sln") do echo Solution     : %%f
    echo.
    echo To rebuild (skip reconfigure):
    echo   generate_migraphx.bat --vs %VS_CONFIG% --build-only
) else (
    echo To rebuild (skip reconfigure):
    echo   generate_migraphx.bat %PRESET% --build-only
)
echo.
echo To run DxGML driver tests:
echo   test\dxgml\run_dxgml_tests.bat
echo.
if "%USE_VS%"=="true" (
    echo To run parse unit tests:
    echo   ctest --test-dir "%BUILD_DIR%" -C %VS_CONFIG% -R test_dxgml -V
) else (
    echo To run parse unit tests:
    echo   ctest --test-dir "%BUILD_DIR%" -R test_dxgml -V
)
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
echo   --vs                       Visual Studio 2022 generator (default: Ninja)
echo   --local-rocmlir [DIR]      Use local rocMLIR build (C:\Develop\rocMLIR\build\^<config^>
echo                              for Ninja; build_vs_clang\^<config^> for VS) instead of
echo                              C:\opt\rocMLIR. Optionally supply an explicit DIR.
echo   --build-only               Skip configure, rebuild changed files only
echo   --skip-configure           Alias for --build-only
echo   --no-build                 Configure/generate only, do not build
echo   --skip-build               Alias for --no-build
echo   --run-tests                Run parse tests + all 35 DxGML driver tests after build
echo   --help, -h, /?             Show this help
echo.
echo ENVIRONMENT VARIABLES (all have defaults):
echo   GPU_TARGETS          Semicolon-separated targets  (default: gfx1100;gfx1150;...;gfx1201)
echo   ROCM_DIR             ROCm install path            (default: C:\opt\rocm)
echo   ROCM_INSTALL_DIR     rocMLIR install root         (default: C:\opt)
echo   DXGML_DROP_DIR       DxGML drop directory         (default: Documents\shared_drive\...)
echo   ROCMLIR_LOCAL_DIR    Same as --local-rocmlir but set as env var (persistent)
echo.
echo EXAMPLES:
echo   Full Ninja build + test:
echo     generate_migraphx.bat WinRelWithDebInfo --run-tests
echo.
echo   Ninja rebuild only (after code change):
echo     generate_migraphx.bat WinRelWithDebInfo --build-only
echo.
echo   Link to local rocMLIR at C:\Develop\rocMLIR (auto path):
echo     generate_migraphx.bat WinRelWithDebInfo --local-rocmlir
echo.
echo   Link to a specific local rocMLIR build dir:
echo     generate_migraphx.bat WinRelWithDebInfo --local-rocmlir C:\Develop\rocMLIR\build\Release
echo.
echo   Or set the env var persistently before calling the bat:
echo     set ROCMLIR_LOCAL_DIR=C:\Develop\rocMLIR\build\Release
echo     generate_migraphx.bat WinRelWithDebInfo
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
echo   VS with local rocMLIR (auto-uses build_vs_clang\RelWithDebInfo):
echo     generate_migraphx.bat --vs --local-rocmlir
echo.
exit /b 0
