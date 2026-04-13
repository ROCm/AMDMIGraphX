@echo off
REM ============================================================
REM run_dxgml_tests.bat
REM
REM Run DxGML MLIR dialect tests using migraphx-driver.exe,
REM plus the compiled unit-test binaries in test\dxgml\parse\.
REM Dump output goes to: <script-dir>\dump\
REM
REM Usage:
REM   run_dxgml_tests.bat                   - Run EVERYTHING (mlir + parse unit tests)
REM   run_dxgml_tests.bat parse             - Run only parse unit tests (C++ test binaries)
REM   run_dxgml_tests.bat mlir              - Run only migraphx-driver tests (all .mlir models)
REM   run_dxgml_tests.bat dump              - Run all tests AND dump intermediate output
REM   run_dxgml_tests.bat simple_gemm       - Run simple_gemm/model.mlir only
REM   run_dxgml_tests.bat <model>           - Run model/<model>.mlir or matching file
REM
REM   run_dxgml_tests.bat mlir  dump        - Run driver tests + dump output
REM   run_dxgml_tests.bat parse dump        - Run unit tests + print results
REM   run_dxgml_tests.bat all   dump gfx1201 - Run all with dump for gfx1201
REM
REM Dump files go to: <script-dir>\dump\<model>_migraphx_ops.txt
REM ============================================================

setlocal enabledelayedexpansion

REM ---- Parse arguments ----
set SUITE=%1
set OPT2=%2
set ARCH=%3
if "%ARCH%"=="" set ARCH=gfx1201

REM Normalize SUITE
if "%SUITE%"==""    set SUITE=all
if /i "%SUITE%"=="all" set SUITE=all

REM Check for "dump" keyword in second arg or suite
set DO_DUMP=0
if /i "%OPT2%"=="dump" set DO_DUMP=1
if /i "%SUITE%"=="dump" (
    set SUITE=all
    set DO_DUMP=1
)

REM ---- Locate directories ----
set SCRIPT_DIR=%~dp0
set ROOT_DIR=%SCRIPT_DIR%..\..
set MLIR_DIR=%SCRIPT_DIR%mlir
set DUMP_DIR=%SCRIPT_DIR%dump

REM Create dump directory
if not exist "%DUMP_DIR%" mkdir "%DUMP_DIR%"

REM ---- Locate migraphx-driver.exe ----
set DRIVER_PATH=
set BIN_DIR=

for %%D in (WinRelWithDebInfo WinRelease WinRelMinSizeRel WinDebug) do (
    if exist "%ROOT_DIR%\build\%%D\bin\migraphx-driver.exe" (
        set DRIVER_PATH=%ROOT_DIR%\build\%%D\bin\migraphx-driver.exe
        set BIN_DIR=%ROOT_DIR%\build\%%D\bin
        goto FoundDriver
    )
)
if exist "%ROOT_DIR%\build\bin\migraphx-driver.exe" (
    set DRIVER_PATH=%ROOT_DIR%\build\bin\migraphx-driver.exe
    set BIN_DIR=%ROOT_DIR%\build\bin
    goto FoundDriver
)
echo [ERROR] migraphx-driver.exe not found.
echo         Build MIGraphX first:  build_migraphx.bat
exit /b 1
:FoundDriver

REM ---- Extend PATH so driver can find migraphx.dll, amdhip64_6.dll etc. ----
set PATH=%BIN_DIR%;C:\opt\rocm\bin;%PATH%

REM ---- Parse unit test binaries live alongside the driver ----
set PARSE_BIN_DIR=%BIN_DIR%

REM ---- Counters ----
set TOTAL_PASS=0
set TOTAL_FAIL=0
set TOTAL_SKIP=0

REM ============================================================
REM Dispatch
REM ============================================================

if /i "%SUITE%"=="all"   goto RunAll
if /i "%SUITE%"=="mlir"  goto RunMlir
if /i "%SUITE%"=="parse" goto RunParse

REM Single-model shortcut
goto RunSingleModel

REM ============================================================
:RunAll
REM ============================================================
echo =====================================================
echo DxGML Test Suite - FULL RUN
echo =====================================================
echo Arch:   %ARCH%
echo Driver: %DRIVER_PATH%
echo Dump:   %DUMP_DIR%  (enabled: %DO_DUMP%)
echo.

call :RunParseSuite
echo.
call :RunMlirSuite
goto ShowResults

REM ============================================================
:RunMlir
REM ============================================================
echo =====================================================
echo DxGML Test Suite - MLIR Driver Tests
echo =====================================================
echo Arch:   %ARCH%
echo Driver: %DRIVER_PATH%
echo Dump:   %DUMP_DIR%  (enabled: %DO_DUMP%)
echo.
call :RunMlirSuite
goto ShowResults

REM ============================================================
:RunParse
REM ============================================================
echo =====================================================
echo DxGML Test Suite - Parse Unit Tests
echo =====================================================
echo BinDir: %PARSE_BIN_DIR%
echo.
call :RunParseSuite
goto ShowResults

REM ============================================================
:RunSingleModel
REM ============================================================
set FOUND_MODEL=0

REM Check subdirectory model.mlir convention first
if exist "%MLIR_DIR%\%SUITE%\model.mlir" (
    call :RunDriverTest "%MLIR_DIR%\%SUITE%\model.mlir" "%SUITE%"
    set FOUND_MODEL=1
    goto ShowResults
)

REM Check by exact filename (without extension)
for /r "%MLIR_DIR%" %%F in (*.mlir) do (
    if /i "%%~nF"=="%SUITE%" (
        call :RunDriverTest "%%F" "%SUITE%"
        set FOUND_MODEL=1
        goto ShowResults
    )
)

if !FOUND_MODEL!==0 (
    echo [ERROR] Unknown test or model: %SUITE%
    echo.
    echo Usage examples:
    echo   run_dxgml_tests.bat                 - Run all tests
    echo   run_dxgml_tests.bat parse           - C++ parse unit tests only
    echo   run_dxgml_tests.bat mlir            - Driver tests only
    echo   run_dxgml_tests.bat dump            - All tests + dump output
    echo   run_dxgml_tests.bat simple_gemm     - Single model test
    echo   run_dxgml_tests.bat mlir dump       - Driver tests + dump output
    exit /b 1
)
goto ShowResults

REM ============================================================
REM :RunParseSuite
REM Run the four C++ parse unit-test executables.
REM ============================================================
:RunParseSuite
echo --- Parse Unit Tests (C++ test binaries) ---
call :RunUnitTest "test_dxgml_conv_relu_test.exe"          "ConvRelu parse"
call :RunUnitTest "test_dxgml_gelu_test.exe"               "Gelu parse"
call :RunUnitTest "test_dxgml_relu_erf_test.exe"           "ReluErf parse"
call :RunUnitTest "test_dxgml_standalone_cluster_test.exe" "StandaloneCluster parse"
goto :eof

REM ============================================================
REM :RunMlirSuite
REM Run all .mlir files through migraphx-driver --dxgml.
REM ============================================================
:RunMlirSuite
echo --- MLIR Driver Tests ---

echo [simple models]
call :RunDriverTest "%MLIR_DIR%\simple_gemm\model.mlir"      "simple_gemm"
call :RunDriverTest "%MLIR_DIR%\conv_example\model.mlir"     "conv_example"
call :RunDriverTest "%MLIR_DIR%\test\test_dxgml.mlir"        "test_dxgml"
call :RunDriverTest "%MLIR_DIR%\test\test_model1_clean.mlir" "test_model1_clean"

echo [CNN models]
call :RunDriverTest "%MLIR_DIR%\model1\model.mlir"       "model1"
call :RunDriverTest "%MLIR_DIR%\model1\model_test.mlir"  "model1_test"
call :RunDriverTest "%MLIR_DIR%\model2\model.mlir"       "model2"
call :RunDriverTest "%MLIR_DIR%\model2\model_test.mlir"  "model2_test"
call :RunDriverTest "%MLIR_DIR%\model3\model.mlir"       "model3"
call :RunDriverTest "%MLIR_DIR%\model3\model_test.mlir"  "model3_test"

echo [vision models]
call :RunDriverTest "%MLIR_DIR%\audio2face\model.mlir"      "audio2face"
call :RunDriverTest "%MLIR_DIR%\audio2face\model_test.mlir" "audio2face_test"

echo [LLM models]
call :RunDriverTest "%MLIR_DIR%\llama32\llama32_dxgml_static_decoder.mlir"       "llama32_decoder"
call :RunDriverTest "%MLIR_DIR%\llama32\llama32_dxgml_static_pre-fill.mlir"      "llama32_prefill"
call :RunDriverTest "%MLIR_DIR%\llama32\llama32_dxgml_static_decoder_test.mlir"  "llama32_decoder_test"
call :RunDriverTest "%MLIR_DIR%\llama32\llama32_dxgml_static_pre-fill_test.mlir" "llama32_prefill_test"
call :RunDriverTest "%MLIR_DIR%\nemotron\model_decoder.mlir"       "nemotron_decoder"
call :RunDriverTest "%MLIR_DIR%\nemotron\model_pre-fill.mlir"      "nemotron_prefill"
call :RunDriverTest "%MLIR_DIR%\nemotron\model_decoder_test.mlir"  "nemotron_decoder_test"
call :RunDriverTest "%MLIR_DIR%\nemotron\model_pre-fill_test.mlir" "nemotron_prefill_test"
call :RunDriverTest "%MLIR_DIR%\phi_silica_qdq\model.mlir" "phi_silica_qdq"
goto :eof

REM ============================================================
REM :RunUnitTest <exe_name> <description>
REM ============================================================
:RunUnitTest
    set "EXE_NAME=%~1"
    set "DESC=%~2"
    set "EXE_PATH=!PARSE_BIN_DIR!\!EXE_NAME!"

    if not exist "!EXE_PATH!" (
        echo   [SKIP] !DESC! - binary not found: !EXE_PATH!
        set /a TOTAL_SKIP+=1
        goto :eof
    )

    "!EXE_PATH!" >nul 2>&1
    if !errorlevel!==0 (
        echo   [PASS] !DESC!
        set /a TOTAL_PASS+=1
    ) else (
        echo   [FAIL] !DESC!
        "!EXE_PATH!" 2>&1
        set /a TOTAL_FAIL+=1
    )
    goto :eof

REM ============================================================
REM :RunDriverTest <mlir_file> <name>
REM   Parses with migraphx-driver read --dxgml --skip-unknown-operators
REM   If DO_DUMP==1, also writes MIGraphX ops text to dump\<name>_migraphx_ops.txt
REM ============================================================
:RunDriverTest
    set "MLIR_FILE=%~1"
    set "MODEL_NAME=%~2"

    if not exist "!MLIR_FILE!" (
        echo   [SKIP] !MODEL_NAME! - file not found
        set /a TOTAL_SKIP+=1
        goto :eof
    )

    REM Parse test
    "!DRIVER_PATH!" read "!MLIR_FILE!" --dxgml --skip-unknown-operators >nul 2>&1
    if !errorlevel!==0 (
        echo   [PASS] !MODEL_NAME!
        set /a TOTAL_PASS+=1
    ) else (
        echo   [FAIL] !MODEL_NAME!
        "!DRIVER_PATH!" read "!MLIR_FILE!" --dxgml --skip-unknown-operators 2>&1
        set /a TOTAL_FAIL+=1
    )

    REM Dump MIGraphX ops representation if requested
    if "%DO_DUMP%"=="1" (
        set "DUMP_FILE=!DUMP_DIR!\!MODEL_NAME!_migraphx_ops.txt"
        echo          ^> !DUMP_FILE!
        "!DRIVER_PATH!" read "!MLIR_FILE!" --dxgml --skip-unknown-operators --text > "!DUMP_FILE!" 2>&1
    )
    goto :eof

REM ============================================================
:ShowResults
REM ============================================================
echo.
echo =====================================================
echo Results: !TOTAL_PASS! passed, !TOTAL_FAIL! failed, !TOTAL_SKIP! skipped
echo =====================================================
if %DO_DUMP%==1 (
    echo Dump files: %DUMP_DIR%
)
if !TOTAL_FAIL! GTR 0 (
    echo.
    echo To debug a failure:
    echo   "!DRIVER_PATH!" read ^<file.mlir^> --dxgml --skip-unknown-operators --text
    exit /b 1
)
echo All tests passed!
exit /b 0
