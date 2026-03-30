@echo off
REM Run DxGML MLIR dialect tests using migraphx-driver.exe
REM Models after C:\Develop\rocMLIR.WML\examples\dxml-dialect\run_examples.bat
REM
REM Usage: run_dxgml_tests.bat [test] [arch]
REM
REM   run_dxgml_tests.bat                      - Run all DxGML tests (parse only)
REM   run_dxgml_tests.bat all gfx1201          - Run all tests with specified arch
REM   run_dxgml_tests.bat ConvRelu             - Run ConvRelu test only
REM   run_dxgml_tests.bat model1               - Run model1 test only
REM
REM Available tests:
REM   --- CompilationInput (4 original parse tests) ---
REM   ConvRelu        - ConvRelu.CompilationInput.mlir
REM   Gelu            - Gelu.CompilationInput.mlir
REM   ReluErf         - ReluErf.CompilationInput.mlir
REM   StandaloneCluster - StandaloneCluster.CompilationInput.mlir
REM
REM   --- Simple Models (from rocMLIR examples) ---
REM   model1          - model1.mlir  (CNN with depth-to-space)
REM   model2          - model2.mlir  (CNN variant)
REM   model3          - model3.mlir  (CNN variant)
REM   simple_gemm     - simple_gemm.mlir  (GEMM + bias + relu)
REM   conv_example    - conv_example.mlir  (Conv+BN+ReLU+MaxPool)
REM
REM   --- Vision Models ---
REM   audio2face      - audio2face.mlir  (with reduce ops)
REM
REM   --- LLM Models ---
REM   llama32_dec     - llama32_decoder.mlir  (LLaMA 3.2 decoder, GQA)
REM   llama32_pre     - llama32_prefill.mlir  (LLaMA 3.2 pre-fill, GQA)
REM   nemotron_dec    - nemotron_decoder.mlir
REM   nemotron_pre    - nemotron_prefill.mlir
REM   phi_silica      - phi_silica_qdq.mlir  (quantized)

setlocal enabledelayedexpansion

set TEST=%1
set ARCH=%2
if "%ARCH%"=="" set ARCH=gfx1201

REM Locate migraphx-driver.exe relative to this script
set SCRIPT_DIR=%~dp0
set ROOT_DIR=%SCRIPT_DIR%..\..
set DRIVER_PATH=

if exist "%ROOT_DIR%\build\WinRelWithDebInfo\bin\migraphx-driver.exe" (
    set DRIVER_PATH=%ROOT_DIR%\build\WinRelWithDebInfo\bin\migraphx-driver.exe
) else if exist "%ROOT_DIR%\build\WinRelease\bin\migraphx-driver.exe" (
    set DRIVER_PATH=%ROOT_DIR%\build\WinRelease\bin\migraphx-driver.exe
) else if exist "%ROOT_DIR%\build\WinDebug\bin\migraphx-driver.exe" (
    set DRIVER_PATH=%ROOT_DIR%\build\WinDebug\bin\migraphx-driver.exe
) else if exist "%ROOT_DIR%\build\bin\migraphx-driver.exe" (
    set DRIVER_PATH=%ROOT_DIR%\build\bin\migraphx-driver.exe
)

if "%DRIVER_PATH%"=="" (
    echo Error: migraphx-driver.exe not found.
    echo Please build MIGraphX first:
    echo   build_migraphx.bat
    echo   - or -
    echo   cmake --build build\WinRelWithDebInfo --target driver
    exit /b 1
)

set MLIR_DIR=%SCRIPT_DIR%mlir

set PASS_COUNT=0
set FAIL_COUNT=0

if "%TEST%"=="" goto RunAll
if /i "%TEST%"=="all" goto RunAll

REM --- CompilationInput tests ---
if /i "%TEST%"=="ConvRelu"          call :RunTest "ConvRelu.CompilationInput.mlir"          "ConvRelu (conv + relu)"              & goto show_results
if /i "%TEST%"=="Gelu"              call :RunTest "Gelu.CompilationInput.mlir"              "Gelu (erf-based GELU)"               & goto show_results
if /i "%TEST%"=="ReluErf"           call :RunTest "ReluErf.CompilationInput.mlir"           "ReluErf (relu + erf)"                & goto show_results
if /i "%TEST%"=="StandaloneCluster" call :RunTest "StandaloneCluster.CompilationInput.mlir" "StandaloneCluster (pre/post-conv)"   & goto show_results

REM --- Simple model tests ---
if /i "%TEST%"=="model1"            call :RunTest "model1.mlir"          "model1 (CNN with depth-to-space)"    & goto show_results
if /i "%TEST%"=="model2"            call :RunTest "model2.mlir"          "model2 (CNN variant)"                & goto show_results
if /i "%TEST%"=="model3"            call :RunTest "model3.mlir"          "model3 (CNN variant)"                & goto show_results
if /i "%TEST%"=="simple_gemm"       call :RunTest "simple_gemm.mlir"     "simple_gemm (GEMM + bias + relu)"    & goto show_results
if /i "%TEST%"=="conv_example"      call :RunTest "conv_example.mlir"    "conv_example (Conv+BN+ReLU+MaxPool)" & goto show_results

REM --- Vision model tests ---
if /i "%TEST%"=="audio2face"        call :RunTest "audio2face.mlir"      "audio2face (with reduce ops)"        & goto show_results

REM --- LLM model tests ---
if /i "%TEST%"=="llama32_dec"       call :RunTest "llama32_decoder.mlir" "llama32 decoder (GQA, dequantize)"   & goto show_results
if /i "%TEST%"=="llama32_pre"       call :RunTest "llama32_prefill.mlir" "llama32 pre-fill (GQA, dequantize)"  & goto show_results
if /i "%TEST%"=="nemotron_dec"      call :RunTest "nemotron_decoder.mlir" "nemotron decoder"                   & goto show_results
if /i "%TEST%"=="nemotron_pre"      call :RunTest "nemotron_prefill.mlir" "nemotron pre-fill"                  & goto show_results
if /i "%TEST%"=="phi_silica"        call :RunTest "phi_silica_qdq.mlir"  "phi_silica_qdq (quantized)"          & goto show_results

echo Unknown test: %TEST%
echo.
echo Run without arguments to run all tests.
exit /b 1

REM ======================================
REM :RunAll - Run every test
REM ======================================
:RunAll
echo ======================================
echo DxGML MLIR Dialect Tests
echo ======================================
echo Arch:   %ARCH%
echo Driver: %DRIVER_PATH%
echo.

echo --- CompilationInput Tests ---
call :RunTest "ConvRelu.CompilationInput.mlir"          "ConvRelu (conv + relu)"
call :RunTest "Gelu.CompilationInput.mlir"              "Gelu (erf-based GELU)"
call :RunTest "ReluErf.CompilationInput.mlir"           "ReluErf (relu + erf)"
call :RunTest "StandaloneCluster.CompilationInput.mlir" "StandaloneCluster (pre/post-conv)"
echo.

echo --- Simple Model Tests ---
call :RunTest "model1.mlir"       "model1 (CNN with depth-to-space)"
call :RunTest "model2.mlir"       "model2 (CNN variant)"
call :RunTest "model3.mlir"       "model3 (CNN variant)"
call :RunTest "simple_gemm.mlir"  "simple_gemm (GEMM + bias + relu)"
call :RunTest "conv_example.mlir" "conv_example (Conv+BN+ReLU+MaxPool)"
echo.

echo --- Vision Model Tests ---
call :RunTest "audio2face.mlir"   "audio2face (with reduce ops)"
echo.

echo --- LLM Model Tests ---
call :RunTest "llama32_decoder.mlir"  "llama32 decoder (GQA, dequantize)"
call :RunTest "llama32_prefill.mlir"  "llama32 pre-fill (GQA, dequantize)"
call :RunTest "nemotron_decoder.mlir" "nemotron decoder"
call :RunTest "nemotron_prefill.mlir" "nemotron pre-fill"
call :RunTest "phi_silica_qdq.mlir"   "phi_silica_qdq (quantized)"
goto show_results

REM ======================================
REM :show_results
REM ======================================
:show_results
echo.
echo ======================================
echo Results: !PASS_COUNT! passed, !FAIL_COUNT! failed
echo ======================================
if !FAIL_COUNT! GTR 0 (
    echo.
    echo To debug a failure, run:
    echo   "!DRIVER_PATH!" "!MLIR_DIR!\<test>.mlir" --dxgml --parse
    exit /b 1
)
echo.
echo All tests passed!
exit /b 0

REM ======================================
REM Subroutine: RunTest <filename> <description>
REM ======================================
:RunTest
    set "MLIR_FILE=!MLIR_DIR!\%~1"
    set "DESC=%~2"

    if not exist "!MLIR_FILE!" (
        echo   [SKIP] !DESC! - file not found: !MLIR_FILE!
        goto :eof
    )

    "!DRIVER_PATH!" "!MLIR_FILE!" --dxgml --parse >nul 2>&1

    if !errorlevel! == 0 (
        echo   [PASS] !DESC!
        set /a PASS_COUNT+=1
    ) else (
        echo   [FAIL] !DESC!
        "!DRIVER_PATH!" "!MLIR_FILE!" --dxgml --parse 2>&1
        set /a FAIL_COUNT+=1
    )
    goto :eof
