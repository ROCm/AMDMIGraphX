@echo off
REM ============================================================
REM run_dxgml_tests.bat  — thin wrapper around run_dxgml_tests.ps1
REM
REM Usage:
REM   run_dxgml_tests.bat                   - Run all tests
REM   run_dxgml_tests.bat parse             - C++ parse unit tests only
REM   run_dxgml_tests.bat mlir              - Driver tests only
REM   run_dxgml_tests.bat mlir --gpu        - Driver tests on GPU
REM   run_dxgml_tests.bat mlir --verify     - Driver tests using verify command
REM   run_dxgml_tests.bat dump              - All tests + dump output
REM   run_dxgml_tests.bat simple_gemm       - Single model test
REM   run_dxgml_tests.bat mlir dump gfx1201 - Driver tests + dump + arch
REM ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dxgml_tests.ps1" %*
exit /b %ERRORLEVEL%
