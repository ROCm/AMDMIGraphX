@echo off
REM ============================================================
REM run_dxgml_gpu_tests.bat  — thin wrapper around run_dxgml_gpu_tests.ps1
REM
REM Usage:
REM   run_dxgml_gpu_tests.bat                        - Verify all (GPU vs CPU ref)
REM   run_dxgml_gpu_tests.bat verify                 - Same as above
REM   run_dxgml_gpu_tests.bat parse                  - Parse only (no GPU)
REM   run_dxgml_gpu_tests.bat compile                - Compile for GPU, no run
REM   run_dxgml_gpu_tests.bat run                    - GPU run, no validation
REM   run_dxgml_gpu_tests.bat profile                - GPU profiling (rocprof/rocprofv2)
REM   run_dxgml_gpu_tests.bat verify dump            - Verify all + dump outputs
REM   run_dxgml_gpu_tests.bat verify conv_act_add    - Verify single model
REM   run_dxgml_gpu_tests.bat compile conv_example   - Compile single model
REM   run_dxgml_gpu_tests.bat profile simple_gemm    - Profile single model
REM ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dxgml_gpu_tests.ps1" %1 %2 %3 %4
exit /b %ERRORLEVEL%
