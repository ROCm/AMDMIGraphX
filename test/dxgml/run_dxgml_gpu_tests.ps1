# run_dxgml_gpu_tests.ps1
#
# Run DxGML GPU tests in one of five modes:
#
#   parse   - Parse each model with migraphx-driver read --dxgml (no GPU)
#   compile - Compile each model for GPU with migraphx-driver compile --dxgml --gpu
#   run     - Compile + run on GPU (no CPU reference validation)
#   verify  - Compile + run + validate GPU output vs CPU reference (default)
#   profile - Compile + run under rocprofv2 (writes CSV trace to dump\)
#
# Usage:
#   .\run_dxgml_gpu_tests.ps1                       - verify all models
#   .\run_dxgml_gpu_tests.ps1 verify                - same as above
#   .\run_dxgml_gpu_tests.ps1 compile               - compile-only sweep
#   .\run_dxgml_gpu_tests.ps1 parse                 - parse-only sweep
#   .\run_dxgml_gpu_tests.ps1 run                   - GPU run without validation
#   .\run_dxgml_gpu_tests.ps1 profile               - profile all models
#   .\run_dxgml_gpu_tests.ps1 verify dump           - verify all + dump outputs
#   .\run_dxgml_gpu_tests.ps1 verify conv_act_add   - verify single model
#   .\run_dxgml_gpu_tests.ps1 compile conv_example  - compile single model
#   .\run_dxgml_gpu_tests.ps1 profile simple_gemm   - profile single model
#
# Validation (verify mode only):
#   --atol 1e-2 --rtol 1e-2  for fp16 models (half precision rounding)
#   --atol 1e-4 --rtol 1e-4  for fp32 models
#
# Profiling (profile mode):
#   Uses rocprofv2 if available, falls back to rocprof.
#   Output: dump\profile_<model>\ directory (CSV kernel trace + stats).

param(
    [string]$Mode  = "verify",   # parse | compile | run | verify | profile
    [string]$Suite = "all",      # all | dump | <model-name>
    [string]$Opt3  = ""
)

$ErrorActionPreference = "Continue"

# --- Normalize args ---
# Allow: .\script.ps1 dump            (Mode=verify, Suite=all, DoDump=true)
# Allow: .\script.ps1 verify dump     (Mode=verify, Suite=all, DoDump=true)
# Allow: .\script.ps1 verify modelX   (Mode=verify, Suite=modelX)
$ValidModes = @("parse","compile","run","verify","profile")
if ($ValidModes -notcontains $Mode.ToLower()) {
    # First arg is not a mode — treat as old-style (verify all / verify <model>)
    $Opt3  = $Suite
    $Suite = $Mode
    $Mode  = "verify"
}

$DoDump = ($Opt3 -ieq "dump") -or ($Suite -ieq "dump")
if ($Suite -ieq "dump") { $Suite = "all" }
$Mode = $Mode.ToLower()

# --- Locate directories ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir   = (Resolve-Path "$ScriptDir\..\..").Path
$MlirDir   = "$ScriptDir\mlir"
$DumpDir   = "$ScriptDir\dump"
if (!(Test-Path $DumpDir)) { New-Item -ItemType Directory -Path $DumpDir | Out-Null }
$LogFile   = "$DumpDir\gpu_run_results.log"
"" | Set-Content $LogFile

# --- Locate migraphx-driver.exe ---
$DriverPath = $null
$BinDir     = $null
foreach ($config in @("WinRelWithDebInfo","WinRelease","WinMinSizeRel","WinDebug")) {
    $c = "$RootDir\build\$config\bin\migraphx-driver.exe"
    if (Test-Path $c) { $DriverPath = $c; $BinDir = "$RootDir\build\$config\bin"; break }
}
if (!$DriverPath) {
    foreach ($config in @("RelWithDebInfo","Release","Debug","MinSizeRel")) {
        $c = "$RootDir\build_vs\bin\$config\migraphx-driver.exe"
        if (Test-Path $c) { $DriverPath = $c; $BinDir = "$RootDir\build_vs\bin\$config"; break }
    }
}
if (!$DriverPath) {
    Write-Host "[ERROR] migraphx-driver.exe not found. Build MIGraphX first."
    exit 1
}

$env:PATH = "$BinDir;C:\opt\rocm\bin;C:\Program Files\AMD\ROCm\6.2\bin;$env:PATH"

# --- Select discrete GPU >= gfx1100 ---
# Only attempt GPU execution on devices with gcnArchName >= gfx1100.
# Integrated GPUs and older discrete GPUs (gfx1036, etc.) lack the resources
# needed to JIT-compile and run these kernels.
$HipInfoCmd = Get-Command "hipInfo.exe" -ErrorAction SilentlyContinue
$HipInfoExe = if ($HipInfoCmd) { $HipInfoCmd.Source } else { $null }
if (!$HipInfoExe) {
    foreach ($cand in @("C:\opt\rocm\bin\hipInfo.exe","C:\Program Files\AMD\ROCm\6.2\bin\hipInfo.exe")) {
        if (Test-Path $cand) { $HipInfoExe = $cand; break }
    }
}

function Get-GfxNumber([string]$arch) {
    # Extract numeric part from gfxNNNN, e.g. "gfx1100" -> 1100
    if ($arch -match "gfx(\d+)") { return [int]$Matches[1] }
    return 0
}

$GpuDeviceId   = $null
$GpuDeviceName = $null
$GpuDeviceArch = $null
if ($HipInfoExe) {
    $hipOut  = & $HipInfoExe 2>&1
    $devIdx  = -1
    $curArch = ""
    $curName = ""
    foreach ($line in $hipOut) {
        if ($line -match "^device#\s+(\d+)")        { $devIdx = [int]$Matches[1]; $curArch = ""; $curName = "" }
        if ($line -match "Name:\s+(.+)")             { $curName = $Matches[1].Trim() }
        if ($line -match "gcnArchName:\s+(\S+)")     { $curArch = $Matches[1].Trim() }
        # Accept device once we have both name and arch for this index
        if ($devIdx -ge 0 -and $curArch -ne "") {
            if ((Get-GfxNumber $curArch) -ge 1100) {
                $GpuDeviceId   = $devIdx
                $GpuDeviceName = $curName
                $GpuDeviceArch = $curArch
            }
            $curArch = ""  # reset so we don't re-match same device
        }
    }
}

if ($null -ne $GpuDeviceId) {
    $env:HIP_VISIBLE_DEVICES  = "$GpuDeviceId"
    $env:ROCR_VISIBLE_DEVICES = "$GpuDeviceId"
} else {
    Write-Host "[WARN] No GPU >= gfx1100 found. GPU tests may fail."
}

# --- Locate ROCm profiler (profile mode) ---
$RocprofPath = $null
foreach ($cand in @("rocprofv2.exe","rocprof.exe")) {
    $found_prof = Get-Command $cand -ErrorAction SilentlyContinue
    if ($found_prof) { $RocprofPath = $found_prof.Source; break }
}
if ($Mode -eq "profile" -and !$RocprofPath) {
    Write-Host "[WARN] rocprofv2/rocprof not found on PATH. Profile mode will run without instrumentation."
}

# --- Logging ---
function Log([string]$msg = "") {
    Write-Host $msg
    Add-Content $LogFile $msg
}

# --- Counters ---
$pass = 0; $fail = 0; $skip = 0

# -------------------------------------------------------------------
# RunTest: run migraphx-driver in the chosen mode for one model file
#
# $mlirFile   - path to .mlir file
# $name       - display name
# $atol/$rtol - tolerances (verify mode, fp16: 1e-2, fp32: 1e-4)
# $timeoutSec - per-test timeout
# -------------------------------------------------------------------
function RunTest([string]$mlirFile, [string]$name,
                 [string]$atol = "1e-2", [string]$rtol = "1e-2",
                 [int]$timeoutSec = 180) {
    if (!(Test-Path $mlirFile)) {
        Log "  [SKIP] $name - file not found"
        $script:skip++; return
    }

    # Build argument list based on mode
    $execPath = $script:DriverPath
    $execArgs = @()

    switch ($script:Mode) {
        "parse" {
            $execArgs = @("read", $mlirFile, "--dxgml")
        }
        "compile" {
            $execArgs = @("compile", $mlirFile, "--dxgml", "--gpu")
        }
        "run" {
            $execArgs = @("run", $mlirFile, "--dxgml", "--gpu")
        }
        "profile" {
            # rocprofv2: --output-directory <dir> --kernel-trace -- <exe> <args>
            # rocprof   (legacy): -o <csv> <exe> <args>
            $profDir = "$script:DumpDir\profile_$name"
            if (!(Test-Path $profDir)) { New-Item -ItemType Directory -Path $profDir | Out-Null }

            $driverCore = @("run", $mlirFile, "--dxgml", "--gpu")

            if ($script:RocprofPath -and ($script:RocprofPath -match "rocprofv2")) {
                $execPath = $script:RocprofPath
                $execArgs = @(
                    "--output-directory", $profDir,
                    "--kernel-trace",
                    "--",
                    $script:DriverPath
                ) + $driverCore
            } elseif ($script:RocprofPath) {
                # legacy rocprof
                $execPath = $script:RocprofPath
                $execArgs = @("-o", "$profDir\trace.csv",
                              $script:DriverPath) + $driverCore
            } else {
                # no profiler — run plain, still useful for timing
                $execArgs = $driverCore
            }
        }
        default {  # verify
            $execArgs = @(
                "verify", $mlirFile,
                "--dxgml", "--gpu",
                "--atol", $atol,
                "--rtol", $rtol
            )
        }
    }

    # Use .NET ProcessStartInfo with RedirectStandardOutput/Error so that
    # env vars set in this PS session (HIP_VISIBLE_DEVICES etc.) are propagated
    # to the child process without shell quoting issues.
    $psi                         = [System.Diagnostics.ProcessStartInfo]::new($execPath)
    $psi.UseShellExecute         = $false
    $psi.RedirectStandardOutput  = $true
    $psi.RedirectStandardError   = $true
    $psi.CreateNoWindow          = $true
    # Build argument string (quote args that contain spaces)
    $psi.Arguments = ($execArgs | ForEach-Object {
        if ($_ -match '\s') { "`"$_`"" } else { $_ }
    }) -join " "
    # Explicitly copy key env vars so the child sees our HIP_VISIBLE_DEVICES
    if ($env:HIP_VISIBLE_DEVICES)  { $psi.EnvironmentVariables["HIP_VISIBLE_DEVICES"]  = $env:HIP_VISIBLE_DEVICES }
    if ($env:ROCR_VISIBLE_DEVICES) { $psi.EnvironmentVariables["ROCR_VISIBLE_DEVICES"] = $env:ROCR_VISIBLE_DEVICES }
    $psi.EnvironmentVariables["PATH"] = $env:PATH

    $proc = [System.Diagnostics.Process]::Start($psi)

    # Read stdout and stderr asynchronously to prevent deadlock
    $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
    $stderrTask = $proc.StandardError.ReadToEndAsync()

    $finished = $proc.WaitForExit($timeoutSec * 1000)

    if (!$finished) {
        $proc.Kill()
        Log "  [TIMEOUT] $name  (>${timeoutSec}s)"
        $script:fail++
        return
    }

    $exitCode = $proc.ExitCode
    $stdout   = $stdoutTask.Result
    $stderr   = $stderrTask.Result
    $combined = "$stdout`n$stderr"

    if ($exitCode -eq 0) {
        Log "  [PASS] $name"
        $script:pass++
    } else {
        $errLine = ($combined -split "`n" | Where-Object { $_ -match "Error|error|FAIL|bad_alloc|mismatch" } | Select-Object -First 1)
        if (!$errLine) { $errLine = "(exit $exitCode - run with dump flag for full output)" }
        Log "  [FAIL] $name"
        Log "         $errLine"
        $script:fail++
    }

    if ($DoDump -or ($script:Mode -eq "profile")) {
        if ($script:Mode -eq "profile") {
            $dumpFile = "$script:DumpDir\profile_${name}\driver_output.txt"
        } else {
            $dumpFile = "$script:DumpDir\gpu_${name}_output.txt"
        }
        $combined | Set-Content $dumpFile
        Log "         > $dumpFile"
    }

    if ($script:Mode -eq "profile" -and (Test-Path "$script:DumpDir\profile_$name")) {
        $csvFiles = Get-ChildItem "$script:DumpDir\profile_$name" -Filter "*.csv" -ErrorAction SilentlyContinue
        if ($csvFiles) {
            foreach ($csv in $csvFiles) { Log "         > $($csv.FullName)" }
        }
    }
}

# -------------------------------------------------------------------
# Test definitions — all models that currently compile successfully.
# atol/rtol 1e-2 for fp16 (half precision), 1e-4 for fp32.
# Large models get a longer timeout.
# -------------------------------------------------------------------
function RunAllTests {
    Log "--- Compilation inputs (original fixtures) ---"
    RunTest "$MlirDir\ConvRelu.CompilationInput.mlir"      "ConvRelu"          -atol "1e-2" -rtol "1e-2" -timeoutSec 120
    RunTest "$MlirDir\Gelu.CompilationInput.mlir"          "Gelu"              -atol "1e-2" -rtol "1e-2" -timeoutSec 120
    RunTest "$MlirDir\ReluErf.CompilationInput.mlir"       "ReluErf"           -atol "1e-2" -rtol "1e-2" -timeoutSec 120
    RunTest "$MlirDir\StandaloneCluster.CompilationInput.mlir" "StandaloneCluster" -atol "1e-2" -rtol "1e-2" -timeoutSec 120

    Log ""
    Log "--- Simple models ---"
    RunTest "$MlirDir\simple_gemm\model.mlir"              "simple_gemm"       -atol "1e-2" -rtol "1e-2"
    RunTest "$MlirDir\conv_example\model.mlir"             "conv_example"      -atol "1e-4" -rtol "1e-4"
    RunTest "$MlirDir\test\test_dxgml.mlir"                "test_dxgml"        -atol "1e-2" -rtol "1e-2"

    Log ""
    Log "--- Standalone unit models ---"
    RunTest "$MlirDir\standalone\conv_act_add.mlir"            "conv_act_add"           -atol "1e-2" -rtol "1e-2"
    RunTest "$MlirDir\standalone\conv_relu_mul.mlir"           "conv_relu_mul"          -atol "1e-2" -rtol "1e-2"
    RunTest "$MlirDir\standalone\gemm_add.mlir"                "gemm_add"               -atol "1e-2" -rtol "1e-2"
    RunTest "$MlirDir\standalone\gemm_relu_add.mlir"           "gemm_relu_add"          -atol "1e-2" -rtol "1e-2"
    RunTest "$MlirDir\standalone\gemm_gemm_add.mlir"           "gemm_gemm_add"          -atol "1e-2" -rtol "1e-2"
    RunTest "$MlirDir\standalone\transpose_conv_transpose.mlir" "transpose_conv_transpose" -atol "1e-2" -rtol "1e-2"
    RunTest "$MlirDir\standalone\group_query_attention.mlir"   "group_query_attention"  -atol "1e-2" -rtol "1e-2"
    RunTest "$MlirDir\standalone\qkv_projection.mlir"          "qkv_projection"         -atol "1e-2" -rtol "1e-2"
    RunTest "$MlirDir\standalone\qkv_projection_2.mlir"        "qkv_projection_2"       -atol "1e-2" -rtol "1e-2"

    Log ""
    Log "--- CNN models ---"
    RunTest "$MlirDir\model3\model.mlir"                       "model3"                 -atol "1e-2" -rtol "1e-2" -timeoutSec 300
    RunTest "$MlirDir\model3\model_test.mlir"                  "model3_test"            -atol "1e-2" -rtol "1e-2" -timeoutSec 300
}

# -------------------------------------------------------------------
# Dispatch
# -------------------------------------------------------------------
$totalStart = Get-Date

$modeLabel = switch ($Mode) {
    "parse"   { "DxGML Parse Tests (no GPU)" }
    "compile" { "DxGML Compile Tests (GPU kernel compile)" }
    "run"     { "DxGML GPU Run Tests (no validation)" }
    "profile" { "DxGML GPU Profile Tests (rocprofv2 kernel trace)" }
    default   { "DxGML GPU Verify Tests (GPU vs CPU reference)" }
}

switch -Wildcard ($Suite.ToLower()) {
    "all" {
        Log "====================================================="
        Log $modeLabel
        Log "====================================================="
        Log "Driver:  $DriverPath"
        Log "Mode:    $Mode"
        if ($null -ne $GpuDeviceId) {
            Log "GPU:     device $GpuDeviceId - $GpuDeviceName ($GpuDeviceArch)"
        } else {
            Log "GPU:     (no device >= gfx1100 detected)"
        }
        if ($Mode -eq "profile") {
            if ($RocprofPath) { Log "Profiler: $RocprofPath" }
            else              { Log "Profiler: (not found - running without instrumentation)" }
        }
        Log "Log:     $LogFile"
        Log "Dump:    $DoDump"
        Log ""
        RunAllTests
    }
    default {
        # Single model by name
        $found = $false
        Get-ChildItem -Path $MlirDir -Recurse -Filter "*.mlir" | ForEach-Object {
            if ($_.BaseName -ieq $Suite -or $_.Name -ieq $Suite) {
                $atol = "1e-2"; $rtol = "1e-2"
                if ($_.Name -match "conv_example") { $atol = "1e-4"; $rtol = "1e-4" }
                RunTest $_.FullName $Suite -atol $atol -rtol $rtol -timeoutSec 300
                $script:found = $true
            }
        }
        if (!$found) {
            Log "[ERROR] No .mlir file matching '$Suite' found under $MlirDir"
            exit 1
        }
    }
}

$elapsed = ((Get-Date) - $totalStart).TotalSeconds

Log ""
Log "====================================================="
Log "Results [$Mode]: $pass passed, $fail failed, $skip skipped  ($([math]::Round($elapsed,1))s)"
Log "====================================================="
if ($DoDump) { Log "Dump files: $DumpDir" }
Log "Full log:   $LogFile"

if ($fail -gt 0) { exit 1 }
Log "All tests passed!"
exit 0
