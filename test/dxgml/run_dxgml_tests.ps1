# run_dxgml_tests.ps1
#
# Run DxGML MLIR dialect tests using migraphx-driver.exe.
# Works from any shell (PowerShell, bash, cmd).
#
# Usage:
#   .\run_dxgml_tests.ps1                     - Run all tests
#   .\run_dxgml_tests.ps1 parse               - C++ parse unit tests only
#   .\run_dxgml_tests.ps1 mlir                - Driver tests only
#   .\run_dxgml_tests.ps1 mlir --gpu          - Driver tests on GPU
#   .\run_dxgml_tests.ps1 mlir --verify        - Driver tests using verify command
#   .\run_dxgml_tests.ps1 dump                - All tests + dump output
#   .\run_dxgml_tests.ps1 simple_gemm         - Single model test
#   .\run_dxgml_tests.ps1 mlir dump gfx1201   - Driver tests + dump + arch
#   .\run_dxgml_tests.ps1 all --gpu dump      - Full run, MLIR tests on GPU, with dump

param(
    [string]$Suite  = "all",
    [string]$Opt2   = "",
    [string]$Arch   = "gfx1201",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Continue"

# --- Normalize args ---
$rawArgs = @()
foreach ($token in @($Suite, $Opt2, $Arch) + $ExtraArgs) {
    if (-not [string]::IsNullOrWhiteSpace($token)) {
        $rawArgs += $token
    }
}

$DoDump   = ($rawArgs | Where-Object { $_ -ieq "dump" }).Count -gt 0
$DoGpu    = ($rawArgs | Where-Object { $_ -ieq "--gpu" -or $_ -ieq "-g" }).Count -gt 0
$DoVerify = ($rawArgs | Where-Object { $_ -ieq "--verify" -or $_ -ieq "-v" }).Count -gt 0

$positionalArgs = $rawArgs | Where-Object {
    ($_ -ine "dump") -and ($_ -ine "--gpu") -and ($_ -ine "-g") -and ($_ -ine "--verify") -and ($_ -ine "-v")
}

if ($positionalArgs.Count -ge 1) {
    $Suite = $positionalArgs[0]
} else {
    $Suite = "all"
}

if ($positionalArgs.Count -ge 2) {
    $Arch = $positionalArgs[1]
} else {
    $Arch = "gfx1201"
}

if ($Suite -ieq "dump") { $Suite = "all" }

# --- Locate directories ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir   = (Resolve-Path "$ScriptDir\..\.." ).Path
$MlirDir   = "$ScriptDir\mlir"
$DumpDir   = "$ScriptDir\dump"

if (!(Test-Path $DumpDir)) { New-Item -ItemType Directory -Path $DumpDir | Out-Null }
$LogFile = "$DumpDir\run_results.log"
"" | Set-Content $LogFile

# --- Logging helper ---
function Log([string]$msg = "") {
    Write-Host $msg
    Add-Content $LogFile $msg
}

# --- Locate migraphx-driver.exe ---
$DriverPath = $null
$BinDir     = $null
foreach ($config in @("WinRelWithDebInfo","WinRelease","WinRelMinSizeRel","Debug", "WinDebug")) {
    $candidate = "$RootDir\build\$config\bin\migraphx-driver.exe"
    if (Test-Path $candidate) { $DriverPath = $candidate; $BinDir = "$RootDir\build\$config\bin"; break }
}
if (!$DriverPath -and (Test-Path "$RootDir\build\bin\migraphx-driver.exe")) {
    $DriverPath = "$RootDir\build\bin\migraphx-driver.exe"
    $BinDir     = "$RootDir\build\bin"
}
# Visual Studio multi-config layout: build_vs\bin\<Config>\
if (!$DriverPath) {
    foreach ($config in @("RelWithDebInfo","Release","Debug","MinSizeRel")) {
        $candidate = "$RootDir\build_vs\bin\$config\migraphx-driver.exe"
        if (Test-Path $candidate) { $DriverPath = $candidate; $BinDir = "$RootDir\build_vs\bin\$config"; break }
    }
}
if (!$DriverPath) {
    Log "[ERROR] migraphx-driver.exe not found. Build MIGraphX first."
    exit 1
}

# Extend PATH so driver can find DLLs
$env:PATH = "$BinDir;C:\opt\rocm\bin;$env:PATH"
$ParseBinDir = $BinDir

# --- Counters ---
$pass = 0; $fail = 0; $skip = 0

# --- Test runners ---
function RunDriverTest([string]$mlirFile, [string]$name) {
    if (!(Test-Path $mlirFile)) {
        Log "  [SKIP] $name - file not found"
        $script:skip++; return
    }

    if ($script:DoVerify) {
        $driverArgs = @("verify", $mlirFile, "--dxgml", "--skip-unknown-operators")
    } elseif ($script:DoGpu) {
        $driverArgs = @("compile", $mlirFile, "--dxgml", "--skip-unknown-operators", "--gpu")
    } else {
        $driverArgs = @("read", $mlirFile, "--dxgml", "--skip-unknown-operators")
    }

    $out = & $DriverPath @driverArgs 2>&1
    if ($LASTEXITCODE -eq 0) {
        Log "  [PASS] $name"
        $script:pass++
    } else {
        Log "  [FAIL] $name"
        $out | ForEach-Object { Log "         $_" }
        $script:fail++
    }
    if ($DoDump) {
        $dumpFile = "$DumpDir\${name}_migraphx_ops.txt"
        Log "         > $dumpFile"
        if ($script:DoGpu) {
            & $DriverPath compile $mlirFile --dxgml --skip-unknown-operators --gpu --text 2>&1 | Set-Content $dumpFile
        } else {
            & $DriverPath read $mlirFile --dxgml --skip-unknown-operators --text 2>&1 | Set-Content $dumpFile
        }
    }
}

function RunUnitTest([string]$exeName, [string]$desc) {
    $exePath = "$ParseBinDir\$exeName"
    if (!(Test-Path $exePath)) {
        Log "  [SKIP] $desc - binary not found: $exePath"
        $script:skip++; return
    }
    $out = & $exePath 2>&1
    if ($LASTEXITCODE -eq 0) {
        Log "  [PASS] $desc"
        $script:pass++
    } else {
        Log "  [FAIL] $desc"
        $out | ForEach-Object { Log "         $_" }
        $script:fail++
    }
}

function RunParseSuite {
    Log "--- Parse Unit Tests (C++ test binaries) ---"
    RunUnitTest "test_dxgml_conv_relu_test.exe"          "ConvRelu parse"
    RunUnitTest "test_dxgml_gelu_test.exe"               "Gelu parse"
    RunUnitTest "test_dxgml_relu_erf_test.exe"           "ReluErf parse"
    RunUnitTest "test_dxgml_standalone_cluster_test.exe" "StandaloneCluster parse"
}

function RunMlirSuite {
    Log "--- MLIR Driver Tests ---"

    Log "[simple models]"
    RunDriverTest "$MlirDir\simple_gemm\model.mlir"      "simple_gemm"
    RunDriverTest "$MlirDir\conv_example\model.mlir"     "conv_example"
    RunDriverTest "$MlirDir\test\test_dxgml.mlir"        "test_dxgml"
    RunDriverTest "$MlirDir\test\test_model1_clean.mlir" "test_model1_clean"

    Log "[standalone tests]"
    RunDriverTest "$MlirDir\standalone\conv_act_add.mlir"           "conv_act_add"
    RunDriverTest "$MlirDir\standalone\conv_relu_mul.mlir"          "conv_relu_mul"
    RunDriverTest "$MlirDir\standalone\gemm_add.mlir"               "gemm_add"
    RunDriverTest "$MlirDir\standalone\gemm_relu_add.mlir"          "gemm_relu_add"
    RunDriverTest "$MlirDir\standalone\gemm_gemm_add.mlir"            "gemm_gemm_add"
    RunDriverTest "$MlirDir\standalone\transpose_conv_transpose.mlir" "transpose_conv_transpose"
    RunDriverTest "$MlirDir\standalone\group_query_attention.mlir"    "group_query_attention"
    RunDriverTest "$MlirDir\standalone\qkv_projection.mlir"          "qkv_projection"
    RunDriverTest "$MlirDir\standalone\qkv_projection_2.mlir"        "qkv_projection_2"
    RunDriverTest "$MlirDir\standalone\gqa_attention.mlir"           "gqa_attention"

    Log "[CNN models]"
    RunDriverTest "$MlirDir\model1\model.mlir"      "model1"
    RunDriverTest "$MlirDir\model1\model_test.mlir" "model1_test"
    RunDriverTest "$MlirDir\model2\model.mlir"      "model2"
    RunDriverTest "$MlirDir\model2\model_test.mlir" "model2_test"
    RunDriverTest "$MlirDir\model3\model.mlir"      "model3"
    RunDriverTest "$MlirDir\model3\model_test.mlir" "model3_test"

    # Log "[vision models]"
    # RunDriverTest "$MlirDir\audio2face\model.mlir"      "audio2face"
    # RunDriverTest "$MlirDir\audio2face\model_test.mlir" "audio2face_test"

    # Log "[LLM models]"
    # RunDriverTest "$MlirDir\llama32\llama32_dxgml_static_decoder.mlir"       "llama32_decoder"
    # RunDriverTest "$MlirDir\llama32\llama32_dxgml_static_pre-fill.mlir"      "llama32_prefill"
    # RunDriverTest "$MlirDir\llama32\llama32_dxgml_static_decoder_test.mlir"  "llama32_decoder_test"
    # RunDriverTest "$MlirDir\llama32\llama32_dxgml_static_pre-fill_test.mlir" "llama32_prefill_test"
    # RunDriverTest "$MlirDir\nemotron\model_decoder.mlir"      "nemotron_decoder"
    # RunDriverTest "$MlirDir\nemotron\model_pre-fill.mlir"     "nemotron_prefill"
    # RunDriverTest "$MlirDir\nemotron\model_decoder_test.mlir" "nemotron_decoder_test"
    # RunDriverTest "$MlirDir\nemotron\model_pre-fill_test.mlir" "nemotron_prefill_test"
    # RunDriverTest "$MlirDir\phi_silica_qdq\model.mlir"        "phi_silica_qdq"
}

# --- Dispatch ---
switch -Wildcard ($Suite.ToLower()) {
    "all" {
        Log "====================================================="
        Log "DxGML Test Suite - FULL RUN"
        Log "====================================================="
        Log "Arch:   $Arch"
        Log "GPU:    $DoGpu"
        Log "Driver: $DriverPath"
        Log "Dump:   $DumpDir  (enabled: $DoDump)"
        Log "Log:    $LogFile"
        Log ""
        RunParseSuite
        Log ""
        RunMlirSuite
    }
    "mlir" {
        Log "====================================================="
        Log "DxGML Test Suite - MLIR Driver Tests"
        Log "====================================================="
        Log "Arch:   $Arch"
        Log "GPU:    $DoGpu"
        Log "Driver: $DriverPath"
        Log "Log:    $LogFile"
        Log ""
        RunMlirSuite
    }
    "parse" {
        Log "====================================================="
        Log "DxGML Test Suite - Parse Unit Tests"
        Log "====================================================="
        Log "BinDir: $ParseBinDir"
        Log "Log:    $LogFile"
        Log ""
        RunParseSuite
    }
    default {
        # Single model
        $found = $false
        $candidate = "$MlirDir\$Suite\model.mlir"
        if (Test-Path $candidate) {
            RunDriverTest $candidate $Suite
            $found = $true
        } else {
            Get-ChildItem -Path $MlirDir -Recurse -Filter "*.mlir" | ForEach-Object {
                if ($_.BaseName -ieq $Suite) {
                    RunDriverTest $_.FullName $Suite
                    $script:found = $true
                }
            }
        }
        if (!$found) {
            Log "[ERROR] Unknown test or model: $Suite"
            exit 1
        }
    }
}

# --- Results ---
Log ""
Log "====================================================="
Log "Results: $pass passed, $fail failed, $skip skipped"
Log "====================================================="
if ($DoDump) { Log "Dump files: $DumpDir" }
Log "Full log: $LogFile"

if ($fail -gt 0) {
    Log ""
    Log "To debug a failure:"
    if ($DoGpu) {
        Log "  migraphx-driver.exe compile <file.mlir> --dxgml --skip-unknown-operators --gpu --text"
    } else {
        Log "  migraphx-driver.exe read <file.mlir> --dxgml --skip-unknown-operators --text"
    }
    exit 1
}
Log "All tests passed!"
exit 0
