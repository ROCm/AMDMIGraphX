# Copyright (C) 2025 Advanced Micro Devices Inc.

param(
    [ValidateScript({ Test-Path -Path $_ })]
    [string]$sourceDir,
    [string]$buildDir,
    [string]$installDir,
#    [ValidateScript({
#        $allowedConfigs = @('Release', 'RelWithDebInfo', 'MinSizeRel', 'Debug')
#        $configurations = $_ -split ',' | ForEach-Object { $_.Trim() }
#        ($configurations -le $allowedConfigs) -and ($configurations.Count -eq ($configurations | Select-Object -Unique).Count)
#    })]
    [string]$buildType,
    [string[]]$defines,
    [ValidateSet("default", "hipSdk", "clangCl")]
    [string]$toolchain,
    [switch]$force = $false,
    [string]$configJson,
    [switch]$minimal = $false,
    [switch]$binSkim = $false,
    [string]$targets,
    [ValidateScript({ Test-Path -Path $_ })]
    [string]$hipPath,
    [ValidateScript({ Test-Path -Path $_ })]
    [string]$rocmlirPath,
    [switch]$skipConfigure = $false,
    [switch]$skipInstall = $false,
    [int]$jobs = [Math]::Max([Environment]::ProcessorCount - 2, 1)
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$PSNativeCommandUseErrorActionPreference = $true

function Remove-File {
    param (
        [string]$BasePath,
        [string]$FileName
    )
    $Path = Join-Path -Path $BasePath -ChildPath $FileName
    if (Test-Path -Path $Path) {
        Remove-Item -Path $Path -Force -ProgressAction SilentlyContinue
    }
}

function Invoke-Call {
    param (
        [scriptblock]$ScriptBlock,
        [string]$ErrorCode = $ErrorActionPreference
    )
    & @ScriptBlock
    if (($LASTEXITCODE -ne 0) -and $ErrorAction -eq 'Stop') {
        exit $LASTEXITCODE
    }
}

if (-not $sourceDir -or $sourceDir.Trim() -eq '') {
    $sourceDir = (Get-Location).Path
}
if (-not $buildDir -or $buildDir.Trim() -eq '') {
    $buildDir = Join-Path -Path $sourceDir -ChildPath 'build'
}
if (-not $installDir -or $installDir.Trim() -eq '') {
    $installDir = Join-Path -Path $sourceDir -ChildPath 'install'
}
$configurations = @('Debug', 'Release')
if ($buildType -and $buildType -ne '') {
    $configurations = @($buildType -split ',' | Where-Object { $_.Trim() -ne '' })
}
$parentDir = Split-Path -Path $installDir -Parent
if ($minimal) {
    $configJsonDefault = 'minimal'
} else {
    $configJsonDefault = 'default'
}
if (-not $configJson -or $configJson.Trim() -eq '') {
    $configJson = $configJsonDefault
}
if ($binSkim) {
    $configJson = "$configJson.binskim"
}
$configJson = Join-Path -Path $sourceDir -ChildPath "$configJson.json"
if (-not $hipPath -or $hipPath.Trim() -eq '') {
    if ($env:HIP_PATH) {
        $hipPath = "$env:HIP_PATH"
    } elseif ($env:ROCM_PATH) {
        $hipPath = "$env:ROCM_PATH"
    } else {
        $hipPath = "$parentDir\rocm"
    }
}
if($skipConfigure -and $force) {
    Write-Error 'SkipConfigure and Force cannot be used together'
    Exit
}
$jsonContent = @{}
if (Test-Path -Path $configJson) {
    $jsonContent = Get-Content -Path $configJson -Raw | ConvertFrom-Json
}
$defaultToolchain = 'default'
if ($jsonContent -and $jsonContent.PSObject.Properties.Name -contains 'toolchain') {
    $defaultToolchain = $jsonContent.toolchain
}
if (-not $toolchain -or $toolchain.Trim() -eq '') {
    $toolchain = $defaultToolchain
}
$buildDict = @{}
if ($toolchain -eq 'hipSdk') {
    $buildDict['CMAKE_C_COMPILER'] = "$hipPath\bin\clang.exe"
    $buildDict['CMAKE_CXX_COMPILER'] = "$hipPath\bin\clang++.exe"
} elseif ($toolchain -eq 'clangCl') {
    $installDir = "$installDir.cl"
    $buildDir = "$buildDir.cl"
    $buildDict['CMAKE_C_COMPILER'] = "$hipPath\bin\clang-cl.exe"
    $buildDict['CMAKE_CXX_COMPILER'] = "$hipPath\bin\clang-cl.exe"
}
if ($binSkim) {
    $buildDir = "$buildDir.binskim"
    $installDir = "$installDir.binskim"
}
if ($jsonContent -and $jsonContent.PSObject.Properties.Name -contains 'compileWarningAsError') {
    if ($jsonContent.compileWarningAsError) {
        $buildDict["CMAKE_COMPILE_WARNING_AS_ERROR"] = "ON"
    }
}
$targetsDefault = @("all")
if (-not $targets -or $targets.Trim() -eq '') {
    if ($jsonContent -and $jsonContent.PSObject.Properties.Name -contains 'targets') {
        $listTargets = $jsonContent.targets
    } else {
        $listTargets = $targetsDefault
    }
} else {
    $listTargets = @($targets -split ',' | Where-Object { $_.Trim() -ne '' })
}
$depPrefix = ''
if ($binSkim) {
    $depPrefix = '.binskim'
}
$configurations | ForEach-Object {
    $buildType = $_
    Write-Host "Building configuration '$buildType'...";
    $buildPath = Join-Path -Path $buildDir -ChildPath $buildType
    if (-Not (Test-Path -Path $buildPath)) {
        New-Item -ItemType Directory -Path $buildPath -Force | Out-Null
    } elseif ($force) {
        Remove-Item -Path $buildPath -Recurse -Force -ProgressAction SilentlyContinue
    } elseif (-not $skipConfigure) {
        Remove-File -BasePath $buildPath -FileName "CMakeCache.txt"
        Remove-File -BasePath $buildPath -FileName "CMakeFiles\\cmake.check_cache"
    }
    $buildDict["CMAKE_BUILD_TYPE"] = "$buildType"
    if ($jsonContent -and $jsonContent.PSObject.Properties.Name -contains "cacheVariables") {
        foreach ($key in $jsonContent.cacheVariables.PSObject.Properties.Name) {
            $buildDict[$key] = $ExecutionContext.InvokeCommand.ExpandString($jsonContent.cacheVariables.$key)
        }
    }
    if ($defines -and $defines.Trim() -ne '') {
        $defines | ForEach-Object {
            $s = $_.Trim() -split "="
            $buildDict[$s[0].Trim()] = $ExecutionContext.InvokeCommand.ExpandString($s[-1].Trim())
        }
    }
    $cmakePrefixPath = "$sourceDir\depend\$buildType"
    if ($jsonContent -and $jsonContent.PSObject.Properties.Name -contains "depends") {
        $cmakePrefixPath += $jsonContent.depends.GetEnumerator() | ForEach-Object { if ($_ -eq "rocmlir" -and $rocmlirPath) { ";$rocmlirPath\$buildType" } else { ";$parentDir\$_$depPrefix\$buildType" } }
    }
    $buildDict["CMAKE_PREFIX_PATH"] = $cmakePrefixPath
    $buildDefines = $buildDict.GetEnumerator() | ForEach-Object { "-D$($_.Key)=$($_.Value)" }
    Write-Output $buildDefines
    if (-not $skipConfigure) {
        Invoke-Call -ScriptBlock { cmake -S $sourceDir -B $buildPath -G Ninja $buildDefines -Wno-deprecated -Wno-dev }
    }
    Invoke-Call -ScriptBlock { cmake --build $buildPath --config $buildType -j $jobs --target $listTargets }
    if (-not $skipInstall) {
        $prefixPath = Join-Path -Path $installDir -ChildPath $buildType
        if (Test-Path -Path $prefixPath) {
           Remove-Item -Path $prefixPath -Recurse -Force -ProgressAction SilentlyContinue
        }
        Invoke-Call -ScriptBlock { cmake --install $buildPath --prefix $prefixPath --config $buildType }
    }
}
