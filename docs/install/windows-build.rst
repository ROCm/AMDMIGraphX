.. meta::
  :description: Building MIGraphX on Windows with TheRock ROCm
  :keywords: install, build, MIGraphX, AMD, ROCm, Windows, TheRock

********************************************************************
Building MIGraphX on Windows
********************************************************************

This guide describes how to build MIGraphX on Windows using TheRock ROCm.

Prerequisites
====================================================================

Before beginning, ensure you have the following installed:

* **Visual Studio 2022** (or Visual Studio 2022 Build Tools)
  
  - In the installer, select "Desktop development with C++" under the Desktop & Mobile section
  - In Individual Components, select "C++ Clang Compiler for Windows" and "MSBuild support for LLVM (clang-cl)" toolset

* **CMake 3.15 or later**
  
  - CMake version 4 may cause compatibility issues; version 3.x is recommended

* **Python 3.x**
  
  - Python 3.12.x is recommended, as newer versions may have compatibility issues with the cget tool

* **Git**

ROCm installation
====================================================================

1. Download the appropriate TheRock nightly tarball for your GPU architecture from the 
   `TheRock releases page <https://github.com/ROCm/TheRock/blob/main/RELEASES.md>`__.

   Example for gfx1151:
   
   .. code:: text

      https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-windows-gfx1151-<version>.tar.gz

2. Extract the contents to ``C:\opt\rocm``.

.. note::

   If you choose a different installation path, you will need to adjust the compiler paths 
   in the build command and add ``-DCMAKE_PREFIX_PATH=<your_path>`` to the build command.

Build MIGraphX
====================================================================

Open Windows PowerShell and navigate to the AMDMIGraphX directory:

.. code:: powershell

   cd AMDMIGraphX

Create and activate a Python virtual environment:

.. code:: powershell

   python -m venv venv_rbuild
   .\venv_rbuild\Scripts\Activate.ps1

.. note::

   If you encounter script execution errors, run:
   
   .. code:: powershell
   
      Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

Install rbuild:

.. code:: powershell

   pip install https://github.com/RadeonOpenCompute/rbuild/archive/windows.tar.gz

Build MIGraphX:

.. code:: powershell

   rbuild build -d depend -B build -DGPU_TARGETS=<target> -DCMAKE_C_COMPILER=C:/opt/rocm/lib/llvm/bin/clang.exe -DCMAKE_CXX_COMPILER=C:/opt/rocm/lib/llvm/bin/clang++.exe


Runtime setup
====================================================================

After the build completes successfully, copy the required ROCm runtime DLLs to the 
MIGraphX build directory:

.. code:: powershell

   Copy-Item "C:\opt\rocm\bin\amdhip64_7.dll" ".\build\bin\" -Force
   Copy-Item "C:\opt\rocm\bin\hiprtc0702.dll" ".\build\bin\" -Force
   Copy-Item "C:\opt\rocm\bin\hiprtc-builtins0702.dll" ".\build\bin\" -Force
   Copy-Item "C:\opt\rocm\bin\amd_comgr0702.dll" ".\build\bin\" -Force

Set the PATH environment variable to include the ROCm bin directory:

.. code:: powershell

   $env:PATH = "C:\opt\rocm\bin;" + $env:PATH

Verify the build
====================================================================

Test the build by running:

.. code:: powershell

   .\build\bin\migraphx-driver.exe perf <path_to_onnx_model>

.. note::

   If nothing happens when you run this command, verify that ``$env:PATH`` includes 
   the ROCm bin directory.

Troubleshooting
====================================================================

Script execution policy error
--------------------------------------------------------------------

If PowerShell blocks script execution, run:

.. code:: powershell

   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

Custom ROCm installation path
--------------------------------------------------------------------

If ROCm is installed to a different path than ``C:\opt\rocm``, adjust the build command:

.. code:: powershell

   rbuild build -d depend -B build -DGPU_TARGETS=<target> -DCMAKE_PREFIX_PATH=<your_rocm_path> -DCMAKE_C_COMPILER=<your_rocm_path>/lib/llvm/bin/clang.exe -DCMAKE_CXX_COMPILER=<your_rocm_path>/lib/llvm/bin/clang++.exe
