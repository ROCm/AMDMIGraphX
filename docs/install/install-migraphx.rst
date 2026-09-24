.. meta::
  :description: Installing MIGraphX for ROCm
  :keywords: install, pip, build, MIGraphX, AMD, ROCm, development, contributing

.. |MIGRAPHX_VERSION| replace:: 2.16.0
.. |ROCM_VERSION| replace:: 7.14.1
.. |PKG_REPO| replace:: https://rocm.frameworks.amd.com/whl-multi-arch/
.. |WHL| replace:: migraphx==2.16.0+rocm7.14.1

****************
Install MIGraphX
****************

This page describes how to install MIGraphX |MIGRAPHX_VERSION| using pip.
Installation using your Linux distribution's package manager will be fully
supported in a future release.

Prerequisites
=============

MIGraphX requires ROCm to be installed on your system first. For instructions,
see `Install AMD ROCm`_ |ROCM_VERSION|.

.. _Install AMD ROCm: https://rocm.docs.amd.com/en/docs-|ROCM_VERSION|/install/rocm.html?fam=all

MIGraphX is currently supported on:

* ``gfx950`` AMD Instinct MI355X and MI350X

* ``gfx942`` AMD Instinct MI325X and MI300X

See the `ROCm compatibility matrix`_ for more information.

.. _ROCm compatibility matrix: https://rocm.docs.amd.com/en/docs-|ROCM_VERSION|/compatibility/compatibility-matrix.html

Ensure your system has Python 3.12 installed and accessible.

Install MIGraphX using pip
===========================

This method installs MIGraphX into a Python virtual environment.

1. Create and activate a virtual environment.

   .. code-block:: bash

      python3.12 -m venv .venv
      source .venv/bin/activate

2. Install the MIGraphX wheel.

   .. code-block:: bash
      :substitutions:

      python -m pip install --index-url |PKG_REPO| \
          |WHL| \
          https://rocm.frameworks.amd.com/whl-multi-arch/migraphx/migraphx_libs-2.16.0%2Brocm7.14.1-py3-none-manylinux_2_28_x86_64.whl

3. ONNX Runtime accelerates machine learning inference using the MIGraphX
   execution provider on ROCm-supported GPUs. See the `ONNX Runtime
   installation guide`_.

.. _ONNX Runtime installation guide: https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/onnxruntime.html?rocm-ver=|ROCM_VERSION|

Build MIGraphX from source
===========================

.. note::

   This method for building MIGraphX requires using ``sudo``.

1. Install ``rocm-cmake``, ``pip3``, ``rocblas``, and ``miopen-hip``:

   .. code-block:: shell

      sudo apt install -y rocm-cmake python3-pip rocblas miopen-hip

2. Install `rbuild <https://github.com/RadeonOpenCompute/rbuild>`__:

   .. code-block:: shell

      pip3 install --prefix /usr/local https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

3. Build MIGraphX source code:

   .. code-block:: shell

      sudo rbuild build -d depend -B build -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')
