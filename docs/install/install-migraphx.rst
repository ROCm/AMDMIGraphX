.. meta::
  :description: Installing MIGraphX for ROCm
  :keywords: install, build, MIGraphX, AMD, ROCm, package installer, development, contributing

********************************************************************
MIGraphX on ROCm installation
********************************************************************

ROCm must be installed before installing MIGraphX. See `ROCm installation 
for Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`__ 
for instructions.

Installing MIGraphX using a package installer is sufficient for most users 
who want to use the MIGraphX API. If you plan to develop for MIGraphX or 
contribute to the source code, see `Developing for MIGraphX <../dev/contributing-to-migraphx.html>`_

Install MIGraphX with a package installer
====================================================================

The package installer will install all the prerequisites you need for MIGraphX.

Use the following command to install MIGraphX: 

.. code:: shell

   sudo apt update && sudo apt install -y migraphx

Build MIGraphX from source
====================================================================

.. note::

   This method for building MIGraphX requires using ``sudo``.

1. Install ``rocm-cmake``, ``pip3``, ``rocblas``, and ``miopen-hip``:

.. code:: shell

   sudo apt install -y rocm-cmake python3-pip rocblas miopen-hip

2. Install `rbuild <https://github.com/RadeonOpenCompute/rbuild>`__:

.. code:: shell

   pip3 install --prefix /usr/local https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

3. Build MIGraphX source code:

.. code:: shell

   sudo rbuild build -d depend -B build -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')