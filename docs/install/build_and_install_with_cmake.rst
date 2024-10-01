.. meta::
  :description: Build and install MIGraphX using CMake
  :keywords: build, install, MIGraphX, AMD, ROCm, CMake

********************************************************************
Build and install MIGraphX using CMake
********************************************************************

ROCm must be installed before installing MIGraphX. See `ROCm installation for Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_ for information on how to install ROCm on Linux.

.. note::
  
  This method for building MIGraphX requires using ``sudo``.


1. Install the dependencies:

    .. code:: shell
    
        sudo rbuild build -d depend -B build -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')

    .. note:: 

        If ``rbuild`` is not installed on your system, install it with:

            .. code:: shell

                pip3 install --prefix /usr/local https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

2. Create a build directory and change directory to it:

    .. code:: shell

        mkdir build
        cd build

3. Configure CMake:

    .. code:: shell
    
        CXX=/opt/rocm/llvm/bin/clang++ cmake .. -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')
    
4. Build MIGraphX source code:

    .. code:: shell

        make -j$(nproc)


    You can verify this using:

    .. code:: shell

        make -j$(nproc) check
    

5. Install MIGraphX libraries:

    .. code:: shell

        make install
    