.. meta::
  :description: Installing MIGraphX using Docker
  :keywords: install, MIGraphX, AMD, ROCm, Docker

********************************************************************
Installing MIGraphX using Docker
********************************************************************

ROCm must be installed before installing MIGraphX. See `ROCm installation for Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_ for information on how to install ROCm on Linux.

.. note::
  
  Docker commands are run using ``sudo``. 

1. Build the Docker image. This command will install all the prerequisites required to install MIGraphX. Ensure that you are running this in the same directory as ``Dockerfile``. 

    .. code:: shell
    
      sudo docker build -t migraphx .
  

2. Create and run the container. Once this command is run, you will be in the ``/code/AMDMIGraphX`` directory of a pseudo-tty.

    .. code:: shell
    
      sudo docker run --device='/dev/kfd' --device='/dev/dri' -v=`pwd`:/code/AMDMIGraphX -w /code/AMDMIGraphX --group-add video -it migraphx

3. In the ``/code/AMDMIGraphX``, create a ``build`` directory, then change directory to ``/code/AMDMIGraphX/build``:

    .. code:: shell

      mkdir build
      cd build


4. Configure CMake:

    .. code:: shell

      CXX=/opt/rocm/llvm/bin/clang++ cmake .. -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')
  
4. Build the MIGraphX libraries:

    .. code:: shell
    
      make -j$(nproc)

5. Install the libraries:

    .. code:: shell

      make install