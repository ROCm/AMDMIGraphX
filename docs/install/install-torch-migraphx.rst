.. meta::
  :description: Installing torch-migraphx
  :keywords: install, build, MIGraphX, torch_migraphx, AMD, ROCm, package installer, development, contributing

********************************************************************
Torch-MIGraphX installation
********************************************************************

Prerequisites:
- MIGraphX installed. See `MIGraphX installation <./install-migraphx.html>`_
- Pytorch installed. See `Pytorch installation <https://rocm.docs.amd.com/projects/install-on-linux/en/develop/how-to/3rd-party/pytorch-install.html#using-a-wheels-package>`_


Install torch_migraphx with PIP
====================================================================
Use the following command to install torch_migraphx using PIP:

  .. code:: shell
  
    pip install torch_migraphx


Build torch_migraphx from source
====================================================================
Use the following command to build torch_migraphx from source:

  .. code:: shell
  
    git clone https://github.com/ROCm/torch_migraphx.git
    cd torch_migraphx
    pip install . --no-build-isolation

Refer to the `torch_migraphx repository <https://github.com/ROCm/torch_migraphx>`_ for more information regarding local builds and docker environments.
