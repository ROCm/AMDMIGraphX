.. meta::
  :description: Installing Torch-MIGraphX
  :keywords: install, build, MIGraphX, torch_migraphx, AMD, ROCm, package installer, development, contributing

********************************************************************
Torch-MIGraphX installation
********************************************************************

MIGraphX can be integrated with PyTorch workflows by using the Torch-MIGraphX library.
It includes the ``torch.compile`` API, so you can compile PyTorch models using MIGraphX.

Prerequisites:

- ROCm must be installed before installing MIGraphX. See `ROCm installation 
  for Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`__ 
  for instructions.

- Installing MIGraphX using a package installer is sufficient if you want to 
  use the MIGraphX API. See `MIGraphX installation <./install-migraphx.html>`__ 
  for instructions.

- Pytorch must be installed. See `PyTorch installation <https://rocm.docs.amd.com/projects/install-on-linux/en/develop/how-to/3rd-party/pytorch-install.html#using-a-wheels-package>`__
  for instructions.

Install Torch-MIGraphX
====================================================================

Use the following command to install ``torch_migraphx`` using ``pip``:

.. code:: shell
  
   pip install torch_migraphx

Build Torch-MIGraphX from source
====================================================================

Use the following command to build ``torch_migraphx`` from source:

.. code:: shell
  
   git clone https://github.com/ROCm/torch_migraphx.git
   cd torch_migraphx
   pip install . --no-build-isolation

Refer to the `https://github.com/ROCm/torch_migraphx/ <https://github.com/ROCm/torch_migraphx/>`__ repository for more information on local builds and Docker environments.
