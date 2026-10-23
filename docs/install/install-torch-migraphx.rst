:selector-toc2: Installation environment
:selector-toc2-icon: fa-solid fa-computer

.. meta::
  :description: Installing Torch-MIGraphX
  :keywords: install, pip, build, MIGraphX, torch_migraphx, PyTorch, AMD, ROCm, development, contributing

.. |WHL| replace:: "torch_migraphx==1.2+rocm10.1.0"

**********************
Install Torch-MIGraphX
**********************

MIGraphX can be integrated with PyTorch workflows by using the Torch-MIGraphX
library. It includes the ``torch.compile`` API, so you can compile PyTorch
models using MIGraphX. This page describes how to install Torch-MIGraphX
|TORCH_MIGRAPHX_VERSION| for ROCm |ROCM_VERSION| using pip.

.. selector:: Installation method
   :key: i

   .. selector-option:: pip
      :value: pip
      :width: 12

Prerequisites
=============

Torch-MIGraphX |TORCH_MIGRAPHX_VERSION| is supported on the same AMD GPUs as
MIGraphX |MIGRAPHX_VERSION|. See the :doc:`MIGraphX installation page
<install-migraphx>` for the list of supported architectures and the
|ROCM_COMPAT_DOC| for more information.

.. selected:: i=pip

   * Ensure your system has Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed
     and accessible.

   * A ROCm-enabled PyTorch build must be installed in the same environment.

Install Torch-MIGraphX on Linux
===============================

Install ROCm
------------

For instructions, see |ROCM_INSTALL_DOC|. Use the selector
panel on that page to view instructions appropriate for your system
environment.

.. selected:: i=pip
   :heading: Install Torch-MIGraphX 1.2 using pip
   :heading-level: 3

   After installing ROCm, install Torch-MIGraphX. This method installs
   Torch-MIGraphX into a Python virtual environment.

   1. Create and activate a virtual environment or activate an existing ROCm
      |ROCM_VERSION| environment.

      .. tab-set::

         .. tab-item:: Python 3.14
            :sync: py314

            .. code-block:: bash

               python3.14 -m venv .venv
               source .venv/bin/activate

         .. tab-item:: Python 3.13
            :sync: py313

            .. code-block:: bash

               python3.13 -m venv .venv
               source .venv/bin/activate

         .. tab-item:: Python 3.12
            :sync: py312

            .. code-block:: bash

               python3.12 -m venv .venv
               source .venv/bin/activate

         .. tab-item:: Python 3.11
            :sync: py311

            .. code-block:: bash

               python3.11 -m venv .venv
               source .venv/bin/activate

         .. tab-item:: Python 3.10
            :sync: py310

            .. code-block:: bash

               python3.10 -m venv .venv
               source .venv/bin/activate

   2. Install a ROCm-enabled PyTorch release into the environment. See
      |TORCH_INSTALL_DOC|.

   3. Install the ``torch_migraphx`` wheel. This also installs the matching
      MIGraphX |MIGRAPHX_VERSION| wheels.

      .. code-block:: bash
         :substitutions:

         python -m pip install --extra-index-url |WHL_PKG_REPO| \
             |WHL|

   4. Verify your installation.

      Confirm that the package imports and reports the expected version.

      .. code-block:: bash

         python -c "import torch_migraphx, importlib.metadata as m; print(m.version('torch_migraphx'))"
         python -c "import torch, torch_migraphx; print('migraphx' in torch._dynamo.list_backends())"

      This prints ``True`` if the MIGraphX backend is registered as expected.

      .. important::

         If you installed ROCm using your Linux distribution's package manager
         instead of pip, both commands fail with ``libhsa-runtime64.so.1:
         cannot open shared object file``. The ROCm libraries are in
         ``/opt/rocm/core-10.1/lib``, which isn't on the default linker path,
         so add it before running them:

         .. code-block:: bash

            export ROCM_PATH=/opt/rocm/core-10.1
            export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

         To make this permanent, add these lines to your shell startup script.

.. selected:: i=pip
   :heading: Uninstall Torch-MIGraphX
   :heading-level: 3

   Remove the package from your virtual environment.

   .. code-block:: bash

      python -m pip uninstall torch_migraphx

   To also remove the MIGraphX libraries that were installed as dependencies:

   .. code-block:: bash

      python -m pip uninstall migraphx migraphx-libs

Build Torch-MIGraphX from source
================================

Use the following commands to build ``torch_migraphx`` from source:

.. code-block:: shell-session

   git clone https://github.com/ROCm/torch_migraphx.git
   cd torch_migraphx
   pip install . --no-build-isolation

Refer to the `torch_migraphx <https://github.com/ROCm/torch_migraphx/>`__
repository for more information on local builds and Docker environments.
