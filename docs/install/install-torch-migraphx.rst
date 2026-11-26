:selector-toc2: Installation environment
:selector-toc2-icon: fa-solid fa-computer

.. meta::
  :description: Installing Torch-MIGraphX
  :keywords: install, pip, build, MIGraphX, torch_migraphx, PyTorch, AMD, ROCm, development, contributing

**********************
Install Torch-MIGraphX
**********************

MIGraphX can be integrated with PyTorch workflows by using the Torch-MIGraphX
library. It includes the ``torch.compile`` API, so you can compile PyTorch
models using MIGraphX. This page describes how to install Torch-MIGraphX
|TORCH_MIGRAPHX_VERSION| for ROCm |ROCM_VERSION| using pip.

Prerequisites
=============

Torch-MIGraphX |TORCH_MIGRAPHX_VERSION| is supported on the same AMD GPUs as
MIGraphX |MIGRAPHX_VERSION|. See the :doc:`MIGraphX installation page
<install-migraphx>` for the list of supported architectures and the
|ROCM_COMPAT_DOC| for more information.

* A ROCm-enabled PyTorch build must be installed in the same environment.

Install Torch-MIGraphX on Linux
===============================

Install ROCm
------------

For instructions, see |ROCM_INSTALL_DOC|. Select **pip** as the
installation method and install ROCm |ROCM_VERSION| into a Python virtual
environment. Use the selector panel on that page to view instructions
appropriate for your system environment.

Install Torch-MIGraphX 1.2 using pip
--------------------------------------

After installing ROCm, install Torch-MIGraphX. This method installs
Torch-MIGraphX into a Python virtual environment.

1. Activate the ROCm |ROCM_VERSION| virtual environment you created
   while installing ROCm using pip.

   .. code-block:: bash

      source .venv/bin/activate

   .. note::

      Substitute ``.venv`` with the path to your virtual environment if
      you used a different name or location.

2. Install a ROCm-enabled PyTorch release into the environment. See
   |TORCH_INSTALL_DOC|, select **pip** and ROCm |ROCM_VERSION| on the
   selector panel, and install into the same virtual environment. The
   ``torch_migraphx`` wheel in the next step requires a matching ROCm
   |ROCM_VERSION| installation.

3. Install the ``torch_migraphx`` wheel. This also installs the matching
   MIGraphX |MIGRAPHX_VERSION| wheel.

   .. code-block:: bash
      :substitutions:

      python -c "import torch; print(torch.__version__); assert torch.version.hip, 'ROCm PyTorch required'" && \
      python -m pip install --extra-index-url |WHL_PKG_REPO| \
          |WHL|

4. Verify your installation.

   Confirm that the package imports and reports the expected version.

   .. code-block:: bash

      python -c "import torch_migraphx, importlib.metadata as m; print(m.version('torch_migraphx'))"
      python -c "import torch, torch_migraphx; print('migraphx' in torch._dynamo.list_backends())"

   This prints ``True`` if the MIGraphX backend is registered as expected.

   .. important::

      If either command fails with ``libhiprtc.so.7: cannot open shared
      object file``, update ``LD_LIBRARY_PATH`` to link to the required
      ROCm libraries. ROCm's pip wheels place their shared libraries
      inside your virtual environment's ``site-packages`` instead of
      ``/opt/rocm``, so add them to ``LD_LIBRARY_PATH`` before running
      the commands. Run this command again in each new shell session
      after activating the virtual environment.

      .. code-block:: bash

         ROCM_SDK_CORE=$(python3 -c "import _rocm_sdk_core, os; print(os.path.dirname(_rocm_sdk_core.__file__))")
         ROCM_SDK_LIBRARIES=$(python3 -c "import _rocm_sdk_libraries, os; print(os.path.dirname(_rocm_sdk_libraries.__file__))")
         export LD_LIBRARY_PATH=$ROCM_SDK_CORE/lib:$ROCM_SDK_CORE/lib/rocm_sysdeps/lib:$ROCM_SDK_LIBRARIES/lib:$LD_LIBRARY_PATH

Uninstall Torch-MIGraphX
--------------------------

Remove the package from your virtual environment.

.. code-block:: bash

   python -m pip uninstall torch_migraphx

To also remove the MIGraphX library that was installed as a dependency:

.. code-block:: bash

   python -m pip uninstall migraphx

Build Torch-MIGraphX from source
================================

Use the following commands to build ``torch_migraphx`` from source:

.. code-block:: shell-session

   git clone https://github.com/ROCm/torch_migraphx.git
   cd torch_migraphx
   pip install . --no-build-isolation

Refer to the `torch_migraphx <https://github.com/ROCm/torch_migraphx/>`__
repository for more information on local builds and Docker environments.
