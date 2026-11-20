:selector-toc2: Installation environment
:selector-toc2-icon: fa-solid fa-computer

.. meta::
  :description: Installing MIGraphX for ROCm
  :keywords: install, pip, build, MIGraphX, AMD, ROCm, development, contributing

****************
Install MIGraphX
****************

Installing MIGraphX using prebuilt packages is sufficient for most users
who want to use the MIGraphX API. If you plan to develop for MIGraphX or
contribute to the source code, see `Developing for MIGraphX <../dev/contributing-to-migraphx.html>`_.

MIGraphX |MIGRAPHX_VERSION| is installed using pip.

Prerequisites
=============

MIGraphX |MIGRAPHX_VERSION| is currently supported on:

* ``gfx950`` AMD Instinct MI355X and MI350X

* ``gfx942`` AMD Instinct MI325X and MI300X

* ``gfx1200``, ``gfx1201``, ``gfx1100``, ``gfx1101``, and ``gfx1102`` Radeon GPUs.

See the |ROCM_COMPAT_DOC| for more information.

* Ensure your system has Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed
  and accessible.

.. _migraphx-package-install:

Install MIGraphX on Linux
=========================

MIGraphX requires ROCm to be installed on your system first.

Install ROCm
------------

For instructions, see |ROCM_INSTALL_DOC|. Use the selector
panel on that page to view instructions appropriate for your system
environment.

Install MIGraphX using pip
---------------------------

After installing ROCm, install MIGraphX. This method installs MIGraphX |MIGRAPHX_VERSION| into
a Python virtual environment.

1. Create and activate a virtual environment or activate an existing ROCm
   |ROCM_VERSION| environment.

   .. code-block:: bash

      python3 -m venv .venv
      source .venv/bin/activate

   .. note::

      This command may vary depending on the Python version installed on
      your system (for example, ``python3.12 -m venv .venv``).

2. Install the MIGraphX wheel.

   .. code-block:: bash
      :substitutions:

      python -m pip install --index-url |WHL_PKG_REPO| \
          "migraphx==|MIGRAPHX_VERSION|+rocm|ROCM_VERSION|"

3. Importing MIGraphX may fail with a missing shared library error, such
   as ``libhiprtc.so.7: cannot open shared object file`` or
   ``libhsa-runtime64.so.1: cannot open shared object file``. As a
   workaround, update ``LD_LIBRARY_PATH`` to link to the required ROCm
   libraries. Select the tab that matches how you installed ROCm.

   .. tab-set::

      .. tab-item:: Installed ROCm using pip
         :sync: rocm-pip

         ROCm's pip wheels place their shared libraries inside your
         virtual environment's ``site-packages`` instead of
         ``/opt/rocm``. Set the environment variable for the current
         shell session with the virtual environment active, so the
         paths resolve correctly. Run this command again in each new
         shell session after activating the virtual environment.

         .. code-block:: bash

            # MIGraphX Environment Setup
            ROCM_SDK_CORE=$(python3 -c "import _rocm_sdk_core, os; print(os.path.dirname(_rocm_sdk_core.__file__))")
            ROCM_SDK_LIBRARIES=$(python3 -c "import _rocm_sdk_libraries, os; print(os.path.dirname(_rocm_sdk_libraries.__file__))")
            export LD_LIBRARY_PATH=$ROCM_SDK_CORE/lib:$ROCM_SDK_CORE/lib/rocm_sysdeps/lib:$ROCM_SDK_LIBRARIES/lib:$LD_LIBRARY_PATH

      .. tab-item:: Installed ROCm using a package manager
         :sync: rocm-pkgman

         .. tab-set::

            .. tab-item:: Active session
               :sync: active

               Set the environment variable for the current shell
               session only. To keep this setting across sessions, use
               System-wide setup or User setup.

               .. code-block:: bash
                  :substitutions:

                  # MIGraphX Environment Setup
                  export ROCM_PATH=/opt/rocm/core-|ROCM_VERSION_PKG_SUFFIX|
                  export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

            .. tab-item:: System-wide setup
               :sync: system

               Create a profile script so that all users inherit the
               MIGraphX environment variables when they start a shell
               session.

               .. code-block:: bash
                  :substitutions:

                  # MIGraphX Environment Setup
                  sudo tee /etc/profile.d/set-migraphx-env.sh << 'EOF'
                  export ROCM_PATH=/opt/rocm/core-|ROCM_VERSION_PKG_SUFFIX|
                  export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
                  EOF

                  sudo chmod +x /etc/profile.d/set-migraphx-env.sh
                  source /etc/profile.d/set-migraphx-env.sh

            .. tab-item:: User setup
               :sync: user

               Configure the MIGraphX environment for your user by
               updating your shell startup configuration file.

               .. tab-set::

                  .. tab-item:: .bashrc
                     :sync: bashrc

                     .. code-block:: bash
                        :substitutions:

                        # MIGraphX Environment Setup
                        tee --append ~/.bashrc << 'EOF'
                        # BEGIN MIGraphX environment configuration
                        export ROCM_PATH=/opt/rocm/core-|ROCM_VERSION_PKG_SUFFIX|
                        export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
                        # END MIGraphX environment configuration
                        EOF

                        source ~/.bashrc

                  .. tab-item:: .profile
                     :sync: profile

                     .. code-block:: bash
                        :substitutions:

                        # MIGraphX Environment Setup
                        tee --append ~/.profile << 'EOF'
                        # BEGIN MIGraphX environment configuration
                        export ROCM_PATH=/opt/rocm/core-|ROCM_VERSION_PKG_SUFFIX|
                        export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
                        # END MIGraphX environment configuration
                        EOF

                        source ~/.profile

4. Verify your installation.

   Confirm that ``migraphx-driver`` and the ``migraphx`` Python module are
   available in your virtual environment and report the expected version.

   .. code-block:: bash

      migraphx-driver --version
      python -c "import migraphx; print(migraphx.__version__)"

   You should see MIGraphX |MIGRAPHX_VERSION| in the output:

   .. code-block:: text
      :substitutions:

      MIGraphX Version: |MIGRAPHX_VERSION|
      |MIGRAPHX_VERSION|+rocm|ROCM_VERSION|

.. seealso::

   ONNX Runtime with ``onnxruntime-ep-migraphx`` accelerates machine
   learning inference using the MIGraphX execution provider on
   ROCm-supported GPUs. See the `ONNX Runtime installation guide`_.

.. _ONNX Runtime installation guide: https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/onnxruntime.html

Uninstall MIGraphX
-------------------

1. Remove the virtual environment containing MIGraphX.

   .. code-block:: bash

      deactivate
      rm -rf .venv

2. Remove the MIGraphX environment configuration.

   **If you added the** ``LD_LIBRARY_PATH`` **workaround as part of
   the installation**, remove it.

   .. tab-set::

      .. tab-item:: System-wide setup
         :sync: system

         If you opted for a system-wide setup during the installation
         process, remove the MIGraphX environment variables.

         .. code-block:: bash

            sudo rm -f /etc/profile.d/set-migraphx-env.sh

      .. tab-item:: User setup
         :sync: user

         .. tab-set::

            .. tab-item:: .bashrc
               :sync: bashrc

               If you opted for a user-specific setup during the installation
               process, remove the MIGraphX environment configuration block --
               the lines between the ``BEGIN`` and ``END`` markers -- from
               ``~/.bashrc``.

            .. tab-item:: .profile
               :sync: profile

               If you opted for a user-specific setup during the installation
               process, remove the MIGraphX environment configuration block --
               the lines between the ``BEGIN`` and ``END`` markers -- from
               ``~/.profile``.

.. _build-from-source:

Build MIGraphX from source
==========================

.. note::

   For reproducibility, we recommend using the provided Docker environment
   with MIGraphX build prerequisites preinstalled. See
   :doc:`./build-migraphx-docker`.

See the `MIGraphX README <https://github.com/ROCm/AMDMIGraphX>`__ for more
information.

1. Complete the appropriate :doc:`ROCm installation prerequisites
   <rocm:install/rocm>` for your system environment
   and register AMD package repositories.

2. Install required ROCm development packages:

   .. tab-set::

      .. tab-item:: Debian-based distros
         :sync: deb

         .. code-block:: shell
            :substitutions:

            sudo apt install amdrocm-runtime-dev|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-blas-dev|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-blas-host|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-dnn-dev|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-dnn-host|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-hipblas-common-dev|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-core|ROCM_VERSION_PKG_SUFFIX|

      .. tab-item:: RHEL-based distros
         :sync: rhel

         .. code-block:: shell
            :substitutions:

            sudo dnf install amdrocm-runtime-devel|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-blas-devel|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-blas-host|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-dnn-devel|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-dnn-host|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-hipblas-common-devel|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-core|ROCM_VERSION_PKG_SUFFIX|

      .. tab-item:: SLES
         :sync: zypper

         .. code-block:: shell
            :substitutions:

            sudo zypper install amdrocm-runtime-devel|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-blas-devel|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-blas-host|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-dnn-devel|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-dnn-host|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-hipblas-common-devel|ROCM_VERSION_PKG_SUFFIX| \
                amdrocm-core|ROCM_VERSION_PKG_SUFFIX|

   ``amdrocm-core`` provides the GPU device code and links ``/opt/rocm/bin``,
   ``/opt/rocm/llvm``, and the other ``/opt/rocm`` entry points to this
   release, so the paths used in the remaining steps resolve.

   .. note::

      :substitution-code:`amdrocm-core|ROCM_VERSION_PKG_SUFFIX|` installs the device code for
      every supported architecture. Append your GPU's architecture to install
      only the device code that architecture needs, which is a much smaller
      download:

      .. code-block:: shell
         :substitutions:

         sudo apt install amdrocm-core|ROCM_VERSION_PKG_SUFFIX|-gfx942

      Run ``rocminfo | grep -o -m1 'gfx.*'`` to find the architecture of an
      installed GPU. The suffixed package sets up the same ``/opt/rocm`` links.

3. Install the host build tools. The ROCm packages don't provide CMake, Git, or
   the Python headers and ``venv`` module that the following steps need.

   .. tab-set::

      .. tab-item:: Debian-based distros
         :sync: deb

         .. code-block:: shell

            sudo apt install build-essential cmake git pkg-config \
                python3-dev python3-pip python3-venv

      .. tab-item:: RHEL-based distros
         :sync: rhel

         .. code-block:: shell

            sudo dnf install gcc-c++ make cmake git \
                python3-devel python3-pip

      .. tab-item:: SLES
         :sync: zypper

         .. code-block:: shell

            sudo zypper install gcc-c++ make cmake git \
                python3-devel python3-pip

4. Clone the MIGraphX source code:

   .. code-block:: shell

      git clone https://github.com/ROCm/AMDMIGraphX.git
      cd AMDMIGraphX

5. Install `rbuild <https://github.com/RadeonOpenCompute/rbuild>`__ into a Python
   virtual environment.

   On Python 3.14 or newer, install ``cget`` from source.

   .. code-block:: shell

      python3 -m venv venv_rbuild
      source venv_rbuild/bin/activate

      pip install https://github.com/pfultz2/cget/archive/master.tar.gz
      pip install https://github.com/ROCm/rbuild/archive/master.tar.gz

6. Build MIGraphX source code. Run this from the root of the checkout, because
   ``rbuild`` builds the source directory you invoke it from:

   .. code-block:: shell

      rbuild build -d depend -B build -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')

Once completed, all prerequisites are in the ``depend`` folder and MIGraphX is in
the ``build`` directory.

If you plan to develop for MIGraphX or contribute to the source code, see
:doc:`Developing for MIGraphX <../dev/contributing-to-migraphx>`.
