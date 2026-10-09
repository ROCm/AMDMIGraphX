:selector-toc2: Installation environment
:selector-toc2-icon: fa-solid fa-computer

.. meta::
  :description: Installing MIGraphX for ROCm
  :keywords: install, pip, package manager, tarball, build, MIGraphX, AMD, ROCm, development, contributing

.. |MIGRAPHX_VERSION| replace:: 2.18.0
.. |ROCM_VERSION| replace:: 10.1.0
.. |PKG_REPO| replace:: https://stable.repo.amd.com/rocm/migraphx/whl-next/
.. |WHL| replace:: "migraphx==2.18.0+rocm10.1.0"
.. |WHL_LIBS| replace:: "migraphx-libs==2.18.0+rocm10.1.0"
.. |TARBALL| replace:: amdrocm10-migraphx-2.18.0.tar.gz
.. |TARBALL_URL| replace:: https://stable.repo.amd.com/rocm/migraphx/tarball/amdrocm10-migraphx-2.18.0.tar.gz

****************
Install MIGraphX
****************

MIGraphX is AMD's graph inference engine for optimizing and executing ONNX
models on AMD GPUs using ROCm. This page describes how to install MIGraphX
|MIGRAPHX_VERSION| for ROCm |ROCM_VERSION|.

.. selector:: Installation method
   :key: i

   .. selector-option:: Package manager
      :value: pkgman
      :width: 4

   .. selector-option:: pip
      :value: pip
      :width: 4

   .. selector-option:: Tarball
      :value: tar
      :width: 4

Prerequisites
=============

MIGraphX |MIGRAPHX_VERSION| is currently supported on:

* ``gfx950`` AMD Instinct MI355X, MI350X, and MI350P

* ``gfx942`` AMD Instinct MI325X, MI300X, and MI300A

* ``gfx90a`` AMD Instinct MI250X, MI250, and MI210

* ``gfx1200``, ``gfx1201``, ``gfx1100``, ``gfx1101``, and ``gfx1102`` Radeon GPUs.

* ``gfx1153``, ``gfx1152``, ``gfx1151``, and ``gfx1150`` Ryzen AI processors.

See the `ROCm compatibility matrix`_ for more information.

.. _ROCm compatibility matrix: https://rocm.docs.amd.com/en/docs-|ROCM_VERSION|/compatibility/compatibility-matrix.html

.. selected:: i=pip

   * Ensure your system has Python 3.12 installed and accessible.

.. _migraphx-package-install:

Install MIGraphX on Linux
=========================

MIGraphX requires ROCm to be installed on your system first.

Install ROCm
------------

For instructions, see `Install AMD ROCm`_ |ROCM_VERSION|. Use the selector
panel on that page to view instructions appropriate for your system
environment.

.. _Install AMD ROCm: https://rocm.docs.amd.com/en/docs-|ROCM_VERSION|/install/rocm.html?fam=all

.. selected:: i=pkgman
   :heading: Install MIGraphX 2.18.0 via package manager
   :heading-level: 3

   Use the following steps to install MIGraphX system-wide using your
   distribution's package manager on top of ROCm core libraries.

   .. selector:: Linux distribution
      :key: os

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 4

      .. selector-option:: Debian
         :value: debian
         :width: 4

      .. selector-option:: RHEL
         :value: rhel
         :width: 4

      .. selector-option:: Oracle Linux
         :value: oracle-linux
         :width: 4

      .. selector-option:: Rocky Linux
         :value: rocky-linux
         :width: 4

      .. selector-option:: SLES
         :value: sles
         :width: 4

   .. selected:: os=ubuntu

      .. selector:: Ubuntu version
         :key: ubuntu-ver

         .. selector-option:: 26.04.1
            :width: 4

         .. selector-option:: 24.04.5
            :width: 4

         .. selector-option:: 22.04.5
            :width: 4

   .. selected:: os=debian

      .. selector:: Debian version
         :key: debian-ver

         .. selector-option:: 13
            :width: 6

         .. selector-option:: 12
            :width: 6

   .. selected:: os=rhel

      .. selector:: RHEL version
         :key: rhel-ver

         .. selector-option:: 10.2
            :width: 2

         .. selector-option:: 10.0
            :width: 2

         .. selector-option:: 9.8
            :width: 2

         .. selector-option:: 9.6
            :width: 2

         .. selector-option:: 9.4
            :width: 2

         .. selector-option:: 8.10
            :width: 2

   .. selected:: os=oracle-linux

      .. selector:: Oracle Linux version
         :key: oracle-linux-ver

         .. selector-option:: 10
            :width: 4

         .. selector-option:: 9
            :width: 4

         .. selector-option:: 8
            :width: 4

   .. selected:: os=rocky-linux

      .. selector:: Rocky Linux version
         :key: rocky-linux-ver

         .. selector-option:: 9
            :width: 12

   .. selected:: os=sles

      .. selector:: SLES version
         :key: sles-ver

         .. selector-option:: 16
            :width: 6

         .. selector-option:: 15.7
            :width: 6

   .. raw:: html

      <br>

   1. Register the ROCm MIGraphX repository.

      .. selected:: os=ubuntu

         .. selected:: ubuntu-ver=26.04.1

            .. code-block:: bash

               sudo mkdir --parents --mode=0755 /etc/apt/keyrings
               wget https://stable.repo.amd.com/rocm/gpg/packages.gpg -O - | \
                   gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null
               sudo tee /etc/apt/sources.list.d/amdrocm-migraphx.sources << 'EOF'
               X-Repo-Id: amdrocm-migraphx
               Types: deb
               URIs: https://stable.repo.amd.com/rocm/migraphx/packages/ubuntu2604/
               Suites: stable
               Components: main
               Architectures: amd64
               Signed-By: /etc/apt/keyrings/amdrocm.gpg
               Enabled: yes
               EOF

               sudo apt update

         .. selected:: ubuntu-ver=24.04.5

            .. code-block:: bash

               sudo mkdir --parents --mode=0755 /etc/apt/keyrings
               wget https://stable.repo.amd.com/rocm/gpg/packages.gpg -O - | \
                   gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null
               sudo tee /etc/apt/sources.list.d/amdrocm-migraphx.sources << 'EOF'
               X-Repo-Id: amdrocm-migraphx
               Types: deb
               URIs: https://stable.repo.amd.com/rocm/migraphx/packages/ubuntu2404/
               Suites: stable
               Components: main
               Architectures: amd64
               Signed-By: /etc/apt/keyrings/amdrocm.gpg
               Enabled: yes
               EOF

               sudo apt update

         .. selected:: ubuntu-ver=22.04.5

            .. code-block:: bash

               sudo mkdir --parents --mode=0755 /etc/apt/keyrings
               wget https://stable.repo.amd.com/rocm/gpg/packages.gpg -O - | \
                   gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null
               sudo tee /etc/apt/sources.list.d/amdrocm-migraphx.sources << 'EOF'
               X-Repo-Id: amdrocm-migraphx
               Types: deb
               URIs: https://stable.repo.amd.com/rocm/migraphx/packages/ubuntu2204/
               Suites: stable
               Components: main
               Architectures: amd64
               Signed-By: /etc/apt/keyrings/amdrocm.gpg
               Enabled: yes
               EOF

               sudo apt update

      .. selected:: os=debian

         .. selected:: debian-ver=13

            .. code-block:: bash

               sudo mkdir --parents --mode=0755 /etc/apt/keyrings
               wget https://stable.repo.amd.com/rocm/gpg/packages.gpg -O - | \
                   gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null
               sudo tee /etc/apt/sources.list.d/amdrocm-migraphx.sources << 'EOF'
               X-Repo-Id: amdrocm-migraphx
               Types: deb
               URIs: https://stable.repo.amd.com/rocm/migraphx/packages/debian13/
               Suites: stable
               Components: main
               Architectures: amd64
               Signed-By: /etc/apt/keyrings/amdrocm.gpg
               Enabled: yes
               EOF

               sudo apt update

         .. selected:: debian-ver=12

            .. code-block:: bash

               sudo mkdir --parents --mode=0755 /etc/apt/keyrings
               wget https://stable.repo.amd.com/rocm/gpg/packages.gpg -O - | \
                   gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null
               sudo tee /etc/apt/sources.list.d/amdrocm-migraphx.sources << 'EOF'
               X-Repo-Id: amdrocm-migraphx
               Types: deb
               URIs: https://stable.repo.amd.com/rocm/migraphx/packages/debian12/
               Suites: stable
               Components: main
               Architectures: amd64
               Signed-By: /etc/apt/keyrings/amdrocm.gpg
               Enabled: yes
               EOF

               sudo apt update

      .. selected:: os=rhel

         .. selected:: rhel-ver=10.2 rhel-ver=10.0

            .. code-block:: bash

               sudo tee /etc/yum.repos.d/amdrocm-migraphx.repo <<EOF
               [amdrocm-migraphx]
               name=AMD ROCm MIGraphX
               baseurl=https://stable.repo.amd.com/rocm/migraphx/packages/rhel10/x86_64
               enabled=1
               gpgcheck=1
               gpgkey=https://stable.repo.amd.com/rocm/gpg/packages.gpg
               priority=50
               EOF
               sudo dnf clean all

         .. selected:: rhel-ver=9.8 rhel-ver=9.6 rhel-ver=9.4

            .. code-block:: bash

               sudo tee /etc/yum.repos.d/amdrocm-migraphx.repo <<EOF
               [amdrocm-migraphx]
               name=AMD ROCm MIGraphX
               baseurl=https://stable.repo.amd.com/rocm/migraphx/packages/rhel9/x86_64
               enabled=1
               gpgcheck=1
               gpgkey=https://stable.repo.amd.com/rocm/gpg/packages.gpg
               priority=50
               EOF
               sudo dnf clean all

         .. selected:: rhel-ver=8.10

            .. code-block:: bash

               sudo tee /etc/yum.repos.d/amdrocm-migraphx.repo <<EOF
               [amdrocm-migraphx]
               name=AMD ROCm MIGraphX
               baseurl=https://stable.repo.amd.com/rocm/migraphx/packages/rhel8/x86_64
               enabled=1
               gpgcheck=1
               gpgkey=https://stable.repo.amd.com/rocm/gpg/packages.gpg
               priority=50
               EOF
               sudo dnf clean all

      .. selected:: os=oracle-linux

         .. selected:: oracle-linux-ver=10

            .. code-block:: bash

               sudo tee /etc/yum.repos.d/amdrocm-migraphx.repo <<EOF
               [amdrocm-migraphx]
               name=AMD ROCm MIGraphX
               baseurl=https://stable.repo.amd.com/rocm/migraphx/packages/rhel10/x86_64
               enabled=1
               gpgcheck=1
               gpgkey=https://stable.repo.amd.com/rocm/gpg/packages.gpg
               priority=50
               EOF
               sudo dnf clean all

         .. selected:: oracle-linux-ver=9

            .. code-block:: bash

               sudo tee /etc/yum.repos.d/amdrocm-migraphx.repo <<EOF
               [amdrocm-migraphx]
               name=AMD ROCm MIGraphX
               baseurl=https://stable.repo.amd.com/rocm/migraphx/packages/rhel9/x86_64
               enabled=1
               gpgcheck=1
               gpgkey=https://stable.repo.amd.com/rocm/gpg/packages.gpg
               priority=50
               EOF
               sudo dnf clean all

         .. selected:: oracle-linux-ver=8

            .. code-block:: bash

               sudo tee /etc/yum.repos.d/amdrocm-migraphx.repo <<EOF
               [amdrocm-migraphx]
               name=AMD ROCm MIGraphX
               baseurl=https://stable.repo.amd.com/rocm/migraphx/packages/rhel8/x86_64
               enabled=1
               gpgcheck=1
               gpgkey=https://stable.repo.amd.com/rocm/gpg/packages.gpg
               priority=50
               EOF
               sudo dnf clean all

      .. selected:: os=rocky-linux

         .. code-block:: bash

            sudo tee /etc/yum.repos.d/amdrocm-migraphx.repo <<EOF
            [amdrocm-migraphx]
            name=AMD ROCm MIGraphX
            baseurl=https://stable.repo.amd.com/rocm/migraphx/packages/rhel9/x86_64
            enabled=1
            gpgcheck=1
            gpgkey=https://stable.repo.amd.com/rocm/gpg/packages.gpg
            priority=50
            EOF
            sudo dnf clean all

      .. selected:: os=sles

         .. selected:: sles-ver=16

            .. code-block:: bash

               sudo tee /etc/zypp/repos.d/amdrocm-migraphx.repo <<EOF
               [amdrocm-migraphx]
               name=AMD ROCm MIGraphX
               baseurl=https://stable.repo.amd.com/rocm/migraphx/packages/sles16/x86_64
               enabled=1
               gpgcheck=1
               gpgkey=https://stable.repo.amd.com/rocm/gpg/packages.gpg
               priority=50
               EOF

               sudo zypper --gpg-auto-import-keys refresh

         .. selected:: sles-ver=15.7

            .. code-block:: bash

               sudo tee /etc/zypp/repos.d/amdrocm-migraphx.repo <<EOF
               [amdrocm-migraphx]
               name=AMD ROCm MIGraphX
               baseurl=https://stable.repo.amd.com/rocm/migraphx/packages/sles15/x86_64
               enabled=1
               gpgcheck=1
               gpgkey=https://stable.repo.amd.com/rocm/gpg/packages.gpg
               priority=50
               EOF

               sudo zypper --gpg-auto-import-keys refresh

   2. Install the MIGraphX packages and ROCm dependencies.

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

            sudo apt install amdrocm10-migraphx amdrocm10-migraphx-dev

      .. selected:: os=rhel os=oracle-linux os=rocky-linux

         .. code-block:: bash

            sudo dnf install amdrocm10-migraphx amdrocm10-migraphx-devel

      .. selected:: os=sles

         .. code-block:: bash

            sudo zypper install amdrocm10-migraphx amdrocm10-migraphx-devel

   3. Complete the following post-installation steps.

      Configure environment variables so that MIGraphX is added to the
      ``PATH`` and ``LD_LIBRARY_PATH``.

      .. tab-set::

         .. tab-item:: User (~/.bashrc)
            :sync: bashrc

            .. code-block:: bash

               # MIGraphX Environment Setup
               tee --append ~/.bashrc << 'EOF'
               # BEGIN MIGraphX environment configuration
               export ROCM_PATH=/opt/rocm/core-10.1
               export MIGRAPHX_PATH=/opt/rocm/extras-10
               export PATH=$MIGRAPHX_PATH/bin:$PATH
               export LD_LIBRARY_PATH=$MIGRAPHX_PATH/lib:$ROCM_PATH/lib:$LD_LIBRARY_PATH
               # END MIGraphX environment configuration
               EOF

               source ~/.bashrc

         .. tab-item:: User (~/.profile)
            :sync: profile

            .. code-block:: bash

               # MIGraphX Environment Setup
               tee --append ~/.profile << 'EOF'
               # BEGIN MIGraphX environment configuration
               export ROCM_PATH=/opt/rocm/core-10.1
               export MIGRAPHX_PATH=/opt/rocm/extras-10
               export PATH=$MIGRAPHX_PATH/bin:$PATH
               export LD_LIBRARY_PATH=$MIGRAPHX_PATH/lib:$ROCM_PATH/lib:$LD_LIBRARY_PATH
               # END MIGraphX environment configuration
               EOF

               source ~/.profile

         .. tab-item:: System-wide
            :sync: system

            .. code-block:: bash

               # MIGraphX Environment Setup
               sudo tee /etc/profile.d/set-migraphx-env.sh << 'EOF'
               export ROCM_PATH=/opt/rocm/core-10.1
               export MIGRAPHX_PATH=/opt/rocm/extras-10
               export PATH=$MIGRAPHX_PATH/bin:$PATH
               export LD_LIBRARY_PATH=$MIGRAPHX_PATH/lib:$ROCM_PATH/lib:$LD_LIBRARY_PATH
               EOF

               sudo chmod +x /etc/profile.d/set-migraphx-env.sh
               source /etc/profile.d/set-migraphx-env.sh

   4. Verify your installation.

      Confirm that ``migraphx-driver`` is on your ``PATH`` and reports the
      expected version.

      .. code-block:: bash

         migraphx-driver --version

      You should see MIGraphX |MIGRAPHX_VERSION| in the output:

      .. code-block:: text
         :substitutions:

         MIGraphX Version: |MIGRAPHX_VERSION|

      .. tip::

         If the command isn't found, the environment variables from the
         previous step aren't set in your current shell. Re-run the
         ``source`` command or open a new shell session.

   5. ONNX Runtime accelerates machine learning inference using the MIGraphX
      execution provider on ROCm-supported GPUs. See the `ONNX Runtime
      installation guide`_.

      .. note::

         The ONNX Runtime MIGraphX execution provider requires MIGraphX to
         be installed using pip. The ``onnxruntime-ep-migraphx`` wheel
         depends on ``migraphx-libs``, which pip can't resolve from a
         package manager installation. To use the execution provider,
         install MIGraphX using pip instead.

.. _ONNX Runtime installation guide: https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/onnxruntime.html?rocm-ver=|ROCM_VERSION|

.. selected:: i=pkgman
   :heading: Uninstall MIGraphX
   :heading-level: 3

   1. Use your package manager to remove the installed packages.

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

            sudo apt remove amdrocm10-migraphx amdrocm10-migraphx-dev

      .. selected:: os=rhel os=oracle-linux os=rocky-linux

         .. code-block:: bash

            sudo dnf remove amdrocm10-migraphx amdrocm10-migraphx-devel

      .. selected:: os=sles

         .. code-block:: bash

            sudo zypper remove amdrocm10-migraphx amdrocm10-migraphx-devel

   2. Remove the MIGraphX repository.

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

            # Remove MIGraphX repository
            sudo rm /etc/apt/sources.list.d/amdrocm-migraphx.sources

            # Clear the cache and clean the system
            sudo apt clean
            sudo apt update

      .. selected:: os=rhel os=oracle-linux os=rocky-linux

         .. code-block:: bash

            # Remove MIGraphX repository
            sudo rm /etc/yum.repos.d/amdrocm-migraphx.repo

            # Clear the cache and clean the system
            sudo dnf clean all

      .. selected:: os=sles

         .. code-block:: bash

            # Remove MIGraphX repository
            sudo rm /etc/zypp/repos.d/amdrocm-migraphx.repo

            # Clear the cache and clean the system
            sudo zypper clean --all
            sudo zypper refresh

   3. Remove the MIGraphX environment configuration.

      .. tab-set::

         .. tab-item:: User (~/.bashrc)
            :sync: bashrc

            If you opted for a user-specific setup during the installation
            process, remove the MIGraphX environment configuration block --
            the lines between the ``BEGIN`` and ``END`` markers -- from
            ``~/.bashrc``.

         .. tab-item:: User (~/.profile)
            :sync: profile

            If you opted for a user-specific setup during the installation
            process, remove the MIGraphX environment configuration block --
            the lines between the ``BEGIN`` and ``END`` markers -- from
            ``~/.profile``.

         .. tab-item:: System-wide
            :sync: system

            If you opted for a system-wide setup during the installation
            process, remove the MIGraphX environment variables.

            .. code-block:: bash

               sudo rm -f /etc/profile.d/set-migraphx-env.sh

.. selected:: i=pip
   :heading: Install MIGraphX 2.18.0 using pip
   :heading-level: 3

   After installing ROCm, install MIGraphX. This method installs MIGraphX into
   a Python virtual environment.

   1. Create and activate a virtual environment or activate an existing ROCm
      |ROCM_VERSION| environment.

      .. tab-set::

         .. tab-item:: Python 3.12
            :sync: py312

            .. code-block:: bash

               python3.12 -m venv .venv
               source .venv/bin/activate

   2. Install the MIGraphX and ``migraphx-libs`` wheels.

      .. code-block:: bash
         :substitutions:

         python -m pip install --index-url |PKG_REPO| \
             |WHL| |WHL_LIBS|

   3. Verify your installation.

      Confirm that ``migraphx-driver`` is available in your virtual
      environment and reports the expected version.

      .. code-block:: bash

         migraphx-driver --version

   4. ONNX Runtime accelerates machine learning inference using the MIGraphX
      execution provider on ROCm-supported GPUs. See the `ONNX Runtime
      installation guide`_.

.. selected:: i=tar
   :heading: Install MIGraphX 2.18.0 using tarball
   :heading-level: 3

   After installing ROCm, install MIGraphX using the tarball method.

   1. Download the MIGraphX tarball.

      .. code-block:: bash
         :substitutions:

         wget |TARBALL_URL|

   2. Extract the tarball.

      MIGraphX is part of the ROCm Extras set of tools that work with the
      ROCm Core SDK and requires configuring the location of a ROCm
      installation. Set the ``ROCM_INSTALL_PATH`` variable to the install
      directory of ROCm. For example, if you installed the ROCm Core SDK
      using your Linux distribution's package manager, set it to
      ``/opt/rocm/core-10.1``. If ROCm was installed via tarball to a custom
      location, set ``ROCM_INSTALL_PATH`` to that location, for example
      ``$HOME/therock-tarball/install``.

      ``MIGRAPHX_INSTALL_PATH`` is set to the installation of MIGraphX to an
      extras location within the ROCm installation.

      .. code-block:: bash
         :substitutions:

         # Set installation paths
         ROCM_INSTALL_PATH="$HOME/therock-tarball/install"
         MIGRAPHX_INSTALL_PATH="$HOME/therock-tarball/install/extras-10"

         # Extract to the MIGraphX directory
         mkdir -p $MIGRAPHX_INSTALL_PATH
         tar -xzf |TARBALL| -C $MIGRAPHX_INSTALL_PATH

      .. note::

         The installation path for MIGraphX above assumes ROCm was installed
         to ``$HOME/therock-tarball/install``.

   3. Complete the following post-installation steps.

      Configure environment variables so that MIGraphX is added to the
      ``PATH`` and ``LD_LIBRARY_PATH``. ``MIGRAPHX_PATH`` and ``ROCM_PATH``
      are set to the values used during extraction.

      .. tab-set::

         .. tab-item:: User (~/.bashrc)
            :sync: bashrc

            .. code-block:: bash

               # MIGraphX Environment Setup
               tee --append ~/.bashrc << EOF
               # BEGIN MIGraphX environment configuration
               export ROCM_PATH=$ROCM_INSTALL_PATH
               export MIGRAPHX_PATH=$MIGRAPHX_INSTALL_PATH
               export PATH=\$MIGRAPHX_PATH/bin:\$PATH
               export LD_LIBRARY_PATH=\$MIGRAPHX_PATH/lib:\$ROCM_PATH/lib:\$LD_LIBRARY_PATH
               # END MIGraphX environment configuration
               EOF

               source ~/.bashrc

         .. tab-item:: User (~/.profile)
            :sync: profile

            .. code-block:: bash

               # MIGraphX Environment Setup
               tee --append ~/.profile << EOF
               # BEGIN MIGraphX environment configuration
               export ROCM_PATH=$ROCM_INSTALL_PATH
               export MIGRAPHX_PATH=$MIGRAPHX_INSTALL_PATH
               export PATH=\$MIGRAPHX_PATH/bin:\$PATH
               export LD_LIBRARY_PATH=\$MIGRAPHX_PATH/lib:\$ROCM_PATH/lib:\$LD_LIBRARY_PATH
               # END MIGraphX environment configuration
               EOF

               source ~/.profile

         .. tab-item:: System-wide
            :sync: system

            .. code-block:: bash

               # MIGraphX Environment Setup
               sudo tee /etc/profile.d/set-migraphx-env.sh << EOF
               export ROCM_PATH=$ROCM_INSTALL_PATH
               export MIGRAPHX_PATH=$MIGRAPHX_INSTALL_PATH
               export PATH=\$MIGRAPHX_PATH/bin:\$PATH
               export LD_LIBRARY_PATH=\$MIGRAPHX_PATH/lib:\$ROCM_PATH/lib:\$LD_LIBRARY_PATH
               EOF

               sudo chmod +x /etc/profile.d/set-migraphx-env.sh
               source /etc/profile.d/set-migraphx-env.sh

   4. Verify your installation.

      Confirm that ``migraphx-driver`` is on your ``PATH`` and reports the
      expected version.

      .. code-block:: bash

         migraphx-driver --version

      .. tip::

         If the command isn't found, the environment variables from the
         previous step aren't set in your current shell. Re-run the
         ``source`` command or open a new shell session.

.. selected:: i=tar
   :heading: Uninstall MIGraphX
   :heading-level: 3

   1. Remove the installation directory.

      To uninstall MIGraphX, remove your MIGraphX installation directory.

      .. important::

         The following command assumes you're working with the
         ``MIGRAPHX_INSTALL_PATH`` directory set to
         ``$HOME/therock-tarball/install/extras-10``. If you chose a
         different directory when installing MIGraphX, adjust the command
         accordingly.

      .. code-block:: bash

         rm -rf "$HOME/therock-tarball/install/extras-10"

   2. Remove the MIGraphX environment configuration.

      .. tab-set::

         .. tab-item:: User (~/.bashrc)
            :sync: bashrc

            If you opted for a user-specific setup during the installation
            process, remove the MIGraphX environment configuration block --
            the lines between the ``BEGIN`` and ``END`` markers -- from
            ``~/.bashrc``.

         .. tab-item:: User (~/.profile)
            :sync: profile

            If you opted for a user-specific setup during the installation
            process, remove the MIGraphX environment configuration block --
            the lines between the ``BEGIN`` and ``END`` markers -- from
            ``~/.profile``.

         .. tab-item:: System-wide
            :sync: system

            If you opted for a system-wide setup during the installation
            process, remove the MIGraphX environment variables.

            .. code-block:: bash

               sudo rm -f /etc/profile.d/set-migraphx-env.sh

Build MIGraphX from source
==========================

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

If you plan to develop for MIGraphX or contribute to the source code, see
:doc:`Developing for MIGraphX <../dev/contributing-to-migraphx>`.
