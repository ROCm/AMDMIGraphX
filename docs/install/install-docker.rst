.. meta::
  :description: Install MIGraphX using Docker
  :keywords: install, Docker, MIGraphX, AMD, ROCm, container

********************************************************************
Install MIGraphX with Docker
********************************************************************

Docker provides a development environment with MIGraphX build prerequisites
preinstalled. Use this approach when you want to build MIGraphX from source
without installing dependencies on your host system.

The default ``Dockerfile`` at the repository root builds against ROCm 7.13 and
newer using TheRock (``amdrocm-*``) packages. Alternative Dockerfiles for
other ROCm releases are available under ``tools/docker/``.

Prerequisites
====================================================================

Before you build or run a MIGraphX Docker image, install the following on
your host:

* `ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`__
  with GPU drivers configured
* Docker with access to ``/dev/kfd`` and ``/dev/dri``

Build the default Docker image
====================================================================

From the repository root, build the image:

.. code-block:: shell

   docker build -t migraphx .

To reduce image size, set ``GPU_ARCH`` to the GPU architecture on your
system. Leaving ``GPU_ARCH`` unset installs device code for all supported
architectures, which is useful when the same image runs on different
ROCm-supported GPUs.

.. code-block:: shell

   docker build -t migraphx --build-arg GPU_ARCH=$(rocminfo | grep -o -m1 'gfx.*') .

The root ``Dockerfile`` accepts the following build arguments:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Build argument
     - Description
   * - ``ROCM_VERSION``
     - ROCm release version for versioned package names (default: ``7.14``).
   * - ``GPU_ARCH``
     - GPU architecture family (for example, ``gfx942``, ``gfx120x``). Leave empty for arch-independent packages.
   * - ``USE_WHL``
     - When set to a non-empty value, installs ROCm from Python wheels instead of system packages.
   * - ``INDEX_URL``
     - pip index URL used when ``USE_WHL`` is set (default: ``https://repo.amd.com/rocm/whl-multi-arch/``).

Run the container
====================================================================

Start an interactive container with your repository mounted at
``/code/AMDMIGraphX``:

.. code-block:: shell

   docker run --device='/dev/kfd' --device='/dev/dri' \
       -v=`pwd`:/code/AMDMIGraphX -w /code/AMDMIGraphX \
       --group-add video -it migraphx

Build MIGraphX inside the container
====================================================================

Inside the container, prerequisites are already installed. Follow the steps in
:doc:`MIGraphX on ROCm installation <./install-migraphx>` to build from source,
starting from the rbuild or CMake build steps.

Alternative Dockerfiles
====================================================================

Use an alternative Dockerfile when your target ROCm release requires a
different base image or package source.

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Dockerfile path
     - Use when
   * - ``tools/docker/legacy.dockerfile``
     - Building against ROCm 7.2.x and older on Ubuntu 22.04.
   * - ``tools/docker/ubuntu_2404.dockerfile``
     - Building on Ubuntu 24.04 with ROCm 7.1.1.
   * - ``tools/docker/ubuntu_2204.dockerfile``
     - Building on Ubuntu 22.04 with ROCm 6.4.2.

Build with the ``-f`` flag to select a Dockerfile. For example:

.. code-block:: shell

   docker build -t migraphx:legacy -f tools/docker/legacy.dockerfile .

Then follow the same ``docker run`` and build steps described above.
