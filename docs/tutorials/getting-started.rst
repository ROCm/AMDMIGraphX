.. meta::
  :description: Get started with MIGraphX after installation
  :keywords: MIGraphX, getting started, tutorial, ROCm

********************************************************************
Get started with MIGraphX
********************************************************************

After you install MIGraphX, use this guide to choose a starting path based on
how you plan to use the library. Each section links to validated examples and
reference material in this repository.

Choose your path
====================================================================

MIGraphX supports several integration patterns. Pick the section that matches
your role and goal.

Application developer using the C++ API
====================================================================

Use this path when you embed MIGraphX directly in a C++ application.

1. Confirm MIGraphX is installed. See :doc:`MIGraphX on ROCm installation <../install/install-migraphx>`.
2. Set ``CMAKE_PREFIX_PATH`` to your MIGraphX installation location and link
   against the C++ API:

   .. code-block:: cmake

      find_package(migraphx)
      target_link_libraries(my_app migraphx::c)

   Replace ``my_app`` with your CMake target name.
3. Work through :doc:`Parse, load, and save a model <./parse-load-save-tutorial>`
   to learn how to load ONNX models and serialize programs.
4. Continue with the C++ MNIST inference example at
   ``examples/vision/cpp_mnist/`` in the
   `MIGraphX GitHub repository <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/vision/cpp_mnist>`__
   for a complete parse, compile, and evaluate workflow.
5. Consult the :ref:`cpp-api-reference` for API details.

Application developer using the Python API
====================================================================

Use this path when you call MIGraphX from Python.

1. Confirm MIGraphX is installed. See :doc:`MIGraphX on ROCm installation <../install/install-migraphx>`.
2. Add the Python module to your environment:

   .. code-block:: shell

      export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH

   Replace ``/opt/rocm/lib`` with the directory that contains the MIGraphX
   Python module if your installation path differs.
3. Browse the Python examples under ``examples/`` in the
   `MIGraphX GitHub repository <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples>`__.
   Start with a domain that matches your model, such as vision or natural
   language processing.
4. Consult the :ref:`python-api-reference` for API details.

Model integrator or inference engineer
====================================================================

Use this path when you want to inspect, compile, verify, or benchmark models
without writing application code.

1. Confirm ``migraphx-driver`` is available at ``/opt/rocm/bin/migraphx-driver``,
   or at ``AMDMIGraphX/build/bin/migraphx-driver`` after a source build.
2. Read :doc:`MIGraphX driver <../migraphx-driver>` for commands such as
   ``read``, ``compile``, ``run``, ``verify``, and ``perf``.
3. Validate a model against the reference implementation:

   .. code-block:: shell

      /opt/rocm/bin/migraphx-driver verify model.onnx --onnx

4. Measure performance:

   .. code-block:: shell

      /opt/rocm/bin/migraphx-driver perf model.onnx --onnx --gpu -n 50

5. See :doc:`Validate model outputs <../how-to/model-validation>` and
   :doc:`Tune model performance <../how-to/model-performance-tuning>` for
   environment variables and tolerance settings.

PyTorch user
====================================================================

Use this path when you integrate MIGraphX with PyTorch workflows through
Torch-MIGraphX.

1. Install Torch-MIGraphX. See
   :doc:`Torch-MIGraphX installation <../install/install-torch-migraphx>`.
2. Refer to the Torch-MIGraphX repository at
   `https://github.com/ROCm/torch_migraphx/ <https://github.com/ROCm/torch_migraphx/>`__
   for model conversion and inference examples.

ONNX Runtime user
====================================================================

Use this path when you run inference through the ONNX Runtime MIGraphX
execution provider.

1. Install MIGraphX. See :doc:`MIGraphX on ROCm installation <../install/install-migraphx>`.
2. Follow the ResNet50 example at
   ``examples/onnxruntime/resnet50/`` in the
   `MIGraphX GitHub repository <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/onnxruntime/resnet50>`__
   for setup and inference steps with ONNX Runtime.

MIGraphX contributor
====================================================================

Use this path when you modify the MIGraphX source code or add operators and
passes.

1. Set up a build environment. See :doc:`Install MIGraphX with Docker <../install/install-docker>`
   or :doc:`MIGraphX on ROCm installation <../install/install-migraphx>`.
2. Read :doc:`Develop for the MIGraphX code base <../dev/contributing-to-migraphx>`.
3. Consult :doc:`MIGraphX environment variables <../reference/MIGraphX-dev-env-vars>`
   for tuning, tracing, and testing options used during development.

Next steps
====================================================================

* :doc:`Parse, load, and save a model <./parse-load-save-tutorial>` walks
  through your first C++ workflow with an ONNX model.
* :doc:`MIGraphX examples <./MIGraphX-examples>` lists documented examples by
  domain and target user.
* :doc:`Deep learning compilation with MIGraphX <../conceptual/deep-learning-compilation>`
  explains how MIGraphX optimizes inference graphs.
