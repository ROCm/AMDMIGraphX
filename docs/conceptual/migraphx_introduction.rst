.. meta::
  :description: MIGraphX Introduction
  :keywords: MIGraphX, ROCm, library, API

.. _what-is-migraphx:

********************************************************************
What is a Deep Learning Compiler?
********************************************************************

DL compilers all look to improve the performance of models by analyzing aspects of DL models and automatically applying optimizations.
DL compilers can focus on different areas of the runtime.
For example, there are projects that only optimize kernels commonly used in DL models without providing a way to run a model.
MIGraphX provides an end-to-end solution for optimizing and executing DL models.

Our compilation process applies optimizing transformations to the compute graph of the model and then lowers the graph operations into kernel libraries or kernel code generators.
An overview of the compilation process for MIGraphX is shown in :numref:`compilation-label`.
One type of optimization that MIGraphX performs are kernel fusions such as the Attention fusion seen in :numref:`attention-label`.
Kernel fusions merge compatible operations into the same kernel execution on the accelerator.
Fusing the operations reduces kernel overhead and the number of redundant stores and loads between the host and accelerator, thereby improving overall performance.
By applying graph optimizations and selecting or generating highly performant device kernels, MIGraphX achieves significant performance gains over uncompiled models and similar compiled solutions.

.. figure:: figures/migraphx_compilation_flow.svg
  :scale: 50%
  :alt: compilation flow chart for MIGraphX
  :name: compilation-label

  Simplified overview of the compilation process in MIGraphX.


.. figure:: figures/attention_fusion.svg
  :scale: 50%
  :alt: attention fusion kernel
  :name: attention-label

  Fusion of Attention operations into a single kernel.

********************************************************************
What does MIGraphX offer as a Deep Learning compiler?
********************************************************************

* **Minimal development effort for an end-to-end solution that compiles DL models for improved inference performance on AMD hardware**
* **Open source C++ codebase with Python and C++ APIs**
* **Multiple ways to use MIGraphX:**
    * Direct compilation of ONNX models and Tensorflow models
    * PyTorch execution through the `Torch-MIGraphX <https://github.com/ROCm/torch_migraphx>`_ project 
    * ONNX Runtime execution provider
* **Specialized for AMD hardware:**
    * Compiles for consumer-grade Navi GPUs and server-grade MI GPUs
    * Our internal team works closely with the hardware and kernel teams to provide the best performance
* **Various quantized types support: FP16, BF16, OCP FP8, INT8, INT4**
    * With more types support in development
* **Continual improvement and additional model support as the machine learning landscape changes**
