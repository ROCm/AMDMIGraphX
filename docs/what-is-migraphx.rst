.. meta::
   :description: MIGraphX provides an optimized execution engine for deep learning neural networks
   :keywords: MIGraphX, ROCm, library, API

.. _what-is-migraphx:

=====================
What is MIGraphX?
=====================

AMD MIGraphX is a graph inference engine and graph compiler. MIGraphX accelerates machine-learning models by leveraging several graph-level transformations and optimizations. These optimizations include:

* Operator fusion
* Arithmetic simplifications
* Dead-code elimination
* Common subexpression elimination (CSE)
* Constant propagation

After optimization, MIGraphX generates code for AMD GPUs by calling MIOpen or rocBLAS, or by creating HIP kernels. MIGraphX can also target CPUs using DNNL or ZenDNN libraries.
