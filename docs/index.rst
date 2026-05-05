.. meta::
   :description: MIGraphX provides an optimized execution engine for deep learning neural networks
   :keywords: MIGraphX, ROCm, library, API

.. _index:

===========================
MIGraphX documentation
===========================

MIGraphX is a graph compiler and inference engine for high performance machine learning model inference.
It compiles trained models from end-to-end to optimize for inference performance on AMD hardware.

The MIGraphX public repository is located at `https://github.com/ROCm/AMDMIGraphX/ <https://github.com/ROCm/AMDMIGraphX/>`__.

You can integrate MIGraphX with PyTorch worflows by using the Torch-MIGraphX library.
The public repository is located at `https://github.com/ROCm/torch_migraphx/ <https://github.com/ROCm/torch_migraphx/>`__.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`MIGraphX on ROCm installation <./install/install-migraphx>`
    * :doc:`Torch-MIGraphX installation <./install/install-torch-migraphx>`

  .. grid-item-card:: Conceptual

    * :doc:`Deep learning compilation with MIGraphX <./conceptual/deep-learning-compilation>`

  .. grid-item-card:: Reference

    * :doc:`MIGraphX user reference <./reference/MIGraphX-user-reference>`
      
      * :ref:`cpp-api-reference`
      * :ref:`python-api-reference`
      * :doc:`Supported ONNX Operators <./dev/onnx_operators>`
   
    * :doc:`MIGraphX contributor reference <./reference/MIGraphX-dev-reference>`
   
      * :doc:`Environment variables <./reference/MIGraphX-dev-env-vars>`
      * :doc:`Develop for the MIGraphX code base <./dev/contributing-to-migraphx>` 
      * :ref:`migraphx-driver`
    
  .. grid-item-card:: Examples  

    * :doc:`MIGraphX examples <./tutorials/MIGraphX-examples>` 

To contribute to the documentation refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`__.

Licensing information can be found on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`__ page.

