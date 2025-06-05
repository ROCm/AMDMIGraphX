.. meta::
   :description: MIGraphX provides an optimized execution engine for deep learning neural networks
   :keywords: MIGraphX, ROCm, library, API

.. _index:

===========================
MIGraphX documentation
===========================

MIGraphX is a graph inference engine and graph compiler. MIGraphX accelerates machine-learning models by leveraging several graph-level transformations and optimizations. These optimizations include:

* Operator fusion
* Arithmetic simplifications
* Dead-code elimination
* Common subexpression elimination (CSE)
* Constant propagation

After optimization, MIGraphX generates code for AMD GPUs by calling various ROCm libraries to create the fastest combinations of HIP kernels.

The MIGraphX public repository is located at `https://github.com/ROCm/AMDMIGraphX/ <https://github.com/ROCm/AMDMIGraphX/>`_

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Installing MIGraphX with the package installer <./install/installing_with_package>`
    * :doc:`Building and installing MIGraphX from source code <./install/building_migraphx>`

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
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

Licensing information can be found on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
