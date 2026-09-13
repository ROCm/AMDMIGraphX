.. meta::
  :description: MIGraphX examples
  :keywords: MIGraphX, AMD, ROCm, examples

********************************************************************
MIGraphX examples
********************************************************************

The MIGraphX repository includes examples for common inference workflows.
Each example below is documented in its README file. Follow the linked path
in the `MIGraphX GitHub repository <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples>`__
to build and run the example.

To swap in your own model, replace the model file path named in the example
README or source code with your ONNX or serialized MIGraphX program. Adjust
input dimensions and data loading to match your model's parameter shapes.

Getting started examples
====================================================================

These examples teach core MIGraphX workflows. Start here if you are new to
the library.

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Example
     - Target user
     - What it demonstrates
   * - :doc:`Parse, load, and save a model <./parse-load-save-tutorial>`
     - C++ application developers
     - Parse ONNX models, inspect graphs, and save or load programs in MessagePack or JSON format.
   * - `cpp_mnist <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/vision/cpp_mnist>`__
     - C++ application developers
     - End-to-end inference: parse ONNX, compile for GPU or CPU, quantize, prepare inputs, and evaluate.
   * - `migraphx_driver <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/migraphx/migraphx_driver>`__
     - Model integrators
     - Command-line usage of ``migraphx-driver`` for read, compile, run, verify, and perf without writing code.

MIGraphX API examples
====================================================================

Examples under ``examples/migraphx/`` cover MIGraphX-specific utilities and
advanced C++ API usage.

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Example
     - Target user
     - What it demonstrates
   * - `cpp_parse_load_save <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/migraphx/cpp_parse_load_save>`__
     - C++ application developers
     - Parse, load, and save graph programs. Swap in your ONNX file as the input argument.
   * - `cpp_dynamic_batch <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/migraphx/cpp_dynamic_batch>`__
     - C++ application developers
     - Run programs with dynamic batch sizes using ``dynamic_dimension`` objects. Replace the ONNX input file with your dynamic-batch model.
   * - `cpp_trace_callback <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/migraphx/cpp_trace_callback>`__
     - MIGraphX contributors and debuggers
     - Inspect operator output buffers during evaluation with ``program::run_trace``.
   * - `export_frozen_graph_tf1 <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/migraphx/export_frozen_graph_tf1>`__
     - Model integrators
     - Export frozen TensorFlow 1 graphs for MIGraphX ingestion.
   * - `export_frozen_graph_tf2 <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/migraphx/export_frozen_graph_tf2>`__
     - Model integrators
     - Export frozen TensorFlow 2 graphs for MIGraphX ingestion.

Vision examples
====================================================================

Examples under ``examples/vision/`` cover image classification, detection,
segmentation, and super resolution.

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Example
     - Target user
     - What it demonstrates
   * - `cpp_mnist <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/vision/cpp_mnist>`__
     - C++ application developers
     - MNIST digit classification with the C++ API.
   * - `python_resnet50 <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/vision/python_resnet50>`__
     - Python application developers
     - ResNet50 V2 inference in Python via a Jupyter notebook. Replace the bundled model with your own pre-trained classification model.
   * - `python_nfnet <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/vision/python_nfnet>`__
     - Python application developers
     - NFNet inference and ONNX Runtime comparison.
   * - `python_super_resolution <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/vision/python_super_resolution>`__
     - Python application developers
     - Super resolution inference.
   * - `python_unet <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/vision/python_unet>`__
     - Python application developers
     - U-Net segmentation inference.
   * - `python_3dunet <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/vision/python_3dunet>`__
     - Python application developers
     - 3D U-Net inference.
   * - `python_yolov4 <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/vision/python_yolov4>`__
     - Python application developers
     - YOLOv4 object detection inference.
   * - `python_yolo26 <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/vision/python_yolo26>`__
     - Python application developers
     - YOLO26 object detection inference.

Natural language processing examples
====================================================================

Examples under ``examples/nlp/`` cover question answering and speech
recognition.

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Example
     - Target user
     - What it demonstrates
   * - `python_bert_squad <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/nlp/python_bert_squad>`__
     - Python application developers
     - BERT question answering on SQuAD. Replace ``bertsquad-10.onnx`` and ``inputs.json`` with your model and input data.
   * - `python_rnnt <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/nlp/python_rnnt>`__
     - Python application developers
     - RNN-T speech recognition inference.

Diffusion examples
====================================================================

Examples under ``examples/diffusion/`` cover text-to-image diffusion models.

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Example
     - Target user
     - What it demonstrates
   * - `python_stable_diffusion_21 <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/diffusion/python_stable_diffusion_21>`__
     - Python application developers
     - Stable Diffusion 2.1 text-to-image inference.
   * - `python_stable_diffusion_xl <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/diffusion/python_stable_diffusion_xl>`__
     - Python application developers
     - Stable Diffusion XL text-to-image inference.
   * - `python_stable_diffusion_3 <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/diffusion/python_stable_diffusion_3>`__
     - Python application developers
     - Stable Diffusion 3 text-to-image inference.
   * - `python_flux <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/diffusion/python_flux>`__
     - Python application developers
     - Flux text-to-image inference.
   * - `python_controlnet_canny_sd_15 <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/diffusion/python_controlnet_canny_sd_15>`__
     - Python application developers
     - ControlNet Canny edge guidance with Stable Diffusion 1.5.

Transformer examples
====================================================================

Examples under ``examples/transformers/`` cover large language model and
speech model inference.

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Example
     - Target user
     - What it demonstrates
   * - `python_llama2 <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/transformers/python_llama2>`__
     - Python application developers
     - Llama 2 inference.
   * - `python_whisper <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/transformers/python_whisper>`__
     - Python application developers
     - Whisper automatic speech recognition.

ONNX Runtime examples
====================================================================

Examples under ``examples/onnxruntime/`` show MIGraphX integration through
the ONNX Runtime execution provider.

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Example
     - Target user
     - What it demonstrates
   * - `resnet50 <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/onnxruntime/resnet50>`__
     - ONNX Runtime users
     - ResNet50 inference through ONNX Runtime with the MIGraphX execution provider.
   * - `inceptionV3 <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/onnxruntime/inceptionV3>`__
     - ONNX Runtime users
     - InceptionV3 inference through ONNX Runtime with the MIGraphX execution provider.

See also
====================================================================

* :doc:`Get started with MIGraphX <./getting-started>` for persona-based
  starting paths.
* :doc:`Install MIGraphX with Docker <../install/install-docker>` for a
  containerized build environment.
* :doc:`Validate model outputs <../how-to/model-validation>` to check example
  output against the reference implementation.
