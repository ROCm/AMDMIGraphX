.. meta::
  :description: MIGraphX debug symbols reference
  :keywords: MIGraphX, code base, contribution, developing, debug symbols

===================
Debug Symbols Usage
===================

MIGraphX supports adding debug symbols to instructions and having them propagate through compiler passes.
Debug symbols can be added when parsing an ONNX file or manually adding instructions through the MIGraphX python API.
* Debug symbols are enabled with ONNX by supplying the `use_debug_symbols` flag in `onnx_options` during parsing.
  Example `onnx_options` usage in C++ API::

    #include <migraphx/migraphx.hpp>

    int main(int argc, char** argv)
    {
        migraphx::onnx_options options;
        options.use_debug_symbols = true;
        auto prog = migraphx::parse_onnx("conv_transpose_test.onnx", options);
    }

* Enabling debug symbols for the ONNX parser inserts the parsed ONNX node name into the resultant MIGraphX instructions.
  **Add simple parsed example IR output with ONNX node names here**

* Each instruction has their ONNX node names.
* You can trace through the ONNX node names of the input instructions to get the ONNX node inputs.
