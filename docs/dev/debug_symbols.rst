.. meta::
  :description: MIGraphX debug symbols reference
  :keywords: MIGraphX, code base, contribution, developing, debug symbols

===================
Using Debug Symbols
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

  Using the `--debug-symbols` option in the MIGraphX driver tool::

    <migraphx_driver> read <path_to_model.onnx> --debug-symbols
    <migraphx_driver> compile <path_to_model.onnx> --debug-symbols

* Enabling debug symbols for the ONNX parser inserts the parsed ONNX node name into the resultant MIGraphX instructions.
  Here is some example MIGraphX IR output from parsing a simple classification model `mnist-8.onnx`.
  The model is avaliable at: https://github.com/onnx/models/blob/main/validated/vision/classification/mnist/model/mnist-8.onnx.
  Where the text after the `#` are the debug symbols::

    module: "main"
    Input3 = @param:Input3 -> float_type, {1, 1, 28, 28}, {784, 784, 28, 1}
    @1 = @literal{-0.044856, 0.00779166, 0.0681008, 0.0299937, -0.12641, 0.140219, -0.0552849, -0.0493838, 0.0843221, -0.0545404} -> float_type, {1, 10}, {10, 1} # Parameter194
    @2 = @literal{256, 10} -> int64_type, {2}, {1} # Parameter193_reshape1_shape
    @3 = @literal{1, 256} -> int64_type, {2}, {1} # Pooling160_Output_0_reshape0_shape
    @4 = @literal{ ... } -> float_type, {16, 1, 1}, {1, 1, 1} # Parameter88
    @5 = @literal{-0.16154, -0.433836, 0.0916414, -0.0168522, -0.0650264, -0.131738, 0.0204176, -0.12111} -> float_type, {8, 1, 1}, {1, 1, 1} # Parameter6
    @6 = @literal{ ... } -> float_type, {8, 1, 5, 5}, {25, 25, 5, 1} # Parameter5
    @7 = @literal{ ... } -> float_type, {16, 8, 5, 5}, {200, 25, 5, 1} # Parameter87
    @8 = @literal{ ... } -> float_type, {16, 4, 4, 10}, {160, 40, 10, 1} # Parameter193
    @9 = reshape[dims={256, 10}](@8) -> float_type, {256, 10}, {10, 1} # Times212_reshape1
    @10 = convolution[padding={2, 2, 2, 2},stride={1, 1},dilation={1, 1},group=1,padding_mode=0](Input3,@6) -> float_type, {1, 8, 28, 28}, {6272, 784, 28, 1} # Convolution28
    @11 = multibroadcast[out_lens={1, 8, 28, 28},out_dyn_dims={}](@5) -> float_type, {1, 8, 28, 28}, {0, 1, 0, 0} # Plus30
    @12 = add(@10,@11) -> float_type, {1, 8, 28, 28}, {6272, 784, 28, 1} # Plus30
    @13 = relu(@12) -> float_type, {1, 8, 28, 28}, {6272, 784, 28, 1} # ReLU32
    @14 = pooling[mode=max,padding={0, 0, 0, 0},padding_mode=0,stride={2, 2},lengths={2, 2},dilations={1, 1},ceil_mode=0,count_include_pad=0,lp_order=2,dyn_global=0](@13) -> float_type, {1, 8, 14, 14}, {1568, 196, 14, 1} # Pooling66
    @15 = convolution[padding={2, 2, 2, 2},stride={1, 1},dilation={1, 1},group=1,padding_mode=0](@14,@7) -> float_type, {1, 16, 14, 14}, {3136, 196, 14, 1} # Convolution110
    @16 = multibroadcast[out_lens={1, 16, 14, 14},out_dyn_dims={}](@4) -> float_type, {1, 16, 14, 14}, {0, 1, 0, 0} # Plus112
    @17 = add(@15,@16) -> float_type, {1, 16, 14, 14}, {3136, 196, 14, 1} # Plus112
    @18 = relu(@17) -> float_type, {1, 16, 14, 14}, {3136, 196, 14, 1} # ReLU114
    @19 = pooling[mode=max,padding={0, 0, 0, 0},padding_mode=0,stride={3, 3},lengths={3, 3},dilations={1, 1},ceil_mode=0,count_include_pad=0,lp_order=2,dyn_global=0](@18) -> float_type, {1, 16, 4, 4}, {256, 16, 4, 1} # Pooling160
    @20 = reshape[dims={1, 256}](@19) -> float_type, {1, 256}, {256, 1} # Times212_reshape0
    @21 = dot(@20,@9) -> float_type, {1, 10}, {10, 1} # Times212
    @22 = add(@21,@1) -> float_type, {1, 10}, {10, 1} # Plus214
    @23 = @return(@22) # @output_0:Plus214_Output_0

* The debug symbols will propagate throughout the compilation passes.
  Here is the same `mnist-8.onnx` model compiled IR::

    module: "main"
    @0 = check_context::migraphx::gpu::context -> float_type, {}, {}
    @1 = hip::hip_allocate_memory[shape=int8_type, {31360}, {1},id=main:scratch] -> int8_type, {31360}, {1}
    @2 = hip::hip_copy_literal[id=main:@literal:5] -> float_type, {1, 10}, {10, 1} # Parameter194
    @3 = hip::hip_copy_literal[id=main:@literal:4] -> float_type, {256, 10}, {10, 1} # Times212_reshape1
    @4 = hip::hip_copy_literal[id=main:@literal:3] -> float_type, {16, 1, 1}, {1, 1, 1} # Parameter88
    @5 = hip::hip_copy_literal[id=main:@literal:1] -> float_type, {8, 1, 1}, {1, 1, 1} # Parameter6
    @6 = hip::hip_copy_literal[id=main:@literal:0] -> float_type, {8, 1, 5, 5}, {25, 25, 5, 1} # Convolution28, Parameter5
    @7 = hip::hip_copy_literal[id=main:@literal:2] -> float_type, {16, 8, 5, 5}, {200, 25, 5, 1} # Convolution110, Parameter87
    @8 = load[offset=6272,end=31360](@1) -> float_type, {1, 8, 28, 28}, {6272, 784, 28, 1}
    @9 = multibroadcast[out_lens={1, 8, 28, 28},out_dyn_dims={}](@5) -> float_type, {1, 8, 28, 28}, {0, 1, 0, 0} # Plus30
    Input3 = @param:Input3 -> float_type, {1, 1, 28, 28}, {784, 784, 28, 1} # Convolution28
    @11 = gpu::code_object[code_object=5632,symbol_name=channelwise_conv_add_relu_kernel,global=11520,local=480,](Input3,@6,@9,@8) -> float_type, {1, 8, 28, 28}, {6272, 784, 28, 1} # Convolution28, Plus30, ReLU32
    @12 = load[offset=0,end=6272](@1) -> float_type, {1, 8, 14, 14}, {1568, 196, 14, 1}
    @13 = gpu::code_object[code_object=5384,symbol_name=pooling_kernel,global=6272,local=256,](@11,@12) -> float_type, {1, 8, 14, 14}, {1568, 196, 14, 1} # Convolution110, Pooling66
    @14 = load[offset=6272,end=18816](@1) -> float_type, {1, 16, 14, 14}, {3136, 196, 14, 1}
    @15 = gpu::code_object[code_object=6184,symbol_name=mlir_convolution_add_relu,global=1792,local=256,output_arg=3,](@13,@7,@4,@14) -> float_type, {1, 16, 14, 14}, {3136, 196, 14, 1} # Convolution110, Plus112, ReLU114
    @16 = load[offset=0,end=1024](@1) -> float_type, {1, 16, 4, 4}, {256, 16, 4, 1}
    @17 = gpu::code_object[code_object=5512,symbol_name=pooling_kernel,global=2048,local=256,](@15,@16) -> float_type, {1, 16, 4, 4}, {256, 16, 4, 1} # Pooling160
    main:#output_0 = @param:main:#output_0 -> float_type, {1, 10}, {10, 1} # @output_0:Plus214_Output_0
    @19 = gpu::code_object[code_object=6176,symbol_name=mlir_reshape_dot_add,global=64,local=64,output_arg=3,](@17,@3,@2,main:#output_0) -> float_type, {1, 10}, {10, 1} # Plus214, Times212, Times212_reshape0
    @20 = @return(@19) # @output_0:Plus214_Output_0

* Debug symbols can also be manually added through the Python API when adding instructions::

    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y],
                                 debug_symbols=["sym_a", "sym_b"])
    assert add_ins.get_debug_symbols() == {"sym_a", "sym_b"}

* Or when using macros in the Python API::

    p = migraphx.program()
    mm = p.get_main_module()
    a = mm.add_literal(_make_arg([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    b = mm.add_literal(_make_arg([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
    gemm_mac = migraphx.macro("gemm")
    gemm_result = mm.add_macro(gemm_mac, [a, b])
    einsum_mac = migraphx.macro("einsum", equation="ij,jk->ik")
    einsum_result = mm.insert_macro(gemm_result[0], einsum_mac, [a, b],
                                    debug_symbols=["macro:einsum"])


=======================
Examining Debug Symbols
=======================

* Using the debug symbol option when parsing an ONNX file adds the corresponding ONNX node names to each MIGraphX instruction.
  You can trace through the ONNX node names of the input instructions to get the ONNX node inputs.
  Let us look at the compiled `mnist-8.onnx` IR output and show how to look at an MIGraphX kernel instruction and relate it to the ONNX nodes::

    module: "main"
    @0 = check_context::migraphx::gpu::context -> float_type, {}, {}
    @1 = hip::hip_allocate_memory[shape=int8_type, {31360}, {1},id=main:scratch] -> int8_type, {31360}, {1}
    @2 = hip::hip_copy_literal[id=main:@literal:5] -> float_type, {1, 10}, {10, 1} # Parameter194
    @3 = hip::hip_copy_literal[id=main:@literal:4] -> float_type, {256, 10}, {10, 1} # Times212_reshape1
    arg_2 -> @4 = hip::hip_copy_literal[id=main:@literal:3] -> float_type, {16, 1, 1}, {1, 1, 1} # Parameter88
    @5 = hip::hip_copy_literal[id=main:@literal:1] -> float_type, {8, 1, 1}, {1, 1, 1} # Parameter6
    @6 = hip::hip_copy_literal[id=main:@literal:0] -> float_type, {8, 1, 5, 5}, {25, 25, 5, 1} # Convolution28, Parameter5
    arg_1 -> @7 = hip::hip_copy_literal[id=main:@literal:2] -> float_type, {16, 8, 5, 5}, {200, 25, 5, 1} # Convolution110, Parameter87
    @8 = load[offset=6272,end=31360](@1) -> float_type, {1, 8, 28, 28}, {6272, 784, 28, 1}
    @9 = multibroadcast[out_lens={1, 8, 28, 28},out_dyn_dims={}](@5) -> float_type, {1, 8, 28, 28}, {0, 1, 0, 0} # Plus30
    Input3 = @param:Input3 -> float_type, {1, 1, 28, 28}, {784, 784, 28, 1} # Convolution28
    @11 = gpu::code_object[code_object=5632,symbol_name=channelwise_conv_add_relu_kernel,global=11520,local=480,](Input3,@6,@9,@8) -> float_type, {1, 8, 28, 28}, {6272, 784, 28, 1} # Convolution28, Plus30, ReLU32
    @12 = load[offset=0,end=6272](@1) -> float_type, {1, 8, 14, 14}, {1568, 196, 14, 1}
    arg_0 -> @13 = gpu::code_object[code_object=5384,symbol_name=pooling_kernel,global=6272,local=256,](@11,@12) -> float_type, {1, 8, 14, 14}, {1568, 196, 14, 1} # Convolution110, Pooling66
    arg_4 -> @14 = load[offset=6272,end=18816](@1) -> float_type, {1, 16, 14, 14}, {3136, 196, 14, 1}
    kernel -> @15 = gpu::code_object[code_object=6184,symbol_name=mlir_convolution_add_relu,global=1792,local=256,output_arg=3,](@13,@7,@4,@14) -> float_type, {1, 16, 14, 14}, {3136, 196, 14, 1} # Convolution110, Plus112, ReLU114
    ...

* Looking at instruction `@15` that is marked as the `kernel`, we know from the name of the code_object (`mlir_convolution_add_relu`) that this is a fused convolution with bias and a ReLU activation.
  The instruction has the debug symbols `Convolution110, Plus112, ReLU114`.
  Those node names line up with the ONNX nodes for the original `Conv`, `Add`, and `Relu` that were fused.
  To trace the inputs with respect to the ONNX model, we look at the input instructions to `@15`.
  Which are `(@13, @7, @4, @14)`.
  Instruction `@13` is marked as `arg_0` in the example and is a pooling kernel with the debug symbols `Convolution110, Pooling66`.
  `Pooling66` is the ONNX node name for the `MaxPool` node, so we know it is a max pooling.
  It also has the `Convolution110` debug symbol, which means it was altered by a compilation pass in MIGraphX that also touched `@15`.
  Instruction `@7` is a literal and has the debug symbols `Convolution110, Parameter87`.
  `Parameter87` is the initializer name for the weights tensor into `Convolution110`.


* Note that debug symbols are propagated through compiler passes in MIGraphX such that replaced instructions inherit all of the debug symbols from the instructions they replace.
  This means that a parsed debug symbol can end up in multiple instructions after compilation.
  For example, the `Convolution110` debug symbol appearing in multiple places in the above example.
