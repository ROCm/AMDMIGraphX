// Standalone test: convolution -> activation -> elementwise add
//
// Computes: add(relu(conv(input, weight)), residual)
//   input    : 1x3x8x8  f16  (batch=1, C=3, H=8, W=8)
//   weight   : 16x3x3x3 f16  (16 output channels, 3x3 kernel)
//   residual : 1x16x6x6 f16  (same shape as conv output)
//   conv     : stride=1, pad=0 -> 1x16x6x6 f16  (out = (8-3)/1+1 = 6)
//   result   : 1x16x6x6 f16

module attributes {gpu.container_module} {
  dxgml.module @conv_act_add {
    dxgml.entry_point @conv_relu_add(
        %input    : !dxgml.tensor<1x3x8x8x!dxgml.float16>,
        %residual : !dxgml.tensor<1x16x6x6x!dxgml.float16>
    ) -> !dxgml.tensor<1x16x6x6x!dxgml.float16>
    attributes {
      torch.onnx_meta.ir_version = 8 : si64,
      torch.onnx_meta.opset_version = 17 : si64,
      torch.onnx_meta.producer_name = "pytorch",
      torch.onnx_meta.producer_version = "2.0.0"
    } {
      %weight = dxgml_op.constant(#dxgml.constant_resource<conv.weight : !dxgml.tensor<16x3x3x3x!dxgml.float16>>)
      %null = dxgml_op.null_ptr

      // Step 1: Convolution  1x3x8x8 -> 1x16x6x6
      %conv = dxgml_op.convolution(%input, %weight, %null) {
        group_count   = #dxgml.integer<1 : !dxgml.int64>,
        dilations     = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        start_padding = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        end_padding   = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        strides       = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
      } : (!dxgml.tensor<1x3x8x8x!dxgml.float16>,
           !dxgml.tensor<16x3x3x3x!dxgml.float16>,
           !dxgml.null)
        -> !dxgml.tensor<1x16x6x6x!dxgml.float16>

      // Step 2: ReLU activation
      %act = dxgml_op.relu(%conv)
        : (!dxgml.tensor<1x16x6x6x!dxgml.float16>)
        -> !dxgml.tensor<1x16x6x6x!dxgml.float16>

      // Step 3: Elementwise add (residual connection)
      %result = dxgml_op.add(%act, %residual)
        : (!dxgml.tensor<1x16x6x6x!dxgml.float16>,
           !dxgml.tensor<1x16x6x6x!dxgml.float16>)
        -> !dxgml.tensor<1x16x6x6x!dxgml.float16>

      dxgml.return %result : !dxgml.tensor<1x16x6x6x!dxgml.float16>
    }
  }
  gpu.module @rock_gpu_module {
  }
}
