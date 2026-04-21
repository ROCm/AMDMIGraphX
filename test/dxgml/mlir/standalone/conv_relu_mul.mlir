// Standalone test: convolution -> relu activation -> elementwise multiply
//
// Computes: multiply(relu(conv(input, weight)), scale)
//   input  : 1x4x10x10 f16  (batch=1, C=4, H=10, W=10)
//   weight : 8x4x3x3   f16  (8 output channels, 3x3 kernel)
//   scale  : 1x8x8x8   f16  (per-element scale, same shape as conv output)
//   conv   : stride=1, pad=0 -> 1x8x8x8 f16  (out = (10-3)/1+1 = 8)
//   result : 1x8x8x8   f16

module attributes {gpu.container_module} {
  dxgml.module @conv_relu_mul {
    dxgml.entry_point @conv_relu_mul(
        %input : !dxgml.tensor<1x4x10x10x!dxgml.float16>,
        %scale : !dxgml.tensor<1x8x8x8x!dxgml.float16>
    ) -> !dxgml.tensor<1x8x8x8x!dxgml.float16>
    attributes {
      torch.onnx_meta.ir_version = 8 : si64,
      torch.onnx_meta.opset_version = 17 : si64,
      torch.onnx_meta.producer_name = "pytorch",
      torch.onnx_meta.producer_version = "2.0.0"
    } {
      %weight = dxgml_op.constant(#dxgml.constant_resource<conv.weight : !dxgml.tensor<8x4x3x3x!dxgml.float16>>)
      %null = dxgml_op.null_ptr

      // Step 1: Convolution  1x4x10x10 -> 1x8x8x8
      %conv = dxgml_op.convolution(%input, %weight, %null) {
        group_count   = #dxgml.integer<1 : !dxgml.int64>,
        dilations     = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        start_padding = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        end_padding   = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        strides       = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
      } : (!dxgml.tensor<1x4x10x10x!dxgml.float16>,
           !dxgml.tensor<8x4x3x3x!dxgml.float16>,
           !dxgml.null)
        -> !dxgml.tensor<1x8x8x8x!dxgml.float16>

      // Step 2: ReLU activation
      %act = dxgml_op.relu(%conv)
        : (!dxgml.tensor<1x8x8x8x!dxgml.float16>)
        -> !dxgml.tensor<1x8x8x8x!dxgml.float16>

      // Step 3: Elementwise multiply (e.g. gating or scaling)
      %result = dxgml_op.multiply(%act, %scale)
        : (!dxgml.tensor<1x8x8x8x!dxgml.float16>,
           !dxgml.tensor<1x8x8x8x!dxgml.float16>)
        -> !dxgml.tensor<1x8x8x8x!dxgml.float16>

      dxgml.return %result : !dxgml.tensor<1x8x8x8x!dxgml.float16>
    }
  }
  gpu.module @rock_gpu_module {
  }
}
