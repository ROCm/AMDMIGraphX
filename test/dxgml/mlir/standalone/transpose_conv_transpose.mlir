// Standalone test: transpose -> convolution -> transpose
//
// Computes: NCHW_result = transpose(conv(transpose(input_NHWC), weight), [0,2,3,1])
//   Converts NHWC input to NCHW for convolution, then converts result back to NHWC.
//   This pattern appears when framework-native NHWC tensors pass through a Conv layer.
//
//   input_NHWC : 1x8x8x3  f16  (N=1, H=8, W=8, C=3) — channel-last layout
//   weight     : 16x3x3x3 f16  (16 output channels, 3x3 kernel, NCHW weight)
//   conv input : 1x3x8x8  f16  (transposed to NCHW for convolution)
//   conv output: 1x16x6x6 f16  (stride=1, pad=0: out = (8-3)/1+1 = 6)
//   result     : 1x6x6x16 f16  (transposed back to NHWC)

module attributes {gpu.container_module} {
  dxgml.module @transpose_conv_transpose {
    dxgml.entry_point @nhwc_conv_nhwc(
        %input_nhwc : !dxgml.tensor<1x8x8x3x!dxgml.float16>
    ) -> !dxgml.tensor<1x6x6x16x!dxgml.float16>
    attributes {
      torch.onnx_meta.ir_version = 8 : si64,
      torch.onnx_meta.opset_version = 17 : si64,
      torch.onnx_meta.producer_name = "pytorch",
      torch.onnx_meta.producer_version = "2.0.0"
    } {
      %weight = dxgml_op.constant(#dxgml.constant_resource<conv.weight : !dxgml.tensor<16x3x3x3x!dxgml.float16>>)
      %null = dxgml_op.null_ptr

      // Step 1: Transpose NHWC -> NCHW  [N,H,W,C] -> [N,C,H,W]  (0,3,1,2)
      %input_nchw = dxgml_op.transpose(%input_nhwc) {
        permutation = #dxgml.dense_integer_elements<[0, 3, 1, 2]> : !dxgml.tensor<4x!dxgml.int64>
      } : (!dxgml.tensor<1x8x8x3x!dxgml.float16>)
        -> !dxgml.tensor<1x3x8x8x!dxgml.float16>

      // Step 2: Convolution  1x3x8x8 -> 1x16x6x6
      //   stride=1, pad=0, 3x3 kernel: out = (8-3)/1+1 = 6
      %conv = dxgml_op.convolution(%input_nchw, %weight, %null) {
        group_count   = #dxgml.integer<1 : !dxgml.int64>,
        dilations     = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        start_padding = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        end_padding   = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        strides       = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
      } : (!dxgml.tensor<1x3x8x8x!dxgml.float16>,
           !dxgml.tensor<16x3x3x3x!dxgml.float16>,
           !dxgml.null)
        -> !dxgml.tensor<1x16x6x6x!dxgml.float16>

      // Step 3: Transpose NCHW -> NHWC  [N,C,H,W] -> [N,H,W,C]  (0,2,3,1)
      %result = dxgml_op.transpose(%conv) {
        permutation = #dxgml.dense_integer_elements<[0, 2, 3, 1]> : !dxgml.tensor<4x!dxgml.int64>
      } : (!dxgml.tensor<1x16x6x6x!dxgml.float16>)
        -> !dxgml.tensor<1x6x6x16x!dxgml.float16>

      dxgml.return %result : !dxgml.tensor<1x6x6x16x!dxgml.float16>
    }
  }
  gpu.module @rock_gpu_module {
  }
}
