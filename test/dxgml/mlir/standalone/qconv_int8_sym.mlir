// Weight-only quantized convolution test — symmetric int8 weights (no zero-point).
//
// Pattern exercised:
//   %w_dq = dxgml_op.dequantize_linear(%w_q: int8, %scale: f16)
//   %out  = dxgml_op.convolution(%input: f16, %w_dq: f16)
//
// The DxGML fuse_dxgml_dequant pass should fold this into:
//   qconv(%input: f16, %w_q: int8, %scale: f16)  <- in-kernel dequant
//
// Dimensions:
//   input  : 1x3x8x8  f16   (batch=1, C_in=3, H=8, W=8)
//   weight : 16x3x3x3 int8  (C_out=16, C_in=3, kH=3, kW=3)
//   scale  : 16x1x1x1 f16   (per-output-channel)
//   output : 1x16x6x6 f16   ((8-3)/1+1 = 6)

module attributes {gpu.container_module} {
  dxgml.module @qconv_int8_sym {
    dxgml.entry_point @qconv_int8_sym(
        %input : !dxgml.tensor<1x3x8x8x!dxgml.float16>
    ) -> !dxgml.tensor<1x16x6x6x!dxgml.float16>
    attributes {
      torch.onnx_meta.ir_version = 8 : si64,
      torch.onnx_meta.opset_version = 17 : si64,
      torch.onnx_meta.producer_name = "pytorch",
      torch.onnx_meta.producer_version = "2.0.0"
    } {
      %null = dxgml_op.null_ptr

      // Quantized int8 weight constant (per-output-channel symmetric)
      %w_q = dxgml_op.constant(#dxgml.constant_resource<qconv_int8_sym.weight : !dxgml.tensor<16x3x3x3x!dxgml.int8>>)
      // Per-output-channel scale: one scale per output channel
      %scale = dxgml_op.constant(#dxgml.constant_resource<qconv_int8_sym.scale : !dxgml.tensor<16x1x1x1x!dxgml.float16>>)

      // Dequantize weight: int8 -> f16  (symmetric, no zero-point)
      %w_dq = dxgml_op.dequantize_linear(%w_q, %scale) {
        axis = #dxgml.integer<0 : !dxgml.int64>
      } : (!dxgml.tensor<16x3x3x3x!dxgml.int8>,
           !dxgml.tensor<16x1x1x1x!dxgml.float16>)
        -> !dxgml.tensor<16x3x3x3x!dxgml.float16>

      // Convolution: fp16 input x dequantized-fp16 weight -> fp16 output
      %out = dxgml_op.convolution(%input, %w_dq, %null) {
        group_count   = #dxgml.integer<1 : !dxgml.int64>,
        dilations     = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        start_padding = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        end_padding   = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        strides       = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
      } : (!dxgml.tensor<1x3x8x8x!dxgml.float16>,
           !dxgml.tensor<16x3x3x3x!dxgml.float16>,
           !dxgml.null)
        -> !dxgml.tensor<1x16x6x6x!dxgml.float16>

      dxgml.return %out : !dxgml.tensor<1x16x6x6x!dxgml.float16>
    }
  }
  gpu.module @rock_gpu_module {
  }
}
