// Weight-only quantized convolution test — affine int8 weights with zero-point.
//
// Pattern exercised:
//   %w_dq = dxgml_op.dequantize_linear(%w_q: int8, %scale: f16, %zp: int8)
//   %out  = dxgml_op.convolution(%input: f16, %w_dq: f16)
//
// The DxGML fuse_dxgml_dequant pass should fold this into:
//   qconv(%input: f16, %w_q: int8, %scale: f16, %zp: int8)  <- in-kernel dequant
//
// Dequant formula applied inside the kernel: scale * (weight - zp)
//
// Dimensions:
//   input  : 1x4x10x10 f16   (batch=1, C_in=4, H=10, W=10)
//   weight : 8x4x3x3   int8  (C_out=8, C_in=4, kH=3, kW=3)
//   scale  : 8x1x1x1   f16   (per-output-channel)
//   zp     : 8x1x1x1   int8  (per-output-channel zero-point)
//   output : 1x8x8x8   f16   ((10-3)/1+1 = 8)

module attributes {gpu.container_module} {
  dxgml.module @qconv_int8_zp {
    dxgml.entry_point @qconv_int8_zp(
        %input : !dxgml.tensor<1x4x10x10x!dxgml.float16>
    ) -> !dxgml.tensor<1x8x8x8x!dxgml.float16>
    attributes {
      torch.onnx_meta.ir_version = 8 : si64,
      torch.onnx_meta.opset_version = 17 : si64,
      torch.onnx_meta.producer_name = "pytorch",
      torch.onnx_meta.producer_version = "2.0.0"
    } {
      %null = dxgml_op.null_ptr

      // Quantized int8 weight constant
      %w_q = dxgml_op.constant(#dxgml.constant_resource<qconv_int8_zp.weight : !dxgml.tensor<8x4x3x3x!dxgml.int8>>)
      // Per-output-channel scale
      %scale = dxgml_op.constant(#dxgml.constant_resource<qconv_int8_zp.scale : !dxgml.tensor<8x1x1x1x!dxgml.float16>>)
      // Per-output-channel zero-point (non-zero → affine quantization)
      %zp = dxgml_op.constant(#dxgml.constant_resource<qconv_int8_zp.zero_point : !dxgml.tensor<8x1x1x1x!dxgml.int8>>)

      // Dequantize weight: int8 -> f16  (affine: scale*(q - zp))
      %w_dq = dxgml_op.dequantize_linear(%w_q, %scale, %zp) {
        axis = #dxgml.integer<0 : !dxgml.int64>
      } : (!dxgml.tensor<8x4x3x3x!dxgml.int8>,
           !dxgml.tensor<8x1x1x1x!dxgml.float16>,
           !dxgml.tensor<8x1x1x1x!dxgml.int8>)
        -> !dxgml.tensor<8x4x3x3x!dxgml.float16>

      // Convolution: fp16 input x dequantized-fp16 weight -> fp16 output
      %out = dxgml_op.convolution(%input, %w_dq, %null) {
        group_count   = #dxgml.integer<1 : !dxgml.int64>,
        dilations     = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        start_padding = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        end_padding   = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        strides       = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
      } : (!dxgml.tensor<1x4x10x10x!dxgml.float16>,
           !dxgml.tensor<8x4x3x3x!dxgml.float16>,
           !dxgml.null)
        -> !dxgml.tensor<1x8x8x8x!dxgml.float16>

      dxgml.return %out : !dxgml.tensor<1x8x8x8x!dxgml.float16>
    }
  }
  gpu.module @rock_gpu_module {
  }
}
