// Weight-only quantized convolution test — int8 weights followed by ReLU + residual add.
//
// Pattern exercised:
//   %w_dq = dxgml_op.dequantize_linear(%w_q: int8, %scale: f16)
//   %conv = dxgml_op.convolution(%input: f16, %w_dq: f16)
//   %act  = dxgml_op.relu(%conv)
//   %out  = dxgml_op.add(%act, %residual)
//
// The DxGML fuse_dxgml_dequant pass folds dequantize+conv into:
//   qconv(%input, %w_q, %scale)  <- in-kernel dequant
// The ReLU+add epilogue is handled by the standard GPU fusion pipeline.
//
// Dimensions:
//   input    : 1x8x12x12 f16   (batch=1, C_in=8, H=12, W=12)
//   weight   : 32x8x3x3  int8  (C_out=32, C_in=8, kH=3, kW=3)
//   scale    : 32x1x1x1  f16   (per-output-channel, symmetric)
//   residual : 1x32x10x10 f16  (same shape as conv output)
//   output   : 1x32x10x10 f16  ((12-3)/1+1 = 10)

module attributes {gpu.container_module} {
  dxgml.module @qconv_int8_relu {
    dxgml.entry_point @qconv_int8_relu(
        %input    : !dxgml.tensor<1x8x12x12x!dxgml.float16>,
        %residual : !dxgml.tensor<1x32x10x10x!dxgml.float16>
    ) -> !dxgml.tensor<1x32x10x10x!dxgml.float16>
    attributes {
      torch.onnx_meta.ir_version = 8 : si64,
      torch.onnx_meta.opset_version = 17 : si64,
      torch.onnx_meta.producer_name = "pytorch",
      torch.onnx_meta.producer_version = "2.0.0"
    } {
      %null = dxgml_op.null_ptr

      // Quantized int8 weight constant (per-output-channel symmetric)
      %w_q = dxgml_op.constant(#dxgml.constant_resource<qconv_int8_relu.weight : !dxgml.tensor<32x8x3x3x!dxgml.int8>>)
      // Per-output-channel scale
      %scale = dxgml_op.constant(#dxgml.constant_resource<qconv_int8_relu.scale : !dxgml.tensor<32x1x1x1x!dxgml.float16>>)

      // Dequantize weight: int8 -> f16  (symmetric)
      %w_dq = dxgml_op.dequantize_linear(%w_q, %scale) {
        axis = #dxgml.integer<0 : !dxgml.int64>
      } : (!dxgml.tensor<32x8x3x3x!dxgml.int8>,
           !dxgml.tensor<32x1x1x1x!dxgml.float16>)
        -> !dxgml.tensor<32x8x3x3x!dxgml.float16>

      // Convolution: fp16 input x dequantized-fp16 weight -> fp16 output
      %conv = dxgml_op.convolution(%input, %w_dq, %null) {
        group_count   = #dxgml.integer<1 : !dxgml.int64>,
        dilations     = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        start_padding = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        end_padding   = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        strides       = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
      } : (!dxgml.tensor<1x8x12x12x!dxgml.float16>,
           !dxgml.tensor<32x8x3x3x!dxgml.float16>,
           !dxgml.null)
        -> !dxgml.tensor<1x32x10x10x!dxgml.float16>

      // ReLU activation
      %act = dxgml_op.relu(%conv)
        : (!dxgml.tensor<1x32x10x10x!dxgml.float16>)
        -> !dxgml.tensor<1x32x10x10x!dxgml.float16>

      // Residual add
      %out = dxgml_op.add(%act, %residual)
        : (!dxgml.tensor<1x32x10x10x!dxgml.float16>,
           !dxgml.tensor<1x32x10x10x!dxgml.float16>)
        -> !dxgml.tensor<1x32x10x10x!dxgml.float16>

      dxgml.return %out : !dxgml.tensor<1x32x10x10x!dxgml.float16>
    }
  }
  gpu.module @rock_gpu_module {
  }
}
