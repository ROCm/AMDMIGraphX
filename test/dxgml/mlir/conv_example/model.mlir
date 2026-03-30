// DXML dialect equivalent of migraphx_convolution_example.mlir
//
// Computes: conv(input, weight) -> batch_norm -> relu -> max_pool
//   input   : 1x3x32x32 f32  (batch=1, C=3, H=32, W=32)
//   weight  : 64x3x3x3  f32  (64 output channels, 3x3 kernel)
//   bias    : 64         f32
//   conv    : stride=2, pad=1 -> 1x64x16x16 f32
//   pool    : 3x3 max-pool, stride=2, pad=0 -> 1x64x7x7 f32
//
// To validate (parse only, no arch needed):
//   rocmlir-driver model.mlir
//
// To lower to MIGraphX IR (batch_norm + pooling become migraphx.* ops):
//   rocmlir-driver model.mlir --host-pipeline=dxgml
//
// To lower to TOSA and then GPU (full pipeline):
//   rocmlir-driver model.mlir --host-pipeline=dxgml ^
//       --kernel-pipeline=highlevel,gpu --arch=gfx1201
//
// batch_norm_inference lowers to TOSA primitives (sub/rsqrt/mul/add with
// per-channel broadcast) and max_pooling lowers to tosa.max_pool2d.

module attributes {gpu.container_module} {
  dxgml.module @conv_example {
    dxgml.entry_point @conv_bn_relu_pool(
        %input    : !dxgml.tensor<1x3x32x32x!dxgml.float32>
    ) -> !dxgml.tensor<1x64x7x7x!dxgml.float32>
    attributes {
      torch.onnx_meta.ir_version = 8 : si64,
      torch.onnx_meta.opset_version = 17 : si64,
      torch.onnx_meta.producer_name = "pytorch",
      torch.onnx_meta.producer_version = "2.0.0"
    } {
      // Convolution weights and bias (constant model parameters)
      %weight = dxgml_op.constant(#dxgml.constant_resource<conv.weight : !dxgml.tensor<64x3x3x3x!dxgml.float32>>)
      %bias   = dxgml_op.constant(#dxgml.constant_resource<conv.bias   : !dxgml.tensor<64x!dxgml.float32>>)

      // Batch-norm parameters (scale, B, mean, var)
      %bn_scale = dxgml_op.constant(#dxgml.constant_resource<bn.scale    : !dxgml.tensor<64x!dxgml.float32>>)
      %bn_bias  = dxgml_op.constant(#dxgml.constant_resource<bn.bias     : !dxgml.tensor<64x!dxgml.float32>>)
      %bn_mean  = dxgml_op.constant(#dxgml.constant_resource<bn.mean     : !dxgml.tensor<64x!dxgml.float32>>)
      %bn_var   = dxgml_op.constant(#dxgml.constant_resource<bn.variance : !dxgml.tensor<64x!dxgml.float32>>)

      // Step 1: Convolution  1x3x32x32 -> 1x64x16x16
      //   stride=2, pad=1, 3x3 kernel: out = (32 + 2*1 - 3)/2 + 1 = 16
      %conv = dxgml_op.convolution(%input, %weight, %bias) {
        group_count   = #dxgml.integer<1 : !dxgml.int64>,
        dilations     = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        end_padding   = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        strides       = #dxgml.dense_integer_elements<[2, 2]> : !dxgml.tensor<2x!dxgml.int64>
      } : (!dxgml.tensor<1x3x32x32x!dxgml.float32>,
           !dxgml.tensor<64x3x3x3x!dxgml.float32>,
           !dxgml.tensor<64x!dxgml.float32>)
        -> !dxgml.tensor<1x64x16x16x!dxgml.float32>

      // Step 2: Batch normalization  (epsilon=1e-5)
      %bn = dxgml_op.batch_normalization(%conv, %bn_scale, %bn_bias, %bn_mean, %bn_var) {
        epsilon = #dxgml.float<1.0e-05 : !dxgml.float64>
      } : (!dxgml.tensor<1x64x16x16x!dxgml.float32>,
           !dxgml.tensor<64x!dxgml.float32>,
           !dxgml.tensor<64x!dxgml.float32>,
           !dxgml.tensor<64x!dxgml.float32>,
           !dxgml.tensor<64x!dxgml.float32>)
        -> !dxgml.tensor<1x64x16x16x!dxgml.float32>

      // Step 3: ReLU activation
      %relu = dxgml_op.relu(%bn)
        : (!dxgml.tensor<1x64x16x16x!dxgml.float32>)
        -> !dxgml.tensor<1x64x16x16x!dxgml.float32>

      // Step 4: Max pooling  3x3, stride=2, pad=0  -> 1x64x7x7
      //   out = (16 - 3)/2 + 1 = 7
      %pool = dxgml_op.max_pooling(%relu) {
        strides       = #dxgml.dense_integer_elements<[2, 2]> : !dxgml.tensor<2x!dxgml.int64>,
        window_size   = #dxgml.dense_integer_elements<[3, 3]> : !dxgml.tensor<2x!dxgml.int64>,
        start_padding = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        end_padding   = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>
      } : (!dxgml.tensor<1x64x16x16x!dxgml.float32>)
        -> (!dxgml.tensor<1x64x7x7x!dxgml.float32>)

      dxgml.return %pool : !dxgml.tensor<1x64x7x7x!dxgml.float32>
    }
  }
  gpu.module @rock_gpu_module {
  }
}
