// StandaloneCluster.CompilationInput.mlir
// Minimal DxGML fixture: relu/add/mul -> conv -> relu/add/mul
// Input:  arg0 half[1,4,2160,3840]
// Output: half[1,32,1080,1920]
//
// Instruction order (matches test expectations):
//   [0]  arg0
//   [1]  relu(arg0)
//   [2]  add(relu, arg0)
//   [3]  mul(add, arg0)
//   [4]  _conv1.weight
//   [5]  _conv1.bias
//   [6]  conv(mul, weight, bias)
//   [7]  relu(conv)
//   [8]  add(relu, conv)
//   [9]  mul(add, relu)
//   [10] return

module {
  dxgml.module @standalone_cluster {
    dxgml.entry_point @forward(
        %arg0 : !dxgml.tensor<1x4x2160x3840x!dxgml.float16>
    ) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16> {
      %pre_relu = dxgml_op.relu(%arg0)            : (!dxgml.tensor<1x4x2160x3840x!dxgml.float16>) -> !dxgml.tensor<1x4x2160x3840x!dxgml.float16>
      %pre_add  = dxgml_op.add(%pre_relu, %arg0)  : (!dxgml.tensor<1x4x2160x3840x!dxgml.float16>, !dxgml.tensor<1x4x2160x3840x!dxgml.float16>) -> !dxgml.tensor<1x4x2160x3840x!dxgml.float16>
      %pre_mul  = dxgml_op.multiply(%pre_add, %arg0) : (!dxgml.tensor<1x4x2160x3840x!dxgml.float16>, !dxgml.tensor<1x4x2160x3840x!dxgml.float16>) -> !dxgml.tensor<1x4x2160x3840x!dxgml.float16>
      %weight   = dxgml_op.constant(#dxgml.constant_resource<_conv1.weight : !dxgml.tensor<32x4x3x3x!dxgml.float16>>)
      %bias     = dxgml_op.constant(#dxgml.constant_resource<_conv1.bias   : !dxgml.tensor<32x!dxgml.float16>>)
      %conv = dxgml_op.convolution(%pre_mul, %weight, %bias) {
        group_count   = #dxgml.integer<1 : !dxgml.int64>,
        dilations     = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        end_padding   = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        strides       = #dxgml.dense_integer_elements<[2, 2]> : !dxgml.tensor<2x!dxgml.int64>
      } : (!dxgml.tensor<1x4x2160x3840x!dxgml.float16>, !dxgml.tensor<32x4x3x3x!dxgml.float16>, !dxgml.tensor<32x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
      %post_relu = dxgml_op.relu(%conv)              : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
      %post_add  = dxgml_op.add(%post_relu, %conv)   : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
      %post_mul  = dxgml_op.multiply(%post_add, %post_relu) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
      dxgml.return %post_mul : !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    }
  }
}
