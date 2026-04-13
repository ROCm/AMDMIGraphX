// ConvRelu.CompilationInput.mlir
// Minimal DxGML fixture: conv(stride=2,pad=1) -> relu
// Input:  arg0 half[1,4,2160,3840]
// Weight: _conv1.weight half[32,4,3,3]
// Bias:   _conv1.bias   half[32]
// Output: half[1,32,1080,1920]

module {
  dxgml.module @conv_relu {
    dxgml.entry_point @forward(
        %arg0 : !dxgml.tensor<1x4x2160x3840x!dxgml.float16>
    ) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16> {
      %weight = dxgml_op.constant(#dxgml.constant_resource<_conv1.weight : !dxgml.tensor<32x4x3x3x!dxgml.float16>>)
      %bias   = dxgml_op.constant(#dxgml.constant_resource<_conv1.bias   : !dxgml.tensor<32x!dxgml.float16>>)
      %conv = dxgml_op.convolution(%arg0, %weight, %bias) {
        group_count   = #dxgml.integer<1 : !dxgml.int64>,
        dilations     = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        end_padding   = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        strides       = #dxgml.dense_integer_elements<[2, 2]> : !dxgml.tensor<2x!dxgml.int64>
      } : (!dxgml.tensor<1x4x2160x3840x!dxgml.float16>, !dxgml.tensor<32x4x3x3x!dxgml.float16>, !dxgml.tensor<32x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
      %relu = dxgml_op.relu(%conv) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
      dxgml.return %relu : !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    }
  }
}
