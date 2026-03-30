dxgml.module {
  dxgml.entry_point @Standalone_Fusion(%arg0: !dxgml.tensor<1x4x2160x3840x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
  attributes {version = #dxgml.version<v0.0.1>, producer_name = "pytorch", producer_version = "2.1.1"} {

    // Pre-convolution scalar-ish transforms on the input tensor.
    %r0 = dxgml_op.relu(%arg0) : (!dxgml.tensor<1x4x2160x3840x!dxgml.float16>) -> !dxgml.tensor<1x4x2160x3840x!dxgml.float16>
    %a0 = dxgml_op.add(%r0, %r0) : (!dxgml.tensor<1x4x2160x3840x!dxgml.float16>, !dxgml.tensor<1x4x2160x3840x!dxgml.float16>) -> !dxgml.tensor<1x4x2160x3840x!dxgml.float16>
    %m0 = dxgml_op.multiply(%r0, %a0) : (!dxgml.tensor<1x4x2160x3840x!dxgml.float16>, !dxgml.tensor<1x4x2160x3840x!dxgml.float16>) -> !dxgml.tensor<1x4x2160x3840x!dxgml.float16>

    %w1 = dxgml_op.constant(#dxgml.constant_resource<_conv1.weight : !dxgml.tensor<32x4x3x3x!dxgml.float16>>)
    %b1 = dxgml_op.constant(#dxgml.constant_resource<_conv1.bias : !dxgml.tensor<32x!dxgml.float16>>)

    // Convolution (stride 2) increases channels 4 -> 32 and halves spatial dims.
    %c0 = dxgml_op.convolution(%m0, %w1, %b1)
    {
        mode = #dxgml_op.convolution_mode_enum_attr<convolution_mode_cross_correlation>,
        direction = #dxgml_op.convolution_direction_enum_attr<convolution_direction_forward>,
        strides = #dxgml.dense_integer_elements<[2, 2]> : !dxgml.tensor<2x!dxgml.int64>,
        dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        output_padding = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        group_count = #dxgml.integer<1 : !dxgml.int64>
    } :
    (
        !dxgml.tensor<1x4x2160x3840x!dxgml.float16>,
        !dxgml.tensor<32x4x3x3x!dxgml.float16>,
        !dxgml.tensor<32x!dxgml.float16>
    ) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>

    // Post-convolution nonlinear + simple arithmetic chain.
    %r1 = dxgml_op.relu(%c0) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %a1 = dxgml_op.add(%r1, %r1) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %m1 = dxgml_op.multiply(%a1, %a1) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>

    dxgml.return %m1 : !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
  }
}

{-#
dialect_resources: {
  dxgml: {
    _conv1.weight: "0x08000000", // Placeholder for weight data
    _conv1.bias:   "0x08000000"  // Placeholder for bias data
  }
}
#-}
