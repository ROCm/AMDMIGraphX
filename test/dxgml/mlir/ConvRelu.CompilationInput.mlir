dxgml.module {
    dxgml.entry_point @Conv_Relu(%arg0: !dxgml.tensor<1x4x2160x3840x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    {
        %_conv1.weight = dxgml_op.constant(#dxgml.constant_resource<_conv1.weight : !dxgml.tensor<32x4x3x3x!dxgml.float16>>)
        %_conv1.bias = dxgml_op.constant(#dxgml.constant_resource<_conv1.bias : !dxgml.tensor<32x!dxgml.float16>>)

        %0 = dxgml_op.convolution(%arg0, %_conv1.weight, %_conv1.bias) 
        {
            mode = #dxgml_op.convolution_mode_enum_attr<convolution_mode_cross_correlation>, 
            direction = #dxgml_op.convolution_direction_enum_attr<convolution_direction_forward>, 
            strides = #dxgml.dense_integer_elements<[2, 2]> : !dxgml.tensor<2x!dxgml.int64>, 
            dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
            start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
            end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
            group_count = #dxgml.integer<1 : !dxgml.int64>
        } :
        (
            !dxgml.tensor<1x4x2160x3840x!dxgml.float16>,
            !dxgml.tensor<32x4x3x3x!dxgml.float16>,
            !dxgml.tensor<32x!dxgml.float16>
        ) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>

        %1 = dxgml_op.relu(%0) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
        dxgml.return %1 : !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    }
}

{-#
dialect_resources: {
    dxgml: {
        _conv1.weight: "0x08000000", // Placeholder for actual weight data
        _conv1.bias: "0x08000000" // Placeholder for actual bias data
    }
}
#-}
