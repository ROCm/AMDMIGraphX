module {
  dxgml.module @model {
  dxgml.entry_point @torch_jit(%arg0: !dxgml.tensor<1x4x2160x3840x!dxgml.float16>) -> !dxgml.tensor<1x4x2160x3840x!dxgml.float16> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {aimet_torch = 1 : si64}, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "1.13.1"} {
    %_conv1.weight = dxgml_op.constant(#dxgml.constant_resource<_conv1.weight : !dxgml.tensor<32x4x3x3x!dxgml.float16>>)
    %_conv1.bias = dxgml_op.constant(#dxgml.constant_resource<_conv1.bias : !dxgml.tensor<32x!dxgml.float16>>)
    %_RDB1.conv1.weight = dxgml_op.constant(#dxgml.constant_resource<_RDB1.conv1.weight : !dxgml.tensor<32x32x3x3x!dxgml.float16>>)
    %_RDB1.conv1.bias = dxgml_op.constant(#dxgml.constant_resource<_RDB1.conv1.bias : !dxgml.tensor<32x!dxgml.float16>>)
    %_RDB1.conv2.weight = dxgml_op.constant(#dxgml.constant_resource<_RDB1.conv2.weight : !dxgml.tensor<32x32x3x3x!dxgml.float16>>)
    %_RDB1.conv2.bias = dxgml_op.constant(#dxgml.constant_resource<_RDB1.conv2.bias : !dxgml.tensor<32x!dxgml.float16>>)
    %_RDB1.conv3.weight = dxgml_op.constant(#dxgml.constant_resource<_RDB1.conv3.weight : !dxgml.tensor<32x32x3x3x!dxgml.float16>>)
    %_RDB1.conv3.bias = dxgml_op.constant(#dxgml.constant_resource<_RDB1.conv3.bias : !dxgml.tensor<32x!dxgml.float16>>)
    %_RDB2.conv1.weight = dxgml_op.constant(#dxgml.constant_resource<_RDB2.conv1.weight : !dxgml.tensor<32x32x3x3x!dxgml.float16>>)
    %_RDB2.conv1.bias = dxgml_op.constant(#dxgml.constant_resource<_RDB2.conv1.bias : !dxgml.tensor<32x!dxgml.float16>>)
    %_RDB2.conv2.weight = dxgml_op.constant(#dxgml.constant_resource<_RDB2.conv2.weight : !dxgml.tensor<32x32x3x3x!dxgml.float16>>)
    %_RDB2.conv2.bias = dxgml_op.constant(#dxgml.constant_resource<_RDB2.conv2.bias : !dxgml.tensor<32x!dxgml.float16>>)
    %_RDB2.conv3.weight = dxgml_op.constant(#dxgml.constant_resource<_RDB2.conv3.weight : !dxgml.tensor<32x32x3x3x!dxgml.float16>>)
    %_RDB2.conv3.bias = dxgml_op.constant(#dxgml.constant_resource<_RDB2.conv3.bias : !dxgml.tensor<32x!dxgml.float16>>)
    %_RDB3.conv1.weight = dxgml_op.constant(#dxgml.constant_resource<_RDB3.conv1.weight : !dxgml.tensor<32x32x3x3x!dxgml.float16>>)
    %_RDB3.conv1.bias = dxgml_op.constant(#dxgml.constant_resource<_RDB3.conv1.bias : !dxgml.tensor<32x!dxgml.float16>>)
    %_RDB3.conv2.weight = dxgml_op.constant(#dxgml.constant_resource<_RDB3.conv2.weight : !dxgml.tensor<32x32x3x3x!dxgml.float16>>)
    %_RDB3.conv2.bias = dxgml_op.constant(#dxgml.constant_resource<_RDB3.conv2.bias : !dxgml.tensor<32x!dxgml.float16>>)
    %_RDB3.conv3.weight = dxgml_op.constant(#dxgml.constant_resource<_RDB3.conv3.weight : !dxgml.tensor<32x32x3x3x!dxgml.float16>>)
    %_RDB3.conv3.bias = dxgml_op.constant(#dxgml.constant_resource<_RDB3.conv3.bias : !dxgml.tensor<32x!dxgml.float16>>)
    %_conv_post.weight = dxgml_op.constant(#dxgml.constant_resource<_conv_post.weight : !dxgml.tensor<96x32x3x3x!dxgml.float16>>)
    %_conv_post.bias = dxgml_op.constant(#dxgml.constant_resource<_conv_post.bias : !dxgml.tensor<96x!dxgml.float16>>)
    %_conv_final.weight = dxgml_op.constant(#dxgml.constant_resource<_conv_final.weight : !dxgml.tensor<16x96x1x1x!dxgml.float16>>)
    %_conv_final.bias = dxgml_op.constant(#dxgml.constant_resource<_conv_final.bias : !dxgml.tensor<16x!dxgml.float16>>)
    %0 = dxgml_op.convolution(%arg0, %_conv1.weight, %_conv1.bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>, 
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      strides = #dxgml.dense_integer_elements<[2, 2]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x4x2160x3840x!dxgml.float16>, !dxgml.tensor<32x4x3x3x!dxgml.float16>, !dxgml.tensor<32x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %1 = dxgml_op.convolution(%0, %_RDB1.conv1.weight, %_RDB1.conv1.bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>, 
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      strides = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<32x32x3x3x!dxgml.float16>, !dxgml.tensor<32x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %2 = dxgml_op.relu(%1) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %3 = dxgml_op.add(%2, %0) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %4 = dxgml_op.convolution(%3, %_RDB1.conv2.weight, %_RDB1.conv2.bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>, 
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      strides = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<32x32x3x3x!dxgml.float16>, !dxgml.tensor<32x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %5 = dxgml_op.relu(%4) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %6 = dxgml_op.add(%3, %5) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %7 = dxgml_op.convolution(%6, %_RDB1.conv3.weight, %_RDB1.conv3.bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>, 
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      strides = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<32x32x3x3x!dxgml.float16>, !dxgml.tensor<32x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %8 = dxgml_op.add(%7, %0) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %9 = dxgml_op.convolution(%8, %_RDB2.conv1.weight, %_RDB2.conv1.bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>, 
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      strides = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<32x32x3x3x!dxgml.float16>, !dxgml.tensor<32x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %10 = dxgml_op.relu(%9) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %11 = dxgml_op.add(%10, %8) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %12 = dxgml_op.convolution(%11, %_RDB2.conv2.weight, %_RDB2.conv2.bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>, 
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      strides = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<32x32x3x3x!dxgml.float16>, !dxgml.tensor<32x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %13 = dxgml_op.relu(%12) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %14 = dxgml_op.add(%11, %13) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %15 = dxgml_op.convolution(%14, %_RDB2.conv3.weight, %_RDB2.conv3.bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>, 
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      strides = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<32x32x3x3x!dxgml.float16>, !dxgml.tensor<32x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %16 = dxgml_op.add(%15, %8) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %17 = dxgml_op.convolution(%16, %_RDB3.conv1.weight, %_RDB3.conv1.bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>, 
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      strides = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<32x32x3x3x!dxgml.float16>, !dxgml.tensor<32x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %18 = dxgml_op.relu(%17) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %19 = dxgml_op.add(%18, %16) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %20 = dxgml_op.convolution(%19, %_RDB3.conv2.weight, %_RDB3.conv2.bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>, 
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      strides = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<32x32x3x3x!dxgml.float16>, !dxgml.tensor<32x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %21 = dxgml_op.relu(%20) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %22 = dxgml_op.add(%19, %21) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %23 = dxgml_op.convolution(%22, %_RDB3.conv3.weight, %_RDB3.conv3.bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>, 
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      strides = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<32x32x3x3x!dxgml.float16>, !dxgml.tensor<32x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %24 = dxgml_op.add(%23, %16) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %25 = dxgml_op.convolution(%24, %_conv_post.weight, %_conv_post.bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>, 
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      strides = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<96x32x3x3x!dxgml.float16>, !dxgml.tensor<96x!dxgml.float16>) -> !dxgml.tensor<1x96x1080x1920x!dxgml.float16>
    %26 = dxgml_op.relu(%25) : (!dxgml.tensor<1x96x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x96x1080x1920x!dxgml.float16>
    %27 = dxgml_op.convolution(%26, %_conv_final.weight, %_conv_final.bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>, 
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>, 
      end_padding = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>, 
      strides = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x96x1080x1920x!dxgml.float16>, !dxgml.tensor<16x96x1x1x!dxgml.float16>, !dxgml.tensor<16x!dxgml.float16>) -> !dxgml.tensor<1x16x1080x1920x!dxgml.float16>
     %28 = dxgml_op.depth_to_space(%27) {
      block_size = #dxgml.integer<2 : !dxgml.int64>,
      depth_space_order = #dxgml_op.depth_space_order_enum_attr<depth_space_order_column_row_depth>
     } : (!dxgml.tensor<1x16x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x4x2160x3840x!dxgml.float16>
     dxgml.return %28 : !dxgml.tensor<1x4x2160x3840x!dxgml.float16>
  }
}

}