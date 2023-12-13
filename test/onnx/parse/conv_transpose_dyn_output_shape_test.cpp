
#include <onnx_test.hpp>


TEST_CASE(conv_transpose_dyn_output_shape_error)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    EXPECT(test::throws(
        [&] { migraphx::parse_onnx("conv_transpose_dyn_output_shape_test.onnx", options); }));
}


