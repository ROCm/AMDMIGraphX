
#include <onnx_test.hpp>


TEST_CASE(conv_transpose_dyn_asym_padding_error)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    EXPECT(test::throws(
        [&] { migraphx::parse_onnx("conv_transpose_dyn_asym_padding_test.onnx", options); }));
}


