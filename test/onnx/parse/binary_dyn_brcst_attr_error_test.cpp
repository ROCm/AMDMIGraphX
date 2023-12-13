
#include <onnx_test.hpp>

TEST_CASE(binary_dyn_brcst_attr_error_test)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    EXPECT(test::throws(
        [&] { migraphx::parse_onnx("binary_dyn_brcst_attr_error_test.onnx", options); }));
}
