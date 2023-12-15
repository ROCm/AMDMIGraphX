
#include <onnx_test.hpp>

TEST_CASE(averagepool_dyn_cip_error_test)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    EXPECT(test::throws(
        [&] { migraphx::parse_onnx("averagepool_dyn_cip_error_test.onnx", options); }));
}
