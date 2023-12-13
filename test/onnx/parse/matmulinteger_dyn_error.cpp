
#include <onnx_test.hpp>

TEST_CASE(matmulinteger_dyn_error)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    EXPECT(test::throws([&] { migraphx::parse_onnx("matmulinteger_dyn_error.onnx", options); }));
}
