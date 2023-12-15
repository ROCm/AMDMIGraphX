
#include <onnx_test.hpp>


TEST_CASE(matmul_dyn_broadcast_error)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    EXPECT(test::throws([&] { migraphx::parse_onnx("matmul_dyn_broadcast_error.onnx", options); }));
}


