
#include <onnx_test.hpp>

TEST_CASE(pad_dyn_reflect_error)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {2, 4, {2}};
    EXPECT(test::throws([&] { migraphx::parse_onnx("pad_dyn_reflect_error.onnx", options); }));
}
