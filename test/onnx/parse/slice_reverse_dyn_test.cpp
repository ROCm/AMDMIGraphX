
#include <onnx_test.hpp>

TEST_CASE(slice_reverse_dyn_test)
{
    // A slice command with negative step on any axis will have a "Reverse" instruction added in
    // parsing. At the time of writing, Reverse doesn't support dynamic shape input.
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    EXPECT(test::throws([&] { migraphx::parse_onnx("slice_reverse_dyn_test.onnx", options); }));
}
