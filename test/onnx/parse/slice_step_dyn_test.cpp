
#include <onnx_test.hpp>

TEST_CASE(slice_step_dyn_test)
{
    // A slice command with non-default steps will have a "Step" instruction added in parsing.
    // At the time of writing, Step doesn't support dynamic shape input.
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    EXPECT(test::throws([&] { migraphx::parse_onnx("slice_step_dyn_test.onnx", options); }));
}
