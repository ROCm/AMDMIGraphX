
#include <onnx_test.hpp>

TEST_CASE(shape_end_less_start_error)
{
    migraphx::onnx_options options;
    options.map_dyn_input_dims["x"] = {{1, 4, {1, 4}}, {4, 4}, {2, 4}, {2, 4}};
    EXPECT(test::throws([&] { migraphx::parse_onnx("shape_end_less_start_error.onnx", options); }));
}
