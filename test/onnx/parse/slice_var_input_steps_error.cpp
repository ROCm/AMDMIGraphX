
#include <onnx_test.hpp>

TEST_CASE(slice_var_input_steps_error)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("slice_var_input_steps_error.onnx"); }));
}
