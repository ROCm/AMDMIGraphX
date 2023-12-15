
#include <onnx_test.hpp>

TEST_CASE(pad_4arg_invalid_axes_error_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("pad_4arg_invalid_axes_error_test.onnx"); }));
}
