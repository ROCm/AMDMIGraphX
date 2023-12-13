
#include <onnx_test.hpp>

TEST_CASE(conv_transpose_auto_pad_error)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("conv_transpose_auto_pad_test.onnx"); }));
}
